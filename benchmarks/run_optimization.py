"""Search one ESA satellite formation with orbital preconditioning and evolution."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.evaluation.cases import CASE_ORDER, get_case


EXPORT_REFERENCE = """
import sys
import numpy as np
from src.evaluation.cases import make_reference_problem

udp = make_reference_problem(sys.argv[1])
lower, upper = udp.get_bounds()
weighted_maps = []
for observation, stm in enumerate(udp.stms):
    inflation = udp.inflation_factor if observation else 1.0
    weighted_maps.append(stm[:3, :] * (udp.grid_size / (2.0 * inflation)))
dynamics = np.concatenate(weighted_maps, axis=0)
_, singular_values, right_vectors = np.linalg.svd(dynamics, full_matrices=False)
floor = singular_values.max() * 1e-8
mutation_map = right_vectors.T * (1.0 / np.maximum(singular_values, floor))[None, :]
np.savez_compressed(
    sys.argv[2],
    stms=udp.stms,
    n_sat=udp.n_sat,
    grid_size=udp.grid_size,
    n_meas=udp.n_meas,
    scaling_factor=udp.scaling_factor,
    inflation_factor=udp.inflation_factor,
    lower_bounds=lower,
    upper_bounds=upper,
    mutation_map=mutation_map,
)
"""

VERIFY_REFERENCE = """
import sys
import numpy as np
from src.evaluation.cases import make_reference_problem
from src.evaluation.objectives import observation_scores

udp = make_reference_problem(sys.argv[1])
with np.load(sys.argv[2]) as saved:
    x = np.asarray(saved["x"], dtype=np.float64)
scores = observation_scores(x, udp)
reference_fitness = float(udp.fitness(x.tolist())[0])
score_fitness = -float(scores.min())
np.savez_compressed(
    sys.argv[3],
    reference_fitness=reference_fitness,
    score_fitness=score_fitness,
    observation_scores=scores,
)
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_ORDER, default="hard")
    parser.add_argument("--iterations", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument(
        "--algorithm",
        choices=("de", "genetic", "block-anneal"),
        default="de",
        help="GPU-batched differential evolution, genetic baseline, or block annealing",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="stop once the best ESA fitness reaches this value or lower",
    )
    parser.add_argument(
        "--resume-rng",
        type=Path,
        default=None,
        help="continue the random stream recorded in a search checkpoint",
    )
    parser.add_argument("--velocity-bound", type=float, default=10.0)
    parser.add_argument(
        "--restart-after",
        type=int,
        default=100,
        help="generations without improvement before refreshing the weaker half",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="parent-selection spread; larger values preserve more population diversity",
    )
    return parser.parse_args()


def random_candidates(count, lower, upper, n_sat, velocity_bound, rng):
    """Draw bounded formations with a stratified spread of velocity magnitudes."""
    candidates = rng.uniform(lower, upper, size=(count, lower.size))
    scales = rng.choice(
        np.asarray([0.05, 0.1, 0.2, 0.4, 0.7, 1.0]),
        size=count,
        p=[0.08, 0.14, 0.22, 0.24, 0.20, 0.12],
    )
    velocity_scales = velocity_bound * scales
    candidates[:, 3 * n_sat :] = rng.uniform(
        -1.0, 1.0, size=(count, 3 * n_sat)
    ) * velocity_scales[:, None]
    return candidates


def canonicalize_candidates(candidates, n_sat):
    """Sort interchangeable satellites by their initial x position.

    Satellite labels do not affect the evaluator. Canonical ordering keeps
    population crossover and differential moves from combining unrelated
    satellites solely because they happened to share an array index.
    """
    values = np.asarray(candidates, dtype=np.float64)
    one_candidate = values.ndim == 1
    batch = values[None, :] if one_candidate else values
    states = batch.reshape(len(batch), 6, n_sat).transpose(0, 2, 1)
    order = np.argsort(states[:, :, 0], axis=1, kind="stable")
    states = np.take_along_axis(states, order[:, :, None], axis=1)
    result = states.transpose(0, 2, 1).reshape(len(batch), 6 * n_sat)
    return result[0] if one_candidate else result


def mutate_candidates(
    parents,
    lower,
    upper,
    n_sat,
    mutation_map,
    rng,
    temperature=1.0,
    step_multiplier=1.0,
):
    """Mutate whole satellite states in coordinates scaled by orbital sensitivity."""
    count = len(parents)
    genes = parents.reshape(count, 6, n_sat).transpose(0, 2, 1).copy()

    mutation_fractions = rng.choice(
        np.asarray([0.025, 0.05, 0.1, 0.2, 0.4, 0.75]),
        size=count,
        p=[0.16, 0.19, 0.23, 0.20, 0.16, 0.06],
    )
    changed = rng.random((count, n_sat)) < mutation_fractions[:, None]
    empty = np.flatnonzero(~changed.any(axis=1))
    if empty.size:
        changed[empty, rng.integers(0, n_sat, size=empty.size)] = True

    cell_steps = np.exp(
        rng.uniform(np.log(0.35), np.log(8.0), size=count)
    ) * np.clip(temperature, 0.15, 2.0) * step_multiplier
    noise = rng.normal(size=(count, n_sat, 6))
    scaled_noise = np.einsum("bnj,ij->bni", noise, mutation_map)
    genes += scaled_noise * cell_steps[:, None, None] * changed[..., None]

    # Occasionally replace one satellite state to create a genuinely new branch.
    reset = rng.random((count, n_sat)) < 0.012
    reset_rows, reset_satellites = np.nonzero(reset)
    if reset_rows.size:
        reset_values = random_candidates(
            reset_rows.size, lower, upper, n_sat, 1.0, rng
        ).reshape(-1, 6, n_sat).transpose(0, 2, 1)
        genes[reset_rows, reset_satellites] = reset_values[
            np.arange(reset_rows.size), reset_satellites
        ]

    candidates = genes.transpose(0, 2, 1).reshape(count, 6 * n_sat)
    return np.clip(candidates, lower, upper)


def make_offspring(
    population,
    fitnesses,
    count,
    lower,
    upper,
    n_sat,
    velocity_bound,
    mutation_map,
    rng,
    temperature,
    step_multiplier,
):
    """Breed a batch using rank selection, satellite-level crossover and mutation."""
    population_size = len(population)
    rank_order = np.argsort(fitnesses, kind="stable")
    rank_weights = np.exp(
        -np.arange(population_size)
        / max(1.0, population_size * np.clip(temperature, 0.05, 2.0))
    )
    rank_weights /= rank_weights.sum()

    first = rng.choice(rank_order, size=count, p=rank_weights)
    second = rng.choice(rank_order, size=count, p=rank_weights)
    children = population[first].copy().reshape(count, 6, n_sat).transpose(0, 2, 1)
    other = population[second].reshape(count, 6, n_sat).transpose(0, 2, 1)

    crossover_rates = rng.choice(
        np.asarray([0.05, 0.15, 0.3, 0.5, 0.75]),
        size=count,
        p=[0.12, 0.2, 0.28, 0.26, 0.14],
    )
    crossover = rng.random((count, n_sat)) < crossover_rates[:, None]
    children[crossover] = other[crossover]
    children = children.transpose(0, 2, 1).reshape(count, 6 * n_sat)
    children = mutate_candidates(
        children,
        lower,
        upper,
        n_sat,
        mutation_map,
        rng,
        temperature=max(temperature, 0.25),
        step_multiplier=step_multiplier,
    )

    immigrant_count = max(1, count // 10)
    immigrant_rows = rng.choice(count, size=immigrant_count, replace=False)
    children[immigrant_rows] = random_candidates(
        immigrant_count, lower, upper, n_sat, velocity_bound, rng
    )
    return children, fitnesses[first], immigrant_rows


def select_population(candidates, fitnesses, population_size, rng):
    """Keep strong elites plus weighted near-best candidates for diversity."""
    tie_break = rng.random(len(fitnesses)) * 1e-10
    order = np.lexsort((tie_break, fitnesses))
    elite_count = max(1, population_size // 2)
    elites = order[:elite_count]

    pool_end = min(len(order), population_size * 4)
    pool = order[elite_count:pool_end]
    needed = population_size - elite_count
    if needed and len(pool) >= needed:
        ranks = np.arange(len(pool), dtype=np.float64)
        weights = np.exp(-ranks / max(1.0, population_size * 0.35))
        weights /= weights.sum()
        diverse = rng.choice(pool, size=needed, replace=False, p=weights)
        selected = np.concatenate((elites, diverse))
    else:
        selected = order[:population_size]
    return candidates[selected].copy(), fitnesses[selected].copy()


def make_de_offspring(
    population,
    count,
    lower,
    upper,
    n_sat,
    velocity_bound,
    rng,
    step_multiplier,
    best_x,
):
    """Generate diverse differential-evolution trials, crossing whole satellites."""
    population_size = len(population)
    random_keys = rng.random((count, population_size))
    indices = np.argsort(random_keys, axis=1)[:, :4]
    targets, first, second, third = indices.T

    factors = rng.uniform(0.35, 1.25, size=count) * step_multiplier
    mutant = population[first] + factors[:, None] * (
        population[second] - population[third]
    )

    current_to_best = rng.random(count) < 0.5
    if current_to_best.any():
        target_x = population[targets[current_to_best]]
        mutant[current_to_best] = target_x + factors[current_to_best, None] * (
            best_x - target_x
        ) + factors[current_to_best, None] * (
            population[first[current_to_best]] - population[second[current_to_best]]
        )

    rates = rng.choice(
        np.asarray([0.05, 0.1, 0.25, 0.5, 0.9]),
        size=count,
        p=[0.10, 0.18, 0.28, 0.28, 0.16],
    )
    satellite_mask = rng.random((count, n_sat)) < rates[:, None]
    empty = np.flatnonzero(~satellite_mask.any(axis=1))
    if empty.size:
        satellite_mask[empty, rng.integers(0, n_sat, size=empty.size)] = True

    targets_x = population[targets].reshape(count, 6, n_sat).transpose(0, 2, 1)
    mutant_x = mutant.reshape(count, 6, n_sat).transpose(0, 2, 1)
    trials = targets_x.copy()
    trials[satellite_mask] = mutant_x[satellite_mask]
    trials = np.clip(
        trials.transpose(0, 2, 1).reshape(count, 6 * n_sat), lower, upper
    )

    immigrant_count = max(1, count // 10)
    immigrant_rows = rng.choice(count, size=immigrant_count, replace=False)
    trials[immigrant_rows] = random_candidates(
        immigrant_count, lower, upper, n_sat, velocity_bound, rng
    )
    return trials, targets, immigrant_rows


def make_block_proposals(
    parents,
    donors,
    lower,
    upper,
    n_sat,
    mutation_map,
    rng,
    temperature,
):
    """Perturb a small number of complete satellite states at a time.

    The score is based on discretized grid occupancies, so changing many
    satellites at once often destroys useful baselines. These proposals make
    mostly one- or two-satellite moves, while occasional larger moves and
    donor-state replacements allow the search to escape local plateaus.
    """
    count = len(parents)
    genes = parents.reshape(count, 6, n_sat).transpose(0, 2, 1).copy()
    donor_genes = donors.reshape(count, 6, n_sat).transpose(0, 2, 1)

    move_sizes = rng.choice(
        np.asarray([1, 2, 3, 5, 8, max(1, n_sat // 2)]),
        size=count,
        p=[0.43, 0.24, 0.14, 0.10, 0.06, 0.03],
    )
    changed = np.zeros((count, n_sat), dtype=bool)
    keys = rng.random((count, n_sat))
    order = np.argsort(keys, axis=1)
    rows = np.arange(count)[:, None]
    changed[rows, order[:, : int(move_sizes.max())]] = (
        np.arange(int(move_sizes.max()))[None, :] < move_sizes[:, None]
    )

    steps = np.exp(rng.uniform(np.log(0.1), np.log(12.0), size=count))
    steps *= np.sqrt(np.clip(temperature / 0.12, 0.08, 1.5))
    noise = rng.normal(size=(count, n_sat, 6))
    scaled_noise = np.einsum("bnj,ij->bni", noise, mutation_map)
    genes += scaled_noise * steps[:, None, None] * changed[..., None]

    # Replace one selected satellite with a donor's whole six-dimensional state.
    donor_rows = np.flatnonzero(rng.random(count) < 0.12)
    if donor_rows.size:
        satellite = rng.integers(0, n_sat, size=donor_rows.size)
        genes[donor_rows, satellite] = donor_genes[donor_rows, satellite]

    proposals = genes.transpose(0, 2, 1).reshape(count, 6 * n_sat)
    return np.clip(proposals, lower, upper)


def main():
    args = parse_args()
    if min(
        args.iterations,
        args.batch_size,
        args.population_size,
        args.restart_after,
    ) <= 0:
        raise ValueError(
            "iterations, batch-size, population-size, and restart-after must be positive."
        )
    if args.velocity_bound <= 0 or args.temperature <= 0:
        raise ValueError("velocity-bound and temperature must be positive.")

    case_id = args.case
    spec = get_case(case_id)
    rng = np.random.default_rng(args.seed)
    if args.resume_rng is not None:
        with np.load(args.resume_rng, allow_pickle=False) as saved_state:
            if "seed" in saved_state.files and int(saved_state["seed"]) != args.seed:
                raise ValueError("RNG checkpoint seed does not match --seed.")
            rng.bit_generator.state = json.loads(
                str(saved_state["rng_state"].item())
            )
        print(f"Continuing RNG stream from: {args.resume_rng}", flush=True)
    best_path = Path("results/cases") / f"{case_id}_min_best.npz"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="esa_gpu_search_") as temp_dir:
        temp_path = Path(temp_dir)
        cache_path = temp_path / "reference.npz"
        candidate_path = temp_path / "candidate.npz"
        verified_path = temp_path / "verified.npz"

        # Construct Heyoka STMs in a short-lived CPU process to avoid OpenMP conflicts.
        subprocess.run(
            [sys.executable, "-c", EXPORT_REFERENCE, case_id, str(cache_path)],
            check=True,
        )

        from src.evaluation.gpu import gpu_fitness_batch
        import torch

        with np.load(cache_path) as data:
            n_sat = int(data["n_sat"])
            lower = np.asarray(data["lower_bounds"], dtype=np.float64)
            upper = np.asarray(data["upper_bounds"], dtype=np.float64)
            mutation_map = np.asarray(data["mutation_map"], dtype=np.float64)
            reference = SimpleNamespace(
                stms=np.asarray(data["stms"], dtype=np.float64),
                n_sat=n_sat,
                grid_size=int(data["grid_size"]),
                n_meas=int(data["n_meas"]),
                scaling_factor=float(data["scaling_factor"]),
                inflation_factor=float(data["inflation_factor"]),
            )

        search_lower = lower.copy()
        search_upper = upper.copy()
        search_lower[3 * n_sat :] = np.maximum(
            search_lower[3 * n_sat :], -args.velocity_bound
        )
        search_upper[3 * n_sat :] = np.minimum(
            search_upper[3 * n_sat :], args.velocity_bound
        )
        def score_batch(candidates):
            return gpu_fitness_batch(candidates, reference).detach().cpu().numpy()

        if best_path.exists():
            with np.load(best_path) as saved:
                best_x = np.asarray(saved["x"], dtype=np.float64)
            best_x = canonicalize_candidates(best_x, n_sat)
            best_x = np.clip(best_x, search_lower, search_upper)
            best_fitness = float(score_batch(best_x)[0])
            print(f"Seeding from saved best: {best_path}", flush=True)
        else:
            best_x = random_candidates(
                1, search_lower, search_upper, n_sat, args.velocity_bound, rng
            )[0]
            best_fitness = float(score_batch(best_x)[0])

        initial_fitness = best_fitness
        population = random_candidates(
            args.population_size,
            search_lower,
            search_upper,
            n_sat,
            args.velocity_bound,
            rng,
        )
        population = canonicalize_candidates(population, n_sat)

        warm_count = max(1, args.population_size // 3)
        warm_parents = np.repeat(best_x[None, :], warm_count, axis=0)
        population[:warm_count] = mutate_candidates(
            warm_parents,
            search_lower,
            search_upper,
            n_sat,
            mutation_map,
            rng,
            temperature=1.0,
        )
        population[0] = best_x
        population = canonicalize_candidates(population, n_sat)
        population_fitness = score_batch(population)
        best_found_at = 0
        scored = len(population)
        stagnant_generations = 0
        step_multiplier = 1.0

        improved = int(np.argmin(population_fitness))
        if population_fitness[improved] < best_fitness:
            best_x = population[improved].copy()
            best_fitness = float(population_fitness[improved])
            best_found_at = scored

        print("Search:", args.algorithm, "| GPU-batched population")
        print("Case:", case_id, "| ESA problem:", spec["problem_id"])
        print("Objective: minimize ESA reference fitness")
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Population:", args.population_size,
            "| Batch:", args.batch_size,
            "| Velocity bound:", args.velocity_bound,
            "| Seed:", args.seed,
        )
        print("Starting fitness:", initial_fitness, flush=True)

        started = time.perf_counter()
        generation = 0
        while scored < args.iterations and (
            args.target_score is None or best_fitness > args.target_score
        ):
            generation += 1
            batch_size = min(args.batch_size, args.iterations - scored)
            if args.algorithm == "de":
                children, target_indices, immigrant_rows = make_de_offspring(
                    population,
                    batch_size,
                    search_lower,
                    search_upper,
                    n_sat,
                    args.velocity_bound,
                    rng,
                    step_multiplier,
                    best_x,
                )
                parent_fitness = population_fitness[target_indices]
            elif args.algorithm == "block-anneal":
                # One proposal per walker per generation keeps the Metropolis
                # accept/reject step unambiguous and avoids overwriting a
                # walker twice in the same GPU batch.
                batch_size = min(batch_size, args.population_size)
                target_indices = rng.choice(
                    args.population_size, size=batch_size, replace=False
                )
                donor_indices = rng.integers(
                    0, args.population_size, size=batch_size
                )
                cycle_length = max(40, args.restart_after * 4)
                cycle_position = (generation - 1) % cycle_length
                cycle_fraction = cycle_position / cycle_length
                current_temperature = args.temperature * (
                    0.05 + 0.95 * (1.0 - cycle_fraction) ** 2
                )
                children = make_block_proposals(
                    population[target_indices],
                    population[donor_indices],
                    search_lower,
                    search_upper,
                    n_sat,
                    mutation_map,
                    rng,
                    current_temperature,
                )
                parent_fitness = population_fitness[target_indices]
                immigrant_rows = np.empty(0, dtype=np.int64)
            else:
                children, parent_fitness, immigrant_rows = make_offspring(
                    population,
                    population_fitness,
                    batch_size,
                    search_lower,
                    search_upper,
                    n_sat,
                    args.velocity_bound,
                    mutation_map,
                    rng,
                    args.temperature,
                    step_multiplier,
                )
                target_indices = None
            if args.algorithm in ("de", "genetic", "block-anneal"):
                children = canonicalize_candidates(children, n_sat)
            child_fitness = score_batch(children)
            scored += batch_size

            eligible = np.ones(batch_size, dtype=bool)
            eligible[immigrant_rows] = False
            success_rate = float(
                np.mean(child_fitness[eligible] < parent_fitness[eligible])
            )

            child_best = int(np.argmin(child_fitness))
            if child_fitness[child_best] < best_fitness:
                best_x = children[child_best].copy()
                best_fitness = float(child_fitness[child_best])
                best_found_at = scored
                stagnant_generations = 0
                step_multiplier = max(0.5, step_multiplier * 0.85)
                print(
                    f"evaluations {scored:>9}: best fitness {best_fitness:.12g}",
                    flush=True,
                )
            else:
                stagnant_generations += 1
                if success_rate > 0.02:
                    step_multiplier = max(0.5, step_multiplier * 0.97)
                else:
                    # Widen exploration during a plateau instead of shrinking to
                    # tiny moves that cannot escape the current grid pattern.
                    step_multiplier = min(8.0, step_multiplier * np.exp(0.02))

            if args.algorithm == "de":
                for target in np.unique(target_indices):
                    trial_rows = np.flatnonzero(target_indices == target)
                    candidate_row = trial_rows[
                        np.argmin(child_fitness[trial_rows])
                    ]
                    if child_fitness[candidate_row] < population_fitness[target]:
                        population[target] = children[candidate_row]
                        population_fitness[target] = child_fitness[candidate_row]
            elif args.algorithm == "block-anneal":
                deltas = child_fitness - parent_fitness
                acceptance = np.ones(batch_size, dtype=bool)
                worse = deltas > 0
                acceptance[worse] = (
                    rng.random(int(worse.sum()))
                    < np.exp(-deltas[worse] / max(current_temperature, 1e-12))
                )
                accepted_rows = np.flatnonzero(acceptance)
                accepted_targets = target_indices[accepted_rows]
                population[accepted_targets] = children[accepted_rows]
                population_fitness[accepted_targets] = child_fitness[accepted_rows]
            elif args.algorithm == "genetic":
                combined_x = np.concatenate((population, children), axis=0)
                combined_fitness = np.concatenate((population_fitness, child_fitness))
                population, population_fitness = select_population(
                    combined_x, combined_fitness, args.population_size, rng
                )

            if stagnant_generations >= args.restart_after:
                refresh_count = max(1, args.population_size // 2)
                refreshed = random_candidates(
                    refresh_count,
                    search_lower,
                    search_upper,
                    n_sat,
                    args.velocity_bound,
                    rng,
                )
                refreshed = canonicalize_candidates(refreshed, n_sat)
                local_count = refresh_count // 2
                if local_count:
                    refreshed[:local_count] = mutate_candidates(
                        np.repeat(best_x[None, :], local_count, axis=0),
                        search_lower,
                        search_upper,
                        n_sat,
                        mutation_map,
                        rng,
                        temperature=1.5,
                    )
                    refreshed = canonicalize_candidates(refreshed, n_sat)
                refreshed_fitness = score_batch(refreshed)
                scored += len(refreshed)
                if args.algorithm in ("de", "block-anneal"):
                    worst = np.argsort(population_fitness)[-refresh_count:]
                    population[worst] = refreshed
                    population_fitness[worst] = refreshed_fitness
                else:
                    combined_x = np.concatenate((population, refreshed), axis=0)
                    combined_fitness = np.concatenate(
                        (population_fitness, refreshed_fitness)
                    )
                    population, population_fitness = select_population(
                        combined_x, combined_fitness, args.population_size, rng
                    )
                refreshed_best = int(np.argmin(refreshed_fitness))
                if refreshed_fitness[refreshed_best] < best_fitness:
                    best_x = refreshed[refreshed_best].copy()
                    best_fitness = float(refreshed_fitness[refreshed_best])
                    best_found_at = scored
                    print(
                        f"restart at {scored:>9}: best fitness {best_fitness:.12g}",
                        flush=True,
                    )
                stagnant_generations = 0

            if generation % 25 == 0 or scored >= args.iterations:
                print(
                    f"progress {scored}/{args.iterations}; "
                    f"best fitness {best_fitness:.12g}; "
                    f"population best {population_fitness.min():.12g}; "
                    + (
                        f"temperature {current_temperature:.4g}"
                        if args.algorithm == "block-anneal"
                        else f"mutation scale {step_multiplier:.3f}"
                    ),
                    flush=True,
                )

        if args.target_score is not None and best_fitness <= args.target_score:
            print(f"Target reached: {best_fitness:.12g} <= {args.target_score}", flush=True)

        np.savez_compressed(candidate_path, x=best_x)
        subprocess.run(
            [
                sys.executable,
                "-c",
                VERIFY_REFERENCE,
                case_id,
                str(candidate_path),
                str(verified_path),
            ],
            check=True,
        )
        with np.load(verified_path) as verified:
            cpu_fitness = float(verified["reference_fitness"])
            score_fitness = float(verified["score_fitness"])
            observation_values = np.asarray(verified["observation_scores"])

    difference = abs(best_fitness - cpu_fitness)
    if difference > 1e-10 or abs(cpu_fitness - score_fitness) > 1e-10:
        raise RuntimeError(
            f"GPU/CPU score mismatch for {case_id}: {best_fitness} vs {cpu_fitness}."
        )

    np.savez_compressed(
        best_path,
        x=best_x,
        fitness=cpu_fitness,
        observation_scores=observation_values,
        initial_fitness=initial_fitness,
        best_found_at=best_found_at,
        seed=args.seed,
        iterations=args.iterations,
        batch_size=args.batch_size,
        population_size=args.population_size,
        velocity_bound=args.velocity_bound,
        restart_after=args.restart_after,
        temperature=args.temperature,
        objective="min",
        case_id=case_id,
    )

    print("CPU verification difference:", difference)
    print("Observation scores:", np.array2string(observation_values, precision=6))
    print("Evaluations:", scored)
    print("Elapsed seconds:", round(time.perf_counter() - started, 2))
    print("Saved best:", best_path)
    print("Final fitness:", cpu_fitness)


if __name__ == "__main__":
    main()

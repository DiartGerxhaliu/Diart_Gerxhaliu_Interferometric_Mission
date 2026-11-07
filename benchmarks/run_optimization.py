"""Search one ESA satellite formation with orbital preconditioning and evolution."""

import argparse
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
    parser.add_argument("--seed", type=int, default=43)
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


def mutate_candidates(
    parents, lower, upper, n_sat, mutation_map, rng, temperature=1.0
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
    ) * np.clip(temperature, 0.15, 2.0)
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
    )

    immigrant_count = max(1, count // 20)
    immigrant_rows = rng.choice(count, size=immigrant_count, replace=False)
    children[immigrant_rows] = random_candidates(
        immigrant_count, lower, upper, n_sat, velocity_bound, rng
    )
    return children


def select_population(candidates, fitnesses, population_size, rng):
    """Keep the strongest candidates, randomizing ties on the flat score landscape."""
    tie_break = rng.random(len(fitnesses)) * 1e-10
    selected = np.lexsort((tie_break, fitnesses))[:population_size]
    return candidates[selected].copy(), fitnesses[selected].copy()


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
        population_fitness = score_batch(population)
        best_found_at = 0
        scored = len(population)
        stagnant_generations = 0

        improved = int(np.argmin(population_fitness))
        if population_fitness[improved] < best_fitness:
            best_x = population[improved].copy()
            best_fitness = float(population_fitness[improved])
            best_found_at = scored

        print("Search: orbital-preconditioned evolutionary population")
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
        while scored < args.iterations:
            generation += 1
            batch_size = min(args.batch_size, args.iterations - scored)
            children = make_offspring(
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
            )
            child_fitness = score_batch(children)
            scored += batch_size

            child_best = int(np.argmin(child_fitness))
            if child_fitness[child_best] < best_fitness:
                best_x = children[child_best].copy()
                best_fitness = float(child_fitness[child_best])
                best_found_at = scored
                stagnant_generations = 0
                print(
                    f"evaluations {scored:>9}: best fitness {best_fitness:.12g}",
                    flush=True,
                )
            else:
                stagnant_generations += 1

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
                refreshed_fitness = score_batch(refreshed)
                scored += len(refreshed)
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
                    f"population best {population_fitness.min():.12g}",
                    flush=True,
                )

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

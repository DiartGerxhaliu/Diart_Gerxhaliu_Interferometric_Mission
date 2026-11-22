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

from benchmarks.build_submission import save_candidate_progress
from benchmarks.coverage_heuristic import (
    make_coverage_guided_offspring,
    satellite_coverage_guidance,
)
from benchmarks.lm_cma_es import (
    LimitedMemoryCMAES,
    default_population_size,
    reflect_to_unit_box,
)
from src.evaluation.cases import CASE_ORDER, get_case
from src.evaluation.constraints import (
    enforce_observation_bounds,
    is_inside_observation_bounds,
)


EXPORT_REFERENCE = """
import sys
import numpy as np
from src.evaluation.cases import make_reference_problem

udp = make_reference_problem(sys.argv[1])
lower, upper = udp.get_bounds()
position_maps = np.stack(
    [
        stm[:3, :] / (udp.inflation_factor if observation else 1.0)
        for observation, stm in enumerate(udp.stms)
    ]
)
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
    position_maps=position_maps,
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

DE_NEUTRAL_ACCEPTANCE = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_ORDER, default="hard")
    parser.add_argument("--iterations", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="population size (default: dimension-scaled for LM-CMA-ES, otherwise 256)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("de", "genetic", "block-anneal", "lm-cma-es", "heuristic"),
        default="de",
        help=(
            "GPU-batched differential evolution, genetic, block annealing, "
            "LM-CMA-ES, or coverage-guided heuristic search"
        ),
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
    parser.add_argument(
        "--save-rng-checkpoint",
        type=Path,
        default=None,
        help="save the progressed random stream when the run ends",
    )
    parser.add_argument(
        "--rng-jump",
        type=int,
        default=0,
        help="advance the seeded PCG64 stream before the next search segment",
    )
    parser.add_argument("--velocity-bound", type=float, default=10.0)
    parser.add_argument(
        "--restart-after",
        type=int,
        default=100,
        help=(
            "stagnation generations before population refresh; LM-CMA-ES uses "
            "the equivalent evaluation budget"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="parent-selection spread for genetic search",
    )
    parser.add_argument(
        "--anneal-temperature",
        type=float,
        default=0.005,
        help="base temperature for block annealing",
    )
    parser.add_argument(
        "--skip-cpu-verification",
        action="store_true",
        help="defer CPU verification to a separate launcher process",
    )
    return parser.parse_args()


def random_candidates(count, lower, upper, n_sat, velocity_bound, rng):
    """Draw bounded formations with broad coverage of velocity magnitudes."""
    candidates = rng.uniform(lower, upper, size=(count, lower.size))
    scales = rng.choice(
        np.asarray([0.1, 0.3, 0.5, 0.75, 1.0]),
        size=count,
        p=[0.10, 0.20, 0.20, 0.25, 0.25],
    )
    velocity_scales = velocity_bound * scales
    candidates[:, 3 * n_sat :] = rng.uniform(
        -1.0, 1.0, size=(count, 3 * n_sat)
    ) * velocity_scales[:, None]
    return candidates


def sample_orbital_perturbations(count, n_sat, mutation_map, rng):
    """Scale each satellite's six-state perturbation by orbital sensitivity."""
    mutation_map = np.asarray(mutation_map, dtype=np.float64)
    if mutation_map.shape != (6, 6):
        raise ValueError(
            f"Mutation map must have shape (6, 6), got {mutation_map.shape}."
        )
    noise = rng.normal(size=(count, n_sat, 6))
    return np.einsum("bnj,ij->bni", noise, mutation_map)


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
        p=[0.08, 0.12, 0.16, 0.20, 0.24, 0.20],
    )
    changed = rng.random((count, n_sat)) < mutation_fractions[:, None]
    empty = np.flatnonzero(~changed.any(axis=1))
    if empty.size:
        changed[empty, rng.integers(0, n_sat, size=empty.size)] = True

    cell_steps = np.exp(
        rng.uniform(np.log(0.35), np.log(8.0), size=count)
    ) * np.clip(temperature, 0.15, 2.0) * step_multiplier
    scaled_noise = sample_orbital_perturbations(
        count, n_sat, mutation_map, rng
    )
    genes += scaled_noise * cell_steps[:, None, None] * changed[..., None]

    # Occasionally replace one satellite state to create a genuinely new branch.
    reset = rng.random((count, n_sat)) < 0.025
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
    secondary_scores=None,
):
    """Breed a batch using rank selection, satellite-level crossover and mutation."""
    population_size = len(population)
    if secondary_scores is not None:
        secondary_scores = np.asarray(secondary_scores)
    rank_order = rank_fitnesses(fitnesses, rng, secondary_scores)
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
        p=[0.08, 0.12, 0.2, 0.25, 0.35],
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
    result = children, fitnesses[first], immigrant_rows
    if secondary_scores is not None:
        return result + (secondary_scores[first],)
    return result


def select_population(
    candidates,
    fitnesses,
    population_size,
    rng,
    secondary_scores=None,
):
    """Keep strong elites plus weighted near-best candidates for diversity."""
    if secondary_scores is not None:
        secondary_scores = np.asarray(secondary_scores)
    order = rank_fitnesses(fitnesses, rng, secondary_scores)
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
    result = candidates[selected].copy(), fitnesses[selected].copy()
    if secondary_scores is not None:
        return result + (secondary_scores[selected].copy(),)
    return result


def rank_fitnesses(fitnesses, rng, secondary_scores=None):
    """Rank by official fitness, then secondary progress, then random ties."""
    fitnesses = np.asarray(fitnesses)
    random_tie_breakers = rng.random(len(fitnesses))
    if secondary_scores is None:
        return np.lexsort((random_tie_breakers, fitnesses))
    secondary_scores = np.asarray(secondary_scores)
    if secondary_scores.shape != fitnesses.shape:
        raise ValueError("Fitnesses and secondary scores must have matching shapes.")
    return np.lexsort(
        (random_tie_breakers, -secondary_scores, fitnesses)
    )


def sample_de_indices(count, population_size, rng):
    """Draw four distinct population indices per trial without sorting the pool."""
    if population_size < 4:
        raise ValueError("Differential evolution requires population-size >= 4.")
    if count < 0:
        raise ValueError("Trial count must be non-negative.")

    indices = rng.integers(population_size, size=(count, 4))
    while count:
        duplicate_rows = (
            (indices[:, 0] == indices[:, 1])
            | (indices[:, 0] == indices[:, 2])
            | (indices[:, 0] == indices[:, 3])
            | (indices[:, 1] == indices[:, 2])
            | (indices[:, 1] == indices[:, 3])
            | (indices[:, 2] == indices[:, 3])
        )
        if not np.any(duplicate_rows):
            break
        indices[duplicate_rows] = rng.integers(
            population_size, size=(int(duplicate_rows.sum()), 4)
        )
    return indices.T


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
    targets, first, second, third = sample_de_indices(
        count, population_size, rng
    )

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
        p=[0.05, 0.10, 0.20, 0.30, 0.35],
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


def update_de_population(
    population,
    population_fitness,
    children,
    child_fitness,
    target_indices,
    rng,
    neutral_acceptance=DE_NEUTRAL_ACCEPTANCE,
    population_secondary=None,
    child_secondary=None,
):
    """Keep better trials per target, preferring secondary gains on score ties."""
    if (population_secondary is None) != (child_secondary is None):
        raise ValueError("Population and child secondary scores must be provided together.")
    if population_secondary is not None:
        population_secondary = np.asarray(population_secondary)
        child_secondary = np.asarray(child_secondary)
        if population_secondary.shape != population_fitness.shape:
            raise ValueError("Population secondary scores must match population fitness.")
        if child_secondary.shape != child_fitness.shape:
            raise ValueError("Child secondary scores must match child fitness.")
    novel = np.any(children != population[target_indices], axis=1)
    if population_secondary is None or child_secondary is None:
        order = np.lexsort(
            (rng.random(len(child_fitness)), ~novel, child_fitness, target_indices)
        )
    else:
        order = np.lexsort(
            (
                rng.random(len(child_fitness)),
                ~novel,
                -child_secondary,
                child_fitness,
                target_indices,
            )
        )
    ordered_targets = target_indices[order]
    best_rows = order[
        np.concatenate(
            (
                np.asarray([0]),
                np.flatnonzero(ordered_targets[1:] != ordered_targets[:-1]) + 1,
            )
        )
    ]
    best_targets = target_indices[best_rows]
    best_fitness = child_fitness[best_rows]
    target_fitness = population_fitness[best_targets]
    improving = best_fitness < target_fitness
    equal_fitness = best_fitness == target_fitness
    if population_secondary is None or child_secondary is None:
        secondary_improving = np.zeros(len(best_rows), dtype=bool)
        equal_secondary = np.ones(len(best_rows), dtype=bool)
    else:
        best_secondary = child_secondary[best_rows]
        target_secondary = population_secondary[best_targets]
        secondary_improving = equal_fitness & (best_secondary > target_secondary)
        equal_secondary = best_secondary == target_secondary
    neutral = (
        equal_fitness
        & equal_secondary
        & novel[best_rows]
        & (rng.random(len(best_rows)) < neutral_acceptance)
    )
    accepted = improving | secondary_improving | neutral
    accepted_targets = best_targets[accepted]
    accepted_rows = best_rows[accepted]
    population[accepted_targets] = children[accepted_rows]
    population_fitness[accepted_targets] = child_fitness[accepted_rows]
    if population_secondary is not None and child_secondary is not None:
        population_secondary[accepted_targets] = child_secondary[accepted_rows]


def adapt_de_step_multiplier(step_multiplier, success_rate):
    """Adjust DE's scale conservatively so plateau ties do not cause blowups."""
    if success_rate < 0.1:
        return min(2.0, step_multiplier * 1.03)
    if success_rate > 0.2:
        return max(0.25, step_multiplier * 0.9)
    return step_multiplier


def adapt_genetic_step_multiplier(step_multiplier, success_rate):
    """Change genetic mutation scale slowly while preserving useful local moves."""
    if success_rate > 0.02:
        return max(0.5, step_multiplier * 0.97)
    return min(2.0, step_multiplier * np.exp(0.005))


def block_annealing_temperature(base_temperature, generation, restart_after):
    """Use a cyclic cooling schedule whose scale is separate from genetic selection."""
    cycle_length = max(40, restart_after * 4)
    cycle_position = (generation - 1) % cycle_length
    cycle_fraction = cycle_position / cycle_length
    return base_temperature * (0.05 + 0.95 * (1.0 - cycle_fraction) ** 2)


def parent_improvement_rate(
    child_fitness,
    parent_fitness,
    excluded_rows=(),
    neutral_acceptance=0.0,
    child_secondary=None,
    parent_secondary=None,
):
    """Estimate accepted-trial rate, including optional neutral drift."""
    child_fitness = np.asarray(child_fitness)
    parent_fitness = np.asarray(parent_fitness)
    if child_fitness.shape != parent_fitness.shape:
        raise ValueError("Child and parent fitness arrays must have matching shapes.")
    if child_secondary is not None:
        child_secondary = np.asarray(child_secondary)
    if parent_secondary is not None:
        parent_secondary = np.asarray(parent_secondary)
    if (child_secondary is None) != (parent_secondary is None):
        raise ValueError("Child and parent secondary scores must be provided together.")
    if child_secondary is not None and (
        child_secondary.shape != child_fitness.shape
        or parent_secondary.shape != parent_fitness.shape
    ):
        raise ValueError("Secondary score arrays must match the fitness shapes.")
    eligible = np.ones(len(child_fitness), dtype=bool)
    eligible[np.asarray(excluded_rows, dtype=np.int64)] = False
    if not np.any(eligible):
        return 0.0
    improving = child_fitness[eligible] < parent_fitness[eligible]
    equal_fitness = child_fitness[eligible] == parent_fitness[eligible]
    if child_secondary is None or parent_secondary is None:
        secondary_improving = np.zeros(np.count_nonzero(eligible), dtype=bool)
        neutral = equal_fitness
    else:
        secondary_improving = (
            equal_fitness
            & (child_secondary[eligible] > parent_secondary[eligible])
        )
        neutral = equal_fitness & (
            child_secondary[eligible] == parent_secondary[eligible]
        )
    return float(
        np.mean(improving | secondary_improving)
        + neutral_acceptance * np.mean(neutral)
    )


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

    move_sizes = np.minimum(
        rng.choice(
            np.asarray([1, 2, 3, 5, 8, max(1, n_sat // 2), n_sat]),
            size=count,
            p=[0.35, 0.20, 0.12, 0.10, 0.08, 0.10, 0.05],
        ),
        n_sat,
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
    scaled_noise = sample_orbital_perturbations(
        count, n_sat, mutation_map, rng
    )
    genes += scaled_noise * steps[:, None, None] * changed[..., None]

    # Replace one selected satellite with a donor's whole six-dimensional state.
    donor_rows = np.flatnonzero(rng.random(count) < 0.18)
    if donor_rows.size:
        satellite = rng.integers(0, n_sat, size=donor_rows.size)
        genes[donor_rows, satellite] = donor_genes[donor_rows, satellite]

    proposals = genes.transpose(0, 2, 1).reshape(count, 6 * n_sat)
    return np.clip(proposals, lower, upper)


def main():
    args = parse_args()
    if args.population_size is None:
        if args.algorithm == "lm-cma-es":
            dimension = 6 * get_case(args.case)["n_sat"]
            args.population_size = default_population_size(dimension)
        else:
            args.population_size = 256
    if min(
        args.iterations,
        args.batch_size,
        args.population_size,
        args.restart_after,
    ) <= 0:
        raise ValueError(
            "iterations, batch-size, population-size, and restart-after must be positive."
        )
    if (
        args.velocity_bound <= 0
        or args.temperature <= 0
        or args.anneal_temperature <= 0
    ):
        raise ValueError(
            "velocity-bound, temperature, and anneal-temperature must be positive."
        )
    if args.algorithm in ("de", "lm-cma-es") and args.population_size < 4:
        raise ValueError(
            f"{args.algorithm} requires population-size >= 4."
        )
    if args.iterations <= args.population_size:
        raise ValueError(
            "iterations must exceed population-size to include initialization."
        )
    if args.rng_jump < 0:
        raise ValueError("rng-jump must be non-negative.")

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
    if args.rng_jump:
        rng.bit_generator.advance(args.rng_jump)
        print(f"Advanced RNG stream by {args.rng_jump:,} draws", flush=True)
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
            position_maps = np.asarray(data["position_maps"], dtype=np.float64)
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
        def score_batch(candidates, return_observation_scores=False):
            enforce_observation_bounds(candidates, position_maps, n_sat)
            candidates[...] = canonicalize_candidates(candidates, n_sat)
            result = gpu_fitness_batch(
                candidates,
                reference,
                return_observation_scores=return_observation_scores,
            )
            if return_observation_scores:
                fitnesses, observation_scores = result
                return (
                    fitnesses.detach().cpu().numpy(),
                    observation_scores.detach().cpu().numpy(),
                )
            return result.detach().cpu().numpy()

        def score_cma_batch(candidates, return_observation_scores=False):
            scores = []
            observations = []
            for start in range(0, len(candidates), args.batch_size):
                batch = candidates[start : start + args.batch_size]
                enforce_observation_bounds(batch, position_maps, n_sat)
                result = gpu_fitness_batch(
                    batch,
                    reference,
                    return_observation_scores=return_observation_scores,
                )
                if return_observation_scores:
                    fitnesses, observation_scores = result
                    scores.append(fitnesses.detach().cpu().numpy())
                    observations.append(observation_scores.detach().cpu().numpy())
                else:
                    scores.append(result.detach().cpu().numpy())
            if return_observation_scores:
                return np.concatenate(scores), np.concatenate(observations)
            return np.concatenate(scores)

        coordinate_width = search_upper - search_lower

        def normalize_candidates(candidates):
            return 2.0 * (candidates - search_lower) / coordinate_width - 1.0

        def denormalize_candidates(candidates):
            return search_lower + (candidates + 1.0) * coordinate_width / 2.0

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
        if args.algorithm == "lm-cma-es":
            cma_strategy = LimitedMemoryCMAES(
                normalize_candidates(best_x), args.population_size
            )
            initial_z, initial_directions = cma_strategy.ask(
                args.population_size, rng
            )
            initial_normalized = reflect_to_unit_box(
                cma_strategy.mean
                + cma_strategy.sigma * initial_directions
            )
            population = denormalize_candidates(initial_normalized)
            population_fitness, observation_scores = score_cma_batch(
                population, return_observation_scores=True
            )
            population_secondary = observation_scores.sum(axis=1)
            feasible_normalized = normalize_candidates(population)
            effective_directions = (
                feasible_normalized - cma_strategy.mean
            ) / cma_strategy.sigma
            cma_strategy.tell(
                initial_z,
                effective_directions,
                population_fitness,
                rng,
                secondary_scores=population_secondary,
            )
        else:
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
            population_fitness, observation_scores = score_batch(
                population, return_observation_scores=True
            )
            population_secondary = observation_scores.sum(axis=1)
        best_found_at = 0
        scored = len(population) + 1
        stagnant_generations = 0
        stagnant_evaluations = 0
        cma_restart_count = 0
        heuristic_restart_count = 0
        heuristic_scale = 1.0
        heuristic_guidance = None
        step_multiplier = 1.0

        improved = int(np.argmin(population_fitness))
        if population_fitness[improved] < best_fitness:
            best_x = population[improved].copy()
            best_fitness = float(population_fitness[improved])
            best_found_at = scored
            save_candidate_progress(
                case_id,
                best_x,
                best_fitness,
                objective="min",
                algorithm=args.algorithm,
                seed=args.seed,
                iterations=scored,
                best_found_at=best_found_at,
            )
            print(f"evaluation {scored:,}: best {best_fitness:.12g}", flush=True)

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
                parent_secondary = population_secondary[target_indices]
            elif args.algorithm == "lm-cma-es":
                batch_size = min(args.population_size, args.iterations - scored)
                lm_z, lm_directions = cma_strategy.ask(batch_size, rng)
                normalized = reflect_to_unit_box(
                    cma_strategy.mean + cma_strategy.sigma * lm_directions
                )
                children = denormalize_candidates(normalized)
                child_fitness, child_observations = score_cma_batch(
                    children, return_observation_scores=True
                )
                child_secondary = child_observations.sum(axis=1)
                feasible_normalized = normalize_candidates(children)
                effective_lm_directions = (
                    feasible_normalized - cma_strategy.mean
                ) / cma_strategy.sigma
                parent_fitness = np.empty(batch_size, dtype=np.float64)
                parent_secondary = np.empty(batch_size, dtype=np.float64)
                immigrant_rows = np.empty(0, dtype=np.int64)
                target_indices = None
            elif args.algorithm == "heuristic":
                if heuristic_guidance is None or generation % 25 == 0:
                    heuristic_guidance = satellite_coverage_guidance(
                        best_x, position_maps, reference.grid_size, n_sat
                    )
                weakest_observation, _, _, satellite_weights = heuristic_guidance
                candidate_parents = np.repeat(
                    best_x[None, :], batch_size, axis=0
                )
                diverse_parent_count = batch_size // 3
                if diverse_parent_count:
                    diverse_parent_indices = rng.integers(
                        len(population), size=diverse_parent_count
                    )
                    candidate_parents[:diverse_parent_count] = population[
                        diverse_parent_indices
                    ]
                children = make_coverage_guided_offspring(
                    candidate_parents,
                    batch_size,
                    search_lower,
                    search_upper,
                    n_sat,
                    mutation_map,
                    satellite_weights,
                    rng,
                    step_multiplier=heuristic_scale,
                )
                parent_fitness = np.full(batch_size, best_fitness)
                parent_secondary = np.full(batch_size, 0.0)
                immigrant_rows = np.empty(0, dtype=np.int64)
                target_indices = None
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
                current_temperature = block_annealing_temperature(
                    args.anneal_temperature,
                    generation,
                    args.restart_after,
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
                parent_secondary = population_secondary[target_indices]
                immigrant_rows = np.empty(0, dtype=np.int64)
            else:
                (
                    children,
                    parent_fitness,
                    immigrant_rows,
                    parent_secondary,
                ) = make_offspring(
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
                    secondary_scores=population_secondary,
                )
                target_indices = None
            if args.algorithm in ("de", "genetic", "block-anneal", "heuristic"):
                children = canonicalize_candidates(children, n_sat)
            if args.algorithm != "lm-cma-es":
                child_fitness, child_observations = score_batch(
                    children, return_observation_scores=True
                )
                child_secondary = child_observations.sum(axis=1)
            scored += batch_size

            if args.algorithm in ("lm-cma-es", "block-anneal", "heuristic"):
                success_rate = 0.0
            else:
                neutral_acceptance = (
                    DE_NEUTRAL_ACCEPTANCE if args.algorithm == "de" else 0.0
                )
                success_rate = parent_improvement_rate(
                    child_fitness,
                    parent_fitness,
                    immigrant_rows,
                    neutral_acceptance,
                    child_secondary=child_secondary,
                    parent_secondary=parent_secondary,
                )

            child_best = int(
                rank_fitnesses(child_fitness, rng, child_secondary)[0]
            )
            if child_fitness[child_best] < best_fitness:
                best_x = children[child_best].copy()
                best_fitness = float(child_fitness[child_best])
                best_found_at = scored
                stagnant_generations = 0
                stagnant_evaluations = 0
                if args.algorithm == "heuristic":
                    heuristic_scale = 1.0
                    heuristic_restart_count = 0
                    heuristic_guidance = satellite_coverage_guidance(
                        best_x, position_maps, reference.grid_size, n_sat
                    )
                if args.algorithm == "genetic":
                    step_multiplier = max(0.5, step_multiplier * 0.85)
                save_candidate_progress(
                    case_id,
                    best_x,
                    best_fitness,
                    objective="min",
                    algorithm=args.algorithm,
                    seed=args.seed,
                    iterations=scored,
                    best_found_at=best_found_at,
                )
                print(f"evaluation {scored:,}: best {best_fitness:.12g}", flush=True)
            else:
                stagnant_generations += 1
                stagnant_evaluations += batch_size
                if args.algorithm == "genetic":
                    step_multiplier = adapt_genetic_step_multiplier(
                        step_multiplier, success_rate
                    )

            if args.algorithm == "de":
                step_multiplier = adapt_de_step_multiplier(
                    step_multiplier, success_rate
                )

            if args.algorithm == "de":
                update_de_population(
                    population,
                    population_fitness,
                    children,
                    child_fitness,
                    target_indices,
                    rng,
                    population_secondary=population_secondary,
                    child_secondary=child_secondary,
                )
            elif args.algorithm == "block-anneal":
                deltas = child_fitness - parent_fitness
                acceptance = deltas < 0
                worse = deltas > 0
                acceptance[worse] = (
                    rng.random(int(worse.sum()))
                    < np.exp(-deltas[worse] / max(current_temperature, 1e-12))
                )
                equal = deltas == 0
                secondary_improvement = (
                    child_secondary[equal] > parent_secondary[equal]
                )
                neutral = child_secondary[equal] == parent_secondary[equal]
                equal_acceptance = secondary_improvement.copy()
                equal_acceptance[neutral] = (
                    rng.random(int(neutral.sum())) < DE_NEUTRAL_ACCEPTANCE
                )
                acceptance[equal] = equal_acceptance
                accepted_rows = np.flatnonzero(acceptance)
                accepted_targets = target_indices[accepted_rows]
                population[accepted_targets] = children[accepted_rows]
                population_fitness[accepted_targets] = child_fitness[accepted_rows]
                population_secondary[accepted_targets] = child_secondary[accepted_rows]
            elif args.algorithm == "genetic":
                combined_x = np.concatenate((population, children), axis=0)
                combined_fitness = np.concatenate((population_fitness, child_fitness))
                combined_secondary = np.concatenate(
                    (population_secondary, child_secondary)
                )
                (
                    population,
                    population_fitness,
                    population_secondary,
                ) = select_population(
                    combined_x,
                    combined_fitness,
                    args.population_size,
                    rng,
                    secondary_scores=combined_secondary,
                )
            elif args.algorithm == "heuristic":
                combined_x = np.concatenate((population, children), axis=0)
                combined_fitness = np.concatenate((population_fitness, child_fitness))
                combined_secondary = np.concatenate(
                    (population_secondary, child_secondary)
                )
                (
                    population,
                    population_fitness,
                    population_secondary,
                ) = select_population(
                    combined_x,
                    combined_fitness,
                    args.population_size,
                    rng,
                    secondary_scores=combined_secondary,
                )
            elif args.algorithm == "lm-cma-es" and batch_size == args.population_size:
                cma_strategy.tell(
                    lm_z,
                    effective_lm_directions,
                    child_fitness,
                    rng,
                    secondary_scores=child_secondary,
                )

            if args.algorithm == "heuristic":
                if (
                    stagnant_generations >= args.restart_after
                    and scored < args.iterations
                ):
                    heuristic_scales = (1.0, 2.0, 4.0)
                    heuristic_scale = heuristic_scales[
                        heuristic_restart_count % len(heuristic_scales)
                    ]
                    heuristic_restart_count += 1
                    stagnant_generations = 0
                    heuristic_guidance = None
                    print(
                        f"Coverage-heuristic restart {heuristic_restart_count}; "
                        f"mutation scale {heuristic_scale:g}; "
                        f"weakest observation {weakest_observation}",
                        flush=True,
                    )
            elif args.algorithm == "lm-cma-es":
                cma_restart_after = args.restart_after * args.batch_size
                if (
                    stagnant_evaluations >= cma_restart_after
                    and scored < args.iterations
                ):
                    restart_sigmas = (0.2, 0.5, 1.0)
                    restart_sigma = restart_sigmas[
                        cma_restart_count % len(restart_sigmas)
                    ]
                    cma_strategy.restart(
                        normalize_candidates(best_x), restart_sigma
                    )
                    cma_restart_count += 1
                    stagnant_evaluations = 0
                    print(
                        f"LM-CMA-ES stagnation restart {cma_restart_count}; "
                        f"sigma {restart_sigma:.3g}",
                        flush=True,
                    )
            elif (
                stagnant_generations >= args.restart_after
                and scored < args.iterations
            ):
                refresh_count = min(
                    max(1, args.population_size // 2),
                    args.iterations - scored,
                )
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
                refreshed_fitness, refreshed_observations = score_batch(
                    refreshed, return_observation_scores=True
                )
                refreshed_secondary = refreshed_observations.sum(axis=1)
                scored += len(refreshed)
                if args.algorithm in ("de", "block-anneal"):
                    worst = np.argsort(population_fitness)[-refresh_count:]
                    population[worst] = refreshed
                    population_fitness[worst] = refreshed_fitness
                    population_secondary[worst] = refreshed_secondary
                else:
                    combined_x = np.concatenate((population, refreshed), axis=0)
                    combined_fitness = np.concatenate(
                        (population_fitness, refreshed_fitness)
                    )
                    combined_secondary = np.concatenate(
                        (population_secondary, refreshed_secondary)
                    )
                    (
                        population,
                        population_fitness,
                        population_secondary,
                    ) = select_population(
                        combined_x,
                        combined_fitness,
                        args.population_size,
                        rng,
                        secondary_scores=combined_secondary,
                    )
                    if args.algorithm == "genetic":
                        explorer_count = min(
                            max(1, args.population_size // 10),
                            len(refreshed) - local_count,
                        )
                        explorer_start = local_count
                        if explorer_count:
                            population[-explorer_count:] = refreshed[
                                explorer_start : explorer_start + explorer_count
                            ]
                            population_fitness[-explorer_count:] = refreshed_fitness[
                                explorer_start : explorer_start + explorer_count
                            ]
                            population_secondary[-explorer_count:] = (
                                refreshed_secondary[
                                    explorer_start : explorer_start + explorer_count
                                ]
                            )
                refreshed_best = int(
                    rank_fitnesses(
                        refreshed_fitness, rng, refreshed_secondary
                    )[0]
                )
                if refreshed_fitness[refreshed_best] < best_fitness:
                    best_x = refreshed[refreshed_best].copy()
                    best_fitness = float(refreshed_fitness[refreshed_best])
                    best_found_at = scored
                    save_candidate_progress(
                        case_id,
                        best_x,
                        best_fitness,
                        objective="min",
                        algorithm=args.algorithm,
                        seed=args.seed,
                        iterations=scored,
                        best_found_at=best_found_at,
                    )
                    print(
                        f"evaluation {scored:,}: best {best_fitness:.12g}",
                        flush=True,
                    )
                stagnant_generations = 0
                if args.algorithm in ("de", "genetic"):
                    step_multiplier = 1.0

            if generation % 25 == 0 or scored >= args.iterations:
                population_best = (
                    best_fitness
                    if args.algorithm == "lm-cma-es"
                    else population_fitness.min()
                )
                progress_detail = (
                    (
                        f"weakest observation {weakest_observation}; "
                        f"mutation scale {heuristic_scale:g}"
                    )
                    if args.algorithm == "heuristic"
                    else f"sigma {cma_strategy.sigma:.4g}"
                    if args.algorithm == "lm-cma-es"
                    else (
                        f"temperature {current_temperature:.4g}"
                        if args.algorithm == "block-anneal"
                        else f"mutation scale {step_multiplier:.3f}"
                    )
                )
                print(
                    f"progress {scored}/{args.iterations}; "
                    f"best fitness {best_fitness:.12g}; "
                    f"population best {population_best:.12g}; "
                    f"{progress_detail}",
                    flush=True,
                )

        if args.target_score is not None and best_fitness <= args.target_score:
            print(f"Target reached: {best_fitness:.12g} <= {args.target_score}", flush=True)

        observation_values = None
        if args.skip_cpu_verification:
            cpu_fitness = best_fitness
            print("CPU verification deferred to launcher.", flush=True)
        else:
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
                    f"GPU/CPU score mismatch for {case_id}: "
                    f"{best_fitness} vs {cpu_fitness}."
                )
    if not is_inside_observation_bounds(best_x, position_maps, n_sat):
        raise RuntimeError(f"Best {case_id} candidate violates observation bounds.")

    save_metadata = {
        "initial_fitness": initial_fitness,
        "best_found_at": best_found_at,
        "seed": args.seed,
        "iterations": scored,
        "batch_size": args.batch_size,
        "population_size": args.population_size,
        "velocity_bound": args.velocity_bound,
        "restart_after": args.restart_after,
        "temperature": args.temperature,
        "objective": "min",
    }
    if observation_values is not None:
        save_metadata["observation_scores"] = observation_values
    save_candidate_progress(
        case_id,
        best_x,
        cpu_fitness,
        algorithm=args.algorithm,
        **save_metadata,
    )

    if not args.skip_cpu_verification:
        print("CPU verification difference:", difference)
        print("Observation scores:", np.array2string(observation_values, precision=6))
    else:
        print("Final GPU fitness:", cpu_fitness)
    print("Evaluations:", scored)
    print("Elapsed seconds:", round(time.perf_counter() - started, 2))
    print("Saved best:", best_path)
    print("Final fitness:", cpu_fitness)
    rng_checkpoint_path = args.save_rng_checkpoint or (
        Path("results/cases") / f"{case_id}_optimizer_rng.npz"
    )
    rng_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        rng_checkpoint_path,
        case_id=case_id,
        seed=args.seed,
        rng_state=json.dumps(rng.bit_generator.state),
        best_x=best_x,
        best_fitness=cpu_fitness,
        evaluations=scored,
    )
    print("Saved RNG checkpoint:", rng_checkpoint_path)


if __name__ == "__main__":
    main()

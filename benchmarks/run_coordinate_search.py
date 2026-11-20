"""Checkpointed GPU search guided by propagated grid-cell boundaries."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.build_submission import save_candidate_progress
from benchmarks.run_optimization import VERIFY_REFERENCE
from src.evaluation.cases import CASE_ORDER, get_case
from src.evaluation.constraints import (
    enforce_observation_bounds,
    is_inside_observation_bounds,
)

CELL_STEPS = (1, 2, 4, 8, 16)
CHECKPOINT_VERSION = 7
RANDOM_RESTART_EVERY = 1_000_000
PARTIAL_RANDOMIZE_FRACTION = 0.25


def evaluations_until_random_restart(evaluations_done):
    return RANDOM_RESTART_EVERY - (evaluations_done % RANDOM_RESTART_EVERY)


def make_partial_random_restart_candidate(current, lower, upper, n_sat, rng):
    candidate = np.asarray(current, dtype=np.float64).copy()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (
        candidate.shape != (6 * n_sat,)
        or lower.shape != candidate.shape
        or upper.shape != candidate.shape
    ):
        raise ValueError("Candidate and bounds must each have shape (6*n_sat,).")

    count = max(1, int(np.ceil(n_sat * PARTIAL_RANDOMIZE_FRACTION)))
    satellites = rng.choice(n_sat, size=count, replace=False)
    states = candidate.reshape(6, n_sat)
    lower_states = lower.reshape(6, n_sat)
    upper_states = upper.reshape(6, n_sat)
    states[:, satellites] = rng.uniform(
        lower_states[:, satellites], upper_states[:, satellites]
    )
    return candidate, satellites


EXPORT_BOUNDARY_REFERENCE = """
import sys
import numpy as np
from src.evaluation.cases import make_reference_problem

udp = make_reference_problem(sys.argv[1])
lower, upper = udp.get_bounds()
position_maps = np.stack(
    [
        stm[:3, :] / (udp.inflation_factor if index else 1.0)
        for index, stm in enumerate(udp.stms)
    ]
)
inverse_maps = np.stack([np.linalg.pinv(position_map) for position_map in position_maps])
np.savez_compressed(
    sys.argv[2],
    stms=udp.stms,
    position_maps=position_maps,
    inverse_maps=inverse_maps,
    n_sat=udp.n_sat,
    grid_size=udp.grid_size,
    n_meas=udp.n_meas,
    scaling_factor=udp.scaling_factor,
    inflation_factor=udp.inflation_factor,
    lower_bounds=lower,
    upper_bounds=upper,
)
"""


def boundary_actions(
    incumbent, position_maps, inverse_maps, grid_size, lower, upper
):
    """Find single-satellite moves that cross one propagated grid boundary."""
    incumbent = np.asarray(incumbent, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    n_sat = incumbent.size // 6
    states = incumbent.reshape(6, n_sat).T
    state_lower = lower.reshape(6, n_sat).T
    state_upper = upper.reshape(6, n_sat).T
    cell_width = 2.0 / grid_size
    epsilon = cell_width * 1e-5
    center = int(grid_size / 2)
    actions = [[] for _ in range(n_sat)]

    for position_map, inverse_map in zip(position_maps, inverse_maps):
        positions = np.sum(
            states[:, None, :] * position_map[None, :, :], axis=2
        )

        for satellite in range(n_sat):
            for axis in range(3):
                position = positions[satellite, axis]
                if position <= -1.0:
                    directions = (1,)
                elif position >= 1.0:
                    directions = (-1,)
                else:
                    directions = (-1, 1)

                cell = int(np.floor(position / cell_width + center))
                for direction in directions:
                    boundary_cell = cell + (1 if direction > 0 else 0)
                    boundary = (boundary_cell - center) * cell_width
                    for cell_step in CELL_STEPS:
                        if position <= -1.0:
                            target = -1.0 + epsilon + (cell_step - 1) * cell_width
                        elif position >= 1.0:
                            target = 1.0 - epsilon - (cell_step - 1) * cell_width
                        else:
                            target = boundary + direction * (
                                (cell_step - 1) * cell_width + epsilon
                            )

                        state = states[satellite] + inverse_map[:, axis] * (
                            target - position
                        )
                        state = np.clip(
                            state, state_lower[satellite], state_upper[satellite]
                        )
                        reached = float(np.sum(position_map[axis] * state))
                        if direction > 0 and reached <= boundary:
                            continue
                        if direction < 0 and reached >= boundary:
                            continue
                        if direction > 0 and reached < target - epsilon / 2:
                            continue
                        if direction < 0 and reached > target + epsilon / 2:
                            continue
                        if np.array_equal(state, states[satellite]):
                            continue
                        actions[satellite].append(state)

    return actions


def propose_boundary_candidates(
    incumbent,
    count,
    n_sat,
    lower,
    upper,
    position_maps,
    inverse_maps,
    grid_size,
    move_size,
    rng,
):
    """Cross nearby projected cell boundaries, combining moves after stalls."""
    actions = boundary_actions(
        incumbent, position_maps, inverse_maps, grid_size, lower, upper
    )
    available = np.flatnonzero([bool(options) for options in actions])
    if available.size == 0:
        raise RuntimeError("No feasible grid-boundary moves from the incumbent.")

    if move_size == 1:
        max_actions_per_satellite = (
            len(position_maps) * 3 * 2 * len(CELL_STEPS)
        )
        candidates = np.repeat(
            incumbent[None, :],
            available.size * max_actions_per_satellite,
            axis=0,
        )
        row = 0
        states = candidates.reshape(-1, 6, n_sat)
        for satellite in available:
            for state in actions[satellite]:
                states[row, :, satellite] = state
                row += 1
        return candidates[:row]

    candidates = np.repeat(incumbent[None, :], count, axis=0)
    states = candidates.reshape(count, 6, n_sat)
    moves_per_candidate = min(move_size, available.size)
    for row in range(count):
        satellites = rng.choice(
            available, size=moves_per_candidate, replace=False
        )
        for satellite in satellites:
            state = actions[satellite][rng.integers(len(actions[satellite]))]
            states[row, :, satellite] = state
    return np.clip(candidates, lower, upper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_ORDER, default="hard")
    parser.add_argument("--evaluations", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=239817)
    parser.add_argument("--target-score", type=float, default=-1.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--anneal-temperature",
        type=float,
        default=0.005,
        help="maximum fitness increase allowed during plateau exploration",
    )
    args = parser.parse_args()
    if min(
        args.evaluations,
        args.batch_size,
        args.checkpoint_every,
        args.anneal_temperature,
    ) <= 0:
        raise ValueError(
            "evaluations, batch-size, checkpoint-every, and "
            "anneal-temperature must be positive"
        )

    case_id = args.case
    spec = get_case(case_id)
    best_path = Path("results/cases") / f"{case_id}_min_best.npz"
    checkpoint_path = Path("results/cases") / f"{case_id}_boundary_checkpoint.npz"
    with np.load(best_path) as saved:
        incumbent = np.asarray(saved["x"], dtype=np.float64)
        incumbent_fitness = float(saved["fitness"])
    current = incumbent.copy()
    current_fitness = incumbent_fitness

    rng = np.random.default_rng(args.seed)
    evaluations_done = 0
    move_level = 0
    stagnant_batches = 0
    levels = [1, 2, 3, 5, 8, 12, 20]
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            checkpoint_version = (
                int(saved["checkpoint_version"])
                if "checkpoint_version" in saved.files
                else 1
            )
            if (
                str(saved["case_id"].item()) != case_id
                or int(saved["seed"]) != args.seed
            ):
                raise ValueError("Coordinate-search checkpoint case/seed does not match.")
            if checkpoint_version in (6, CHECKPOINT_VERSION):
                incumbent = np.asarray(saved["incumbent"], dtype=np.float64)
                incumbent_fitness = float(saved["incumbent_fitness"])
                current = np.asarray(saved["current"], dtype=np.float64)
                current_fitness = float(saved["current_fitness"])
                evaluations_done = int(saved["evaluations_done"])
                move_level = int(saved["move_level"])
                stagnant_batches = int(saved["stagnant_batches"])
                rng.bit_generator.state = json.loads(str(saved["rng_state"].item()))
                print(
                    f"Resuming after {evaluations_done:,} evaluations; "
                    f"best {incumbent_fitness:.12g}; seed {args.seed}",
                    flush=True,
                )
            else:
                print(
                    "Ignoring an older boundary-search checkpoint; "
                    f"restarting from saved best {incumbent_fitness:.12g}.",
                    flush=True,
                )
    elif (Path("results/cases") / f"{case_id}_random_checkpoint.npz").exists():
        random_checkpoint = Path("results/cases") / f"{case_id}_random_checkpoint.npz"
        with np.load(random_checkpoint, allow_pickle=False) as saved:
            if str(saved["case_id"].item()) == case_id and int(saved["seed"]) == args.seed:
                rng.bit_generator.state = json.loads(str(saved["rng_state"].item()))
                print(
                    f"Continuing seed {args.seed} after "
                    f"{int(saved['evaluations_done']):,} global-search evaluations; "
                    f"starting from saved best {incumbent_fitness:.12g}",
                    flush=True,
                )
    else:
        print(
            f"Starting from saved best {incumbent_fitness:.12g}; seed {args.seed}",
            flush=True,
        )

    def save_checkpoint():
        temp_checkpoint = checkpoint_path.with_name(checkpoint_path.stem + ".tmp.npz")
        np.savez_compressed(
            temp_checkpoint,
            checkpoint_version=CHECKPOINT_VERSION,
            case_id=case_id,
            seed=args.seed,
            incumbent=incumbent,
            incumbent_fitness=incumbent_fitness,
            current=current,
            current_fitness=current_fitness,
            evaluations_done=evaluations_done,
            move_level=move_level,
            stagnant_batches=stagnant_batches,
            rng_state=json.dumps(rng.bit_generator.state),
        )
        temp_checkpoint.replace(checkpoint_path)

    lower = np.asarray([-1.0] * (3 * spec["n_sat"]) + [-10.0] * (3 * spec["n_sat"]))
    upper = -lower
    local_evaluations = 0
    best_found_at = evaluations_done

    with tempfile.TemporaryDirectory(prefix="esa_coordinate_search_") as temp_dir:
        temp_path = Path(temp_dir)
        cache_path = temp_path / "reference.npz"
        candidate_path = temp_path / "candidate.npz"
        verified_path = temp_path / "verified.npz"
        subprocess.run(
            [
                sys.executable,
                "-c",
                EXPORT_BOUNDARY_REFERENCE,
                case_id,
                str(cache_path),
            ],
            check=True,
        )
        with np.load(cache_path) as data:
            n_sat = int(data["n_sat"])
            stms = np.asarray(data["stms"], dtype=np.float64)
            position_maps = np.asarray(data["position_maps"], dtype=np.float64)
            inverse_maps = np.asarray(data["inverse_maps"], dtype=np.float64)
            grid_size = int(data["grid_size"])
            lower = np.asarray(data["lower_bounds"], dtype=np.float64)
            upper = np.asarray(data["upper_bounds"], dtype=np.float64)
            reference = SimpleNamespace(
                stms=stms,
                n_sat=n_sat,
                grid_size=grid_size,
                n_meas=int(data["n_meas"]),
                scaling_factor=float(data["scaling_factor"]),
                inflation_factor=float(data["inflation_factor"]),
            )

        from src.evaluation.gpu import gpu_fitness_batch

        def score_batch(candidates):
            enforce_observation_bounds(candidates, position_maps, n_sat)
            return gpu_fitness_batch(candidates, reference).detach().cpu().numpy()

        enforce_observation_bounds(incumbent, position_maps, n_sat)
        enforce_observation_bounds(current, position_maps, n_sat)
        incumbent_fitness = float(score_batch(incumbent)[0])
        local_evaluations += 1
        evaluations_done += 1
        if np.array_equal(current, incumbent):
            current_fitness = incumbent_fitness
        else:
            current_fitness = float(score_batch(current)[0])
            local_evaluations += 1
            evaluations_done += 1
        best_found_at = evaluations_done
        with np.load(best_path, allow_pickle=False) as saved:
            saved_fitness = float(saved["fitness"])
        if incumbent_fitness < saved_fitness:
            save_candidate_progress(
                case_id,
                incumbent,
                incumbent_fitness,
                objective="min",
                algorithm="coordinate-search",
                seed=args.seed,
                iterations=evaluations_done,
                best_found_at=best_found_at,
            )
            print(
                f"evaluation {evaluations_done:,}: best {incumbent_fitness:.12g}",
                flush=True,
            )
        save_checkpoint()

        while local_evaluations < args.evaluations and incumbent_fitness > args.target_score:
            remaining = args.evaluations - local_evaluations
            evaluations_to_restart = evaluations_until_random_restart(
                evaluations_done
            )
            if evaluations_to_restart == 1:
                current, randomized_satellites = (
                    make_partial_random_restart_candidate(
                        incumbent, lower, upper, n_sat, rng
                    )
                )
                current_fitness = float(score_batch(current)[0])
                local_evaluations += 1
                evaluations_done += 1
                if current_fitness < incumbent_fitness:
                    incumbent = current.copy()
                    incumbent_fitness = current_fitness
                    best_found_at = evaluations_done
                    save_candidate_progress(
                        case_id,
                        incumbent,
                        incumbent_fitness,
                        objective="min",
                        algorithm="coordinate-search",
                        seed=args.seed,
                        iterations=evaluations_done,
                        best_found_at=best_found_at,
                    )
                    print(
                        f"evaluation {evaluations_done:,}: best "
                        f"{incumbent_fitness:.12g}",
                        flush=True,
                    )
                move_level = 0
                stagnant_batches = 0
                save_checkpoint()
                print(
                    f"Randomized {len(randomized_satellites)}/{n_sat} satellite "
                    f"states at evaluation {evaluations_done:,}; saved best remains "
                    f"{incumbent_fitness:.12g}",
                    flush=True,
                )
                continue

            move_size = levels[min(move_level, len(levels) - 1)]
            count = min(args.batch_size, remaining, evaluations_to_restart - 1)
            candidates = propose_boundary_candidates(
                current,
                count,
                n_sat,
                lower,
                upper,
                position_maps,
                inverse_maps,
                grid_size,
                move_size,
                rng,
            )
            candidates = candidates[:count]
            count = len(candidates)
            fitnesses = score_batch(candidates)
            local_evaluations += count
            evaluations_done += count
            best_index = int(np.argmin(fitnesses))
            batch_best = float(fitnesses[best_index])
            improved_incumbent = batch_best < incumbent_fitness
            if improved_incumbent:
                incumbent_fitness = batch_best
                incumbent = candidates[best_index].copy()
                best_found_at = evaluations_done
                move_level = 0
                stagnant_batches = 0
                save_candidate_progress(
                    case_id,
                    incumbent,
                    incumbent_fitness,
                    objective="min",
                    algorithm="coordinate-search",
                    seed=args.seed,
                    iterations=evaluations_done,
                    best_found_at=best_found_at,
                )
                print(
                    f"evaluation {evaluations_done:,}: best {incumbent_fitness:.12g}",
                    flush=True,
                )
            if batch_best < current_fitness:
                current_fitness = batch_best
                current = candidates[best_index].copy()
            else:
                near_best = np.flatnonzero(fitnesses <= batch_best + 0.01)
                proposal_index = int(rng.choice(near_best))
                proposal_fitness = float(fitnesses[proposal_index])
                increase = proposal_fitness - current_fitness
                if increase <= 0 or rng.random() < np.exp(
                    -increase / args.anneal_temperature
                ):
                    current_fitness = proposal_fitness
                    current = candidates[proposal_index].copy()

            if not improved_incumbent:
                stagnant_batches += 1
                stagnation_limit = 1 if move_size == 1 else 5
                if stagnant_batches >= stagnation_limit:
                    if move_level == len(levels) - 1:
                        move_level = 0
                    else:
                        move_level += 1
                    stagnant_batches = 0
            else:
                stagnant_batches = 0

            batches_done = local_evaluations // args.batch_size
            if batches_done % args.checkpoint_every == 0:
                save_checkpoint()
                print(
                    f"progress {local_evaluations:,}/{args.evaluations:,}; "
                    f"best {incumbent_fitness:.12g}; "
                    f"boundary moves per proposal {move_size}",
                    flush=True,
                )
        np.savez_compressed(candidate_path, x=incumbent)
        subprocess.run(
            [sys.executable, "-c", VERIFY_REFERENCE, case_id, str(candidate_path), str(verified_path)],
            check=True,
        )
        with np.load(verified_path) as verified:
            cpu_fitness = float(verified["reference_fitness"])
            observation_scores = np.asarray(verified["observation_scores"])

    if abs(incumbent_fitness - cpu_fitness) > 1e-10:
        raise RuntimeError(f"CPU/GPU verification mismatch: {cpu_fitness} vs {incumbent_fitness}.")
    if not is_inside_observation_bounds(incumbent, position_maps, n_sat):
        raise RuntimeError(f"Best {case_id} candidate violates observation bounds.")
    with np.load(best_path) as saved:
        saved_fitness = float(saved["fitness"])
        saved_x = np.asarray(saved["x"], dtype=np.float64)
    if (
        not is_inside_observation_bounds(saved_x, position_maps, n_sat)
        or cpu_fitness < saved_fitness
    ):
        np.savez_compressed(
            best_path,
            x=incumbent,
            fitness=cpu_fitness,
            observation_scores=observation_scores,
            objective="min",
            case_id=case_id,
            algorithm="coordinate-search",
            seed=args.seed,
            iterations=evaluations_done,
            best_found_at=best_found_at,
        )
    save_checkpoint()
    print("CPU verification difference:", abs(incumbent_fitness - cpu_fitness))
    print("Observation scores:", observation_scores)
    print("Total evaluations:", evaluations_done)
    print("Target:", args.target_score)
    print("Final fitness:", cpu_fitness)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()

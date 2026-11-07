"""Continue a checkpointed GPU random search from the saved best solution."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.run_optimization import EXPORT_REFERENCE, VERIFY_REFERENCE, random_candidates
from src.evaluation.cases import CASE_ORDER, get_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_ORDER, default="hard")
    parser.add_argument("--evaluations", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=239817)
    parser.add_argument("--target-score", type=float, default=-1.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    if min(args.evaluations, args.batch_size, args.checkpoint_every) <= 0:
        raise ValueError("evaluations, batch-size, and checkpoint-every must be positive")

    import torch

    case_id = args.case
    spec = get_case(case_id)
    best_path = Path("results/cases") / f"{case_id}_min_best.npz"
    checkpoint_path = Path("results/cases") / f"{case_id}_random_checkpoint.npz"

    def save_checkpoint(best_x, best_fitness, evaluations_done, rng):
        temporary_checkpoint = checkpoint_path.with_name(
            checkpoint_path.stem + ".tmp.npz"
        )
        np.savez_compressed(
            temporary_checkpoint,
            case_id=case_id,
            seed=args.seed,
            evaluations_done=evaluations_done,
            best_fitness=best_fitness,
            best_x=best_x,
            rng_state=json.dumps(rng.bit_generator.state),
        )
        temporary_checkpoint.replace(checkpoint_path)

    with np.load(best_path) as saved:
        best_x = np.asarray(saved["x"], dtype=np.float64)
        best_fitness = float(saved["fitness"])

    rng = np.random.default_rng(args.seed)
    evaluations_done = 0
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            old_case = str(saved["case_id"].item())
            old_seed = int(saved["seed"])
            if old_case != case_id or old_seed != args.seed:
                raise ValueError(
                    f"Checkpoint is for {old_case} seed {old_seed}, not {case_id} seed {args.seed}."
                )
            best_x = np.asarray(saved["best_x"], dtype=np.float64)
            best_fitness = float(saved["best_fitness"])
            evaluations_done = int(saved["evaluations_done"])
            rng.bit_generator.state = json.loads(str(saved["rng_state"].item()))
        print(
            f"Resuming checkpoint after {evaluations_done:,} evaluations; "
            f"best {best_fitness:.12g}; seed {args.seed}",
            flush=True,
        )
    else:
        print(
            f"Starting from verified saved candidate {best_fitness:.12g}; seed {args.seed}",
            flush=True,
        )

    lower = np.asarray([-1.0] * (3 * spec["n_sat"]) + [-10.0] * (3 * spec["n_sat"]))
    upper = -lower
    search_lower = lower.copy()
    search_upper = upper.copy()
    search_lower[3 * spec["n_sat"] :] = np.maximum(
        search_lower[3 * spec["n_sat"] :], -10.0
    )
    search_upper[3 * spec["n_sat"] :] = np.minimum(
        search_upper[3 * spec["n_sat"] :], 10.0
    )
    local_evaluations = 0
    best_found_at = evaluations_done

    with tempfile.TemporaryDirectory(prefix="esa_random_search_") as temp_dir:
        temp_path = Path(temp_dir)
        cache_path = temp_path / "reference.npz"
        candidate_path = temp_path / "candidate.npz"
        verified_path = temp_path / "verified.npz"
        subprocess.run(
            [sys.executable, "-c", EXPORT_REFERENCE, case_id, str(cache_path)],
            check=True,
        )
        with np.load(cache_path) as data:
            n_sat = int(data["n_sat"])
            reference = SimpleNamespace(
                stms=np.asarray(data["stms"], dtype=np.float64),
                n_sat=n_sat,
                grid_size=int(data["grid_size"]),
                n_meas=int(data["n_meas"]),
                scaling_factor=float(data["scaling_factor"]),
                inflation_factor=float(data["inflation_factor"]),
            )

        from src.evaluation.gpu import gpu_fitness_batch

        while local_evaluations < args.evaluations and best_fitness > args.target_score:
            count = min(args.batch_size, args.evaluations - local_evaluations)
            candidates = random_candidates(
                count, search_lower, search_upper, n_sat, 10.0, rng
            )
            fitnesses = gpu_fitness_batch(candidates, reference).detach().cpu().numpy()
            local_evaluations += count
            evaluations_done += count
            best_index = int(np.argmin(fitnesses))
            if float(fitnesses[best_index]) < best_fitness:
                best_fitness = float(fitnesses[best_index])
                best_x = candidates[best_index].copy()
                best_found_at = evaluations_done
                print(
                    f"evaluation {evaluations_done:,}: best {best_fitness:.12g}",
                    flush=True,
                )

            batches_done = local_evaluations // args.batch_size
            if batches_done % args.checkpoint_every == 0:
                save_checkpoint(best_x, best_fitness, evaluations_done, rng)
                print(
                    f"progress {local_evaluations:,}/{args.evaluations:,}; "
                    f"best {best_fitness:.12g}",
                    flush=True,
                )

        np.savez_compressed(candidate_path, x=best_x)
        subprocess.run(
            [sys.executable, "-c", VERIFY_REFERENCE, case_id, str(candidate_path), str(verified_path)],
            check=True,
        )
        with np.load(verified_path) as verified:
            cpu_fitness = float(verified["reference_fitness"])
            observation_scores = np.asarray(verified["observation_scores"])

    if abs(best_fitness - cpu_fitness) > 1e-10:
        raise RuntimeError(f"CPU/GPU verification mismatch: {cpu_fitness} vs {best_fitness}.")
    if cpu_fitness < float(np.load(best_path)["fitness"]):
        np.savez_compressed(
            best_path,
            x=best_x,
            fitness=cpu_fitness,
            observation_scores=observation_scores,
            objective="min",
            case_id=case_id,
            algorithm="random-search",
            seed=args.seed,
            iterations=evaluations_done,
            best_found_at=best_found_at,
        )

    save_checkpoint(best_x, best_fitness, evaluations_done, rng)
    print("CPU verification difference:", abs(best_fitness - cpu_fitness))
    print("Observation scores:", observation_scores)
    print("Total random evaluations:", evaluations_done)
    print("Target:", args.target_score)
    print("Final fitness:", cpu_fitness)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()

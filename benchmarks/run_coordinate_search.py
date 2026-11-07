"""Checkpointed GPU search that replaces or perturbs small satellite groups."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.run_optimization import EXPORT_REFERENCE, VERIFY_REFERENCE
from src.evaluation.cases import CASE_ORDER, get_case


def propose_from_incumbent(
    incumbent, count, n_sat, lower, upper, mutation_map, move_size, rng
):
    """Create exact-score candidates by changing a few whole satellite states."""
    genes = np.repeat(
        incumbent.reshape(6, n_sat).T[None, :, :], count, axis=0
    ).copy()
    move_sizes = rng.choice(
        np.asarray([move_size, min(n_sat, move_size + 1), min(n_sat, move_size * 2)]),
        size=count,
        p=[0.65, 0.25, 0.10],
    )
    keys = rng.random((count, n_sat))
    order = np.argsort(keys, axis=1)
    rows = np.arange(count)[:, None]
    max_moves = int(move_sizes.max())
    chosen = order[:, :max_moves]
    active = np.arange(max_moves)[None, :] < move_sizes[:, None]
    selected_rows = np.broadcast_to(rows, chosen.shape)[active]
    selected_sats = chosen[active]

    random_replace = rng.random(len(selected_rows)) < 0.30
    random_values = rng.uniform(-1.0, 1.0, size=(len(selected_rows), 6))
    random_values[:, 3:] *= 10.0
    if random_replace.any():
        genes[selected_rows[random_replace], selected_sats[random_replace]] = (
            random_values[random_replace]
        )

    local_rows = selected_rows[~random_replace]
    local_sats = selected_sats[~random_replace]
    if local_rows.size:
        noise = rng.normal(size=(local_rows.size, 6))
        scale = np.exp(rng.uniform(np.log(0.05), np.log(10.0), size=local_rows.size))
        genes[local_rows, local_sats] += np.einsum(
            "bi,ji->bj", noise, mutation_map
        ) * scale[:, None]

    candidates = genes.transpose(0, 2, 1).reshape(count, 6 * n_sat)
    return np.clip(candidates, lower, upper)


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

    case_id = args.case
    spec = get_case(case_id)
    best_path = Path("results/cases") / f"{case_id}_min_best.npz"
    checkpoint_path = Path("results/cases") / f"{case_id}_coordinate_checkpoint.npz"
    with np.load(best_path) as saved:
        incumbent = np.asarray(saved["x"], dtype=np.float64)
        incumbent_fitness = float(saved["fitness"])

    rng = np.random.default_rng(args.seed)
    evaluations_done = 0
    move_level = 0
    stagnant_batches = 0
    levels = [1, 2, 3, 5, 8, 12, 20]
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            if str(saved["case_id"].item()) != case_id or int(saved["seed"]) != args.seed:
                raise ValueError("Coordinate-search checkpoint case/seed does not match.")
            incumbent = np.asarray(saved["incumbent"], dtype=np.float64)
            incumbent_fitness = float(saved["incumbent_fitness"])
            evaluations_done = int(saved["evaluations_done"])
            move_level = int(saved["move_level"])
            stagnant_batches = int(saved["stagnant_batches"])
            rng.bit_generator.state = json.loads(str(saved["rng_state"].item()))
        print(
            f"Resuming after {evaluations_done:,} evaluations; "
            f"best {incumbent_fitness:.12g}; seed {args.seed}",
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
            case_id=case_id,
            seed=args.seed,
            incumbent=incumbent,
            incumbent_fitness=incumbent_fitness,
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
            [sys.executable, "-c", EXPORT_REFERENCE, case_id, str(cache_path)],
            check=True,
        )
        with np.load(cache_path) as data:
            n_sat = int(data["n_sat"])
            mutation_map = np.asarray(data["mutation_map"], dtype=np.float64)
            reference = SimpleNamespace(
                stms=np.asarray(data["stms"], dtype=np.float64),
                n_sat=n_sat,
                grid_size=int(data["grid_size"]),
                n_meas=int(data["n_meas"]),
                scaling_factor=float(data["scaling_factor"]),
                inflation_factor=float(data["inflation_factor"]),
            )

        from src.evaluation.gpu import gpu_fitness_batch

        while local_evaluations < args.evaluations and incumbent_fitness > args.target_score:
            count = min(args.batch_size, args.evaluations - local_evaluations)
            move_size = levels[min(move_level, len(levels) - 1)]
            candidates = propose_from_incumbent(
                incumbent, count, n_sat, lower, upper, mutation_map, move_size, rng
            )
            fitnesses = gpu_fitness_batch(candidates, reference).detach().cpu().numpy()
            local_evaluations += count
            evaluations_done += count
            best_index = int(np.argmin(fitnesses))
            batch_best = float(fitnesses[best_index])
            if batch_best < incumbent_fitness:
                incumbent_fitness = batch_best
                incumbent = candidates[best_index].copy()
                best_found_at = evaluations_done
                move_level = 0
                stagnant_batches = 0
                print(
                    f"evaluation {evaluations_done:,}: best {incumbent_fitness:.12g}",
                    flush=True,
                )
            else:
                stagnant_batches += 1
                if stagnant_batches >= 100:
                    move_level = min(move_level + 1, len(levels) - 1)
                    stagnant_batches = 0

            batches_done = local_evaluations // args.batch_size
            if batches_done % args.checkpoint_every == 0:
                save_checkpoint()
                print(
                    f"progress {local_evaluations:,}/{args.evaluations:,}; "
                    f"best {incumbent_fitness:.12g}; group size {move_size}",
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
    with np.load(best_path) as saved:
        saved_fitness = float(saved["fitness"])
    if cpu_fitness < saved_fitness:
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

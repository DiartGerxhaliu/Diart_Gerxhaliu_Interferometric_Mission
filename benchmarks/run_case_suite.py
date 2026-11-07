"""Run the score-guided random-improvement search for all three cases."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.evaluation.cases import CASE_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--restart-after", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--velocity-bound", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases that already have a saved, CPU-verified best candidate",
    )
    args = parser.parse_args()

    for index, case_id in enumerate(CASE_ORDER):
        best_path = f"results/cases/{case_id}_min_best.npz"
        if args.resume and Path(best_path).exists():
            print(f"Skipping {case_id}: found {best_path}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "benchmarks.run_optimization",
            "--case",
            case_id,
            "--seed",
            str(args.seed + index),
            "--iterations",
            str(args.iterations),
            "--batch-size",
            str(args.batch_size),
            "--restart-after",
            str(args.restart_after),
            "--velocity-bound",
            str(args.velocity_bound),
            "--temperature",
            str(args.temperature),
        ]
        subprocess.run(command, check=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.build_submission",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

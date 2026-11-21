"""Run an ESA case optimizer and update its submission when the score improves."""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks.build_submission import VERIFY_SUBMISSION, submission_record
from benchmarks.lm_cma_es import default_population_size
from src.evaluation.cases import get_case


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CASE_RESULTS_DIR = RESULTS_DIR / "cases"
SUBMISSION_DIR = RESULTS_DIR / "submissions"
LOG_DIR = RESULTS_DIR / "logs"
ALGORITHMS = (
    "coordinate-search",
    "de",
    "genetic",
    "block-anneal",
    "lm-cma-es",
)
DEFAULT_ALGORITHMS = {
    "easy": "coordinate-search",
    "medium": "de",
    "hard": "de",
}
SEARCH_CYCLES = {
    "easy": (
        "coordinate-search",
        "de",
        "genetic",
        "block-anneal",
        "lm-cma-es",
    ),
    "medium": (
        "de",
        "coordinate-search",
        "genetic",
        "block-anneal",
        "lm-cma-es",
    ),
    "hard": (
        "de",
        "coordinate-search",
        "genetic",
        "block-anneal",
        "lm-cma-es",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search one ESA case continuously until Ctrl+C. Scores are minimized; "
            "each new feasible best updates that case's submission JSON."
        )
    )
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument("--easy", dest="case", action="store_const", const="easy")
    cases.add_argument("--medium", dest="case", action="store_const", const="medium")
    cases.add_argument("--hard", dest="case", action="store_const", const="hard")
    parser.add_argument(
        "--algorithm",
        choices=("auto",) + ALGORITHMS,
        default="auto",
        help="auto switches algorithms after a segment with no improvement",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="remove the score cutoff after this score or lower is reached",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help=(
            "population size (hard DE defaults to 4096; LM-CMA-ES uses a "
            "dimension-scaled default; otherwise 256)"
        ),
    )
    parser.add_argument("--velocity-bound", type=float, default=10.0)
    parser.add_argument("--restart-after", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="genetic selection temperature",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--anneal-temperature",
        type=float,
        default=0.005,
        help="base temperature for coordinate and block annealing",
    )
    parser.add_argument(
        "--resume-rng",
        type=Path,
        default=None,
        help="use this RNG checkpoint instead of the selected case's default",
    )
    parser.add_argument("--rng-jump", type=int, default=0)
    return parser.parse_args()


def checkpoint_seed(path, fallback):
    if not path.is_file():
        return fallback
    with np.load(path, allow_pickle=False) as saved:
        if "seed" in saved.files:
            return int(saved["seed"])
    return fallback


def starting_search_score(case_id, algorithm):
    best_path = CASE_RESULTS_DIR / f"{case_id}_min_best.npz"
    with np.load(best_path, allow_pickle=False) as saved:
        starting_best = float(saved["fitness"])
    if algorithm == "coordinate-search":
        checkpoint_path = CASE_RESULTS_DIR / f"{case_id}_boundary_checkpoint.npz"
        if checkpoint_path.exists():
            with np.load(checkpoint_path, allow_pickle=False) as saved:
                if "incumbent_fitness" in saved.files:
                    starting_best = min(
                        starting_best, float(saved["incumbent_fitness"])
                    )
    return starting_best


def next_search_algorithm(case_id, current_algorithm):
    cycle = SEARCH_CYCLES[case_id]
    return cycle[(cycle.index(current_algorithm) + 1) % len(cycle)]


def build_search_command(args, algorithm=None):
    selected_algorithm = algorithm or (
        DEFAULT_ALGORITHMS[args.case]
        if args.algorithm == "auto"
        else args.algorithm
    )
    coordinate_search = selected_algorithm == "coordinate-search"
    if args.rng_jump < 0:
        raise ValueError("rng-jump must be non-negative.")
    if (
        coordinate_search
        and args.algorithm != "auto"
        and (args.resume_rng is not None or args.rng_jump)
    ):
        raise ValueError("--resume-rng and --rng-jump apply only to population algorithms.")
    if args.algorithm == "auto":
        budget = 5_000_000
    else:
        budget = (
            10_000_000
            if coordinate_search
            or (args.case == "hard" and selected_algorithm == "de")
            else 2_000_000
        )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    population_size = args.population_size
    if population_size is None:
        if args.case == "hard" and selected_algorithm == "de":
            population_size = 4096
        elif selected_algorithm == "lm-cma-es":
            dimension = 6 * get_case(args.case)["n_sat"]
            population_size = default_population_size(dimension)
        else:
            population_size = 256

    if coordinate_search:
        checkpoint = CASE_RESULTS_DIR / f"{args.case}_boundary_checkpoint.npz"
        seed = args.seed
        if seed is None:
            seed = checkpoint_seed(checkpoint, 777)
        command = [
            sys.executable,
            "-u",
            "-m",
            "benchmarks.run_coordinate_search",
            "--case",
            args.case,
            "--evaluations",
            str(budget),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--target-score="
            + str(args.target_score if args.target_score is not None else -1e300),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--anneal-temperature",
            str(args.anneal_temperature),
            "--skip-cpu-verification",
        ]
    else:
        default_rng = CASE_RESULTS_DIR / f"{args.case}_optimizer_rng.npz"
        resume_rng = args.resume_rng
        if resume_rng is not None:
            resume_rng = resume_rng.expanduser()
            if not resume_rng.is_absolute():
                resume_rng = (Path.cwd() / resume_rng).resolve()
            if not resume_rng.is_file():
                raise FileNotFoundError(f"RNG checkpoint not found: {resume_rng}")
        else:
            resume_rng = default_rng
        seed = args.seed
        if seed is None:
            seed = checkpoint_seed(resume_rng, 43)
        command = [
            sys.executable,
            "-u",
            "-m",
            "benchmarks.run_optimization",
            "--case",
            args.case,
            "--iterations",
            str(budget),
            "--batch-size",
            str(args.batch_size),
            "--population-size",
            str(population_size),
            "--algorithm",
            selected_algorithm,
            "--seed",
            str(seed),
            "--velocity-bound",
            str(args.velocity_bound),
            "--restart-after",
            str(args.restart_after),
            "--temperature",
            str(args.temperature),
            "--anneal-temperature",
            str(args.anneal_temperature),
            "--skip-cpu-verification",
        ]
        if resume_rng.is_file():
            command.extend(["--resume-rng", str(resume_rng)])
        if args.target_score is not None:
            command.append(f"--target-score={args.target_score}")
        if args.rng_jump:
            if args.rng_jump < 0:
                raise ValueError("rng-jump must be non-negative.")
            command.extend(["--rng-jump", str(args.rng_jump)])

    return command


def run_search_and_log(command, case_id, algorithm, segment_number):
    starting_best = starting_search_score(case_id, algorithm)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    log_path = LOG_DIR / (
        f"{case_id}_{algorithm}_segment{segment_number:04d}_"
        f"{started.strftime('%Y%m%d_%H%M%S_%f')}.log"
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {started.isoformat(timespec='seconds')}\n")
        log.write(
            f"Segment: {segment_number}; case: {case_id}; algorithm: {algorithm}\n"
        )
        log.write(
            f"Starting best candidate/checkpoint: {starting_best:.12g}\n\n"
        )
        print(f"Run log: {log_path}", flush=True)

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture optimizer output.")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith("evaluation ") and ": best " in line:
                    write_candidate_submission(case_id)
        except KeyboardInterrupt:
            log.write("\nInterrupted by user before this segment completed.\n")
            log.flush()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            raise
        return_code = process.wait()
        if return_code != 0:
            log.write(f"\nSearch exited with code {return_code}.\n")
            raise subprocess.CalledProcessError(return_code, command)

        best_path = CASE_RESULTS_DIR / f"{case_id}_min_best.npz"
        with np.load(best_path, allow_pickle=False) as saved:
            run_best = float(saved["fitness"])
        improved = run_best < starting_best
        summary = (
            f"\nRun-local best (including starting candidate): {run_best:.12g}\n"
            f"Improved saved best this run: {'yes' if improved else 'no'}\n"
        )
        print(summary, end="", flush=True)
        log.write(summary)

    print(f"Full run log saved to: {log_path}", flush=True)
    return improved


def write_submission_if_better(case_id):
    candidate_path = CASE_RESULTS_DIR / f"{case_id}_min_best.npz"
    submission_path = SUBMISSION_DIR / f"{case_id}_min_submission.json"
    with tempfile.TemporaryDirectory(prefix="esa_submission_verification_") as folder:
        verified_path = Path(folder) / "verified.npz"
        subprocess.run(
            [
                sys.executable,
                "-c",
                VERIFY_SUBMISSION,
                case_id,
                str(candidate_path),
                str(submission_path),
                str(verified_path),
            ],
            cwd=ROOT,
            check=True,
        )
        with np.load(verified_path, allow_pickle=False) as verified:
            candidate_feasible = bool(verified["candidate_feasible"])
            cpu_fitness = float(verified["candidate_fitness"])
            previous_present = bool(verified["previous_present"])
            previous_feasible = bool(verified["previous_feasible"])
            previous_fitness = float(verified["previous_fitness"])

    if not candidate_feasible:
        print(
            "Submission not updated: the candidate is not feasible at every "
            "observation."
        )
        return
    if previous_present and previous_feasible and cpu_fitness >= previous_fitness:
        print(
            f"Submission unchanged: {cpu_fitness:.12g} is not better than "
            f"{previous_fitness:.12g}."
        )
        return

    with np.load(candidate_path, allow_pickle=False) as saved:
        candidate = np.asarray(saved["x"], dtype=np.float64)
    record = submission_record(case_id, candidate, cpu_fitness)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = submission_path.with_name(submission_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump([record], output, indent=6)
            output.write("\n")
        temporary_path.replace(submission_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Updated {submission_path} with CPU-verified score {cpu_fitness:.12g}.")


def write_candidate_submission(case_id):
    candidate_path = CASE_RESULTS_DIR / f"{case_id}_min_best.npz"
    with np.load(candidate_path, allow_pickle=False) as saved:
        candidate = np.asarray(saved["x"], dtype=np.float64)
        fitness = float(saved["fitness"])
    if candidate.ndim != 1 or not np.isfinite(candidate).all():
        raise ValueError(f"Invalid candidate vector in {candidate_path}.")

    record = submission_record(case_id, candidate, fitness)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_DIR / f"{case_id}_min_submission.json"
    temporary_path = submission_path.with_name(submission_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump([record], output, indent=6)
            output.write("\n")
        temporary_path.replace(submission_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Submission updated with new best candidate score {fitness:.12g}.")


def main():
    args = parse_args()
    if args.target_score is not None:
        best_path = CASE_RESULTS_DIR / f"{args.case}_min_best.npz"
        with np.load(best_path, allow_pickle=False) as saved:
            if float(saved["fitness"]) <= args.target_score:
                print(
                    "Target already reached; continuing without a score cutoff.",
                    flush=True,
                )
                args.target_score = None

    segment_number = 0
    active_algorithm = (
        DEFAULT_ALGORITHMS[args.case]
        if args.algorithm == "auto"
        else args.algorithm
    )
    print(
        f"Running {active_algorithm} for {args.case} continuously"
        f"{' with automatic algorithm switching' if args.algorithm == 'auto' else ''}. "
        "Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            segment_number += 1
            command = build_search_command(args, active_algorithm)
            print(
                f"\nStarting segment {segment_number} with {active_algorithm}.",
                flush=True,
            )
            run_search_and_log(
                command, args.case, active_algorithm, segment_number
            )
            write_submission_if_better(args.case)
            if args.algorithm == "auto":
                next_algorithm = next_search_algorithm(
                    args.case, active_algorithm
                )
                print(
                    f"Segment complete; switching from {active_algorithm} "
                    f"to {next_algorithm}. Auto rotates every 5,000,000 "
                    "evaluations regardless of improvement.",
                    flush=True,
                )
                active_algorithm = next_algorithm
            if args.target_score is not None:
                with np.load(
                    CASE_RESULTS_DIR / f"{args.case}_min_best.npz",
                    allow_pickle=False,
                ) as saved:
                    if float(saved["fitness"]) <= args.target_score:
                        print(
                            "Target reached; continuing without a score cutoff.",
                            flush=True,
                        )
                        args.target_score = None
    except KeyboardInterrupt:
        print(
            "\nStopped by Ctrl+C. Completed segments and their verified "
            "submissions are saved.",
            flush=True,
        )
        candidate_path = CASE_RESULTS_DIR / f"{args.case}_min_best.npz"
        with np.load(candidate_path, allow_pickle=False) as saved:
            candidate = np.asarray(saved["x"], dtype=np.float64)
        write_submission_if_better(args.case)


if __name__ == "__main__":
    main()

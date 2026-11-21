"""Build one ESA submission JSON per problem from verified case candidates."""

import json
import errno
import hashlib
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.evaluation.cases import CASE_ORDER, CHALLENGE_ID, get_case, make_reference_problem
from src.evaluation.constraints import (
    is_inside_observation_bounds,
    observation_position_maps,
)


OUTPUT_DIR = Path("results/submissions")

VERIFY_SUBMISSION = """
import json
import sys
from pathlib import Path
import numpy as np
from src.evaluation.cases import get_case, make_reference_problem
from src.evaluation.constraints import (
    is_inside_observation_bounds,
    observation_position_maps,
)

case_id, candidate_path, previous_path, output_path = sys.argv[1:]
spec = get_case(case_id)
udp = make_reference_problem(case_id)
position_maps = observation_position_maps(udp.stms, udp.inflation_factor)
lower = np.asarray([-1.0] * (3 * spec["n_sat"]) + [-10.0] * (3 * spec["n_sat"]))
upper = -lower

with np.load(candidate_path, allow_pickle=False) as saved:
    candidate = np.asarray(saved["x"], dtype=np.float64)
    saved_fitness = float(saved["fitness"])
if candidate.shape != (6 * spec["n_sat"],) or not np.isfinite(candidate).all():
    raise ValueError(f"{case_id} candidate has an invalid decision vector.")
if np.any(candidate < lower) or np.any(candidate > upper):
    raise ValueError(f"{case_id} candidate is outside the ESA input bounds.")
candidate_feasible = is_inside_observation_bounds(
    candidate, position_maps, udp.n_sat
)
candidate_fitness = float(udp.fitness(candidate.tolist())[0])
if not np.isclose(saved_fitness, candidate_fitness, rtol=1e-8, atol=1e-10):
    raise ValueError(
        f"{case_id} saved fitness {saved_fitness} does not match CPU fitness "
        f"{candidate_fitness}."
    )

previous_path = Path(previous_path)
previous_present = previous_path.exists()
previous_feasible = False
previous_fitness = np.nan
if previous_present:
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
        or "decisionVector" not in payload[0]
    ):
        raise ValueError(f"Unexpected submission format in {previous_path}.")
    previous = np.asarray(payload[0]["decisionVector"], dtype=np.float64)
    if previous.shape != (6 * spec["n_sat"],) or not np.isfinite(previous).all():
        raise ValueError(f"Invalid decision vector in {previous_path}.")
    if np.any(previous < lower) or np.any(previous > upper):
        raise ValueError(f"Decision vector in {previous_path} is outside ESA bounds.")
    previous_feasible = is_inside_observation_bounds(
        previous, position_maps, udp.n_sat
    )
    previous_fitness = float(udp.fitness(previous.tolist())[0])

np.savez_compressed(
    output_path,
    candidate_feasible=candidate_feasible,
    candidate_fitness=candidate_fitness,
    previous_present=previous_present,
    previous_feasible=previous_feasible,
    previous_fitness=previous_fitness,
)
"""


def solution_path(case_id):
    return Path("results/cases") / f"{case_id}_min_best.npz"


@contextmanager
def _lock_candidate(path):
    lock_id = hashlib.sha256(
        os.path.normcase(str(path.resolve())).encode("utf-8")
    ).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"esa_candidate_{lock_id}.lock"
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_candidate_progress(case_id, candidate, fitness, *, replace_if=None, **metadata):
    path = solution_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = np.asarray(candidate, dtype=np.float64)
    fitness = float(fitness)
    if not np.isfinite(candidate).all() or not np.isfinite(fitness):
        raise ValueError("Candidate and fitness must be finite.")

    with _lock_candidate(path):
        fields = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                fields = {
                    key: np.asarray(saved[key]).copy() for key in saved.files
                }
            if "fitness" in fields:
                saved_fitness = float(fields["fitness"])
                if (
                    np.isfinite(saved_fitness)
                    and saved_fitness <= fitness
                    and (replace_if is None or not replace_if(fields))
                ):
                    return path

        fields["case_id"] = np.asarray(case_id)
        fields["x"] = candidate
        fields["fitness"] = np.asarray(fitness)
        if "observation_scores" not in metadata:
            fields.pop("observation_scores", None)
        fields.update(
            {key: np.asarray(value) for key, value in metadata.items()}
        )

        temporary_path = path.with_name(
            f"{path.stem}.{os.getpid()}.tmp.npz"
        )
        try:
            np.savez_compressed(temporary_path, **fields)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return path


def submission_record(case_id, candidate, fitness):
    spec = get_case(case_id)
    return {
        "decisionVector": candidate.tolist(),
        "problem": spec["problem_id"],
        "challenge": CHALLENGE_ID,
        "name": f"Interferometric Mission {case_id.title()}",
        "description": (
            f"Locally optimized feasible {case_id} candidate; "
            f"published minimum-observation objective; "
            f"fitness={fitness:.12g}."
        ),
    }


def make_submission_record(case_id, candidate, fitness):
    spec = get_case(case_id)
    candidate = np.asarray(candidate, dtype=np.float64)
    expected_size = 6 * spec["n_sat"]
    if candidate.shape != (expected_size,) or not np.isfinite(candidate).all():
        raise ValueError(
            f"{case_id} candidate must have {expected_size} finite values."
        )
    lower = np.asarray([-1.0] * (3 * spec["n_sat"]) + [-10.0] * (3 * spec["n_sat"]))
    upper = -lower
    if np.any(candidate < lower) or np.any(candidate > upper):
        raise ValueError(f"{case_id} candidate is outside the ESA bounds.")
    reference = make_reference_problem(case_id)
    position_maps = observation_position_maps(
        reference.stms, reference.inflation_factor
    )
    if not is_inside_observation_bounds(
        candidate, position_maps, reference.n_sat
    ):
        raise ValueError(
            f"{case_id} candidate has satellites outside the evaluator bounds."
        )
    cpu_fitness = float(reference.fitness(candidate.tolist())[0])
    if not np.isclose(fitness, cpu_fitness, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{case_id} saved fitness does not match the CPU reference.")

    return submission_record(case_id, candidate, cpu_fitness), cpu_fitness


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for case_id in CASE_ORDER:
        path = solution_path(case_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {case_id} solution at {path}. Run that case first."
            )

        with np.load(path) as saved:
            candidate = np.asarray(saved["x"], dtype=np.float64)
            fitness = float(saved["fitness"])

        record, _ = make_submission_record(case_id, candidate, fitness)
        output_path = OUTPUT_DIR / f"{case_id}_min_submission.json"
        with output_path.open("w", encoding="utf-8") as output:
            json.dump([record], output, indent=6)
        output_paths.append(output_path)

    print("Created one ESA submission file per case:")
    for output_path in output_paths:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()

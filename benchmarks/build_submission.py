"""Build one ESA submission JSON per problem from verified case candidates."""

import json
from pathlib import Path

import numpy as np

from src.evaluation.cases import CASE_ORDER, CHALLENGE_ID, get_case, make_reference_problem


OUTPUT_DIR = Path("results/submissions")


def solution_path(case_id):
    return Path("results/cases") / f"{case_id}_min_best.npz"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for case_id in CASE_ORDER:
        spec = get_case(case_id)
        path = solution_path(case_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {case_id} solution at {path}. Run that case first."
            )

        with np.load(path) as saved:
            candidate = np.asarray(saved["x"], dtype=np.float64)
            fitness = float(saved["fitness"])

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
        cpu_fitness = float(reference.fitness(candidate.tolist())[0])
        if not np.isclose(fitness, cpu_fitness, rtol=1e-8, atol=1e-10):
            raise ValueError(f"{case_id} saved fitness does not match the CPU reference.")

        record = {
            "decisionVector": candidate.tolist(),
            "problem": spec["problem_id"],
            "challenge": CHALLENGE_ID,
            "name": f"Interferometric Mission {case_id.title()}",
            "description": (
                f"Locally optimized and CPU-verified {case_id} candidate; "
                f"published minimum-observation objective; fitness={fitness:.12g}."
            ),
        }
        output_path = OUTPUT_DIR / f"{case_id}_min_submission.json"
        with output_path.open("w", encoding="utf-8") as output:
            json.dump([record], output, indent=6)
        output_paths.append(output_path)

    print("Created one ESA submission file per case:")
    for output_path in output_paths:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

import benchmarks.build_submission as build_submission


@contextmanager
def unlocked_candidate_file(_):
    yield


class CandidateSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "hard_min_best.npz"
        self.solution_path_patch = patch.object(
            build_submission, "solution_path", return_value=self.path
        )
        self.lock_patch = patch.object(
            build_submission, "_lock_candidate", unlocked_candidate_file
        )
        self.solution_path_patch.start()
        self.lock_patch.start()

    def tearDown(self):
        self.lock_patch.stop()
        self.solution_path_patch.stop()
        self.temp_dir.cleanup()

    def test_stale_candidate_cannot_replace_better_saved_score(self):
        build_submission.save_candidate_progress(
            "hard", [1.0, 2.0], -1.4069, algorithm="new"
        )
        build_submission.save_candidate_progress(
            "hard", [3.0, 4.0], -1.1844, algorithm="stale"
        )

        with np.load(self.path, allow_pickle=False) as saved:
            self.assertEqual(float(saved["fitness"]), -1.4069)
            np.testing.assert_array_equal(saved["x"], [1.0, 2.0])
            self.assertEqual(str(saved["algorithm"]), "new")

    def test_strictly_better_score_replaces_saved_candidate(self):
        build_submission.save_candidate_progress(
            "hard", [1.0, 2.0], -1.4069, algorithm="old"
        )
        build_submission.save_candidate_progress(
            "hard", [5.0, 6.0], -1.5, algorithm="new"
        )

        with np.load(self.path, allow_pickle=False) as saved:
            self.assertEqual(float(saved["fitness"]), -1.5)
            np.testing.assert_array_equal(saved["x"], [5.0, 6.0])
            self.assertEqual(str(saved["algorithm"]), "new")

    def test_invalid_saved_candidate_can_be_replaced_for_repair(self):
        build_submission.save_candidate_progress(
            "hard", [1.0, 2.0], -2.0, feasible=False
        )
        build_submission.save_candidate_progress(
            "hard",
            [5.0, 6.0],
            -1.5,
            replace_if=lambda saved: not bool(saved["feasible"]),
            feasible=True,
        )

        with np.load(self.path, allow_pickle=False) as saved:
            self.assertEqual(float(saved["fitness"]), -1.5)
            np.testing.assert_array_equal(saved["x"], [5.0, 6.0])
            self.assertTrue(bool(saved["feasible"]))


if __name__ == "__main__":
    unittest.main()

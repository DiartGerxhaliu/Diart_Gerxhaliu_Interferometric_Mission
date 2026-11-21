import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.run_coordinate_search import (
    RANDOM_RESTART_EVERY,
    evaluations_until_random_restart,
    make_partial_random_restart_candidate,
    propose_boundary_candidates,
)


class CoordinateRestartTests(unittest.TestCase):
    def test_restart_count_is_based_on_stagnant_evaluations(self):
        self.assertEqual(evaluations_until_random_restart(0), RANDOM_RESTART_EVERY)
        self.assertEqual(evaluations_until_random_restart(999_999), 1)

    def test_restart_randomizes_a_random_partial_number_of_satellites(self):
        n_sat = 40
        candidate, satellites = make_partial_random_restart_candidate(
            np.zeros(6 * n_sat),
            -np.ones(6 * n_sat),
            np.ones(6 * n_sat),
            n_sat,
            np.random.default_rng(42),
        )

        self.assertGreaterEqual(len(satellites), 1)
        self.assertLessEqual(len(satellites), 10)
        changed = np.flatnonzero(
            np.any(candidate.reshape(6, n_sat) != 0, axis=0)
        )
        np.testing.assert_array_equal(np.sort(changed), np.sort(satellites))

    def test_single_satellite_boundary_moves_are_sampled_across_their_actions(self):
        n_sat = 4
        actions = [
            [np.full(6, satellite + 1.0)] for satellite in range(n_sat)
        ]
        rng = np.random.default_rng(42)
        expected = np.sort(rng.choice(n_sat, size=2, replace=False))

        with patch(
            "benchmarks.run_coordinate_search.boundary_actions",
            return_value=actions,
        ):
            candidates = propose_boundary_candidates(
                np.zeros(6 * n_sat),
                2,
                n_sat,
                -np.ones(6 * n_sat),
                np.ones(6 * n_sat),
                [np.eye(3)],
                [np.eye(3)],
                21,
                1,
                np.random.default_rng(42),
            )

        changed_satellites = np.flatnonzero(
            np.any(candidates.reshape(2, 6, n_sat) != 0, axis=(0, 1))
        )
        self.assertEqual(len(candidates), 2)
        np.testing.assert_array_equal(changed_satellites, expected)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from benchmarks.run_coordinate_search import (
    RANDOM_RESTART_EVERY,
    evaluations_until_random_restart,
    make_partial_random_restart_candidate,
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


if __name__ == "__main__":
    unittest.main()

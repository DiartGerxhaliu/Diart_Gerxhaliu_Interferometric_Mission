import unittest

import numpy as np

from benchmarks.diverse_archive import (
    candidate_distance,
    select_archive_seed,
    update_diverse_archive,
)


class DiverseArchiveTests(unittest.TestCase):
    def test_distance_is_normalized(self):
        self.assertEqual(candidate_distance([0.0, 0.0], [1.0, 1.0]), np.sqrt(1.0))

    def test_archive_starts_with_first_candidate(self):
        x, fitness, secondary = update_diverse_archive(
            None, None, None, [0.0, 0.0], -1.0, 2.0
        )
        np.testing.assert_array_equal(x, [[0.0, 0.0]])
        np.testing.assert_array_equal(fitness, [-1.0])
        np.testing.assert_array_equal(secondary, [2.0])

    def test_archive_replaces_nearby_improvement(self):
        x = np.asarray([[0.0, 0.0]])
        fitness = np.asarray([-1.0])
        secondary = np.asarray([1.0])
        x, fitness, secondary = update_diverse_archive(
            x,
            fitness,
            secondary,
            [0.01, 0.01],
            -2.0,
            3.0,
            minimum_distance=0.05,
        )
        np.testing.assert_allclose(x, [[0.01, 0.01]])
        np.testing.assert_array_equal(fitness, [-2.0])
        np.testing.assert_array_equal(secondary, [3.0])

    def test_archive_adds_distinct_basin_and_respects_limit(self):
        x = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        fitness = np.asarray([-1.0, -2.0])
        secondary = np.asarray([2.0, 1.0])
        x, fitness, secondary = update_diverse_archive(
            x,
            fitness,
            secondary,
            [0.0, 1.0],
            -1.5,
            1.5,
            max_size=2,
            minimum_distance=0.05,
        )
        self.assertEqual(len(x), 2)

    def test_archive_seed_is_returned_from_the_archive(self):
        x = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        fitness = np.asarray([-1.0, -2.0])
        secondary = np.asarray([2.0, 1.0])
        seed, _, _ = select_archive_seed(
            x, fitness, secondary, np.random.default_rng(123)
        )
        self.assertTrue(any(np.all(seed == candidate) for candidate in x))


if __name__ == "__main__":
    unittest.main()

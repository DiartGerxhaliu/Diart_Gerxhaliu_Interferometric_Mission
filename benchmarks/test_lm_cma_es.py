import unittest

import numpy as np

from benchmarks.lm_cma_es import LimitedMemoryCMAES, reflect_to_unit_box


class LimitedMemoryCMATests(unittest.TestCase):
    def test_sampling_and_update_use_limited_memory(self):
        rng = np.random.default_rng(21)
        strategy = LimitedMemoryCMAES(np.zeros(12), 8, sigma=0.2, memory=3)

        z, directions = strategy.ask(8, rng)
        strategy.tell(z, directions, np.arange(8, dtype=np.float64), rng)

        self.assertEqual(strategy.transforms.shape, (3, 12))
        self.assertEqual(strategy.generation, 1)
        self.assertTrue(np.isfinite(strategy.mean).all())
        self.assertTrue(np.isfinite(strategy.sigma))
        self.assertGreater(strategy.sigma, 0.0)

    def test_ranked_candidates_move_the_mean_toward_better_steps(self):
        strategy = LimitedMemoryCMAES(np.zeros(6), 4, sigma=0.2, memory=2)
        z = np.zeros((4, 6))
        directions = np.zeros((4, 6))
        directions[:2, 0] = 1.0

        strategy.tell(
            z,
            directions,
            np.asarray([0.0, 0.0, 1.0, 2.0]),
            np.random.default_rng(22),
        )

        self.assertGreater(strategy.mean[0], 0.0)
        np.testing.assert_array_equal(strategy.mean[1:], np.zeros(5))

    def test_equal_fitnesses_do_not_drive_random_mean_or_sigma_updates(self):
        rng = np.random.default_rng(23)
        strategy = LimitedMemoryCMAES(np.zeros(12), 8, sigma=0.2, memory=3)
        z, directions = strategy.ask(8, rng)
        initial_mean = strategy.mean.copy()
        initial_path = strategy.path.copy()
        initial_transforms = strategy.transforms.copy()

        strategy.tell(z, directions, np.ones(8), rng)

        np.testing.assert_array_equal(strategy.mean, initial_mean)
        np.testing.assert_array_equal(strategy.path, initial_path)
        np.testing.assert_array_equal(strategy.transforms, initial_transforms)
        self.assertEqual(strategy.sigma, 0.2)
        self.assertEqual(strategy.generation, 1)

    def test_secondary_scores_guide_updates_on_official_fitness_ties(self):
        strategy = LimitedMemoryCMAES(np.zeros(6), 4, sigma=0.2, memory=2)
        z = np.zeros((4, 6))
        directions = np.zeros((4, 6))
        directions[:2, 0] = 1.0

        strategy.tell(
            z,
            directions,
            np.ones(4),
            np.random.default_rng(231),
            secondary_scores=np.asarray([4.0, 3.0, 2.0, 1.0]),
        )

        self.assertGreater(strategy.mean[0], 0.0)

    def test_strategy_improves_a_smooth_objective(self):
        rng = np.random.default_rng(42)
        strategy = LimitedMemoryCMAES(np.full(12, 2.0), 12, sigma=0.4, memory=5)
        initial_score = float(np.dot(strategy.mean, strategy.mean))
        best_score = initial_score

        for _ in range(100):
            z, directions = strategy.ask(12, rng)
            candidates = strategy.mean + strategy.sigma * directions
            scores = np.sum(candidates * candidates, axis=1)
            best_score = min(best_score, float(scores.min()))
            strategy.tell(z, directions, scores, rng)

        self.assertLess(best_score, initial_score * 1e-6)

    def test_restart_reinitializes_paths_and_reflection_respects_bounds(self):
        strategy = LimitedMemoryCMAES(np.ones(6), 4, sigma=0.2, memory=2)
        strategy.path.fill(1.0)
        strategy.transforms.fill(1.0)
        strategy.generation = 5

        strategy.restart(np.zeros(6), 0.3)
        reflected = reflect_to_unit_box(np.asarray([-5.0, -1.0, 0.0, 1.0, 5.0]))

        np.testing.assert_array_equal(strategy.mean, np.zeros(6))
        self.assertEqual(strategy.sigma, 0.3)
        self.assertEqual(strategy.generation, 0)
        self.assertTrue(np.all(strategy.path == 0.0))
        self.assertTrue(np.all(strategy.transforms == 0.0))
        self.assertTrue(np.all(reflected >= -1.0))
        self.assertTrue(np.all(reflected <= 1.0))

    def test_requires_a_population_with_multiple_parents(self):
        with self.assertRaisesRegex(ValueError, "population-size >= 4"):
            LimitedMemoryCMAES(np.zeros(6), 3)


if __name__ == "__main__":
    unittest.main()

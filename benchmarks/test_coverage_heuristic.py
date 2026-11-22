import unittest
from types import SimpleNamespace

import numpy as np

from benchmarks.coverage_heuristic import (
    make_coverage_guided_offspring,
    satellite_coverage_guidance,
)
from src.evaluation.objectives import observation_scores


class CoverageHeuristicTests(unittest.TestCase):
    def make_problem(self):
        rng = np.random.default_rng(31)
        n_sat = 8
        grid_size = 11
        position_maps = np.zeros((3, 3, 6), dtype=np.float64)
        position_maps[0, :, :3] = np.eye(3)
        position_maps[1, :, :3] = np.eye(3) * 0.8
        position_maps[1, :, 3:] = np.eye(3) * 0.1
        position_maps[2, :, :3] = np.eye(3) * 0.6
        position_maps[2, :, 3:] = np.eye(3) * -0.15
        stms = np.zeros((3, 6, 6), dtype=np.float64)
        stms[:, :3] = position_maps
        stms[:, 3:, 3:] = np.eye(3)
        problem = SimpleNamespace(
            n_sat=n_sat,
            grid_size=grid_size,
            n_meas=3,
            scaling_factor=1.0,
            inflation_factor=1.0,
            stms=stms,
        )
        states = rng.uniform(-0.7, 0.7, size=(n_sat, 6))
        return problem, position_maps, states.T.reshape(-1)

    def test_guidance_scores_match_the_official_cpu_objective(self):
        problem, position_maps, candidate = self.make_problem()

        weakest, scores, contributions, weights = satellite_coverage_guidance(
            candidate, position_maps, problem.grid_size, problem.n_sat
        )

        np.testing.assert_allclose(scores, observation_scores(candidate, problem))
        self.assertEqual(weakest, int(np.argmin(scores)))
        self.assertEqual(contributions.shape, (problem.n_sat,))
        self.assertTrue(np.all(contributions >= 0.0))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_guided_offspring_are_bounded_and_bias_selected_satellites(self):
        n_sat = 4
        weights = np.asarray([0.7, 0.1, 0.1, 0.1])
        candidates = make_coverage_guided_offspring(
            np.zeros(6 * n_sat),
            512,
            np.full(6 * n_sat, -10.0),
            np.full(6 * n_sat, 10.0),
            n_sat,
            np.eye(6) * 0.1,
            weights,
            np.random.default_rng(32),
        )

        changed = np.any(candidates.reshape(-1, 6, n_sat) != 0.0, axis=1)
        self.assertTrue(np.isfinite(candidates).all())
        self.assertTrue(np.all(candidates >= -10.0))
        self.assertTrue(np.all(candidates <= 10.0))
        self.assertGreater(changed[:, 0].mean(), changed[:, 1:].mean())
        greedy_changes = changed[-64:]
        np.testing.assert_array_equal(greedy_changes.sum(axis=1), np.ones(64))

    def test_greedy_probes_are_centered_on_their_diverse_parent(self):
        n_sat = 4
        count = 16
        parents = np.repeat(
            np.arange(count, dtype=np.float64)[:, None] * 0.01,
            6 * n_sat,
            axis=1,
        )
        candidates = make_coverage_guided_offspring(
            parents,
            count,
            np.full(6 * n_sat, -10.0),
            np.full(6 * n_sat, 10.0),
            n_sat,
            np.eye(6) * 0.1,
            np.full(n_sat, 1.0 / n_sat),
            np.random.default_rng(33),
        )

        greedy_start = count - count // 8
        for row in range(greedy_start, count):
            changed = np.any(
                candidates[row].reshape(6, n_sat)
                != parents[row].reshape(6, n_sat),
                axis=0,
            )
            self.assertEqual(int(changed.sum()), 1)


if __name__ == "__main__":
    unittest.main()

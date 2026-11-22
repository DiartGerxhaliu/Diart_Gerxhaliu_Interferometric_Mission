import unittest

import numpy as np

from benchmarks.run_optimization import (
    adapt_de_step_multiplier,
    adapt_genetic_step_multiplier,
    block_annealing_temperature,
    make_block_proposals,
    make_de_offspring,
    make_offspring,
    mutate_candidates,
    parent_improvement_rate,
    random_candidates,
    rank_fitnesses,
    sample_de_indices,
    update_de_population,
)


class PopulationAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.n_sat = 8
        self.lower = np.full(6 * self.n_sat, -1.0)
        self.upper = np.full(6 * self.n_sat, 1.0)
        self.mutation_map = np.random.default_rng(7).normal(size=(6, 6)) * 0.01

    def test_genetic_mutation_uses_satellite_state_matrix(self):
        parents = np.zeros((4, 6 * self.n_sat))
        children = mutate_candidates(
            parents,
            self.lower,
            self.upper,
            self.n_sat,
            self.mutation_map,
            np.random.default_rng(8),
        )

        self.assertEqual(children.shape, parents.shape)
        self.assertTrue(np.isfinite(children).all())
        self.assertTrue(np.any(children != 0.0))
        self.assertTrue(np.all(children >= self.lower))
        self.assertTrue(np.all(children <= self.upper))

    def test_random_candidates_cover_more_of_the_velocity_bounds(self):
        lower = self.lower.copy()
        upper = self.upper.copy()
        lower[3 * self.n_sat :] = -10.0
        upper[3 * self.n_sat :] = 10.0
        candidates = random_candidates(
            1024, lower, upper, self.n_sat, 10.0, np.random.default_rng(18)
        )

        velocity_coverage = np.mean(
            np.abs(candidates[:, 3 * self.n_sat :]) / 10.0
        )
        self.assertGreater(velocity_coverage, 0.29)

    def test_genetic_mutations_change_broader_satellite_subsets(self):
        parents = np.zeros((1024, 6 * self.n_sat))
        children = mutate_candidates(
            parents,
            np.full(6 * self.n_sat, -100.0),
            np.full(6 * self.n_sat, 100.0),
            self.n_sat,
            np.eye(6) * 0.01,
            np.random.default_rng(19),
        )
        states_changed = np.any(children.reshape(-1, 6, self.n_sat) != 0, axis=1)

        self.assertGreater(states_changed.sum(axis=1).mean() / self.n_sat, 0.30)

    def test_mutation_map_must_match_reference_state_matrix(self):
        with self.assertRaisesRegex(ValueError, r"shape \(6, 6\)"):
            mutate_candidates(
                np.zeros((1, 6 * self.n_sat)),
                self.lower,
                self.upper,
                self.n_sat,
                np.zeros((6 * self.n_sat, 9)),
                np.random.default_rng(12),
            )

    def test_genetic_and_block_anneal_proposals_are_bounded(self):
        population = np.zeros((8, 6 * self.n_sat))
        population[4:] = 0.25
        fitnesses = np.arange(8, dtype=np.float64)
        rng = np.random.default_rng(9)

        children, _, _ = make_offspring(
            population,
            fitnesses,
            4,
            self.lower,
            self.upper,
            self.n_sat,
            1.0,
            self.mutation_map,
            rng,
            0.35,
            1.0,
        )
        proposals = make_block_proposals(
            population[:4],
            population[4:],
            self.lower,
            self.upper,
            self.n_sat,
            self.mutation_map,
            rng,
            0.12,
        )

        for candidates in (children, proposals):
            self.assertEqual(candidates.shape, (4, 6 * self.n_sat))
            self.assertTrue(np.isfinite(candidates).all())
            self.assertTrue(np.all(candidates >= self.lower))
            self.assertTrue(np.all(candidates <= self.upper))

    def test_block_annealing_includes_full_formation_moves(self):
        population = np.zeros((2000, 6 * self.n_sat))
        proposals = make_block_proposals(
            population,
            population,
            np.full(6 * self.n_sat, -1.0),
            np.full(6 * self.n_sat, 1.0),
            self.n_sat,
            np.eye(6) * 0.01,
            np.random.default_rng(20),
            0.12,
        )
        changed_counts = np.any(
            proposals.reshape(-1, 6, self.n_sat) != 0.0, axis=1
        ).sum(axis=1)

        self.assertTrue(np.any(changed_counts == self.n_sat))

    def test_differential_evolution_generates_small_batches(self):
        population = np.random.default_rng(10).uniform(
            self.lower, self.upper, size=(4, 6 * self.n_sat)
        )
        children, targets, immigrants = make_de_offspring(
            population,
            1,
            self.lower,
            self.upper,
            self.n_sat,
            1.0,
            np.random.default_rng(11),
            1.0,
            population[0],
        )

        self.assertEqual(children.shape, (1, 6 * self.n_sat))
        self.assertEqual(targets.shape, (1,))
        self.assertEqual(len(immigrants), 1)

    def test_de_samples_four_distinct_indices_per_trial(self):
        indices = sample_de_indices(256, 4, np.random.default_rng(13))

        self.assertEqual(indices.shape, (4, 256))
        self.assertTrue(
            all(np.unique(indices[:, column]).size == 4 for column in range(256))
        )

    def test_de_keeps_only_the_best_improving_trial_per_target(self):
        population = np.zeros((4, 2))
        population_fitness = np.zeros(4)
        children = np.asarray([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
        child_fitness = np.asarray([-1.0, -2.0, 1.0, -1.0, -2.0])
        target_indices = np.asarray([1, 1, 1, 3, 3])

        update_de_population(
            population,
            population_fitness,
            children,
            child_fitness,
            target_indices,
            np.random.default_rng(14),
        )

        np.testing.assert_array_equal(population[1], children[1])
        np.testing.assert_array_equal(population[3], children[4])
        np.testing.assert_array_equal(population[[0, 2]], np.zeros((2, 2)))
        np.testing.assert_array_equal(population_fitness, [0.0, -2.0, 0.0, -2.0])

    def test_de_can_drift_to_a_novel_candidate_on_a_fitness_plateau(self):
        population = np.zeros((2, 2))
        children = np.asarray([[1.0, 0.0]])
        population_fitness = np.zeros(2)
        child_fitness = np.zeros(1)

        update_de_population(
            population,
            population_fitness,
            children,
            child_fitness,
            np.asarray([0]),
            np.random.default_rng(15),
            neutral_acceptance=1.0,
        )

        np.testing.assert_array_equal(population[0], children[0])

    def test_de_prefers_secondary_gain_when_official_fitness_is_tied(self):
        population = np.zeros((1, 2))
        population_fitness = np.zeros(1)
        population_secondary = np.asarray([1.0])
        children = np.asarray([[1.0, 0.0]])
        child_fitness = np.zeros(1)
        child_secondary = np.asarray([2.0])

        update_de_population(
            population,
            population_fitness,
            children,
            child_fitness,
            np.asarray([0]),
            np.random.default_rng(151),
            neutral_acceptance=0.0,
            population_secondary=population_secondary,
            child_secondary=child_secondary,
        )

        np.testing.assert_array_equal(population[0], children[0])
        np.testing.assert_array_equal(population_secondary, [2.0])

    def test_equal_fitnesses_receive_randomized_genetic_ranks(self):
        fitnesses = np.ones(32)
        first = rank_fitnesses(fitnesses, np.random.default_rng(16))
        second = rank_fitnesses(fitnesses, np.random.default_rng(17))

        np.testing.assert_array_equal(np.sort(first), np.arange(len(fitnesses)))
        self.assertFalse(np.array_equal(first, second))

    def test_secondary_scores_rank_candidates_only_when_official_scores_tie(self):
        fitnesses = np.asarray([-1.0, -1.0, -0.5])
        secondary = np.asarray([1.0, 2.0, 100.0])

        order = rank_fitnesses(
            fitnesses, np.random.default_rng(161), secondary
        )

        np.testing.assert_array_equal(order, [1, 0, 2])

    def test_secondary_gain_counts_as_parent_improvement(self):
        self.assertEqual(
            parent_improvement_rate(
                [0.0],
                [0.0],
                child_secondary=[2.0],
                parent_secondary=[1.0],
            ),
            1.0,
        )

    def test_adaptive_mutation_scales_stay_bounded_on_plateaus(self):
        de_scale = 1.0
        genetic_scale = 1.0
        for _ in range(500):
            de_scale = adapt_de_step_multiplier(de_scale, 0.0)
            genetic_scale = adapt_genetic_step_multiplier(genetic_scale, 0.0)
        self.assertLessEqual(de_scale, 2.0)
        self.assertLessEqual(genetic_scale, 2.0)

        for _ in range(500):
            de_scale = adapt_de_step_multiplier(de_scale, 1.0)
            genetic_scale = adapt_genetic_step_multiplier(genetic_scale, 1.0)
        self.assertGreaterEqual(de_scale, 0.25)
        self.assertGreaterEqual(genetic_scale, 0.5)

    def test_block_annealing_uses_its_own_cooling_temperature(self):
        self.assertEqual(block_annealing_temperature(0.005, 1, 100), 0.005)
        self.assertAlmostEqual(
            block_annealing_temperature(0.005, 400, 100), 0.0002500296875
        )

    def test_no_eligible_parent_trials_have_zero_improvement_rate(self):
        self.assertEqual(parent_improvement_rate([-2.0], [-1.0], [0]), 0.0)
        self.assertEqual(parent_improvement_rate([-2.0], [-1.0]), 1.0)
        self.assertAlmostEqual(
            parent_improvement_rate(
                [0.0], [0.0], neutral_acceptance=0.2
            ),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()

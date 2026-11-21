import unittest
from types import SimpleNamespace

from benchmarks.lm_cma_es import default_population_size
from main import build_search_command, next_search_algorithm
from src.evaluation.cases import get_case


class MainAlgorithmSelectionTests(unittest.TestCase):
    def auto_args(self):
        return SimpleNamespace(
            case="medium",
            algorithm="auto",
            rng_jump=0,
            batch_size=2048,
            population_size=None,
            seed=42,
            velocity_bound=10.0,
            restart_after=100,
            temperature=0.35,
            target_score=None,
            anneal_temperature=0.005,
            checkpoint_every=25,
            resume_rng=None,
        )

    def test_medium_auto_starts_with_de_then_cycles_to_coordinate_search(self):
        args = self.auto_args()
        command = build_search_command(args)
        algorithm_index = command.index("--algorithm")
        self.assertEqual(command[algorithm_index + 1], "de")
        self.assertEqual(next_search_algorithm("medium", "de"), "coordinate-search")

    def test_auto_gives_every_algorithm_a_five_million_evaluation_segment(self):
        for algorithm in (
            "coordinate-search",
            "de",
            "genetic",
            "block-anneal",
            "lm-cma-es",
        ):
            command = build_search_command(self.auto_args(), algorithm)
            budget_argument = (
                "--evaluations" if algorithm == "coordinate-search" else "--iterations"
            )
            self.assertEqual(
                command[command.index(budget_argument) + 1], "5000000"
            )

    def test_auto_cycle_visits_all_five_algorithms(self):
        algorithm = "de"
        visited = []
        for _ in range(5):
            algorithm = next_search_algorithm("hard", algorithm)
            visited.append(algorithm)

        self.assertEqual(
            visited,
            [
                "coordinate-search",
                "genetic",
                "block-anneal",
                "lm-cma-es",
                "de",
            ],
        )

    def test_lm_cma_es_uses_dimension_scaled_population(self):
        args = self.auto_args()
        command = build_search_command(args, "lm-cma-es")
        expected = default_population_size(6 * get_case(args.case)["n_sat"])

        self.assertEqual(
            command[command.index("--population-size") + 1], str(expected)
        )

    def test_population_search_passes_separate_annealing_temperature(self):
        args = self.auto_args()
        command = build_search_command(args, "block-anneal")

        self.assertEqual(
            command[command.index("--anneal-temperature") + 1],
            str(args.anneal_temperature),
        )


if __name__ == "__main__":
    unittest.main()

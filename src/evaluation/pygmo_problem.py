
import numpy as np

from src.evaluation.gpu import gpu_fitness_batch


class GPUProblem:

    def __init__(self, reference_problem, batch_size=512):
        self.reference_problem = reference_problem
        self.batch_size = batch_size

    def fitness(self, x):
        # CPU fallback for individual evaluations.
        return self.reference_problem.fitness(x)

    def batch_fitness(self, dvs):
        # PyGMO supplies all decision vectors as one flattened array.
        n_variables = len(self.reference_problem.get_bounds()[0])

        candidates = np.asarray(
            dvs,
            dtype=np.float64
        ).reshape(-1, n_variables)

        results = []

        # Process candidates in GPU-friendly batches.
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]

            fitness_values = gpu_fitness_batch(
                batch,
                self.reference_problem
            )

            results.extend(
                fitness_values.detach().cpu().numpy().tolist()
            )

        return np.asarray(results, dtype=np.float64)

    def get_bounds(self):
        lower, upper = self.reference_problem.get_bounds()

        n_sat = self.reference_problem.n_sat

        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        lower[3 * n_sat:] = -0.5
        upper[3 * n_sat:] = 0.5

        return lower.tolist(), upper.tolist()

    def get_nix(self):
        return self.reference_problem.get_nix()

    def get_name(self):
        return "ESA Interferometric Mission - GPU"

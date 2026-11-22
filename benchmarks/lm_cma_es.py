"""Limited-memory matrix-adaptation evolution strategy for normalized vectors."""

import numpy as np


def default_population_size(dimension):
    """Return a modest logarithmic offspring count for the search dimension."""
    if dimension < 1:
        raise ValueError("Dimension must be positive.")
    return max(4, 4 + int(3 * np.log(dimension)))


class LimitedMemoryCMAES:
    """Rank-based LM-CMA-ES with O(memory * dimension) covariance storage.

    The update follows the limited-memory matrix adaptation strategy described
    by Loshchilov, Glasmachers, and Beyer (IEEE TEC, 2019).
    """

    def __init__(self, mean, population_size, sigma=0.15, memory=None):
        mean = np.asarray(mean, dtype=np.float64)
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("Mean must be a non-empty one-dimensional vector.")
        if population_size < 4:
            raise ValueError("LM-CMA-ES requires population-size >= 4.")
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        self.mean = mean.copy()
        self.dimension = mean.size
        self.population_size = int(population_size)
        self.n_parents = self.population_size // 2
        raw_weights = np.log(self.n_parents + 0.5) - np.log(
            np.arange(1, self.n_parents + 1)
        )
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        self.memory = (
            max(1, 4 + int(3 * np.log(self.dimension)))
            if memory is None
            else int(memory)
        )
        if self.memory < 1:
            raise ValueError("Memory must be positive.")

        self.c_sigma = min(2.0 * self.population_size / self.dimension, 1.0)
        self.path_decay = 1.0 - self.c_sigma
        self.path_gain = np.sqrt(
            self.mu_eff * self.c_sigma * (2.0 - self.c_sigma)
        )
        path_indices = np.arange(self.memory)
        self.path_rates = np.minimum(
            self.population_size
            / (self.dimension * np.power(4.0, path_indices)),
            1.0,
        )
        self.direction_rates = 1.0 / (
            self.dimension * np.power(1.5, path_indices)
        )
        self.sigma = float(sigma)
        self.path = np.zeros(self.dimension, dtype=np.float64)
        self.transforms = np.zeros(
            (self.memory, self.dimension), dtype=np.float64
        )
        self.generation = 0

    def ask(self, count, rng):
        """Sample standard-normal vectors and their LM-CMA transformed steps."""
        if count < 1 or count > self.population_size:
            raise ValueError("Sample count must be between 1 and population-size.")

        z = rng.standard_normal((count, self.dimension))
        directions = z.copy()
        for index in range(min(self.generation, self.memory)):
            path = self.transforms[index]
            projection = directions @ path
            rate = self.direction_rates[index]
            directions = (1.0 - rate) * directions + (
                rate * projection[:, None] * path
            )
        return z, directions

    def tell(self, z, directions, fitnesses, rng, secondary_scores=None):
        """Update the mean, limited-memory paths, and global step size."""
        expected_shape = (self.population_size, self.dimension)
        if z.shape != expected_shape or directions.shape != expected_shape:
            raise ValueError(
                f"LM-CMA-ES updates require arrays of shape {expected_shape}."
            )
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        if fitnesses.shape != (self.population_size,):
            raise ValueError(
                f"Fitnesses must have shape ({self.population_size},)."
            )
        if not np.isfinite(fitnesses).all():
            raise ValueError("Fitnesses must be finite.")
        if secondary_scores is None:
            secondary_scores = np.zeros(self.population_size, dtype=np.float64)
        else:
            secondary_scores = np.asarray(secondary_scores, dtype=np.float64)
            if secondary_scores.shape != (self.population_size,):
                raise ValueError(
                    f"Secondary scores must have shape ({self.population_size},)."
                )
            if not np.isfinite(secondary_scores).all():
                raise ValueError("Secondary scores must be finite.")
        if np.all(fitnesses == fitnesses[0]) and np.all(
            secondary_scores == secondary_scores[0]
        ):
            self.generation += 1
            return

        order = np.lexsort(
            (rng.random(self.population_size), -secondary_scores, fitnesses)
        )
        selected = order[: self.n_parents]
        weighted_z = self.weights @ z[selected]
        weighted_directions = self.weights @ directions[selected]
        self.mean += self.sigma * weighted_directions
        self.path = self.path_decay * self.path + self.path_gain * weighted_z

        for index, rate in enumerate(self.path_rates):
            gain = np.sqrt(self.mu_eff * rate * (2.0 - rate))
            self.transforms[index] = (
                (1.0 - rate) * self.transforms[index] + gain * weighted_z
            )

        exponent = (
            self.c_sigma
            / 2.0
            * (np.dot(self.path, self.path) / self.dimension - 1.0)
        )
        self.sigma = float(
            np.clip(self.sigma * np.exp(np.clip(exponent, -50.0, 50.0)), 1e-6, 2.0)
        )
        self.generation += 1

    def restart(self, mean, sigma):
        """Recenter after stagnation while retaining the configured strategy."""
        mean = np.asarray(mean, dtype=np.float64)
        if mean.shape != (self.dimension,) or not np.isfinite(mean).all():
            raise ValueError("Restart mean must be finite and match the dimension.")
        if sigma <= 0:
            raise ValueError("Restart sigma must be positive.")
        self.mean = mean.copy()
        self.sigma = float(np.clip(sigma, 1e-6, 2.0))
        self.path.fill(0.0)
        self.transforms.fill(0.0)
        self.generation = 0


def reflect_to_unit_box(candidates):
    """Reflect normalized candidates into [-1, 1] instead of clipping them."""
    values = np.asarray(candidates, dtype=np.float64)
    folded = np.remainder(values + 1.0, 4.0)
    return np.where(folded <= 2.0, folded - 1.0, 3.0 - folded)

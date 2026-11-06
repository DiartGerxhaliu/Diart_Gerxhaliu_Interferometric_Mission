
import numpy as np
import pygmo as pg  # pyright: ignore[reportMissingImports]

from reference.esa_reference import udp
from src.evaluation.pygmo_problem import GPUProblem


gpu_udp = GPUProblem(udp, batch_size=512)

# Wrap our problem in PyGMO.
problem = pg.problem(gpu_udp)

print("PyGMO version:", pg.__version__)
print("Problem:", problem.get_name())
print("Decision variables:", problem.get_nx())
print("Objectives:", problem.get_nf())
print("Batch fitness supported:", problem.has_batch_fitness())

# Generate 32 random candidates.
lower, upper = udp.get_bounds()
rng = np.random.default_rng(42)

candidates = rng.uniform(
    lower,
    upper,
    size=(32, len(lower))
)

# PyGMO expects flattened decision vectors.
results = problem.batch_fitness(candidates.flatten())

# Compare against the original CPU reference.
cpu_results = np.array([
    udp.fitness(candidate.tolist())[0]
    for candidate in candidates
])

print("Candidates tested:", len(candidates))
print("Maximum difference:", np.max(np.abs(results - cpu_results)))
print("All close:", np.allclose(results, cpu_results))

assert problem.has_batch_fitness()
assert np.allclose(results, cpu_results)

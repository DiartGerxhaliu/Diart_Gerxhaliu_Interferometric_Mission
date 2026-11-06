
import time
from pathlib import Path

import numpy as np
import pygmo as pg  # type: ignore[reportMissingImports]

from reference.esa_reference import udp
from src.evaluation.pygmo_problem import GPUProblem


POPULATION_SIZE = 512
GENERATIONS = 20
BATCH_SIZE = 512
SEED = 42


# Initialize the GPU problem.
gpu_udp = GPUProblem(udp, batch_size=BATCH_SIZE)
problem = pg.problem(gpu_udp)

# Configure batch fitness evaluation.
bfe = pg.bfe()

# Generate the initial population.
population = pg.population(
    problem,
    size=POPULATION_SIZE,
    b=bfe,
    seed=SEED
)

initial_best = float(population.champion_f[0])

print("Optimization algorithm: CMA-ES")
print("Population size:", POPULATION_SIZE)
print("Generations:", GENERATIONS)
print("Initial best fitness:", initial_best)


# Configure CMA-ES.
cmaes = pg.cmaes(
    gen=GENERATIONS,
    sigma0=1,
    ftol=0.0,
    xtol=0.0,
    force_bounds=True,
    seed=SEED
)

cmaes.set_bfe(bfe)

algorithm = pg.algorithm(cmaes)
algorithm.set_verbosity(1)


# Run optimization.
start = time.perf_counter()

population = algorithm.evolve(population)

elapsed = time.perf_counter() - start


# Retrieve the best solution.
best_x = population.champion_x
best_fitness = float(population.champion_f[0])


# Verify against the original CPU implementation.
cpu_fitness = float(udp.fitness(best_x.tolist())[0])

print("\nOptimization finished")
print("----------------------")
print("Initial best fitness:", initial_best)
print("Final best fitness:", best_fitness)
print("CPU verification:", cpu_fitness)
print("Verification difference:", abs(best_fitness - cpu_fitness))
print("Execution time:", round(elapsed, 2), "seconds")


# Save the best solution.
Path("results").mkdir(exist_ok=True)

np.savez(
    "results/best_solution.npz",
    x=best_x,
    fitness=best_fitness
)

print("Saved: results/best_solution.npz")

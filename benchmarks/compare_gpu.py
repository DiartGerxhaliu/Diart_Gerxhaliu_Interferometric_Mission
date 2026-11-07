import numpy as np
import torch

from reference.esa_reference import udp
from src.evaluation.gpu import gpu_fitness_batch

print("GPU:", torch.cuda.get_device_name(0))

# Generate 32 random candidates within the ESA bounds.
lower, upper = udp.get_bounds()
rng = np.random.default_rng(42)

candidates = rng.uniform(
    lower,
    upper,
    size=(32, len(lower))
)

# CPU reference evaluations.
cpu_results = np.array([
    udp.fitness(candidate.tolist())[0]
    for candidate in candidates
])

# Batched GPU evaluation.
gpu_results = gpu_fitness_batch(candidates, udp).cpu().numpy()

differences = np.abs(cpu_results - gpu_results)

print("Candidates tested:", len(candidates))
print("Maximum absolute difference:", differences.max())
print("Mean absolute difference:", differences.mean())
print("All close:", np.allclose(cpu_results, gpu_results, rtol=1e-8, atol=1e-10))

import time
import numpy as np
import torch

from reference.esa_reference import udp
from src.evaluation.gpu import gpu_fitness_batch


print("GPU:", torch.cuda.get_device_name(0))

lower, upper = udp.get_bounds()
rng = np.random.default_rng(42)

batch_sizes = [32, 128, 512, 1024]
runs = 3

# Warm up CUDA before measuring.
warmup = rng.uniform(lower, upper, size=(8, len(lower)))
gpu_fitness_batch(warmup, udp)
torch.cuda.synchronize()

print("\nBenchmark results")
print("-" * 65)
print(f"{'Batch':>8} {'CPU eval/s':>15} {'GPU eval/s':>15} {'Speedup':>12}")

for batch_size in batch_sizes:

    candidates = rng.uniform(
        lower,
        upper,
        size=(batch_size, len(lower))
    )

    # CPU benchmark
    start = time.perf_counter()

    for _ in range(runs):
        for candidate in candidates:
            udp.fitness(candidate.tolist())

    cpu_elapsed = time.perf_counter() - start
    cpu_throughput = (batch_size * runs) / cpu_elapsed

    # GPU benchmark
    gpu_fitness_batch(candidates, udp)
    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(runs):
        gpu_fitness_batch(candidates, udp)

    torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - start

    gpu_throughput = (batch_size * runs) / gpu_elapsed
    speedup = gpu_throughput / cpu_throughput

    print(
        f"{batch_size:>8} "
        f"{cpu_throughput:>15.2f} "
        f"{gpu_throughput:>15.2f} "
        f"{speedup:>11.2f}x"
    )

print("-" * 65)
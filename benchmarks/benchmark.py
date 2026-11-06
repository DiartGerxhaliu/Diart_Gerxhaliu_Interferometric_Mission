import time
from reference.esa_reference import udp

solution = [0.0] * 240

# Warm-up
udp.fitness(solution)

runs = 5
start = time.perf_counter()

for _ in range(runs):
    udp.fitness(solution)

elapsed = time.perf_counter() - start

print(f"Average CPU evaluation: {elapsed / runs:.6f} seconds")
print(f"Evaluations per second: {runs / elapsed:.2f}")
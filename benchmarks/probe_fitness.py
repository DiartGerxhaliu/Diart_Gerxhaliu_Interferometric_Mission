import numpy as np

from reference.esa_reference import udp


rng = np.random.default_rng(42)

n = udp.n_sat
samples = 128

print("Velocity range | Nonzero scores | Best fitness | Median fitness")
print("-" * 65)

for velocity_range in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:

    candidates = np.zeros((samples, 6 * n))

    # Position perturbations.
    candidates[:, :3 * n] = rng.uniform(
        -1, 1, (samples, 3 * n)
    )

    # Velocity perturbations.
    candidates[:, 3 * n:] = rng.uniform(
        -velocity_range,
        velocity_range,
        (samples, 3 * n)
    )

    scores = np.array([
        udp.fitness(candidate.tolist())[0]
        for candidate in candidates
    ])

    print(
        f"{velocity_range:>14.2f} | "
        f"{np.count_nonzero(scores):>14} | "
        f"{scores.min():>13.8f} | "
        f"{np.median(scores):>14.8f}"
    )
import numpy as np
import torch
from scipy.signal import correlate

from src.evaluation.gpu import autocorrelation_fill

rng = np.random.default_rng(42)

for i in range(10):
    projection = rng.integers(0, 2, size=(21, 21))

    cpu = correlate(projection, projection, mode="full")
    cpu_fill = np.count_nonzero(cpu) / cpu.size

    gpu = autocorrelation_fill(
        torch.tensor(projection, dtype=torch.float64, device="cuda")
    ).item()

    print(
        f"Test {i + 1}: "
        f"CPU={cpu_fill:.8f}, "
        f"GPU={gpu:.8f}, "
        f"Difference={abs(cpu_fill - gpu):.8f}"
    )
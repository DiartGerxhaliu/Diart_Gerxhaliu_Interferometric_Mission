# ESA SPOC-3 – Interferometric Mission

**GPU-Accelerated Optimization of Satellite Formations for Space-Based Interferometry**

## Overview

This project focuses on solving the European Space Agency's SPOC-3 Interferometric Mission challenge, specifically the **Large Observatory in Halo** problem.

The objective is to optimize a formation of satellites orbiting around a mothership in a halo orbit, with the goal of improving interferometric observation coverage and image reconstruction.

The implementation aims to leverage GPU acceleration to evaluate large populations of candidate satellite configurations efficiently.

## Project Goals

- Reproduce and understand the original ESA reference implementation.
- Simulate satellite formations using the Circular Restricted Three-Body Problem (CR3BP).
- Calculate State Transition Matrices (STMs) for orbital propagation.
- Implement GPU-accelerated fitness evaluation using PyTorch and CUDA.
- Develop a Differential Evolution optimization algorithm.
- Evaluate and improve interferometric coverage across observation periods.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| PyTorch | GPU-accelerated computation |
| CUDA | NVIDIA GPU acceleration |
| Heyoka | Orbital dynamics and numerical integration |
| PyGMO | Optimization algorithms |
| NumPy | Numerical computations |
| SciPy | Scientific computing |
| Matplotlib | Visualization |
| Pillow | Image processing |

## Hardware

Development hardware:

- **GPU:** NVIDIA GeForce RTX 5070 Laptop GPU
- **CPU:** Intel Core Ultra 7 255HX
- **VRAM:** 8 GB

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Diart_Gerxhaliu_Interferometric_Mission
```

### 2. Create the Conda environment

```bash
conda create -n esa python=3.12 numpy scipy matplotlib pillow pygmo heyoka.py -c conda-forge
```

### 3. Activate the environment

```bash
conda activate esa
```

### 4. Install PyTorch with CUDA support

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### 5. Verify the installation

```python
import torch
import heyoka
import pygmo
import numpy
import scipy

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Optimization Approach

The planned architecture separates orbital simulation from computationally intensive optimization.

1. **Orbital propagation:** Calculate the reference trajectory and STMs.
2. **Candidate generation:** Generate populations of possible satellite formations.
3. **GPU fitness evaluation:** Propagate candidate formations and calculate interferometric coverage using PyTorch.
4. **Evolutionary optimization:** Apply Differential Evolution to improve candidate configurations.
5. **Validation:** Compare GPU results against the original ESA implementation.

## Project Status

<ins>Currently in development.</ins>

- [x] Python environment configured
- [x] NVIDIA GPU detected through PyTorch CUDA
- [ ] ESA reference implementation reproduced
- [ ] GPU fitness evaluator implemented
- [ ] Differential Evolution optimizer implemented
- [ ] Performance benchmarking
- [ ] Final solution validation

## Challenge

[ESA SPOC-3 – Interferometric Mission: Large Observatory in Halo](https://optimise.esa.int/challenge/spoc-3-interferometric-mission/p/large-observatory-in-halo)

---

**Author:** Diart Gërxhaliu

**Project:** ESA SPOC-3 Interferometric Mission Optimization
# ESA SPOC-3 – Interferometric Mission Optimization

**Project:** Large Observatory in Halo  
**Organization:** European Space Agency (ESA)  
**Status:** Environment configured; reference implementation under validation  
**Language:** Python  
**GPU Framework:** PyTorch  
**Target GPU:** NVIDIA GeForce RTX 5070 Laptop GPU

---

## 1. Project Overview

This project focuses on solving the European Space Agency's SPOC-3 Interferometric Mission optimization challenge, specifically the **Large Observatory in Halo** problem.

The objective is to optimize the initial relative positions and velocities of a formation of satellites orbiting around a mothership following a nominal trajectory in the Circular Restricted Three-Body Problem (CR3BP).

The satellite formation must produce good interferometric measurements at multiple observation points. The quality of these measurements is evaluated using Golomb array patterns and their autocorrelations.

The main goal is to develop an optimization implementation that uses GPU acceleration to evaluate candidate satellite formations efficiently while preserving the mathematical behavior of the original ESA reference implementation.

**Official challenge:**  
https://optimise.esa.int/challenge/spoc-3-interferometric-mission/p/large-observatory-in-halo

---

## 2. Development Objectives

The project is being developed in stages:

1. Understand and validate the original ESA reference implementation.
2. Reproduce the original fitness calculation.
3. Develop a GPU-accelerated fitness evaluator using PyTorch.
4. Implement a population-based optimization algorithm.
5. Compare CPU and GPU performance.
6. Validate GPU results against the original ESA implementation.
7. Optimize performance and improve the final satellite formation.

The mathematical correctness of the original evaluation is prioritized over performance improvements.

---

## 3. Problem Configuration

The current reference problem uses the following configuration:

| Parameter | Value |
|---|---:|
| Number of satellites | 40 |
| Decision variables per satellite | 6 |
| Total decision variables | 240 |
| Number of observations | 3 |
| Grid dimensions | 21 × 21 × 21 |
| CR3BP gravitational parameter (μ) | 0.01215058560962404 |
| Scaling factor | 0.0001 |
| Inflation factor | 1.23 |
| Halo orbit period | 2.383491010514447 |
| Total propagation time | 4.766982020... |

Each satellite has six optimization variables:

- Initial relative position: X, Y, Z
- Initial relative velocity: VX, VY, VZ

The optimization chromosome is organized into six blocks of 40 values:

```text
[dx1 ... dx40,
 dy1 ... dy40,
 dz1 ... dz40,
 dvx1 ... dvx40,
 dvy1 ... dvy40,
 dvz1 ... dvz40]
```

The decision variable bounds are:

| Variables | Lower bound | Upper bound |
|---|---:|---:|
| Relative positions | -1 | 1 |
| Relative velocities | -10 | 10 |

All 240 decision variables are continuous.

---

## 4. Fitness Function

The original ESA implementation evaluates each candidate formation at three observation points:

- First observation: t = 0
- Second observation: t = 1 orbital period
- Third observation: t = 2 orbital periods

The satellite positions are propagated using precomputed State Transition Matrices (STMs).

For every observation:

1. Propagate the relative satellite positions.
2. Apply the formation inflation factor for observations after the initial measurement.
3. Remove satellites outside the allowed spatial boundaries.
4. Map the remaining positions onto a three-dimensional grid.
5. Generate three two-dimensional projections:
   - XY
   - XZ
   - YZ
6. Calculate the autocorrelation of each projection.
7. Calculate the fill factor of each autocorrelation.

The three fill factors are summed for each observation.

The final fitness function returns:

```python
return [-min(fill_factor)]
```

This means the optimization algorithm minimizes the negative of the weakest observation's combined fill factor.

**The optimization objective is therefore to maximize the worst-performing observation.**

---

## 5. Technology Stack

The project uses the following libraries:

| Library | Purpose |
|---|---|
| Heyoka | CR3BP dynamics and Taylor integration |
| NumPy | Numerical operations and arrays |
| SciPy | Signal processing and autocorrelation |
| Matplotlib | Visualization and plotting |
| Pillow | Image processing |
| PyTorch | GPU-accelerated calculations |
| PyGMO | Available for potential optimization experiments |

The original ESA implementation does not require PyTorch or PyGMO to calculate its reference fitness. PyTorch is installed for the planned GPU implementation.

---

## 6. Hardware

Development and testing are performed on the following machine:

| Component | Specification |
|---|---|
| Laptop | MSI Crosshair 16 X |
| CPU | Intel Core Ultra 7 255HX |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| GPU Memory | 8 GB |
| Operating System | Windows |
| GPU Acceleration | CUDA |

The NVIDIA GPU is detected successfully by PyTorch.

---

## 7. Development Environment

The project uses Miniforge and Conda for Python environment management.

**Environment name:**

```text
esa
```

**Python version:**

```text
Python 3.12
```

**Environment location:**

```text
C:\Users\User\miniforge3\envs\esa
```

The environment is activated using:

```bash
conda activate esa
```

The project directory is:

```text
C:\Users\User\Desktop\Fakultet\Diart_Gerxhaliu_Interferometric_Mission
```

### Installed dependencies

The following packages have been installed and verified:

| Package | Installed version | Source |
|---|---|---|
| heyoka | 7.13.0 | conda-forge |
| heyoka.py | 7.13.2 | conda-forge |
| numpy | 2.5.4 | conda-forge |
| scipy | 1.18.1 | conda-forge |
| matplotlib | 3.11.2 | conda-forge |
| pillow | 13.0.0 | conda-forge |
| pygmo | 2.20.0 | conda-forge |
| torch | 2.15.0+cu132 | pip |
| torchvision | 0.29.2+cu132 | pip |

The Python imports have been tested successfully.

### PyTorch CUDA verification

The following checks were executed:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Verified output:

```text
PyTorch: 2.15.0+cu132
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

This confirms that the Python environment can access the NVIDIA GPU through CUDA.

---

## 8. Original ESA Reference Implementation

The original ESA source code has been saved locally at:

```text
reference/esa_reference.py
```

The implementation contains:

- `propagate_formation()`
- `stm_factory()`
- `orbital_golomb_array`
- `fitness_impl()`
- `plot_simulated_reconstruction()`

The reference implementation uses Heyoka to construct the CR3BP equations of motion and their variational equations.

It calculates the State Transition Matrices once during initialization and reuses them when evaluating different satellite formations.

This is important because it avoids repeatedly integrating the entire orbital system for every candidate solution.

The original implementation is being preserved as the CPU reference against which the GPU implementation will be tested.

---

## 9. Troubleshooting and Compatibility

During initial testing, the original ESA implementation encountered an error:

```text
TypeError: 'NoneType' object is not subscriptable
```

The error occurred in `stm_factory()`:

```python
sol = ta.propagate_grid(t_grid)

ref_state = sol[4][:, :6]
stms = sol[4][:, 6:].reshape(M, 6, 6)
```

The installed Heyoka version uses a return structure in which the propagated state data is expected at index 5 rather than index 4.

The proposed compatibility adjustment is:

```python
ref_state = sol[5][:, :6]
stms = sol[5][:, 6:].reshape(M, 6, 6)
```

This adjustment still needs to be applied and tested against the original implementation.

No changes to the underlying CR3BP equations or fitness mathematics are intended.

---

## 10. Current Project Structure

```text
Interferometric_Mission/
│
├── reference/
│   └── esa_reference.py
│
├── .gitignore
│
└── README.md
```

The original ESA implementation is kept separate from future GPU code to make comparison and validation easier.

---

## 11. Current Development Status

### Completed

- [x] Selected the ESA SPOC-3 Interferometric Mission challenge.
- [x] Identified the Large Observatory in Halo problem.
- [x] Cloned the project repository from GitHub.
- [x] Installed Miniforge.
- [x] Created the Conda environment named `esa`.
- [x] Installed Python 3.12.
- [x] Installed Heyoka and its Python package.
- [x] Installed NumPy, SciPy, Matplotlib and Pillow.
- [x] Installed PyTorch with CUDA support.
- [x] Installed PyGMO.
- [x] Verified all required Python imports.
- [x] Verified CUDA availability.
- [x] Confirmed RTX 5070 GPU detection.
- [x] Obtained and saved the original ESA reference implementation.
- [x] Identified a Heyoka API compatibility issue.

### In progress

- [ ] Apply and verify the Heyoka return-index compatibility fix.
- [ ] Successfully initialize the original ESA UDP.
- [ ] Evaluate the first baseline solution.
- [ ] Establish a reproducible CPU fitness benchmark.

### Planned

- [ ] Implement the GPU fitness evaluator in PyTorch.
- [ ] Validate GPU fitness against CPU fitness.
- [ ] Implement GPU-accelerated Differential Evolution.
- [ ] Experiment with population sizes and batch sizes.
- [ ] Benchmark CPU versus GPU performance.
- [ ] Optimize the best-performing satellite formation.
- [ ] Visualize the resulting satellite formation and interferometric reconstruction.
- [ ] Document final results and performance improvements.

---

## 12. Immediate Next Step

The next task is to get the original ESA reference implementation running successfully.

The first test will evaluate a chromosome containing 240 zero-valued decision variables:

```python
from reference.esa_reference import udp

print("Decision variables:", len(udp.get_bounds()[0]))
print("Baseline fitness:", udp.fitness([0.0] * 240))
```

Expected decision variable count:

```text
240
```

Once this works, the resulting CPU fitness value will serve as the baseline for validating the GPU implementation.

---

## 13. Development Philosophy

The project follows three main principles:

1. **Correctness before optimization:** GPU calculations must reproduce the original ESA fitness function.
2. **Preserve the mathematics:** Performance improvements must not alter the underlying orbital dynamics or evaluation criteria.
3. **Measure actual performance:** GPU acceleration must be benchmarked against the CPU implementation rather than assumed to be faster.

The ultimate goal is to develop a reliable, GPU-accelerated optimization solution for the ESA interferometric satellite formation problem.
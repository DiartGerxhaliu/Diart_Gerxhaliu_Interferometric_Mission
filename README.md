# ESA SPOC-3 – Interferometric Mission Optimization

**Project:** ESA SpOC-3 Interferometric Mission
**Organization:** European Space Agency (ESA)
**Status:** One score-guided random-improvement search handles easy, medium, and hard; GPU checks remain available as optional validation
**Language:** Python
**GPU Framework:** PyTorch  
**Target GPU:** NVIDIA GeForce RTX 5070 Laptop GPU

---

## 1. Project Overview

This project focuses on solving the European Space Agency's SPOC-3 Interferometric Mission optimization challenge across its **easy, medium, and hard** problems.

The objective is to optimize the initial relative positions and velocities of a formation of satellites orbiting around a mothership following a nominal trajectory in the Circular Restricted Three-Body Problem (CR3BP).

The satellite formation must produce good interferometric measurements at multiple observation points. The quality of these measurements is evaluated using Golomb array patterns and their autocorrelations.

The main search scores batches of random formations on the GPU, mutates candidates, and preserves the best score found. It can accept a worse intermediate candidate occasionally to escape plateaus. A CPU child process prepares the reference STMs and independently checks the final result.

**Official challenge:**  
https://optimise.esa.int/challenge/spoc-3-interferometric-mission/p/large-observatory-in-halo

---

## 2. Development Objectives

The project is being developed in stages:

1. Understand and validate the original ESA reference implementation.
2. Reproduce the original fitness calculation.
3. Develop a GPU-accelerated fitness evaluator using PyTorch.
4. Search by proposing mutations to the best candidate and keeping improvements.
5. Check the search against the original ESA CPU evaluator.
6. Keep optional GPU comparisons for performance experiments.

The mathematical correctness of the original evaluation is prioritized over performance improvements.

---

## 3. Problem Configuration

The three challenge cases use the following configurations:

| Parameter | Value |
|---|---:|
| Case | ESA problem | Satellites | Grid | Orbit | Period |
|---|---|---:|---:|---|---:|
| Easy | `tiny-observatory-in-dro` | 5 | 11³ | DRO | 2.6905181697222775 |
| Medium | `large-observatory-in-dro` | 40 | 21³ | DRO | 2.6905181697222775 |
| Hard | `large-observatory-in-halo` | 40 | 21³ | HALO | 2.383491010514447 |

All cases use 3 observations, CR3BP gravitational parameter μ = 0.01215058560962404, scaling factor 0.0001, and inflation factor 1.23. Total propagation time is two orbit periods. The easy case has 30 decision variables; medium and hard each have 240.

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

The challenge README describes the objective as the negative maximum of the three observation scores:

```python
return [-max(fill_factor)]
```

The simple search defaults to the published ESA Python evaluator's `min` objective. The challenge README instead specifies `max`; both formulas remain selectable because they conflict. Outputs include the selected objective in their filenames and metadata.

These are local scores, not official leaderboard results. The submission helper expects one problem per JSON file, so the exporter writes `results/submissions/easy_submission.json`, `medium_submission.json`, and `hard_submission.json`. None has been uploaded to ESA.

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

The checked-in source currently uses:

```python
ref_state = sol[5][:, :6]
stms = sol[5][:, 6:].reshape(M, 6, 6)
```

Reference initialization and a baseline fitness evaluation now run successfully in the `esa` environment.

No changes to the underlying CR3BP equations or fitness mathematics are intended.

---

## 10. Current Project Structure

```text
Interferometric_Mission/
├── reference/esa_reference.py       # ESA CPU reference
├── src/evaluation/gpu.py            # Batched PyTorch evaluator
├── src/evaluation/pygmo_problem.py  # PyGMO wrapper and search bounds
├── src/evaluation/cases.py          # Easy, medium, and hard reference configurations
├── benchmarks/                      # Evaluation, comparison, and optimization scripts
└── results/cases/                   # Per-case, per-objective verified candidates
```

The ESA reference file is kept unchanged as the comparison baseline. Project-specific fixes and extensions belong in the surrounding evaluator and benchmark code.

---

## 11. Current Development Status

### Present in the project

- [x] Selected the ESA SPOC-3 Interferometric Mission challenge.
- [x] Identified and configured all three challenge cases: easy DRO, medium DRO, and hard HALO.
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
- [x] ESA reference reads propagated states from Heyoka's current return slot (`sol[5]`); initialization and fitness evaluation were verified in the `esa` environment.
- [x] Added a batched PyTorch evaluator and a PyGMO wrapper.
- [x] Added scripts for CPU/GPU comparison, fitness inspection, and optimization.
- [x] Added one score-guided random-improvement loop that selects any of the three cases.
- [x] Made the published Python evaluator's `min` objective the search default; `max` remains available for the challenge README's formula.
- [x] Added three ESA-format submission JSON files, one bounds-checked candidate per case.

### Still to verify

- [x] Run CPU/GPU comparisons on 32 fixed random candidates; all scores matched within floating-point tolerance.
- [x] Validate CPU/GPU agreement for both objective definitions on all three cases.
- [x] Recompute saved candidates with the published CPU evaluator and match their stored scores.
- [x] Benchmark throughput on the current machine: at batch size 1024, one run measured about 26,435 GPU evaluations/second versus 1,146 CPU evaluations/second (23.1×); benchmark repeats and records the seed and batch sizes.
- [ ] Confirm the live submission service's aggregation if ESA clarifies its conflict with the challenge README.

### Planned

- [x] Update the optimization runner to save each run and retain the best known candidate.
- [ ] Compare wider velocity bounds and alternative population/batch sizes.
- [ ] Optimize the best-performing satellite formation.
- [ ] Visualize the resulting satellite formation and interferometric reconstruction.
- [ ] Document final results and performance improvements.

---

## 12. Run the Case Suite

Activate the `esa` environment, then try 10,000 candidates per case in batches of 512 and build one local submission JSON per case:

```bash
python -m benchmarks.run_case_suite --iterations 10000 --batch-size 512 --seed 43
```

Add `--resume` to skip cases with saved candidates, then verify and rebuild all three submission files:

```bash
python -m benchmarks.run_case_suite --resume
```

Each case can also be run individually. For example:

```bash
python -m benchmarks.run_optimization --case hard --iterations 10000 --batch-size 512 --seed 43
```

The search uses the published Python evaluator's `min` objective. The challenge README documents `max`, so those two sources disagree. ESA permits positions in `[-1, 1]` and velocities in `[-10, 10]`; the search initializes and mutates velocities within `[-0.5, 0.5]` by default to keep trial formations near the mothership. Set `--velocity-bound 10` to explore the full velocity range. The search cools its acceptance of worse intermediate candidates; use `--temperature` and `--restart-after` to adjust exploration.

After all three selected-objective candidates exist, build or refresh the files in `results/submissions/` with:

```bash
python -m benchmarks.build_submission
```

Each file contains one ESA problem entry and can be uploaded separately through ESA's site. The project only creates these local files; it does not submit them. Current reference-matched files end in `_min_submission.json`; `max` files follow the challenge README's alternate formula.

The CPU reference baseline for a chromosome containing 240 zero-valued decision variables is:

```python
from reference.esa_reference import udp

print("Decision variables:", len(udp.get_bounds()[0]))
print("Baseline fitness:", udp.fitness([0.0] * 240))
```

Expected decision variable count:

```text
240
```

The observed CPU result is `-0.001784651992861392`. The optimizer prints the three observation scores for each run and stores them with its result.

The easy case has 30 decision variables; medium and hard have 240 each.

---

## 13. Development Philosophy

The project follows three main principles:

1. **Correctness before optimization:** GPU calculations must reproduce the original ESA fitness function.
2. **Preserve the mathematics:** Performance improvements must not alter the underlying orbital dynamics or evaluation criteria.
3. **Measure actual performance:** GPU acceleration must be benchmarked against the CPU implementation rather than assumed to be faster.

The goal is to produce reproducible ESA candidate vectors with a clear score-guided search and reference-based verification.

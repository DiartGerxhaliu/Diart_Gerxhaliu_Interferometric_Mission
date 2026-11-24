# ESA SpOC-3 Interferometric Mission Optimizer

A CUDA-accelerated Python search system for ESA's SpOC-3 Interferometric Mission challenge. The project optimizes independent satellite formations for three official cases, verifies candidates against the supplied Heyoka CPU evaluators, and writes ESA submission JSON files.

> **Current status:** the repository includes live candidate archives and search logs. The most recent committed improvement improved the medium case to `-1.8994646044021417`. The hard case remains at `-1.5449137418203451` in the latest saved candidate/log data.

## Cases

| Case | ESA problem | Orbit | Satellites | Grid | Saved fitness |
|---|---|---|---:|---:|---:|
| Easy | `tiny-observatory-in-dro` | DRO | 5 | 11³ | `-0.142857142857` |
| Medium | `large-observatory-in-dro` | DRO | 40 | 21³ | `-1.8994646044021417` |
| Hard | `large-observatory-in-halo` | HALO | 40 | 21³ | `-1.5449137418203451` |

Fitness values come from the repository's candidate archives and ESA reference evaluators. Treat them as the current search incumbent, not a benchmark leaderboard.

## How the optimizer works

### Decision vector

Each candidate is a flat float64 vector containing six blocks:

```text
[dx satellites, dy satellites, dz satellites,
 dvx satellites, dvy satellites, dvz satellites]
```

- Position inputs: `[-1, 1]`
- Velocity inputs: `[-10, 10]`, with a configurable search velocity bound
- Easy: 30 decision variables
- Medium/hard: 240 decision variables

### Evaluation

[`src/evaluation/cases.py`](src/evaluation/cases.py) loads the exact ESA reference problem for each case. It validates `heyoka==3.2.0` and rejects mismatches between the supplied evaluator and the repository's case metadata.

[`src/evaluation/gpu.py`](src/evaluation/gpu.py) evaluates a batch of formations in PyTorch/CUDA:

1. Propagates candidate satellite perturbations through the precomputed state-transition matrices.
2. Converts positions to 3-D grid indices at the three observation times.
3. Builds occupied volume grids.
4. Projects the volume onto XY, XZ, and YZ planes.
5. Measures the autocorrelation fill factor of each projection.
6. Returns the negative sum of plane fill factors at the weakest observation.

[`src/evaluation/constraints.py`](src/evaluation/constraints.py) rescales candidates to keep every propagated position strictly inside the observation bounds.

### Search algorithms

| Algorithm | Use |
|---|---|
| `de` | Whole-satellite differential evolution; strongest general-purpose population search currently in the repository. |
| `genetic` | Rank selection with satellite crossover and broader mutations; it retained several explorers during restarts. |
| `block-anneal` | Block/Metropolis proposals that change mostly one or two satellites. |
| `lm-cma-es` | Limited-memory covariance adaptation in normalized coordinates with repeated restart scales. |
| `heuristic` | Coverage-guided mutations using the weakest observation and satellite-specific grid guidance. |
| `coordinate-search` | Directly explores grid-boundary transitions and selected-satellite changes. |
| `auto` | Runs fixed segments and cycles through coordinate search, DE, genetic, block annealing, LM-CMA-ES, and heuristic search. |

The official score is discrete. All population methods additionally rank by the sum of observation scores when the weakest-observation fitness ties, but this secondary score never replaces CPU fitness in saved candidates.

## Quick start

Create or activate the documented ESA environment, then run from the repository root:

```powershell
$python = 'C:\Users\User\miniforge3\envs\esa_official_311\python.exe'
& $python -u main.py --hard
```

The continuous launcher defaults to the case-specific algorithm:

```powershell
& $python -u main.py --easy
& $python -u main.py --medium
& $python -u main.py --hard
```

Select an algorithm explicitly:

```powershell
& $python -u main.py --hard --algorithm genetic
& $python -u main.py --hard --algorithm de
& $python -u main.py --hard --algorithm coordinate-search
& $python -u main.py --hard --algorithm lm-cma-es
& $python -u main.py --hard --algorithm heuristic
& $python -u main.py --hard --algorithm block-anneal
```

`main.py` writes each completed segment to `results/logs/`, then updates a submission only after CPU verification. Press Ctrl+C to stop a continuous run.

## Direct fixed-budget runs

For experiments and fair algorithm comparisons, invoke the population runner directly. A fixed run retains only that algorithm's population and random state for the full budget.

```powershell
& $python -u -m benchmarks.run_optimization `
  --case hard `
  --algorithm de `
  --iterations 10000000 `
  --population-size 4096 `
  --batch-size 2048 `
  --restart-after 100 `
  --seed 239817
```

Genetic search:

```powershell
& $python -u -m benchmarks.run_optimization `
  --case hard `
  --algorithm genetic `
  --iterations 20000000 `
  --population-size 1024 `
  --batch-size 2048 `
  --restart-after 50 `
  --temperature 0.35 `
  --seed 239820
```

Run coordinate search directly:

```powershell
& $python -u -m benchmarks.run_coordinate_search `
  --case hard `
  --evaluations 10000000 `
  --batch-size 2048 `
  --seed 777
```

The standalone random-search tool is also available:

```powershell
& $python -u -m benchmarks.run_random_search --case hard --evaluations 10000000
```

Use a fresh seed and a separate checkpoint path for each independent experiment. The saved RNG state is not a population checkpoint; only the incumbent candidate and RNG stream are restored.

## Search budgets and behavior

The default budgets are implemented in [`main.py`](main.py):

- `auto`: 5 million evaluations per algorithm segment.
- Explicit coordinate search: 10 million evaluations unless `--iterations` is supplied.
- Explicit hard DE: 10 million evaluations unless `--iterations` is supplied.
- Other explicit population searches: 2 million evaluations unless `--iterations` is supplied.

Useful `main.py` options:

```text
--iterations N       Override the per-segment budget
--batch-size N       GPU batch size
--population-size N  Population size
--seed N             Base seed or fixed population seed
--restart-after N    Stagnation generations before refresh/restart
--temperature F      Genetic parent-selection temperature
--velocity-bound F   Search-space velocity bound
--target-score F     Stop the cutoff after this best score is reached
--resume-rng PATH    Resume a population RNG checkpoint
--rng-jump N         Advance the RNG stream
```

`--target-score` is not a termination flag. Once the target is reached, it removes the score cutoff and the continuous run continues.

## Practical experiment strategy

The saved values show repeated hard/medium searches, but many log segments finish without improvement:

- Keep hard DE or genetic runs long enough to test an entire population basin.
- Change seeds for basin diversity; repeating the incumbent/RNG checkpoint may repeat the same plateau.
- Use genetic or DE for broad search, coordinate search for grid-boundary refinement, and block annealing for small local moves.
- Compare results at equal evaluation budgets and seeds, not just total wall-clock time.
- One GPU search should be the default on the RTX 5070 Laptop GPU because search processes contend for the same device memory and may reduce throughput.

## Verification and submission

CPU-verify every saved candidate and regenerate the three submission JSON files:

```powershell
& $python -m benchmarks.build_submission
```

`build_submission.py`:

- Loads the official UDP for each case.
- Checks candidate shape and ESA input bounds.
- Checks all observation-space feasibility constraints.
- Evaluates the candidate on the CPU reference implementation.
- Confirms it matches the saved GPU fitness.
- Writes `results/submissions/<case>_min_submission.json`.

This command does not upload candidates to ESA.

## Saved artifacts

| Path | Purpose |
|---|---|
| `results/cases/<case>_min_best.npz` | Current best candidate, fitness, and run metadata. |
| `results/cases/<case>_optimizer_rng.npz` | Population RNG checkpoint. |
| `results/cases/<case>_boundary_checkpoint.npz` | Coordinate-search incumbent and stagnation state. |
| `results/cases/<case>_coordinate_checkpoint.npz` | Coordinate-search auxiliary state. |
| `results/cases/<case>_random_checkpoint.npz` | Random-search state. |
| `results/submissions/<case>_min_submission.json` | ESA submission payload. |
| `results/logs/*.log` | Search stdout, progress, interruptions, and final summaries. |

Candidate improvements are protected by file locks in `save_candidate_progress()`. Interrupted runs preserve completed candidate improvements, while an unfinished segment may not have a newly saved RNG checkpoint.

## Tests and GPU diagnostics

Focused unit tests:

```powershell
& $python -m unittest `
  benchmarks.test_population_algorithms `
  benchmarks.test_lm_cma_es `
  benchmarks.test_coverage_heuristic `
  benchmarks.test_main_algorithm_selection `
  benchmarks.test_coordinate_search_restart `
  benchmarks.test_candidate_save
```

Additional GPU diagnostics:

```powershell
& $python benchmarks.compare_gpu
& $python benchmarks.benchmark_batch
& $python benchmarks.test_autocorrelation
& $python benchmarks.test_pygmo_gpu
```

## Repository layout

```text
main.py                         Continuous case and algorithm launcher
benchmarks/
  run_optimization.py           CUDA population search runner
  run_coordinate_search.py      Grid-boundary coordinate search
  run_random_search.py          Independent random search
  coverage_heuristic.py         Coverage-guided heuristic
  lm_cma_es.py                  Limited-memory CMA-ES
  build_submission.py           CPU verification and ESA JSON generation
  test_*.py                     Unit, algorithm, and launcher tests
src/evaluation/
  cases.py                      ESA case metadata and reference loading
  gpu.py                        Batched PyTorch/CUDA evaluator
  constraints.py                Observation-bound enforcement
  objectives.py                 CPU observation scores
reference/                      Supplied ESA/Heyoka reference evaluators
results/cases/                  Saved candidates and optimizer checkpoints
results/submissions/            ESA-ready JSON submissions
results/logs/                   Segment logs and experiment history
```

## Environment

- Python 3.11 (documented environment: `esa_official_311`)
- NumPy
- CUDA-enabled PyTorch
- `heyoka==3.2.0`
- CUDA-capable NVIDIA GPU

The existing test environment is an NVIDIA GeForce RTX 5070 Laptop GPU.

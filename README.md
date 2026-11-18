# ESA SpOC-3 Interferometric Mission Optimizer

This repository searches satellite initial conditions for ESA's Interferometric Mission challenge. Easy, medium, and hard candidates are optimized and saved independently.

## Objective and evaluators

Each problem has its own evaluator/configuration:

| Case | ESA problem ID | Orbit | Satellites | Grid |
|---|---|---|---:|---:|
| Easy | `tiny-observatory-in-dro` | DRO | 5 | 11 × 11 × 11 |
| Medium | `large-observatory-in-dro` | DRO | 40 | 21 × 21 × 21 |
| Hard | `large-observatory-in-halo` | HALO | 40 | 21 × 21 × 21 |

For each of three observation times, the evaluator propagates each satellite's six-component relative state, bins in-range positions onto a 3-D grid, projects occupancy onto XY, XZ and YZ, and counts nonzero autocorrelation cells. It sums those three plane fill factors for each observation. The shipped evaluator returns the **negative of the weakest observation's sum**, `-min(fill_factor)`, so lower (more negative) fitness is better. The challenge README describes `-max`; this conflicts with the executable evaluator. The local CPU reference is the scoring authority used by the search and JSON builder.

The chromosome has six blocks in this order:

```text
[dx_1...dx_N, dy_1...dy_N, dz_1...dz_N,
 dvx_1...dvx_N, dvy_1...dvy_N, dvz_1...dvz_N]
```

Bounds are positions `[-1, 1]` and velocities `[-10, 10]`. Later observations apply the evaluator's inflation factor of 1.23. The hard case uses HALO initial conditions; easy and medium use DRO conditions.

## How the search works

`benchmarks/run_optimization.py` scores candidate batches on the NVIDIA GPU, then checks the best candidate with the local CPU reference before saving it. A GPU fitness value by itself is provisional until that CPU check passes.

The main search is differential evolution (DE): it combines differences between population members to propose new candidates, crosses over whole six-value satellite states, and adapts its mutation scale when progress stalls. It periodically refreshes weaker population members. Before crossover, the optimizer sorts satellites by initial X position. Satellite labels do not affect the score, so this gives equivalent permutations a consistent order and makes the same-index genes more likely to represent similar satellites. It changes representation only; the candidate is still evaluated by the ESA objective.

The genetic variant uses rank-based parent selection, whole-satellite crossover, and mutation in coordinates preconditioned by the linearized state-transition sensitivity. The coordinate search uses the evaluator's discrete grid structure: it directly proposes minimum-norm initial-state changes that cross projected cell boundaries, then combines satellite moves when single-satellite moves stall. Block annealing and independent random search remain available as alternatives.

Fitness is discrete and plateau-prone: many small continuous changes leave all grid occupancies unchanged. Mutation therefore spans multiple scales, and random immigrants/restarts help explore formations outside the current local basin. No score is extrapolated: each reported candidate is an actual vector passed to the evaluator.

## Current hard-case status

Latest CPU-verified hard candidate:

| Item | Value |
|---|---:|
| Fitness | `-1.4223676383105293` |
| Observation fill-factor sums | `[1.424747, 1.422368, 1.422368]` |
| Requested hard target | `<= -1.5` |
| Remaining gap | about `0.077632` |
| CPU/GPU verification difference | `2.22e-16` |

The target is **not met yet**. A completed 10-million-evaluation DE segment (population 4096) improved the hard score from `-1.4211778703152884` to `-1.4223676383105293`. CPU verification passed with a `2.22e-16` difference. No search is running now.

The best candidate is stored internally at `results/cases/hard_min_best.npz`; the stream state is stored at `results/cases/hard_optimizer_rng.npz`. The population is not checkpointed, so resuming after a completed segment retains the best vector and RNG sequence but initializes a new population.

## Current medium-case status

The latest CPU-verified large-observatory-in-DRO candidate scores `-1.7507436049970253`, meeting the requested `-1.75` target. Annealed multiscale boundary-guided search found it after 10,334,428 total evaluations. Its candidate is in `results/cases/medium_min_best.npz`, with resumable walker and search state in `results/cases/medium_boundary_checkpoint.npz`.

To continue the medium search, run:

```powershell
& 'C:\Users\User\miniforge3\envs\esa_official_311\python.exe' -u -m benchmarks.run_coordinate_search --case medium --evaluations 1000000 --batch-size 2048 --seed 777 --target-score -1.75
```

## Run or resume hard search (PowerShell)

From the project directory, run:

```powershell
& 'C:\Users\User\miniforge3\envs\esa_official_311\python.exe' -u -m benchmarks.run_optimization --case hard --algorithm de --iterations 10000000 --batch-size 2048 --population-size 4096 --seed 239817 --velocity-bound 10 --restart-after 300 --temperature 0.35 --target-score -1.5 --resume-rng results\cases\hard_optimizer_rng.npz
```

To try the genetic variant, replace `--algorithm de` with `--algorithm genetic`. Do not launch concurrent GPU searches on this machine; they contend for the same GPU and do not share a population. The seed is checked against the checkpoint. Keeping the seed and `--resume-rng` continues from the *last saved* random state; omitting `--resume-rng` restarts the stream from the seed. If a process is interrupted, its in-memory RNG state is lost, so restarting from the last checkpoint replays random draws from that unfinished segment. Use `--rng-jump` to deliberately move farther along the PCG64 stream when avoiding that replay matters; this skips draws rather than restoring the exact interrupted state.

At completion, check for `CPU verification difference`, `Observation scores`, `Final fitness`, and `Saved RNG checkpoint` in the output. Only then treat a new score as verified.

## Build ESA JSON files

The `.npz` files are optimizer checkpoints, not ESA submissions. After a run completes successfully, build the per-case JSON files with:

```powershell
& 'C:\Users\User\miniforge3\Scripts\conda.exe' run -p 'C:\Users\User\miniforge3\envs\esa' --no-capture-output python -m benchmarks.build_submission
```

This checks the saved easy, medium, and hard vectors against their matching local CPU evaluators and writes:

```text
results/submissions/easy_min_submission.json
results/submissions/medium_min_submission.json
results/submissions/hard_min_submission.json
```

Each JSON contains an ESA `decisionVector`, challenge ID, and matching problem ID. Building these files does not upload them to ESA.

## Important files

| Path | Purpose |
|---|---|
| `reference/esa_reference.py` | Hard HALO CPU reference (`udp`); protected, do not edit |
| `src/evaluation/cases.py` | Case definitions and matching DRO/HALO reference construction |
| `src/evaluation/gpu.py` | Batched PyTorch GPU evaluator |
| `src/evaluation/objectives.py` | CPU observation-score helpers |
| `benchmarks/run_optimization.py` | GPU DE, genetic, and block-anneal search |
| `benchmarks/run_random_search.py` | Checkpointed independent random search |
| `benchmarks/run_coordinate_search.py` | Checkpointed grid-boundary-guided coordinate search |
| `benchmarks/build_submission.py` | CPU verification and JSON exporter |
| `OPTIMIZER_HANDOFF.md` | Search history, rationale, checkpoints, and continuation notes |

ESA's Dependencies page requires `heyoka==3.2.0`; newer releases changed the `propagate_grid` return tuple used by the evaluator files. The CPU evaluator also requires PyGMO, NumPy, SciPy, Pillow, and Matplotlib. GPU search additionally requires CUDA-enabled PyTorch. The tested GPU is an NVIDIA GeForce RTX 5070 Laptop GPU.

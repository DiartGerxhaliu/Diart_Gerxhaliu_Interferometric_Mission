# ESA SpOC-3 Interferometric Mission Optimizer

This repository searches satellite initial conditions for ESA's Interferometric Mission challenge. The current active work is the **hard, 40-satellite HALO case**. The optimizer can also be configured for easy and medium, but treat their saved candidates as separate case-specific results.

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

The genetic variant uses rank-based parent selection, whole-satellite crossover, and mutation in coordinates preconditioned by the linearized state-transition sensitivity. Block annealing, independent random search, and small-group coordinate search are also available as experiments; earlier runs of those alternatives did not improve the hard incumbent.

Fitness is discrete and plateau-prone: many small continuous changes leave all grid occupancies unchanged. Mutation therefore spans multiple scales, and random immigrants/restarts help explore formations outside the current local basin. No score is extrapolated: each reported candidate is an actual vector passed to the evaluator.

## Current hard-case status

Latest CPU-verified hard candidate before the currently running genetic segment:

| Item | Value |
|---|---:|
| Fitness | `-1.4211778703152884` |
| Observation fill-factor sums | `[1.422368, 1.421178, 1.421178]` |
| Requested hard target | `<= -1.5` |
| Remaining gap | about `0.078822` |
| CPU/GPU verification difference | `2.22e-16` |

The target is **not met yet**. The latest CPU-verified hard score remains `-1.4211778703152884` (verification difference `2.22e-16`). A 10-million-evaluation DE segment with population 4096 was stopped at 4,061,184 evaluations, without improving the saved candidate. No search is running now. The candidate and RNG checkpoint from the last completed segment remain intact; the in-progress RNG state from the stopped segment was not saved.

The best candidate is stored internally at `results/cases/hard_min_best.npz`; the stream state is stored at `results/cases/hard_optimizer_rng.npz`. The population is not checkpointed, so resuming after a completed segment retains the best vector and RNG sequence but initializes a new population.

## Run or resume hard search (PowerShell)

From the project directory, run:

```powershell
& 'C:\Users\User\miniforge3\Scripts\conda.exe' run -p 'C:\Users\User\miniforge3\envs\esa' --no-capture-output python -m benchmarks.run_optimization --case hard --algorithm de --iterations 10000000 --batch-size 2048 --population-size 4096 --seed 239817 --velocity-bound 10 --restart-after 300 --temperature 0.35 --target-score -1.5 --resume-rng results\cases\hard_optimizer_rng.npz --rng-jump 25000000000
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

Each JSON contains an ESA `decisionVector`, challenge ID, and matching problem ID. Building these files does not upload them to ESA. The medium search was interrupted before its best could be saved, so its on-disk candidate remains the previous CPU-verified one.

## Important files

| Path | Purpose |
|---|---|
| `reference/esa_reference.py` | Hard HALO CPU reference (`udp`); protected, do not edit |
| `src/evaluation/cases.py` | Case definitions and matching DRO/HALO reference construction |
| `src/evaluation/gpu.py` | Batched PyTorch GPU evaluator |
| `src/evaluation/objectives.py` | CPU observation-score helpers |
| `benchmarks/run_optimization.py` | GPU DE, genetic, and block-anneal search |
| `benchmarks/run_random_search.py` | Checkpointed independent random search |
| `benchmarks/run_coordinate_search.py` | Checkpointed small-group replacement search |
| `benchmarks/build_submission.py` | CPU verification and JSON exporter |
| `OPTIMIZER_HANDOFF.md` | Search history, rationale, checkpoints, and continuation notes |

The Python environment is Miniforge at `C:\Users\User\miniforge3\envs\esa` (environment name `esa`). The tested GPU is an NVIDIA GeForce RTX 5070 Laptop GPU. The optimizer also requires the ESA CPU reference dependencies, including Heyoka and PyGMO.

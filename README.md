# ESA SpOC-3 Interferometric Mission Optimizer

This project searches satellite initial conditions for ESA's SpOC-3 Interferometric Mission challenge. Easy, medium, and hard candidates are optimized and submitted independently.

## Cases and current saved scores

| Case | ESA problem ID | Orbit | Satellites | Grid | CPU-verified fitness |
|---|---|---|---:|---:|---:|
| Easy | `tiny-observatory-in-dro` | DRO | 5 | 11 x 11 x 11 | `-0.142857142857` |
| Medium | `large-observatory-in-dro` | DRO | 40 | 21 x 21 x 21 | `-1.80785246877` |
| Hard | `large-observatory-in-halo` | HALO | 40 | 21 x 21 x 21 | `-1.40690065437` |

These scores were checked against the local CPU evaluators on 2026-11-21. The candidate archives in `results/cases/` and matching JSONs in `results/submissions/` are the live results and may improve after this snapshot.

## Objective and constraints

The decision vector contains six satellite-state blocks:

```text
[dx_1...dx_N, dy_1...dy_N, dz_1...dz_N,
 dvx_1...dvx_N, dvy_1...dvy_N, dvz_1...dvz_N]
```

The input bounds are `[-1, 1]` for positions and `[-10, 10]` for velocities. The evaluator propagates the satellites to three observation times, bins in-range positions on a 3-D grid, projects occupied cells onto XY, XZ, and YZ planes, and counts nonzero autocorrelation cells. Multiple satellites in the same grid cell count as one occupied cell.

Fitness is the negative sum of the three plane fill factors at the weakest observation. The search minimizes fitness, so more negative is better. The score is discrete: small state changes often leave all grid occupancies unchanged. For the 21-cell grids, each fill-factor change is quantized in increments of `1 / 1681` (about `0.000595`); tiny fractional score changes are not expected.

All optimizers constrain every satellite to remain strictly inside the evaluator's `[-1, 1]` position bounds at all three observation times. Candidate improvements are saved immediately. The launcher updates that case's submission JSON immediately, then CPU-verifies the candidate when the segment finishes; treat a result as verified only after that check succeeds.

## Run a search

Run commands from the repository root in the ESA Python environment. For example:

```powershell
$python = 'C:\Users\User\miniforge3\envs\esa_official_311\python.exe'
& $python -u main.py --easy
& $python -u main.py --medium
& $python -u main.py --hard
```

`main.py` runs continuously until `Ctrl+C`; its evaluation budgets are per segment, not a total run limit. `--target-score` does not stop the continuous search: after reaching the target, the launcher removes the cutoff and continues.

Recommended starting algorithms; run one GPU search at a time:

```powershell
& $python -u main.py --easy --algorithm coordinate-search
& $python -u main.py --medium --algorithm de
& $python -u main.py --hard --algorithm de
```

To run LM-CMA-ES directly:

```powershell
& $python -u main.py --hard --algorithm lm-cma-es
```

Run the focused regression tests with:

```powershell
& $python -m unittest benchmarks.test_population_algorithms benchmarks.test_lm_cma_es benchmarks.test_main_algorithm_selection benchmarks.test_coordinate_search_restart benchmarks.test_candidate_save
```

### Algorithms

| Option | Behavior |
|---|---|
| `auto` (default) | Rotates through all five methods after every 5 million evaluations, regardless of improvement. Easy starts with coordinate search; medium and hard start with DE. |
| `coordinate-search` | Proposes changes that cross projected grid-cell boundaries and samples single-satellite moves across the available satellites. |
| `de` | Differential evolution with whole-satellite crossover, bounded adaptive scale, neutral moves across score plateaus, and population refreshes. |
| `genetic` | Rank-based selection with randomized equal-score ties, whole-satellite crossover, and bounded mutation. |
| `block-anneal` | Proposes satellite-state changes and accepts worse candidates using a separate, cooler annealing schedule. |
| `lm-cma-es` | Limited-memory covariance adaptation in normalized coordinates; restarts at several scales after extended stagnation. |

The standalone random-search utility is not a `main.py --algorithm` option; run it with `& $python -m benchmarks.run_random_search --case hard --evaluations 10000000`.

### Recommendations by case

| Case | Recommended algorithm | Why |
|---|---|---|
| Easy | `coordinate-search` | Only five satellites; direct grid-boundary moves target this discrete objective. This is a structure-based recommendation, not a controlled comparison. |
| Medium | `de` | DE produced the current saved best (`-1.80785246877`); coordinate search's recorded medium run plateaued at a weaker score. |
| Hard | `de` | The current saved hard candidate came from DE. Keep `auto` if you want it to cycle through the other methods, including LM-CMA-ES. |

These are practical starting choices, not guarantees; the recorded searches were not equal-budget head-to-head trials. In auto mode, each method gets a 5-million-evaluation segment, then the launcher rotates to the next method whether or not the score improved. Easy starts with coordinate search; medium and hard start with DE. Explicit coordinate-search segments use 10 million evaluations; other explicit population-search segments use 2 million, except explicitly selected hard DE, which uses 10 million. LM-CMA-ES uses `max(4, 4 + floor(3*ln(6*n_sat)))` offspring by default. Explicitly selecting one algorithm disables switching; for example, `--hard --algorithm de` continues DE rather than switching methods.

Hard DE defaults to a population of 4096; LM-CMA-ES uses a dimension-scaled default, and other population searches default to 256. Useful options include `--batch-size`, `--population-size`, `--seed`, `--velocity-bound`, `--restart-after`, `--temperature` (genetic selection), and `--anneal-temperature` (coordinate/block annealing; default `0.005`). Run `& $python main.py --help` for the current CLI options. The launcher does not accept `--iterations`; direct optimizer modules do.

Coordinate search randomizes a partial restart only after 1,000,000 evaluations without a new best, independently of auto's 5-million-evaluation algorithm rotation. It randomly chooses 1 to 25% of the satellite states (1 to 10 for a 40-satellite case), then enforces the observation bounds. A new best resets the stagnation counter, and the saved best is retained unless a feasible candidate scores better.

## Saved state and submissions

- `results/cases/<case>_min_best.npz` stores the best candidate and its score.
- `results/cases/<case>_optimizer_rng.npz` stores the population optimizer's random-number state at the end of a segment. The population itself is not checkpointed.
- `results/cases/<case>_boundary_checkpoint.npz` stores coordinate-search progress.
- `results/submissions/<case>_min_submission.json` stores the ESA submission for that case.
- `results/logs/` contains the live output and final summary for each segment.

For a valid saved candidate, optimizer writes only replace it with a strictly better score. Coordinate search also reconciles a resumed checkpoint with the latest saved candidate.

If interrupted with `Ctrl+C`, saved candidate improvements remain, but the random state from an unfinished segment may not be saved. Resuming restores the last completed random-state checkpoint and creates a new population around the saved best.

To CPU-verify the saved candidates and regenerate the three JSON files:

```powershell
& $python -m benchmarks.build_submission
```

This command does not upload anything to ESA. Do not run multiple GPU searches at the same time; they contend for the same GPU and do not share a population.

## Project files and environment

| Path | Purpose |
|---|---|
| `main.py` | Continuous case runner, auto switching, logs, and submission updates |
| `benchmarks/run_optimization.py` | GPU population optimizers |
| `benchmarks/run_coordinate_search.py` | Grid-boundary-guided coordinate search |
| `benchmarks/run_random_search.py` | Independent random-search utility |
| `benchmarks/build_submission.py` | CPU verification and ESA JSON generation |
| `src/evaluation/cases.py` | Case configuration and matching reference selection |
| `src/evaluation/gpu.py` | Batched PyTorch GPU scoring |
| `src/evaluation/constraints.py` | Observation-bound enforcement and checks |
| `reference/` | Local evaluator files; do not edit |

The evaluator requires `heyoka==3.2.0`. GPU population search additionally requires CUDA-enabled PyTorch. The documented environment is `esa_official_311`; the tested GPU is an NVIDIA GeForce RTX 5070 Laptop GPU.

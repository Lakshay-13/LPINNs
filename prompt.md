# Localisation Experiment Prompt

## Scope

Work only inside `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation`.

Ignore `rl-pinn` and ignore TorchHolo for this phase.

This phase is about building a clean, reproducible localisation study around:

1. Harmonic oscillator (HO)
2. Heat equation
3. 4D problem

The goal is not just lower final loss. The goal is to determine whether localisation gives:

1. Less wall-clock time
2. Lower loss / residual
3. Better recovered geometry / solution shape
4. Smaller networks that match or beat larger baselines

Fixed epoch comparisons are useful, but they are not the main objective. If a localised model reaches a better result sooner, that matters more.

## Core Research Question

Find localiser functions that are better than Gaussian, then test whether localisation helps across:

1. Different localiser families
2. Different domain lengths
3. Different training budgets
4. Same-network comparisons
5. Network width sweeps
6. Smaller-localised vs bigger-baseline comparisons

The final conclusion should be based on both quality and time, not just one scalar metric.

## Existing Function Families

Treat these as the current starting set:

1. `gaussian`
2. `activated`
3. `super_gaussian`
4. `ricker`
5. `boxcar`

Keep the existing Gaussian-style variants where they make sense:

1. fixed parameters
2. learnable sigma only
3. learnable mu and sigma

If a family only supports a subset cleanly, keep the implementation honest instead of forcing symmetry.

## New Function Families To Add

Add a broader function screen before declaring Gaussian best. Include at least these candidates:

1. `laplace`
2. `cauchy`
3. `raised_cosine`
4. `bump`
5. `gabor`
6. `morlet`
7. `inverse_quadratic`
8. `triangular`

Notes:

1. Favor functions that can localise smoothly and remain numerically stable.
2. Prefer functions with interpretable center / width controls.
3. If two functions are effectively duplicates, keep the cleaner one and document why.
4. If a candidate is unstable, mark it as unstable and stop wasting compute on it.

## Comparison Axes

Run all three comparison modes:

1. Same-network comparison
   Baseline and localised models use the exact same architecture.

2. Width / length sweep
   Sweep model size to see where localisation starts helping or stops helping.

3. Smaller-localised vs bigger-baseline comparison
   Localised model is intentionally smaller. This is important because the main claim is efficiency, not just accuracy.

## Architecture Grid

Use a practical, CPU-friendly grid that still spans small to large networks.

For HO, use:

1. Single-layer widths: `[16]`, `[32]`, `[64]`, `[128]`, `[256]`
2. Two-layer widths: `[16,16]`, `[32,32]`, `[64,64]`, `[128,128]`

Mixed-size comparisons for HO:

1. baseline `[32,32]` vs localised `[16,16]`
2. baseline `[64,64]` vs localised `[32,32]`
3. baseline `[128,128]` vs localised `[64,64]`
4. baseline `[256]` vs localised `[128]`

For Heat Equation:

1. Single-layer widths: `[32]`, `[64]`, `[128]`
2. Two-layer widths: `[32,32]`, `[64,64]`, `[128,128]`

Mixed-size comparisons for Heat Equation:

1. baseline `[64,64]` vs localised `[32,32]`
2. baseline `[128,128]` vs localised `[64,64]`

For 4D:

1. Single-layer widths: `[16]`, `[32]`, `[64]`
2. Two-layer widths: `[16,16]`, `[32,32]`, `[64,64]`

Mixed-size comparisons for 4D:

1. baseline `[32,32]` vs localised `[16,16]`
2. baseline `[64,64]` vs localised `[32,32]`

If early pilots show a grid point is clearly too weak or too expensive, document that and prune it.

## Domain Sweeps

### Harmonic Oscillator

Run the HO study on:

1. `pi`
2. `2pi`
3. `3pi`
4. `4pi`

### Heat Equation

Use a larger domain range than HO:

1. `pi`
2. `2pi`
3. `4pi`
4. `8pi`

### 4D

Keep 4D narrower at first because cost grows quickly:

1. `pi`
2. `2pi`
3. `4pi`

If 4D is still cheap enough, extend later. Do not overcommit on the first pass.

## Training Budgets

### Harmonic Oscillator

Run:

1. `1k`
2. `3k`
3. `5k`
4. `10k`
5. `50k`
6. `100k`

### Heat Equation

Run:

1. `1k`
2. `3k`
3. `5k`
4. `10k`
5. `50k`

### 4D

Run:

1. `1k`
2. `3k`
3. `5k`
4. `10k`

The intent is:

1. HO gets the deepest budget sweep.
2. Heat gets enough range to reveal whether localisation still helps at medium and long budgets.
3. 4D stays capped so each full seed-batch resolves within a few hours instead of dragging indefinitely.

## Seeds

Every experiment configuration must run with 10 random seeds.

Use the same fixed seed list everywhere so comparisons are paired and fair.

Recommended seed list:

1. `1`
2. `2`
3. `3`
4. `4`
5. `5`
6. `6`
7. `7`
8. `8`
9. `9`
10. `10`

Do not change the seed set between baseline and localised runs.

## Execution Strategy

Use local CPU execution, not Slurm, for this campaign unless explicitly told otherwise.

Run multiple experiments in parallel so the CPU does not sit idle. Target 10 concurrent workers if the machine stays responsive and memory is stable. If 10 is too aggressive, drop only as much as needed.

Run order:

1. HO function screen
2. HO architecture comparisons
3. Heat Equation function screen
4. Heat Equation architecture comparisons
5. 4D function screen
6. 4D architecture comparisons

Pragmatic rule:

1. If a function is consistently unstable or clearly dominated in HO, do not carry it into the full Heat and 4D sweep.
2. Keep the screening logic documented so pruning does not look arbitrary.

## What To Measure

For every run, record at minimum:

1. wall-clock time
2. total epochs
3. final train loss
4. final validation / residual metric
5. solution error against the reference solution
6. best-so-far checkpoint statistics

For every 10-seed experiment group, report:

1. mean
2. median
3. standard deviation
4. min / max
5. wins out of 10 against the paired baseline

Key evaluation priorities:

1. less clock time is ideal
2. lower loss / residual is ideal
3. better geometry is ideal
4. smaller localised models beating bigger baselines is especially valuable

If `50k` is already better than `100k` in the ways that matter, note that explicitly. Do not assume the longest run is the best run.

## Geometry / Solution Quality

Treat geometry as a first-class output, not an afterthought.

For HO:

1. compare shape against the expected oscillatory solution
2. note amplitude drift
3. note phase drift
4. note whether the curve stays physically clean over larger domains

For Heat Equation:

1. compare smoothness and diffusion behavior
2. note whether the solution stays stable over the full domain
3. note whether localisation sharpens or distorts the solution

For 4D:

1. use consistent low-dimensional slices or projections
2. compare reference vs prediction on the same slice definitions every time
3. prefer a small number of clear visuals over many noisy ones

## Plot Requirements

Use a light theme only. Do not use a black background.

Use this plotting style:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from cycler import cycler
import seaborn as sns
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
import plotly.graph_objects as go

large = 20
medium = 16
small = 12
text_color = "#404040"

colors = ['#009E73', '#0072B2', '#E69F00', '#F0E442', '#D55E00',
          '#CC79A7', '#56B4E9', '#666666']

params = {
    'axes.titlesize': medium,
    'legend.fontsize': small,
    'figure.figsize': (5, 5),
    'axes.labelsize': small,
    'axes.linewidth': 2,
    'xtick.labelsize': small,
    'xtick.color': text_color,
    'ytick.color': text_color,
    'ytick.labelsize': small,
    'axes.edgecolor': text_color,
    'figure.titlesize': small,
    'axes.prop_cycle': cycler(color=colors),
    'axes.titlecolor': text_color,
    'axes.labelcolor': text_color,
}

plt.rcParams.update(params)
```

For each 10-seed experiment group, save:

1. individual loss plots for each seed
2. individual solution plots for each seed
3. one combined loss plot with all 10 thin traces plus a bold mean trace
4. one combined solution plot with all 10 thin traces plus a bold mean trace
5. reference solution overlay wherever applicable

The combined plots do not replace the individual plots.

## Naming And Output Structure

Use clean experiment names. Avoid generic names like `run_1`, `trial_new`, `final_final`.

Use names like:

1. `ho_function_screen_v1`
2. `ho_same_network_v1`
3. `ho_mixed_size_v1`
4. `heat_function_screen_v1`
5. `heat_same_network_v1`
6. `heat_mixed_size_v1`
7. `4d_function_screen_v1`
8. `4d_same_network_v1`
9. `4d_mixed_size_v1`

Inside each experiment root, structure outputs as:

1. localiser family
2. variant
3. domain
4. budget
5. architecture
6. seed

Example:

`ho_function_screen_v1/gaussian/gaussian_mu_sigma/domain_2pi/epochs_10000/arch_64x64/seed_03/`

Each seed folder should contain:

1. metrics JSON
2. raw log
3. loss plot
4. solution plot
5. any saved prediction arrays needed for later aggregation

Each 10-seed group folder should contain:

1. aggregate JSON
2. aggregate CSV
3. combined loss plot
4. combined solution plot
5. a short markdown summary

## Required Summaries

Maintain a concise status summary as the campaign progresses. Keep it decision-oriented.

For each completed experiment group, write:

1. what was tested
2. which function or setup won
3. whether it won on time, quality, or both
4. whether the geometry looked cleaner or worse
5. whether the result is worth carrying forward

Do not let the status document turn into a dump of raw numbers.

## Decision Rules

Use these rules when deciding what to carry forward:

1. If a localiser is slower and worse, drop it.
2. If a localiser is slightly slower but clearly better in quality, keep it for the next stage.
3. If a smaller localised model matches or beats a bigger baseline, prioritize that result.
4. If a function only works on short domains and breaks on longer ones, mark that clearly.
5. If a function is sensitive to learnable `mu` or `sigma`, note that instead of averaging it away.

## Deliverables

At the end of the campaign, the repo should contain:

1. reproducible scripts for the full sweep
2. clean output directory structure
3. aggregate metrics for every 10-seed experiment group
4. individual and combined plots
5. a concise status document
6. a short conclusion identifying the best localiser families and where localisation genuinely helps

## Implementation Handoff

Use this prompt as both the research brief and the execution brief.

Before writing new code, inspect and reuse the useful parts of:

1. `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation/experiments/validate_localisation_matrix.py`
2. `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation/experiments/recreate_major_experiments.py`
3. `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation/run_all_experiments.sh`
4. `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation/run_all_experiments_detached.sh`
5. `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation/run_gaussian_validation.sh`
6. `/Users/lakshaychawla/Downloads/NAS/Mtech/MTP/localisation/run_gaussian_validation_detached.sh`

Do not duplicate existing patterns blindly. Reuse only what actually helps the new campaign.

## Runner Requirements

Build a proper campaign runner that can execute the study described in this file.

The runner must support:

1. HO, Heat Equation, and 4D stages
2. multiple localiser families
3. multiple domains
4. multiple epoch budgets
5. same-network, width-sweep, and mixed-size comparisons
6. 10 paired random seeds
7. bounded local CPU parallelism
8. clean run-root naming
9. per-seed metrics plus per-group aggregates

Implementation requirements:

1. use CLI flags or a clear config structure instead of hardcoding a single case
2. prioritize HO first, then Heat Equation, then 4D
3. add a `--smoke-test` mode
4. add a `--max-workers` control
5. write a run manifest into the run root
6. write a top-level summary file that updates as work progresses
7. save enough metadata that interrupted runs are still understandable

## Smoke Test Requirements

Before any real launch, run a small smoke test that proves:

1. the runner starts correctly
2. the output folder structure is created correctly
3. at least one HO baseline run completes
4. at least one HO localised run completes
5. aggregate bookkeeping works
6. detached launch plumbing works

Keep the smoke test intentionally small. A good default is:

1. HO only
2. 1 or 2 localiser families
3. domain `pi`
4. a very short budget such as `10` to `50` epochs if needed
5. 1 seed
6. 1 same-network comparison

If a slightly larger smoke test is needed to avoid false confidence, keep it minimal and explain why.

## Detached Launch Requirement

The real run must survive closing the app.

Use `screen`, not a foreground shell and not Slurm.

The detached launcher must print and persist:

1. screen session name
2. batch id
3. batch root
4. launcher log path
5. the command or config used to start the run

Do not rely on interactive shell state.

## First Real Run

After the smoke test passes, launch a real detached run immediately.

The first run should be a pragmatic first stage of the campaign, not a reckless full-matrix launch.

Good default:

1. HO first
2. all currently implemented localiser families
3. domains `pi`, `2pi`, `3pi`, `4pi`
4. budgets `1k`, `3k`, `5k`, `10k`
5. same-network plus mixed-size comparisons
6. all 10 seeds

If that is still too large for a safe first launch, reduce only enough to keep the first run sensible, and document the exact subset started.

## Validation Standard

Do not claim success without actual execution checks.

Verify all of the following:

1. the smoke test command succeeds
2. the detached launcher starts successfully
3. the `screen` session exists
4. the launcher log is being written
5. the batch root contains the expected initial files

When reporting completion, include only:

1. which files were added or changed
2. the smoke test command used
3. the detached launch command used
4. the screen session name
5. the batch root
6. the launcher log path
7. any limitations or unfinished parts

## Non-Goals

For this phase:

1. do not include RL-PINN comparisons
2. do not include TorchHolo
3. do not optimize for GPU or cluster execution
4. do not send notifications
5. do not clutter the repo with redundant temporary folders

## Final Standard

The campaign is successful only if the outputs make it easy to answer:

1. Which localiser family is best overall?
2. Is Gaussian actually best, or just the best among what was tried before?
3. Does localisation help only at equal network size, or also when the localised model is smaller?
4. On which equations and domains does localisation remain worthwhile?
5. Is the gain mostly about lower loss, better geometry, less time, or some combination of the three?

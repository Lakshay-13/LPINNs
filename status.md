# Localization Status

Last verified: `2026-07-04`

This file records the current status of the saved runs and experiment branches based on direct inspection of the code and executed notebooks in this folder.

## Overall State

- Repository maturity: exploratory research workspace
- Execution style: notebook-driven
- Automation status: no Slurm scripts, no CLI training entrypoints, no run registry
- Strongest saved branches: harmonic oscillator Gaussian localization, heat-equation Gaussian localization, 4D targeted-sampling branch
- Main blocker: activated localization for the heat equation is currently broken

## Status Scale

- `working`: saved executed notebook exists and the branch appears usable as a research baseline
- `exploratory`: code and runs exist, but results are not centralized or the branch is not yet clean enough to treat as a baseline
- `blocked`: saved notebook shows a concrete failure that must be fixed first

## Branch Summary

| Area | Main branch | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Harmonic oscillator | `HO/gaussian/Single Layer/` | `working` | Executed `final run.ipynb` and a direct `localization vs no localization.ipynb` comparison | Best-documented core branch for localization on the nonlinear HO |
| Harmonic oscillator | `HO/gaussian/Multiple Layers/` | `working` | Executed `final_run.ipynb` | Extends Gaussian localization beyond a single localized layer |
| Harmonic oscillator | `HO/activated localization/` | `exploratory` | Executed `final run.ipynb` exists, but the branch is less standardized than the Gaussian line | Worth preserving, but not yet the primary baseline |
| Harmonic oscillator | `HO/super gaussian/`, `HO/boxcar/`, `HO/ricker wavelet/` | `exploratory` | Multiple exploratory notebooks and separate model files | Kernel-family sweep, useful for idea history rather than current baseline |
| Harmonic oscillator | attention / embedding / multinet variants | `exploratory` | Side notebooks exist but are not integrated into the main lines | Architectural alternatives, not the current main direction |
| Heat equation | `Heat Equation/Gaussian/` | `working` | Executed `final run.ipynb` with localized and non-localized runs at two widths | Main successful PDE extension in the repo |
| Heat equation | `Heat Equation/Activated localization/` | `blocked` | Saved trial fails with matrix-shape mismatch | Needs repair before reuse |
| 4D | `4D/Run 1-4.ipynb` | `exploratory` | Four saved runs with explicit minimum training losses | Latest branch and most likely next target for cleanup |

## Harmonic Oscillator Status

### Main Gaussian Single-Layer Branch

- Core implementation: `HO/gaussian/Single Layer/harmonic_oscillator.py`
- Saved final notebook: `HO/gaussian/Single Layer/final run.ipynb`
- Comparison notebook: `HO/gaussian/Single Layer/localization vs no localization.ipynb`

Saved final configuration:

- domain scaled to `[0, 1]`
- original HO span: `0` to `4pi`
- `input_neurons = [16, 256]`
- activation: custom sine activation
- `mu = False`
- `sigma = False`
- `localization = True`
- `epochs = 5000`

Interpretation:

- This is the cleanest saved HO branch.
- The comparison notebook explicitly runs both `localization=False` and `localization=True` for 10 seeds each.
- The saved final configuration uses fixed localization parameters in that notebook, so the idea of learnable locality is explored elsewhere in the repo but not uniformly used in the final saved HO configuration.

### Multi-Layer Gaussian Branch

- Core implementation: `HO/gaussian/Multiple Layers/harmonic_oscillator.py`
- Saved final notebook: `HO/gaussian/Multiple Layers/final_run.ipynb`

Saved configurations in the final notebook include:

- deep narrow localized network: `[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]`
- wider localized network: `[64, 64]`
- `epochs = 5000`

Interpretation:

- This branch is beyond pure first-layer localization.
- It is useful if the next goal is to test whether locality should stay shallow or become hierarchical.

### Activated Localization Branch

- Core implementation: `HO/activated localization/harmonic_oscillator.py`
- Saved final notebook: `HO/activated localization/final run.ipynb`

Saved final notebook configurations include:

- `[256]`
- `[512]`
- `[1024]`
- `sigma = False`
- `epochs = 5000`

Interpretation:

- This branch has completed saved runs and should be kept.
- It is still less consolidated than the Gaussian HO line.
- An older exploratory notebook, `single_layer.ipynb`, records a long run and is not a good summary metric for branch quality.

## Heat Equation Status

### Gaussian Heat-Equation Branch

- Core implementation: `Heat Equation/Gaussian/heat_equation.py`
- Saved final notebook: `Heat Equation/Gaussian/final run.ipynb`

Saved runs in the final notebook:

- `input_neurons = [128, 128]`, `localization = True`, `epochs = 300000`
- `input_neurons = [128, 128]`, `localization = False`, `epochs = 300000`
- `input_neurons = [512, 512]`, `localization = True`, `epochs = 300000`
- `input_neurons = [512, 512]`, `localization = False`, `epochs = 300000`

Interpretation:

- This is the strongest saved PDE branch in the repo.
- It already contains matched localized vs non-localized runs at two widths.
- This branch is a good candidate for conversion into a non-notebook training pipeline.

### Activated Heat-Equation Branch

- Core implementation: `Heat Equation/Activated localization/heat_equation.py`
- Saved trial: `Heat Equation/Activated localization/trial.ipynb`

Current failure:

- `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x16 and 32x1)`

Interpretation:

- This branch is not currently runnable as saved.
- It should be treated as blocked until the head/fusion dimensions are corrected.

## 4D Status

The 4D branch changes the problem setup in two ways:

- it uses localized layerwise gating through `4D/lnn.py`
- it also biases training samples toward a target region using a Latin-hypercube generator with a concentrated fraction near `(0.8, 0.8, 0.8, 0.8)`

Shared 4D sampling setup in saved runs:

- `Nsamp = 4096`
- `target_point = (0.8, 0.8, 0.8, 0.8)`
- `target_point_std = (0.01, 0.01, 0.01, 0.01)`
- `target_point_frac = 0.3`
- `max_epochs = 150000`

Saved run summary:

| Notebook | Hidden units | Min train loss | Status note |
| --- | --- | ---: | --- |
| `4D/Run 1.ipynb` | `[32, 32]` | `0.7506964377` | baseline 4D localized run |
| `4D/Run 2.ipynb` | `[32, 16, 8]` | `0.6704457263` | improved over Run 1 |
| `4D/Run 3.ipynb` | `[32, 16, 16, 8, 8]` | `0.4110166847` | best saved 4D result |
| `4D/Run 4.ipynb` | `[16, 16, 8, 8]` | `0.6134128902` | latest file, but regresses vs Run 3 |

Interpretation:

- `Run 3` is the strongest saved 4D configuration.
- `Run 4` is the latest iteration by date, but not the best saved run.
- If the next step is Slurm migration, `Run 3` is the more defensible starting point.

## What Is Ready For Migration

If this repo is being converted from notebooks to reproducible job scripts, the best immediate candidates are:

1. `HO/gaussian/Single Layer/`
2. `Heat Equation/Gaussian/`
3. `4D/Run 3.ipynb`

## Main Gaps

- no `requirements.txt` or environment file
- no centralized benchmark table
- no standardized saved metric export
- no Slurm runners
- no automated tests for the custom localized architectures

## Recommended Immediate Next Steps

1. Convert the HO Gaussian single-layer branch into a scriptable baseline.
2. Convert the heat-equation Gaussian branch into a scriptable baseline.
3. Recreate `4D/Run 3` as a single Python entrypoint with fixed config.
4. Repair the activated heat-equation branch only after the Gaussian branches are stabilized.

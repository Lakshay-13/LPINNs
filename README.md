<p align="center">
  <img src="assets/poster.png" alt="LPINNs poster" width="100%">
</p>

# LPINNs

LPINNs: First-Layer Gated Localization for Physics-Informed Neural Networks.

LPINNs multiply the first hidden representation of an otherwise standard dense PINN by input-dependent localization functions. This gives first-layer units local receptive fields while keeping one smooth global network, with no domain partition, interface loss, or extra subnetwork.

## Paper and headline results

- [Submission manuscript PDF](claude_revision/main.pdf)
- [LaTeX source](claude_revision/main.tex)
- [Primary results and provenance notes](results/primary_results.md)
- [Reproducibility and experiment status](status.md)

The completed campaign contains 24,960 configurations and 249,600 process-completed seed records across harmonic-oscillator, heat-equation, and 4D fourth-order settings. The three detailed, ten-seed comparisons in the manuscript are:

| Problem and setting | Localized configuration | Baseline solution RMSE | LPINNs solution RMSE | LPINNs wins |
| --- | --- | ---: | ---: | ---: |
| HO, $2\pi$, 3k epochs | Fixed Gaussian | 0.483688 | 0.008830 | 10/10 |
| Heat, $8\pi$, 10k epochs | Inverse-quadratic, learnable centers and widths | 0.289589 | 0.030589 | 10/10 |
| 4D, $4\pi$, 10k epochs | Fixed bump | 17.5947 | 0.226047 | 10/10 |

These are scoped results, not a claim of universal improvement. The family screen shows that the choice of localization function matters: inverse-quadratic localization is the only family that beats the baseline in all three primary problem columns, while several families are substantially worse and Gaussian-, Ricker-, Gabor-, and Morlet-based 4D runs can become non-finite.

## Method

For a standard first hidden representation $h^{(1)}$, LPINNs use

```text
h_localized = h_first_layer * g(x; localization_parameters)
```

The rest of the network, differential-equation residual, optimizer, collocation scheme, and boundary-condition transform remain unchanged. The completed study screens 13 families:

- Gaussian
- activated
- super-Gaussian
- Ricker
- boxcar
- Laplace
- Cauchy
- raised-cosine
- bump
- Gabor
- Morlet
- inverse-quadratic
- triangular

Each family is evaluated with fixed parameters, learnable widths, and learnable centers and widths where the implementation supports the variant.

## Repository layout

```text
HO/                         Harmonic-oscillator experiments
Heat Equation/              Heat-equation experiments
4D/                         Four-dimensional experiments
experiments/                Reproducible runners and validation tools
claude_revision/            Submission manuscript source, figures, and PDF
results/                    Compact public result summaries
assets/poster.png           LPINNs repository banner
```

The notebook archive contains exploratory history. The reproducible command-line runners and the manuscript’s primary evidence are the recommended entry points for the public release.

## Reproduce

Create an environment with Python 3.11 or a compatible Python version, then install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run a bounded Gaussian validation or the complete experiment recreation with the supplied scripts. Outputs are written to an ignored `repro_runs/` directory by default:

```bash
./run_gaussian_validation.sh --help
./run_all_experiments.sh --help
```

The full campaign used CPU-only multiprocessing and ten paired seeds. The public results summary records the exact primary settings, aggregate values, and interpretation limits. Raw logs, checkpoints, and generated campaign galleries are intentionally not part of the Git release because they are very large.

## Limitations

The evidence covers three equations and a controlled set of domain lengths, budgets, architectures, and localization families. Comparisons are matched on epoch budget, not equal wall-clock cost. The family-screen minima are selected over parameterizations and should not be treated as unbiased model selection. The 4D manufactured target has a simple analytic form, so its result primarily tests optimization under a fourth-order residual. Broader claims require additional coupled, non-separable, and physical problems.

## Citation and release status

This repository is the public LPINNs release accompanying the submission manuscript. The paper source is intentionally kept in submission form; update the author block and venue metadata only when the submission workflow requires it.

# LPINNs release status

Last verified: `2026-09-17`

## Release source of truth

- Public project name: **LPINNs**
- Public repository: <https://github.com/Lakshay-13/LPINNs>
- Paper source: maintained separately from this public code repository
- Compact result record: [`results/primary_results.md`](results/primary_results.md)
- Completed local campaign used for verification: `campaign_runs/project/run_1`

## Verified campaign state

The completed campaign contains 24,960 configurations and 249,600 process-completed seed records with zero process failures. The generated campaign also contains completed-but-non-finite metric records; these are preserved in the local analysis and are not silently converted to finite values.

The manuscript reports three detailed ten-seed paired comparisons:

| Problem | Configuration | Solution RMSE, baseline | Solution RMSE, LPINNs | Paired solution wins |
| --- | --- | ---: | ---: | ---: |
| HO at $2\pi$, 3k epochs | Fixed Gaussian | 0.483688 | 0.00883028 | 10/10 |
| Heat at $8\pi$, 10k epochs | Inverse-quadratic, learnable centers and widths | 0.289589 | 0.0305888 | 10/10 |
| 4D at $4\pi$, 10k epochs | Fixed bump | 17.5947 | 0.226047 | 10/10 |

## Interpretation

- First-layer localization is a useful optimization intervention in the three reported settings.
- It is not a universal win: the family screen is strongly equation-dependent.
- Inverse-quadratic localization is the only screened family that beats the baseline in all three primary problem columns.
- Gaussian, Ricker, Gabor, and Morlet 4D family-screen runs can be non-finite under the tested fourth-order residual.
- Reported comparisons are matched on epoch budget, not equal wall-clock cost.
- The 4D manufactured target is deliberately simple; that result primarily tests optimization under a fourth-order residual.

## Reproducible entry points

- `reproduce.sh` is the single repeatable experiment entry point.
- `experiments/run_experiments.py` contains the complete HO, Heat, and 4D campaign definitions and execution logic.

Default generated outputs go under ignored `repro_runs/`. The large completed campaign tree under `campaign_runs/` stays local; the public release contains the compact, checked result summary instead of raw logs and checkpoints.

## Known limitations

The public release does not claim that one localization family, architecture, or budget is optimal outside the tested settings. Broader evidence is still needed for coupled and non-separable physical problems, more domains and budgets, equal-compute comparisons, and stabilized high-order residuals.

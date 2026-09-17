# LPINNs primary results

This compact record is the public release summary for the completed campaign. It was checked against the per-group `aggregate.json` files in the local campaign root `campaign_runs/project/run_1` on 2026-09-17. The raw campaign logs and checkpoints are intentionally excluded from Git because the generated tree is hundreds of megabytes.

## Campaign coverage

| Quantity | Value |
| --- | ---: |
| Configurations | 24,960 |
| Requested seed records | 249,600 |
| Process-completed seed records | 249,600 |
| Process failures | 0 |
| Localization families | 13 |
| Paired seeds per reported comparison | 10 |

Some completed family-screen records are non-finite. They remain counted as process-completed but are excluded from valid metric aggregates. In particular, Gaussian, Ricker, Gabor, and Morlet 4D runs are recorded as non-finite at the first checkpoint for the tested parameterizations.

## Detailed comparisons

Values are means over ten paired seeds. Lower is better. `Solution wins` counts paired seeds for which the LPINNs solution RMSE is lower than the matched baseline.

| Problem | Domain / budget | Baseline residual RMSE | LPINNs residual RMSE | Baseline solution RMSE | LPINNs solution RMSE | Baseline solution MAE | LPINNs solution MAE | Solution wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Harmonic oscillator | $2\pi$ / 3k | 0.119945 | 0.0263865 | 0.483688 | 0.00883028 | 0.40788 | 0.00705632 | 10/10 |
| Heat equation | $8\pi$ / 10k | 0.704747 | 0.149852 | 0.289589 | 0.0305888 | 0.2319 | 0.0224076 | 10/10 |
| 4D fourth-order | $4\pi$ / 10k | 0.00732024 | 0.000343585 | 17.5947 | 0.226047 | 14.052 | 0.129587 | 10/10 |

The exact configurations are fixed Gaussian localization for HO, inverse-quadratic localization with learnable centers and widths for Heat, and fixed bump localization for 4D. The residual win count is 9/10 for HO and 10/10 for both Heat and 4D; the detailed conclusions use solution accuracy as the primary outcome.

## Family-screen interpretation

At the manuscript’s primary settings, the best finite mean solution-MSE entries by family are:

| Localization family | HO | Heat | 4D |
| --- | ---: | ---: | ---: |
| Baseline | $9.946\times10^{-1}$ | $8.386\times10^{-2}$ | $3.126\times10^{2}$ |
| Gaussian | $8.402\times10^{-5}$ | $7.956\times10^{-2}$ | n/a |
| Activated | $1.788\times10^{1}$ | $7.502\times10^{-2}$ | $1.777\times10^{7}$ |
| Super-Gaussian | $1.541\times10^{1}$ | $7.470\times10^{-2}$ | $1.621\times10^{7}$ |
| Ricker | $1.148\times10^{1}$ | $8.339\times10^{-2}$ | n/a |
| Boxcar | $9.140$ | $7.237\times10^{-2}$ | $1.346\times10^{7}$ |
| Laplace | $1.509\times10^{1}$ | $7.895\times10^{-2}$ | $7.892\times10^{1}$ |
| Cauchy | $1.435$ | $1.640\times10^{-3}$ | $3.613$ |
| Raised-cosine | $9.169$ | $7.162\times10^{-2}$ | $7.233\times10^{6}$ |
| Bump | $1.727\times10^{1}$ | $6.557\times10^{-2}$ | $5.650\times10^{-2}$ |
| Gabor | $1.862\times10^{1}$ | $8.491\times10^{-2}$ | n/a |
| Morlet | $2.280\times10^{1}$ | $8.410\times10^{-2}$ | n/a |
| Inverse-quadratic | $1.963\times10^{-5}$ | $9.780\times10^{-4}$ | $9.924\times10^{-1}$ |
| Triangular | $1.391\times10^{1}$ | $1.176\times10^{-2}$ | $2.598\times10^{6}$ |

Each family value is the minimum over the finite parameterizations tested for that family, whereas the baseline has no such selection. These values are therefore useful for screening and not an unbiased estimate of a pre-registered winner. The defensible release conclusion is that first-layer localization can materially improve solution accuracy in the three detailed settings, but its benefit is equation- and family-dependent.

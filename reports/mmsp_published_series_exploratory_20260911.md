# MMSP published-series exploratory execution — 2026-09-11

## Scope and amendment

Track `mmsp-published-series-exploratory-v1` implements the explicit amendment in
`protocol_amendment_log.md` without changing MMSP admission under the formal
`pv-tsfm-v1-20260909` protocol. It uses the published capacity-normalized `power`
sequence in raw timestamp order and row-index arithmetic. No timezone was assigned,
no UTC conversion or aggregation was performed, and interval semantics remain
`unknown`. Consequently, results refer to a nominal-hourly published sequence and do
not validate true hourly-average PV power.

The frozen roles contain 64 training, 12 validation, and 12 test sites. Every site
contains 12,840 finite target records. The label boundary is row index 10,272
(`floor(0.8 * 12840)`). With 336 context records and 24 future records, each test site
has exactly 2,545 complete origins (indices 10,271--12,815), giving 30,540 total.
Training sites provide 634,432 legal pre-boundary windows; validation sites provide
30,540 held-out origins. Test and validation sites never enter the training sampler.

## E0/E1 results

The table reports nonnegative-postprocessed, equal-site macro MAE/RMSE. All four
methods were evaluated on exactly the same 30,540 origin keys and 732,960 24-lead
target points.

| Method | Prefix | MAE | RMSE |
|---|---:|---:|---:|
| Last value | 1 | 0.062433 | 0.106936 |
| Daily seasonal persistence | 1 | 0.072621 | 0.153869 |
| Past-7-days same-hour mean | 1 | 0.065464 | 0.128380 |
| Chronos-2 E1 | 1 | **0.033989** | **0.072359** |
| Last value | 4 | 0.127598 | 0.211279 |
| Daily seasonal persistence | 4 | 0.072621 | 0.153869 |
| Past-7-days same-hour mean | 4 | 0.065464 | 0.128380 |
| Chronos-2 E1 | 4 | **0.044988** | **0.094849** |
| Last value | 24 | 0.239937 | 0.344201 |
| Daily seasonal persistence | 24 | 0.072477 | 0.153715 |
| Past-7-days same-hour mean | 24 | 0.065451 | 0.128339 |
| Chronos-2 E1 | 24 | **0.059181** | **0.123685** |

Against the best E0 MAE at each prefix, E1 changes MAE by -0.028444 (-45.56%) at
step 1, -0.020477 (-31.28%) at steps 1--4, and -0.006270 (-9.58%) over all 24
steps. Corresponding RMSE changes are -0.034577 (-32.33%), -0.033531 (-26.12%),
and -0.004654 (-3.63%). These are exploratory sequence results, not cross-dataset
transfer results.

## Profiling and training

The 100-step profiles used seed 11 and the first 12,800 rows of its frozen 128,000-row
window stream. Both used microbatch 32, gradient accumulation 4, effective batch 128,
bf16, a fixed 1,000-step scheduler, and one physical GPU per run.

| Method | GPU | LR | Profile time | Seconds/step | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| E4 LoRA | 0 | 1e-5 | 41.60 s | 0.416 | 948,674,560 B |
| E5 Full CPT | 1 | 1e-6 | 29.05 s | 0.290 | 1,639,564,288 B |

No OOM occurred, so no microbatch reduction or two-GPU single-run exception was
used. Formal exploratory training then completed all six locked runs:

| Method | Seed | Steps | GPU | Time | Peak allocated VRAM | Checkpoints |
|---|---:|---:|---:|---:|---:|---|
| E4 LoRA | 11 | 1000 | 0 | 360.14 s | 948,674,560 B | 500, 1000 |
| E4 LoRA | 22 | 1000 | 0 | 360.47 s | 948,674,560 B | 500, 1000 |
| E4 LoRA | 33 | 1000 | 0 | 357.98 s | 948,674,560 B | 500, 1000 |
| E5 Full CPT | 11 | 1000 | 1 | 247.57 s | 1,639,564,288 B | 500, 1000 |
| E5 Full CPT | 22 | 1000 | 1 | 243.69 s | 1,639,564,288 B | 500, 1000 |
| E5 Full CPT | 33 | 1000 | 1 | 239.46 s | 1,639,564,288 B | 500, 1000 |

Every history has contiguous steps 1--1000 and finite loss, learning rate, and
gradient norm. The E4/E5 checkpoints have not yet been scored on validation/test
origins; completion here means the requested training executions completed, not that
adapted-model performance has been established.

## Exceptions and claim limits

- The first manifest command failed before reading data because the script did not
  initially add the repository `src/` directory to its import path; it was fixed and
  the complete manifest was regenerated.
- The first eight-window E1 smoke input used a 2-D tensor, while Chronos-2 requires
  `(series, variate, history)`; the corrected 3-D smoke passed before full evaluation.
- No real evaluation or training run failed, no target value was filled, and no test
  error was used for configuration or site selection.
- StateGrid-to-PVOD remains separate. This MMSP track neither replaces cross-dataset
  validation nor completes the formal two-task MVP.

## Artifacts

- Configuration: `configs/mmsp_published_series_exploratory.yaml`
- Data/split/origin manifest:
  `manifests/mmsp_published_series_exploratory/data_and_origin_manifest.json`
- Frozen training sampler:
  `manifests/mmsp_published_series_exploratory/training_samples/sampling_manifest.json`
- E0/E1 summary and per-site tables:
  `reports/results/mmsp_published_series_exploratory/e01_metrics_summary.csv` and
  `reports/results/mmsp_published_series_exploratory/e01_metrics_by_site.csv`
- Training summary and full execution audit:
  `reports/results/mmsp_published_series_exploratory/training_summary.csv` and
  `reports/results/mmsp_published_series_exploratory/execution_audit.json`
- Raw predictions, profiles, histories and checkpoints (local, git-ignored):
  `outputs/mmsp_published_series_exploratory/`
- Logs: `logs/mmsp_published_series_exploratory/`

No plot was generated because no visualization is needed to establish the requested
origin invariant or execution status.

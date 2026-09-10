# PV-TSFM MVP execution status

Date: 2026-09-10 (Asia/Shanghai)

Protocol: `pv-tsfm-v1-20260909`

## Current result

The scientific question has **not yet been evaluated**. No source or target test
score has been produced, and no 100-step or 1000-step GPU training run has been
started. This is an intentional protocol stop rather than a negative experimental
result.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 1. Local data audit | complete | `reports/data_audit.md`, inventory with paths, sizes, hashes and structural statistics |
| 2. Unified hourly loader | code complete, real admission pending | strict power/energy aggregation and no-fill tests |
| 3. Data/split/origin manifests | generator complete, real manifests blocked | `manifests/*.pending.json`; no guessed grouping or metadata |
| 4. E0 and E1 | code complete | three E0 methods; official `amazon/chronos-2` adapter; local pinned checkpoint smoke-tested on CPU |
| 5. E1 on all test origins | blocked | no protocol-valid split/origin manifest exists |
| 6. E4 LoRA and freeze/reset tests | code complete | 97 target modules, 1,206,912 trainable parameters, strict PEFT failure behavior |
| 7. 100-step profiling | not started | Phase 5 and real source windows are prerequisites |
| 8. E4 seeds 11/22/33 | not started | profiling and protocol-valid manifests are prerequisites |
| 9. E5 seeds 11/22/33 | not started | profiling and protocol-valid manifests are prerequisites |
| 10. MVP result summary | not available | no target scores exist |

## Blocking data facts

1. MMSP has 88 observed site IDs, but a canonical physical-location grouping has
   not been verified. The exact-88 split rule therefore cannot yet be invoked.
2. MMSP target kind, unit, timezone, interval labeling and data-use scope remain
   unverified.
3. StateGrid power is identified as MW and the raw cadence is 15 minutes, but
   interval labeling and authoritative timezone remain unverified. Independence of
   all eight physical locations also requires confirmation.
4. The local AI-PVOD archive contains station00, station01, station07 and station08.
   The frozen external test requests station00, station04, station07 and station08.
   No evidence permits station01 to be relabeled as station04.
5. AI-PVOD power target kind, unit, interval labeling and authoritative timezone
   remain unverified.

These fields control hourly aggregation, chronological boundaries, leakage
prevention and site-out grouping. Substituting assumptions would change the frozen
experiment rather than merely completing an implementation detail.

## Minimum unblock requirements

- Provider/project metadata or authoritative code that resolves the MMSP physical
  location grouping and power/time semantics.
- Provider/project metadata or authoritative code that resolves StateGrid interval
  labeling, timezone and eight-site physical independence.
- The required PVOD-China station04 power series, or an authoritative mapping that
  proves which local member is station04, plus its power/time semantics.
- Confirmation of dataset release/data-use scope recorded in the data manifest.

After these are available, the next legal action is to replace the pending manifests
with admitted data, split and origin manifests, then run E0/E1 over every locked
origin before any training profiling.

## GPU policy state

- Allowed physical IDs: 0 and 1 only.
- Detected models for both IDs: NVIDIA RTX 5880 Ada Generation, 49,140 MiB each.
- Default: one run per GPU, with independent seeds/runs in parallel.
- A two-GPU single run is restricted to E5 Full CPT after a recorded single-GPU OOM
  across legal microbatch reductions that preserve effective batch 128.
- No project experiment has used a GPU as of this report.

## Verification

The local test suite passes 23 tests. It covers hourly aggregation, split integrity,
origin eligibility, no-future-leakage, E0 behavior, strict shared-origin comparison,
MAE/RMSE, LoRA freezing, fresh checkpoint reset, sampling identity and GPU-policy
enforcement.

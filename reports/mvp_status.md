# PV-TSFM MVP execution status

Date: 2026-09-11 (Asia/Shanghai)

Protocol: `pv-tsfm-v1-20260909`

## Current result

The formal two-task scientific question has **not yet been evaluated**. No formal
source or target test score has been produced, and no formal-protocol 100-step or
1000-step GPU training run has been started. This is an intentional protocol stop
rather than a negative experimental result.

Separately, the explicitly amended `mmsp-published-series-exploratory-v1` track has
now completed E0/E1 and all E4/E5 seed 11/22/33 training runs. It uses raw published
sequence indices without asserting timezone or interval semantics and therefore does
not change the formal status above. See
`reports/mmsp_published_series_exploratory_20260911.md`.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 1. Local data audit | complete | `reports/data_audit.md`, inventory with paths, sizes, hashes and structural statistics |
| 1b. Primary-source evidence follow-up | complete | `reports/data_evidence_review_20260910.md`, official-release machine comparison |
| 2. Unified hourly loader | code complete, real admission pending | strict power/energy aggregation and no-fill tests |
| 3. Data/split/origin manifests | MMSP roles precomputed, time manifests blocked | frozen 64/12/12 roles; no guessed time semantics or StateGrid physical grouping |
| 4. E0 and E1 | code complete | three E0 methods; official `amazon/chronos-2` adapter; local pinned checkpoint smoke-tested on CPU |
| 5. E1 on all test origins | blocked | no protocol-valid split/origin manifest exists |
| 6. E4 LoRA and freeze/reset tests | code complete | 97 target modules, 1,206,912 trainable parameters, strict PEFT failure behavior |
| 7. 100-step profiling | not started | Phase 5 and real source windows are prerequisites |
| 8. E4 seeds 11/22/33 | not started | profiling and protocol-valid manifests are prerequisites |
| 9. E5 seeds 11/22/33 | not started | profiling and protocol-valid manifests are prerequisites |
| 10. MVP result summary | not available | no target scores exist |

## Problems that block correct computation

1. MMSP power is now confirmed as capacity-normalized and its 88 IDs are described
   by the paper as 88 geographically dispersed plants. Timestamp timezone, interval
   label, and the published 10-minute-to-hourly reduction remain undocumented.
2. StateGrid is MW at 15-minute granularity, but source timezone and interval
   label/measurement semantics and the official original-to-processed transformation
   remain unverified. Official processed power differs from original on
   30/703/0/241/47/123/122/0 overlapping rows for sites 1--8; site 3 is shortened
   from 52,608 to 20,352 rows; and site 5 has 46 nonnumeric original target markers.
   The local site-4 CSV additionally fills six targets missing in official v4.
3. PVOD power is now confirmed as MW and its original `Timestamp` as UTC. Its
   instantaneous/interval and label semantics remain undocumented.
4. PVOD station01 and station04 are confirmed different physical plants. The refined
   pre-error correction binds official ScienceDB V4 station00/04/07/08; its archive
   is byte-identical across V3--V5. Formal adoption remains pending.

These remaining fields control hourly aggregation, chronological boundaries and
eligible origins. Substituting assumptions would change the frozen experiment rather
than merely complete an implementation detail.

## Problems that limit claims rather than arithmetic

- StateGrid lacks a physical-location group map, so its station-ID transfer track is
  explicitly exploratory and cannot establish location independence.
- MMSP coordinates are anonymized, preventing formal region/climate and geographic
  buffer analyses; the paper nevertheless supports one published ID per plant.
- MMSP file-level use/redistribution scope remains ambiguous.
- AIweather is not a lossless PVOD V4 target copy: it fills absent timestamps for
  stations 0/7/8 and changes one overlapping station00 value without released
  processing code. It is therefore lineage evidence only, not the formal target.
- DKASC member-array lineage, GEFCom evidence, lockbox evidence, and base-model
  exposure remain unresolved. They limit LODO/full-paper claims, not the two MVP
  calculations.

## Minimum unblock requirements

- MMSP source evidence for timestamp timezone and the 10-to-60-minute aggregation /
  interval-label semantics, or a recorded protocol amendment that changes the target.
- StateGrid source evidence for timezone and interval-label/measurement semantics.
  This must also justify the original-versus-processed product and its QC operations.
- PVOD interval-label/measurement semantics and formal adoption of the recorded
  ScienceDB V4 0/4/7/8 member correction.

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

The current local test suite passes 23 tests. It covers hourly aggregation, split
integrity, origin eligibility, no-future-leakage, E0 behavior, strict shared-origin
comparison, MAE/RMSE, LoRA freezing, fresh checkpoint reset, sampling identity and
GPU-policy enforcement. The evidence auditor also reruns successfully and regenerates
`reports/evidence/public_release_audit.json`.

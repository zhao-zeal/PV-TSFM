# Protocol amendment log

## 2026-09-10 — GPU visibility and concurrency ceiling

- Authority: explicit user instruction after Phase 1.
- Change: at most physical GPUs 0 and 1 may be visible/used concurrently. Independent
  single-GPU runs are preferred. A two-GPU single run is allowed only after a legal
  single-GPU Full CPT attempt fails for memory, and must run exclusively.
- Invariants: GPU count cannot change effective batch 128, training steps, sampled
  windows, learning rate, warmup, or any method budget.
- Required logging: CUDA visibility, physical IDs and models, GPU count, peak VRAM per
  GPU, microbatch, accumulation, and effective batch.
- Scientific impact: execution/concurrency policy only; data, model, optimizer,
  endpoint, seed, and selection locks are unchanged.

## 2026-09-10 — proposed PVOD target-member correction (pre-error)

- Status: proposed and recorded before any PV-TSFM target prediction error; not yet a
  runnable formal amendment because interval semantics remain unresolved.
- Evidence: the paper assigns S-1--S-4 to PVOD 0/4/7/8. The Hugging Face
  `full_dataset.zip` assigns S-2 to the physically different station01, while the
  companion `AIweatherdata.zip` contains matched forecast directories and station
  tables for exactly 0/4/7/8. See `reports/data_evidence_review_20260910.md`.
- Proposed change: bind the PVOD external subset to
  `AIweatherdata.zip::station_data/{0,4,7,8}.csv`; exclude
  `full_dataset.zip::All_dataset/S-2.csv` (station01) from formal S-2.
- Invariants: source roles, horizon, origins rule, model, optimizer, seeds, GPU policy,
  effective batch 128, steps, and metrics do not change. No prediction error was used.
- Optional station01 use: a future, separately named exploratory robustness target
  only; never an alias for station04 and never used for selection.

## 2026-09-10 — superseding PVOD release binding (pre-error evidence follow-up)

- Status: proposed before any PV-TSFM target prediction error; supersedes only the
  release binding in the preceding proposal and remains non-runnable until interval
  semantics are resolved.
- New evidence: ScienceDB V3--V5 publish byte-identical official archives. V4 contains
  station00/04/07/08 directly and independently confirms that station01 and station04
  are different plants. The AIweather derivative exactly trims station04, but adds
  97/480/288 timestamps for stations 0/7/8 and changes one overlapping station00
  power value without published transformation code.
- Superseding change: bind the formal power-only target to ScienceDB V4
  `PVODdatasets_v1.0.zip::{station00.csv,station04.csv,station07.csv,station08.csv}` (MD5
  `5cb8ebfb4cdc99973deacf1bea8bacc3`). Retain AIweather only as weather/lineage
  evidence, not as the formal target series. Continue excluding
  `full_dataset.zip::All_dataset/S-2.csv` because it is station01.
- Rationale: the frozen model's allowed input is historical single-site PV power, so
  the official V4 power tables supply the required target without importing opaque
  gap filling. This decision uses release evidence only; no prediction error was
  inspected.
- Invariants: physical target IDs, source roles, horizon, origins rule, model,
  optimizer, seeds, GPU policy, effective batch 128, steps, and metrics do not change.

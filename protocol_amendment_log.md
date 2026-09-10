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

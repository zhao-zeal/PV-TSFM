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

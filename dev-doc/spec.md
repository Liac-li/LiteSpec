# Spec: shared stage context KV

## Configuration and compatibility

Enable `dlite_config.shared_stage_kv` for the v2 student. Old configurations that
omit it use the existing per-layer CHS projection and parameter layout.
`dlite_config.chs_stage_ids` maps the ordered target taps to stages:
`[0,0,1,1,1,2,2,2,2,2,2,2]`. Three nonempty contiguous stages are required.
Persist both fields in checkpoint config. Update the active aligned VLM YAML;
the base text recipe and old checkpoint configs retain their existing behavior.

## Data flow

For CHS from target layer `l`, stage `s`:

`z_l = stage_context_norm[s](h_l + depth_embedding[l] + stage_embedding[s])`.

Stage-specific linear K/V projections each output all existing GQA KV heads.
Concatenate along CHS positions, retaining the original block-major slot order.
Apply a shared context key norm and CHS RoPE once. Pass the same context K/V
tensors to all draft layers without detaching their training graph.

Each draft layer independently computes Q and its local draft-token K/V. Its
local Q/K normalization and RoPE remain layer-local. Joint attention reads
`[shared context KV; layer-local draft KV]` with the existing block mask.
The context tensor is rebuilt for every forward/pivot and is not retained in a
cross-round cache. CHS never becomes a query or enters draft residual updates.

For 4B, 16 Q heads share 4 KV heads, four Q heads per KV pair. Stage separation
does not change this ratio or introduce per-head stage visibility restrictions.

## Acceptance criteria

1. Active config selects the exact 12 taps and 2 / 3 / 7 stage mapping.
2. Both embeddings and all three norms/projection pairs participate in backward.
3. Each stage projection runs once per forward and all draft layers receive the
   same context KV tensors; local draft projections remain independent.
4. Packed multi-anchor blocks preserve isolation and match independent-block
   reference forwards on CPU.
5. New config and weights save/reload with identical outputs; legacy configs
   load without new stage parameters and retain their original computation.
6. Existing optimizer ownership registers stage Linear weights with Muon and
   embeddings/norms with AdamW.
7. Update the experiment document and record verification boundaries. No claim
   of CUDA/FSDP, real-checkpoint, convergence or speed validation from CPU checks.

Malformed stage mappings and stage mode on a non-v2-student fail at construction.
This is a new training architecture; old per-layer projections are not silently
converted to shared stage weights.

## Pure-SFT launch and logging

The active aligned configuration keeps six epochs, block 8, micro batch 1 and
accumulation 4 (global batch 32 on eight GPUs), with the existing CE/L1
objective. Expose the existing SwanLab tracker via `--report-to swanlab`,
`--swanlab-project`, and `--swanlab-name`. Only the coordinator logs; reuse all
current metric names and optimizer-step indexing. Authentication uses
`SWANLAB_API_KEY`; credentials must not enter the uploaded experiment config.

Set `save_epoch_interval: 0.5` in the active VLM YAML. Save at the first completed
optimizer window reaching each half-epoch boundary, and at completion. Disable
the independent fixed-microstep save trigger with `save_interval: 0`; positive
intervals in other recipes retain existing behavior. Keep cross-epoch gradient
accumulation intact. With accumulation 4, boundary overshoot is at most three
microbatches (except when a debug cap ends earlier).

Retain `step_XXXXXXX` directories and atomic `last` updates. Add save reason and
actual fractional epoch progress to checkpoint metadata. Inference weights are
the default; `SAVE_TRAINING_STATE=1` still enables full-state recovery. Restore
the half-epoch save cursor from the training cursor when resuming.

Acceptance: SwanLab CLI/config wiring and credential-free init payload are
inspectable; real log callback scheduling respects half-epoch boundaries,
resume cursor and full accumulation windows. No cloud upload or server training
is claimed from local inspection.

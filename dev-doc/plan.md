# Plan: shared stage context KV

## Inspected integration points

- `litespec/vlm/train.py::make_draft` constructs the VLM draft configuration.
- `scripts/config/vlm/sft_student_aligned.yaml` overrides the original SFT YAML.
- `DLiteDraftModel._apply_chs_depth_embedding` currently applies depth embedding
  and a shared norm; `forward` sends the same normalized CHS to each layer.
- `Qwen3DLiteAttention.forward` currently projects CHS and draft tokens with its
  own K/V weights, applies key norm/RoPE and concatenates the results.
- Training packs CHS by anchor then slot. The existing Flex mask enforces
  own-block visibility; CPU reference and HF video inference call the same draft.
- Optimizer registration already handles Linear, Embedding and RMSNorm modules.

## Steps

1. [x] Inspect active sources and confirm taps/stage boundaries with the user.
2. [x] Add optional stage embedding, norms and context projection modules to the
   existing draft; keep the legacy parameter path when the option is absent.
3. [x] Construct context KV before the layer loop and thread it through decoder
   layers to attention. Preserve packed slot order and existing masks.
4. [x] Wire the two stage configuration fields through the VLM builder and
   enable the selected taps/stages in the active aligned YAML.
5. [x] Inspect shapes, projection counts, gradients, packed-block isolation,
   save/reload, legacy equivalence and optimizer classification with small CPU
   forwards. Do not create test files or run GPU training locally.
6. [x] Update `dev-docs/qwen35-dlite-video.md` and this plan with actual outcomes.

## Risks

Shared context projection is a different parameterization from the old five
independent projections; it requires training. Avoid applying layer-local key
norm or RoPE a second time to shared context. Reordering stages across packed
anchors would break the mask, so retain block-major layout. Keep all gradients
from repeated context reads. Preserve backward compatibility for existing
checkpoint configs. Server CUDA/FSDP execution and latency remain unverified.

## Inspection results

Direct CPU inspection used the active VLM builder/config with a tiny random
draft: five layers, 16 Q / 4 KV heads, 12 CHS, batch 2, three anchors, block 8,
hidden size 64 and head dimension 8. This is not a real 4B checkpoint run.

- All six stage K/V projections executed once; all five layers received the
  same context tensor pair, shaped `[2,4,36,8]` for three packed anchors.
- Original CE/L1 and sequential-head backward produced finite nonzero gradients
  for depth/stage embeddings, all stage norms/projections and context key norm.
- Packed vs independent-block output max absolute difference: `4.768e-7`.
- New checkpoint save/reload output difference: `0`.
- Legacy-mode output vs the exact pre-change attention implementation: `0`.
  Legacy save/reload constructed no stage embedding.
- The unmodified optimizer classification function assigns stage Linear weights
  to Muon and embeddings/norms to AdamW.
- Source syntax and whitespace checks passed. No test files were added.

The local `uv-base` environment has Torch 2.14.0 / Transformers 5.17.0 and no
CUDA; full trainer import also lacks `yunchang`. For optimizer classification,
the existing registration function was executed in isolation from its unrelated
distributed imports. These checks do not validate the locked Torch 2.9.1 /
Transformers 5.20.1 CUDA/FSDP trainer. No environment packages were changed.
Local diagnostic artifacts are under `tests/outputs/stage_kv_inspection/`.

## Pure-SFT launch follow-up

- [x] Confirm global-vs-micro batch 32; preserve six epochs/block 8 and pure SFT.
- [x] Expose the existing SwanLab tracker in `vlm.train_aligned`; pass project
  and experiment name as CLI overrides and omit auth keys from tracker config.
- [x] Set the active YAML to half-epoch saves; allow zero fixed save interval in
  the shared runner. Track fractional boundaries in the VLM callback and resume
  setup; include save reason and actual progress in checkpoint metadata.
- [x] Inspect CLI, tracker init payload and save scheduling without new tests;
  update the experiment document with exact server commands/artifacts.

No DFlash teacher or KL implementation was added. Existing SFT objectives and
optimizer accumulation remain the execution path.

Follow-up inspection confirmed six epochs, global batch 32 on eight GPUs, block
8, SwanLab and 0.5-epoch saves. The actual logging callback, with ten microbatches
per epoch and accumulation four, emitted twelve boundary saves over six epochs;
restoring its cursor did not duplicate a prior save. A real tiny draft export
contained weights/config/run metadata and an atomic `last` symlink; metadata
distinguished requested epoch 0.5 from actual epoch 0.8 in this deliberately
small example. The existing tracker exercised login/init/log/finish against a
recording SDK, excluded auth keys from config and made no rank-one calls. No
cloud request was made; local SwanLab and yunchang packages are absent. Syntax,
whitespace and document-link checks passed; no test files or packages were added.
Artifacts: `tests/outputs/swanlab_launch_inspection/inspection.json`.

## Server launch correction

Observed `train.sh` used unset `DLITE_WORKSPACE`, producing `/runs/...`, and the
VLM launcher defaulted to a nonexistent `.venv-vlm/bin/python`.

- [x] Anchor `train.sh` to its own directory and use repository-local output.
- [x] Invoke active-environment `python` directly in the VLM launcher; keep the
  user's current data/model paths, save-state flag, logging and training recipe.
- [x] Update current launch instructions and inspect shell syntax/path references.
  Do not start training or modify the environment.

Both launchers pass `bash -n` and whitespace inspection. The active training
scripts contain no `.venv-vlm`, `PYTHON_BIN` or `DLITE_WORKSPACE` references.
The user's existing authentication assignment, 100k data path and full-state
save flag were preserved. No training or environment installation was run.

## Parallel validation follow-up

- [x] Add optional original-index selection to `validate_records`, retaining its
  current per-record processor/anchor checks and skip messages.
- [x] Shard fresh validation over current ranks in `preflight`, gather results,
  propagate errors and restore manifest order. Preserve rank-zero reuse/broadcast.
- [x] Inspect serial vs multi-rank behavior and update the training documentation
  and restart instructions. No new test files or environment changes.

Direct inspection used eleven local image records, the real Qwen2VL PIL image
processor with a deterministic local template, and two CPU/Gloo ranks. Valid
indices, excluded records and eligible SHA256 exactly matched serial validation;
rank-zero reuse/broadcast returned the same result without processor access. A
missing image on rank one produced the same rank-attributed error on both ranks.
The sandbox initially blocked local sockets; this inspection completed after
approved local process communication. No GPU or external network was used.
No test files were added. Syntax, whitespace and documentation references passed.
Eight-rank CUDA/NCCL operation and server speed are not locally verified.
Record: `tests/outputs/parallel_validation_inspection/inspection.json`.

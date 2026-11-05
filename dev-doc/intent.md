# Intent: Qwen3.5-4B DLite stage context KV

## Goal

Use target-depth structure to separate DLite conditioning features while keeping
all CHS visible to every draft layer. Continue the current Qwen3.5-4B image/text
direct-SFT to video-evaluation workflow.

## Confirmed scope

- Target taps: `[1,3,7,11,15,18,19,22,23,26,27,31]`.
- Three depth stages: `0–6`, `7–17`, `18–31`, yielding 2 / 3 / 7 CHS.
- Preserve depth embedding and add a shared embedding for each stage.
- Each stage produces context KV once per forward, reused by every draft layer.
- Every layer/head can read all CHS plus its own block's draft tokens.
- Keep the existing 4B target, five draft layers, block 8, RNN rank 320, training
  objective, and target verification behavior.

## Constraints and non-goals

Minimal changes in this checkout. No new tests. Use direct CPU inspection and
forward/backward checks; CUDA/FSDP and performance require the training server.
Do not add historical visual/text tokens, head-specific routing, target-weight
changes, new loss terms, or persistent cross-round context caches.

The motivating measurements are from 9B image samples; improvement on 4B and
video is an experimental hypothesis.

## Current training launch request

Keep pure SFT for comparison with completed checkpoints; DFlash teacher/KL work
was cancelled before any implementation. Use six epochs and block 8, log to
SwanLab, and save at approximately every half epoch on completed optimizer
boundaries. Reuse the existing tracker and provide server commands and an exact
description of the saved artifacts. Batch 32 is confirmed as the global effective
batch: eight GPUs, micro batch 1 per GPU, accumulation 4.

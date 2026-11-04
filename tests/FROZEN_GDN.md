# Frozen DLite + GDN Reader adaptation

Branch: `research/qwen35-gdn-reader-adapt`, based on `research/qwen35-vlm-gdn`.
This tests whether committed native GDN states provide useful supplementary context.
No training objective forces a correct state to outperform a wrong state; that is
measured on independent fixed-prefix probes.

## Model and training

The original local `Qwen3.5-4B-DLite` checkpoint is loaded with strict missing,
unexpected and mismatched-key checks BEFORE installing a new Reader. Its block
size, layer selection, MASK, position scheme and RNN head are retained. Dataset
eligibility is evaluated using the loaded checkpoint's block size, not an assumed
8. The launcher expects the existing 20,000-row image/text training manifest;
the exported effective configuration records the actual eligible count.

Only Reader parameters enter AdamW, with FP32 master parameters. The Target and
original DLite parameters remain frozen. Every export checks a digest of all
non-Reader state tensors against the loaded baseline. No optimizer/RNG checkpoint
is saved. These runs intentionally do not support resume.

| Experiment | CE | Probability L1 | Reader features |
|---|---:|---:|---|
| frozen_reader_legacy_loss | 0.1 | 1.0 | content + log-RMS |
| frozen_reader_ready_loss | 1.0 | 0.1 | content + log-RMS |
| frozen_reader_content_only | 1.0 | 0.1 | content, log-RMS slots zeroed |

All arms use the same backbone, Reader initialization seed, examples, anchors,
position weights and existing position-weighted loss aggregation. This isolates
the weight ratio; it does NOT claim exact reproduction of ReaDy's per-sequence
aggregation, training subsets or two-stage LoRA recipe. Defaults: 3 epochs,
LR 1e-4, cosine schedule, warmup ratio .04, weight decay 0, grad clip 1.0,
rank 8, bottleneck 128, Target layers 14/30, single final-layer post-attention-norm
injection. DLite-specific RNN supervision is retained.

`reader_last_local_backward_norm/*` records the last local backward's gradient
norm BEFORE accumulation normalization/distributed reduction; it is not the global
optimizer gradient. A zero-initialized output projection blocks upstream Reader
gradients at the first backward, so inspect later updates as well.

## Smoke first

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec"
export PYTHON_BIN="$PWD/.venv-vlm/bin/python"
export ORIGINAL_DLITE="$HOME/LMM/lmm_model/Qwen3.5-4B-DLite"
export TARGET_MODEL="$HOME/LMM/lmm_model/Qwen3.5-4B"
export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl"

CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 SMOKE_TEST=1 \
RUN_GROUP="$DLITE_WORKSPACE/runs/frozen_gdn_smoke_$(date +%Y%m%d_%H%M%S)" \
bash tests/train_frozen_gdn.sh
```

The first operation loads the REAL Target and Draft on one visible GPU and checks
BF16 sparse state reconstruction against native prefix caches, zero-init identity,
content-projection gradients after updates, and backbone immutability. It writes
`preflight.json`; these temporary Reader updates are discarded. No checkpoint is
modified on disk. Then torchrun smoke exercises production FlexAttention/FSDP:
32 input records, one epoch, accumulation 1, at most two optimizer steps per arm.
An error stops the launch. No silent fallback is used.

The preflight records the original elementwise atol=.003/rtol=.02 check but gates
on a bounded BF16 policy: relative state L2 <=0.5%, worst-head relative L2 <=2%,
at most 0.01% of elements outside the original tolerance, and no element beyond
4 times its original error budget. ALL these conditions must hold. It additionally
requires <=1% relative error in the initialized Reader's low-rank CONTENT readout,
before log-RMS fusion and the zero-initialized output map. This is a numerical
preflight policy, not a proof of trained-model or lossless decoding equivalence.

Both full-sequence reconstructed states and independently reconstructed SAME-prefix
states are compared with native prefix caches. The two comparisons help separate
length-dependent BF16 backbone differences from recurrence reconstruction errors.
All metrics are printed and written to `preflight.json` BEFORE any gate fails;
the file retains `passed:false` until the entire preflight succeeds. A sparse
pointwise failure alone no longer stops the run, but systematic/head-local/content
errors still do. Do not bypass a failing gate or adjust limits merely to pass.

## Formal training

```bash
export RUN_GROUP="$DLITE_WORKSPACE/runs/frozen_gdn_20k_3ep_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
bash tests/train_frozen_gdn.sh
```

Defaults: microbatch 2 and accumulation 8 => global batch 64 on four GPUs.
For eight GPUs use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8
ACCUMULATION_STEPS=4` to retain global batch 64. The three arms run sequentially.
`EXPERIMENTS` may select a subset of the three names above. Set `TRAIN_DATA` to
the actual prior 20k manifest if your path differs; no data is regenerated.

## Independent content diagnostics

```bash
# Use the saved input manifests of the PRIOR GDN evaluation, not training data.
export DIAG_INPUTS="/absolute/path/to/previous_gdn_run/evaluation/inputs"
DIAG_GPU=7 DIAG_SAMPLES=16 bash tests/diagnose_frozen_gdn.sh
```

This generates one native AR continuation per sample and shares its anchors across
all checkpoints/modes. Modes: normal, off, wrong, wrong_content, content_only,
scale_only, shared_query. `wrong_content` replaces matrix directions while matching
the recipient's raw RMS per layer/head. `content_only`/`scale_only` intervene on the
features, not by setting the raw state to zero (which would change its log-RMS).
`shared_query` repeats the first query within each block; it does not advance state.

Report: `report.md`, `summary.json`, `summary.csv`, `anchors.jsonl`,
`checkpoint_audit.json`, `gradient_probe.json`. First-candidate repairs/breaks are
paired against Reader-off. Baseline-first-rejection repairs require the whole
preceding prefix to remain correct; a changed token behind an earlier new error
does not count as a repair. Reference committed lengths include the anchor. Off on
a verified frozen checkpoint is the original backbone. Fixed-prefix diagnostics
are not throughput or AR equivalence tests. Wrong-state separation alone is not
evidence of benefit: correct-state acceptance must also improve over off.

## Video evaluation

```bash
EVAL_GPU=7 TESTS="VideoDetailCaption Video-MME" EVAL_SAMPLES=16 \
BATCH_SIZES="1 16 64" EVAL_MAX_NEW_TOKENS=256 \
bash tests/eval_frozen_gdn_video.sh
```

Compares AR, original DLite and all selected Reader checkpoints. Uses the existing
video protocol: 16 uniform frames, max 200704 pixels/frame, per-token Target state
recovery, static replicated batches (not online concurrency). Video-MME is evaluated
as description, not official multiple-choice QA. AR mismatches remain visible and
ratios are raw. There is no automatic multi-GPU scheduling or silent OOM reduction.

No Prefill/dual-state reader, LoRA, contrastive state loss or new visual memory is
introduced in this first round. Advance only after useful state content is measured.

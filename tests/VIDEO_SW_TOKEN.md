# Recent Target position windows

Branch: `video-sw-token`, based on `research/qwen35-video`.

This experiment retains K=1/2/5/10 recent **confirmed, processed Target positions**,
including the current pivot. It does not add a visual reader, GDN feature projection,
compressor, or learned parameters. At each position all S selected Target layers are
retained, in time-major order: K × S Draft condition slots (12/24/60/120 for S=12).
The existing layer-depth embeddings and RNN head remain. Earlier positions receive
earlier local RoPE positions; the current pivot-to-query spacing is unchanged.
Invalid left padding is masked. A rejected suffix and the pending bonus token never
enter the window. Prefill initializes it from the last K prompt positions; these
may include visual positions if the prompt ends near an image/video segment.
This is a bounded recent-feature bank, not a full Draft history KV cache.

K=1 preserves the old inference path and parameter schema. K>1 is supported through
`generate_static_batch`, including the existing video adapter; the separate native
`spec_generate` and `generate_steps` APIs reject it explicitly. The scan fixes Target
state recovery to `per_token` and uses greedy decoding. AR match is recorded but
raw speed ratios are not gated by it; these are not a claim of token-exact decoding.

## First: no-training scan

Create a separate worktree so an existing training checkout is unaffected:

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec"
git fetch origin video-sw-token
git worktree add --detach "$DLITE_WORKSPACE/LiteSpec-video-sw-token" FETCH_HEAD
cd "$DLITE_WORKSPACE/LiteSpec-video-sw-token"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RESULT_DIR="$DLITE_WORKSPACE/runs/video_sw_scan_$(date +%Y%m%d_%H%M%S)"
EVAL_GPU=7 EVAL_SAMPLES=16 BATCH_SIZES=1 \
WINDOW_SIZES="1 2 5 10" bash tests/evaluate_sw_token.sh
cat "$RESULT_DIR/comparison.md"
```

By default both parents are evaluated:

- `runs/vlm_three_3ep_20261021_045040_2983971/dlite/last` (early 20k candidate model).
- `runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dlite/last` (100k candidate model).

Set `DLITE_20K` and/or `DLITE_100K` to override/select parents. Symlinks are resolved
before evaluation; overrides never modify saved checkpoint configs.

Defaults: Video-MME, VideoDetailCaption, TextVQA and GSM8K; 16 samples per dataset,
B=1, 256 output tokens, 16 video frames, 448² maximum pixels per frame, 1024 image
tokens, seed 42. `TESTS="Video-MME VideoDetailCaption textvqa text-only"` selects
subsets. `VIDEO_DATA_ROOT` defaults to `$HOME/LMM/liyt/datasets/eval_bench`.
The existing Video-MME **caption prompt** protocol is retained; this does not
measure official Video-MME question-answering accuracy. TextVQA/GSM8K likewise
report drafting/decode metrics rather than answer-quality scores.

Sample selections are frozen once per dataset across all windows/models. Video
asset metadata is checked before reuse. Output includes selection manifests,
per-cell logs/JSON, `comparison.csv`, `comparison.md`, and settings/parent hashes.
Batch is offline static replication of each prompt, not online concurrency.
After the smoke run, use `EVAL_SAMPLES=64 BATCH_SIZES="1 32"` if memory permits.
All jobs run sequentially on `EVAL_GPU`; no distributed serving is introduced.

## Then: matched continued-training controls

A zero-shot K>1 override changes the conditioning distribution; a negative result
alone does not disprove the design. Continue each K from the **same parent**, with
K=1 as a continued-training control:

```bash
export RUN_GROUP="$DLITE_WORKSPACE/runs/video_sw_train_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
WINDOW_SIZES="1 2 5 10" EPOCHS=1 \
MICRO_BATCH_SIZE=2 ACCUMULATION_STEPS=8 \
bash tests/train_sw_token.sh

SW_RUN_GROUP="$RUN_GROUP" EVAL_GPU=7 EVAL_SAMPLES=16 BATCH_SIZES=1 \
WINDOW_SIZES="1 2 5 10" bash tests/evaluate_sw_token.sh
```

`INIT_CHECKPOINT` chooses the shared parent (default early 20k DLite). Parent block
size, layer selection, RNN dimensions and architecture are inherited; only new
window metadata is added. This is a weights-only warm start with a fresh optimizer
and schedule, not optimizer resume. `EPOCHS` means **additional** epochs (default 1;
set 3 if desired). `SW_RUN_GROUP` evaluates saved window settings rather than
applying another inference override.

Training defaults to
`data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl` and image cap 1024.
Override `TRAIN_DATA` when aligning the corpus to a previous experiment; this
validated 20k corpus must not be assumed identical to an older 20k candidate set.
`REUSE_VALIDATION_FROM` may reuse a compatible eligibility manifest; incompatible
fingerprints/image settings fail rather than silently reuse it. Later K runs reuse
the first run's validation. The existing losses, anchors, optimizer and RNN are
retained. All K, including K=1 control, use the same chunked SDPA training path.
`SW_TRAIN_CHUNK_BLOCKS=16` bounds anchor-window materialization and checkpoints
chunks for backward; selected frozen Target states are shared across chunks.

The launcher forces `SAVE_TRAINING_STATE=0`: model weights/config and TensorBoard
are saved, **no `training_state_rank_*` files**. Formal runs should unset inherited
`MAX_STEPS` / `MAX_SAMPLES` debug caps. Parent checkpoints are never overwritten.

## Validation scope

Tiny CPU tests cover window order, padding, accepted-only commits, training/inference
prefix alignment for all K, K=1 parity, chunked gradient parity/future isolation,
real tiny hybrid Qwen3.5 image decoding with per-token state handling, weights-only
training/export, frozen selection and CLI routing. Full checkpoints, actual video
GPU kernels, distributed training memory and throughput still require the server
runs above; no measured acceptance/speed improvement is asserted by this change.

## Source integrity before GPU startup

Both shell entrypoints run `tests/check_sw_token_source.py` first. It checks the
experiment's Python syntax and rejects tool-output truncation markers. The
stdlib-only regression suite can also run without Torch:

```bash
"$PYTHON_BIN" -m unittest discover -s tests -p test_sw_token_source.py -v
```

The initial publication `8c01480` contained a truncated/concatenated
`static_batch.py` despite the local tests having passed. Its repair restores the
file from the video branch and reapplies the window-only edits. The integrity
guard is a publication regression check; it does not validate GPU kernels or
measure performance.

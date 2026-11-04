# Qwen3.5 video benchmark

Branch `research/qwen35-video`, based on `research/qwen35-vlm` (`b72163b`).
Inference only: no training, dataset regeneration or model weight changes.
Existing image/text entrypoints are unchanged. Use the existing `.venv-vlm`.

## Protocol

- Target Qwen3.5-4B; AR vs official DLite, multimodal 100k-candidate DLite and
  DFlash. Defaults reuse `run_vlm_matrix.py` checkpoint paths.
- VideoDetailCaption: original `question`, MP4 or MKV by `video_name`. Exactly
  one matching file required. Original answers are never included in prompts.
- Video-MME: one record per unique `videoID`; repurposed to detailed description
  with a fixed prompt. This is NOT official Video-MME multiple-choice evaluation
  and reports no QA accuracy. Short/medium/long buckets are sampled round-robin
  after seeded within-bucket shuffle (16 gives 6/5/5 if buckets suffice).
- Defaults: 16 unique videos per dataset, seed 42, 16 uniformly selected frames,
  greedy, max_new_tokens=256, B=1/16, `per_token`, SDPA, thinking disabled.
- B=16 means 16 identical replicas of each video/question, not 16 unique concurrent
  online requests. 16 unique videos produce 256 replicated sequences at B=16.
- Frames selected uniformly over source indices, with duplicates if a clip is
  shorter than 16 frames. PyAV decodes MP4/MKV; unknown frame counts are counted
  in a preliminary pass. FPS must be valid; failures stop with an error, never
  silently substitute a different video or resize budget.
- Decoded frames are resized keeping aspect ratio approximately, rounded down to
  spatial patch*merge multiples, with area <=448*448 by default. Tiny/extreme
  aspect videos may require a one-unit minimum. Native processor resizing is
  disabled to enforce this cap regardless of total-video budget defaults. Native
  video normalization, patchification and timestamp insertion are retained, with
  source FPS/indices metadata. This is an explicit per-frame cap, not a claim of
  identical qwen-vl-utils smart-resize heuristics. Grid and token counts are checked.
- Prepared CPU inputs are reused across models and B. Raw frame arrays are released
  after preprocessing. Per-video original/processed sizes, indices, timestamps,
  grid, token counts and preprocessing seconds are saved in `*_inputs.json`.
- Every model/B is warmed up before measurement. GDN recording kernels are warmed
  for each draft/B. Model prefill/decode/total times exclude host transfer, input
  replication, video decode and preprocessing. Those preprocessing costs are
  separately recorded; model end-to-end metrics are NOT request-service latency.
- Accepted draft = committed length minus known anchor. AR equality recorded per
  unique video and scaled by B after the core replica-equality assertion. Ratios
  are raw observations, not lossless claims when output tokens differ.

## Run on the server

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec"
export PYTHON_BIN="$PWD/.venv-vlm/bin/python"
uv pip install --python "$PYTHON_BIN" --no-deps av
# pandas and pyarrow are needed to read local Parquet annotations; if absent:
# uv pip install --python "$PYTHON_BIN" pandas pyarrow
export RESULT_DIR="$DLITE_WORKSPACE/runs/video_matrix_$(date +%Y%m%d_%H%M%S)"
EVAL_GPU=7 EVAL_SAMPLES=16 BATCH_SIZES="1 16" bash tests/run_video_matrix.sh
cat "$RESULT_DIR/comparison.md"
```

Read-only sample/path inspection (no CUDA/models, fresh directory):

```bash
INSPECT_ONLY=1 RESULT_DIR="$DLITE_WORKSPACE/runs/video_inspection_$(date +%Y%m%d_%H%M%S)" \
bash tests/run_video_matrix.sh
```

Real GPU smoke; fresh output, both datasets, AR and all 3 drafts, B=1/16:

```bash
EVAL_GPU=7 EVAL_SAMPLES=1 EVAL_MAX_NEW_TOKENS=16 BATCH_SIZES="1 16" \
RESULT_DIR="$DLITE_WORKSPACE/runs/video_smoke_$(date +%Y%m%d_%H%M%S)" \
bash tests/run_video_matrix.sh
```

Other selectors: `TESTS="VideoDetailCaption"`, `MODELS="dlite_original"`,
`VIDEO_FRAMES`, `VIDEO_MAX_PIXELS`, `VIDEO_DATA_ROOT`, `SAMPLE_SEED`, `WARMUP`,
`TARGET_MODEL`, `ORIGINAL_DLITE`, `DLITE_VLM`, `DFLASH_VLM`.
One GPU runs all cells sequentially to keep comparisons on the same device.
Use `RESUME=1` with the SAME `RESULT_DIR`, samples, checkpoints and selectors to
skip completed cells. Weight stat/config/code/GPU fingerprints must match. A
partially completed cell is rerun; completed cells are not timed again. OOM
raises rather than silently lowering B, frames or pixels. No server GPUs or
real checkpoints are accessible in the development container; run the GPU smoke
before the formal matrix.

Outputs: `settings.json`, `samples.json`, `*_inputs.json`, per-cell JSON, per-video
JSONL text/token traces, incremental `summary.json`, `comparison.csv`,
`comparison.md`. Requests print progress before each generation.

## Local checks

```bash
python -m unittest discover -s tests -p test_video_benchmark.py -v
```

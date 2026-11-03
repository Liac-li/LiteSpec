# Qwen3.5 DLite loop pilot

Branch: `research/qwen35-vlm-looped`.

| Label | Physical layers | Loops | Layer applications |
|---|---:|---:|---:|
| dlite_5l1r | 5 | 1 | 5 |
| dlite_1l5r | 1 | 5 | 5 |
| dlite_1l3r | 1 | 3 | 3 |

All three are independently trained for one epoch with the existing aligned DLite
SFT recipe: seed 42, 12 target CHS layers, RNN rank 320, final CE 0.1 + L1 1.0,
and block size 8 (one known anchor plus seven draft proposals). Only physical
depth and recurrent depth differ. Hidden states pass between loops without an
extra boundary norm; final norm and the RNN head run after the last loop.
Conditioning KV is projected once per physical layer per forward, retaining its
autograd graph. Block QKV is recomputed each loop. No cross-block draft state or
history KV is added. Existing checkpoints without `num_loops` execute one pass.

## Training

Run from the repository with the existing `.venv-vlm` runtime (torch 2.9.1,
Transformers 5.20.1; see `environments/vlm`). No install or environment change is
performed by the launcher. The default manifest is the already regenerated
`../data/finevision_qwen35_sglang_20000/train.jsonl`; override it with `--train-data`.
No additional regeneration is required for that manifest.

```bash
git fetch origin
git switch research/qwen35-vlm-looped
git pull --ff-only origin research/qwen35-vlm-looped

export LOOP_RUN="$PWD/../runs/dlite_loop_20k_1ep_$(date +%Y%m%d_%H%M%S)"
.venv-vlm/bin/python tests/run_dlite_loop.py train \
  --run-dir "$LOOP_RUN" \
  --gpus 0,1,2,3,4,5,6,7
```

The launcher runs the three jobs sequentially on the specified GPUs. Defaults:
micro batch 1/rank, accumulation 4, eight-GPU global batch 32, maximum 20,000
source records, TensorBoard enabled. Input eligibility is validated once and
reused for both loop models; actual eligible counts and schedule are checked
against the baseline. Short/overlength exclusions can reduce the effective
training count; see `effective_config.json` and `skipped_training.jsonl`.
Inference-only exports are the default; no `training_state_rank_*.pt` is written.
The launcher refuses an existing run directory rather than overwriting it.

For a GPU smoke run, use a **different** directory and add `--max-samples 32
--debug-steps 2 --report-to none`. For a command/config preview, add `--dry-run`;
a dry-run directory cannot be evaluated as a trained experiment. Do not reuse
the smoke/dry-run directory for the full experiment.

## Evaluation

```bash
.venv-vlm/bin/python tests/run_dlite_loop.py eval \
  --run-dir "$LOOP_RUN" --gpu 7
```

This evaluates AR and all three exported checkpoints on GSM8K, CharXiv,
Flickr30k, TextVQA, HRBench-4K and HRBench-8K, with B=1 and B=32, 16 unique
prompts per dataset, seed 42, greedy decoding and at most 256 new tokens.
By default, the 1L5R checkpoint is independently evaluated at fixed depths 1–5
and 1L3R at depths 1–3. Labels are `dlite_1l5r_loop1` ... `dlite_1l5r_loop5`
and `dlite_1l3r_loop1` ... `dlite_1l3r_loop3`; the baseline remains `dlite_5l1r`.
This gives nine draft settings plus AR, each tested at B=1 and B=32. These are
separate full decoding runs, not intermediate token predictions emitted inside
one decoding run. The same checkpoint weights are reused, and no retraining or
checkpoint rewriting occurs. Add `--no-loop-sweep` to evaluate only final depths.
CSV/Markdown results include physical layers, trained maximum loops and actual
inference loops (the Markdown table shows layers and inference loops).
Only evaluation prompts/images are passed to the models, never reference answers.
Dataset and image paths use the existing server defaults; override with
`--text-data`, `--data-dir`, `--image-root`, and select groups with `--datasets`.

The default state mode is `per_token` for every speculative model; alternatives
are `--state-mode full_target_replay` and `compact_replay`. B means static
replicated-input batch size: every selected prompt is copied B times and
forwarded as one batch. It is not continuous batching of distinct requests.
B=32 high-resolution inputs require enough device memory; choose an explicit
`--image-max-pixels` limit if needed and use the same limit across methods.
Effective processor sizes and input token counts are saved by the benchmark.
If the default evaluation directory already exists, use a fresh directory:

```bash
.venv-vlm/bin/python tests/run_dlite_loop.py eval \
  --run-dir "$LOOP_RUN" --gpu 7 \
  --result-dir "$LOOP_RUN/evaluation_loops_$(date +%Y%m%d_%H%M%S)"
```

Normal throughput is measured without per-stage instrumentation. A separate
short profile pass (on by default; disable with `--no-profile`) provides draft,
verification and recovery stage times. Outputs under `$LOOP_RUN/evaluation/`:

- `comparison.md`, `comparison.csv`: aggregate throughput, accepted draft tokens,
  anchor-inclusive committed length, memory, raw AR ratios and token matches.
- `position_acceptance.csv`: @0–@6 prefix survival, excluding the known anchor.
  Trial denominators censor positions beyond the output cap or emitted EOS;
  these are prefix acceptance rates, not independent suffix token accuracies.
- `<dataset>/inference_profile.md`: separate instrumented draft/target timings.
- `<dataset>/*_b1.jsonl`, `*_b32.jsonl`: generated outputs and per-prompt records.

Rebuild reports after an interrupted evaluation with:

```bash
.venv-vlm/bin/python tests/run_dlite_loop.py report \
  --result-dir "$LOOP_RUN/evaluation"
```

The benchmark preserves observed throughput ratios when outputs differ. Use
the exported token-match field to distinguish raw observations from matched-output
lossless speedup claims. CPU tiny-model tests are functional checks, not GPU
performance measurements.

## Verification

```bash
.venv-vlm/bin/python -m pytest tests/test_dlite_loop.py \
  tests/test_vlm.py tests/test_vlm_aligned.py tests/test_vlm_batch_training.py \
  tests/test_vlm_static_batch.py -q
```

Tests cover unrolled output/gradient equivalence, shared projection call counts,
old checkpoint behavior, loop checkpoint reload, real tiny Qwen3.5 image training
and text/image generation at B=1/32. Full Qwen3.5-4B CUDA/FSDP training and H100
throughput still require the server and real data.

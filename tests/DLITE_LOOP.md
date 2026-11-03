# Qwen3.5 DLite loop pilot

Branch: `research/qwen35-vlm-looped`.

## Six-model, 20k / five-epoch comparison

The launcher also supports three DFlash controls. Select both families and
override the epoch count explicitly; the original three-model / one-epoch
defaults remain compatible with existing commands and trained runs.

```bash
export LOOP_RUN="$PWD/../runs/dlite_dflash_loop_20k_5ep_$(date +%Y%m%d_%H%M%S)"
.venv-vlm/bin/python tests/run_dlite_loop.py train \
  --run-dir "$LOOP_RUN" --families dlite dflash --epochs 5 \
  --max-samples 20000 --gpus 0,1,2,3,4,5,6,7

# Run after all six training jobs finish.
.venv-vlm/bin/python tests/run_dlite_loop.py eval \
  --run-dir "$LOOP_RUN" \
  --result-dir "$LOOP_RUN/evaluation_img1024_$(date +%Y%m%d_%H%M%S)" \
  --gpus 0,1,2,3,4,5 --batch-sizes 1 32 \
  --samples 16 --max-new-tokens 256 --state-mode per_token \
  --image-max-pixels 1048576
```

Training runs `dlite_5l1r`, `dlite_1l5r`, `dlite_1l3r`, `dflash_5l1r`,
`dflash_1l5r`, `dflash_1l3r` sequentially, each using all selected GPUs and a
fresh initialization. It does not continue the old one-epoch checkpoints.
All six share source records, eligible-set validation, block size, seed, global
batch and five full epochs. The original per-family objectives and optimizers
are retained: DLite SFT uses Muon/AdamW with final CE + L1 and 12 CHS layers;
DFlash uses its upstream AdamW/decayed CE recipe with full target history.
All three DFlash controls use the same five target feature layers, selected by
the existing five-layer DFlash rule, even when the physical draft has one layer.
For a 32-layer target these are `[1, 8, 15, 22, 29]` (zero-based).

DFlash repeats the shared physical backbone on raw hidden states and applies
the final norm / parallel target LM head only after the last loop. Each virtual
layer has its own history KV cache, equivalent to an explicitly unrolled model
with tied weights; this prevents appending a context once per loop into the
same cache. Unlike DLite, it retains full target-history KV. Weight sharing does
not imply a proportional reduction in DFlash history-cache memory. Changing
the inference depth requires a fresh cache; the benchmark creates one per run.

Evaluation reads the model list from the training manifest, includes every fixed
exit for both families (18 draft settings + AR), and keeps the existing dataset
per GPU queue and position-acceptance / profiling outputs. Old manifests without
a model list still evaluate the original three DLite controls. The explicit
pixel budget limits images to about 1024 image tokens with the Qwen3.5 processor;
this is a resized-input benchmark, including for HRBench. It avoids the observed
native-resolution B=32 vision-prefill OOM and must be reported with results.
The original token-match limitation is unchanged by this training extension;
check Match before describing throughput ratios as lossless speedups.

## Original DLite pilot

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

### Parallel evaluation across GPUs

```bash
.venv-vlm/bin/python tests/run_dlite_loop.py eval \
  --run-dir "$LOOP_RUN" \
  --result-dir "$LOOP_RUN/evaluation_multigpu_$(date +%Y%m%d_%H%M%S)" \
  --gpus 0,1,2,3,4,5 --batch-sizes 1 32 --state-mode per_token
```

Use `--gpus` instead of `--gpu`; they are mutually exclusive. Each GPU runs at
most one independent dataset process at a time. With fewer GPUs than datasets,
the next dataset starts on whichever GPU becomes free. Each task runs its own
AR plus every requested loop exit and batch size on the **same** device, so AR
ratios never combine timings from different GPUs. Every B=32 batch still runs
on one GPU. This shortens total evaluation time; it is not tensor parallelism or
a measurement of multi-GPU serving throughput.

The parent prints dataset start/completion messages; detailed output is in
`<result-dir>/<dataset>.log`. `queue_status.json` records dataset/device/PID,
status, command and exit code. Only the parent writes the aggregate reports,
after workers stop, so it never reads another worker's actively written JSON.
On a task failure or Ctrl-C, active workers are stopped, pending jobs remain
unstarted, and completed dataset results are preserved and aggregated. The
manual `report` command also respects queue status and skips unfinished datasets.
No training, model forward, checkpoint weights or generation settings change.
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

The scheduling tests need no GPU, Torch or Transformers:

```bash
.venv-vlm/bin/python -m unittest discover -s tests -p test_dlite_loop_queue.py -v
```

Tests cover unrolled output/gradient equivalence, shared projection call counts,
old checkpoint behavior, loop checkpoint reload, real tiny Qwen3.5 image training
and text/image generation at B=1/32. Full Qwen3.5-4B CUDA/FSDP training and H100
throughput still require the server and real data.

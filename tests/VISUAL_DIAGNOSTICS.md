# Visual evidence and first-candidate diagnostics

Branch: `research/qwen35-vlm-visual-diagnostics`, based on visual-kv commit
`55895f7962cc802df8440c7c8400f096563d751e`.

This phase implements experiments II and III only. It launches no training,
constructs no optimizer, changes no model weights, and writes no training states.
The matched concat-pivot/gated reader versus concat-evidence/block-query training
(experiment I) is deliberately deferred until the ongoing training is complete.
Old gate-reader checkpoints can still use `diagnose_memory.sh`; this new combined
entry point requires `experiment_name` checkpoints from the visual-kv branch.

## Isolated checkout while training continues

Do not switch the checkout used by the active training process. Create another
worktree; keep using the existing runtime rather than installing packages:

```bash
cd "$DLITE_WORKSPACE/LiteSpec"
git fetch origin research/qwen35-vlm-visual-diagnostics
git worktree add --detach "$DLITE_WORKSPACE/LiteSpec-visual-diagnostics" \
  origin/research/qwen35-vlm-visual-diagnostics
cd "$DLITE_WORKSPACE/LiteSpec-visual-diagnostics"
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RUN_GROUP="/absolute/path/to/existing/visual_kv_training_group"
export RESULT_DIR="$RUN_GROUP/diagnostics_$(date +%Y%m%d_%H%M%S)"
EVAL_GPU=4 EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=256 \
EXPERIMENTS="pivot_rnn pivot_visual_rnn pivot_history_rnn" \
TESTS="text-only charxiv_val textvqa" \
bash tests/run_visual_diagnostics.sh
```

Use an idle GPU. Checkpoints must already have an exported `last`. It is resolved
to a fixed step directory before any jobs start; settings include config hashes.
The script does not poll, wait for completion, stop or modify running training.
To test only an already-exported model, set e.g. `EXPERIMENTS=pivot_visual_rnn`.
Missing/unexported checkpoints produce an error before evaluation output creation.
A fixed saved step can be evaluated even if training continues, but it is not
labelled a completed 3-epoch model. This metadata must be checked by the user.

`IMAGE_MAX_TOKENS=1024` must match the checkpoint. Same Target, SDPA backend,
image budget, frozen sampled manifests and AR trace across all cells. Dataset
selection and paths match `run_vlm_matrix.py` (`TEXT_DATA`, `DATA_DIR`, `IMAGE_ROOT`).
Optional tests: `flickr30k hr_bench_4k hr_bench_8k`. Default sample seed 42,
probe stride 8, offline B=1. Extend `EVAL_SAMPLES=64` after the short check.
`PROBE_ONLY=1` generates AR traces and probes, skipping speculative rate cells.

## II: Correct / wrong / disabled evidence

Visual modes get normal, off, first-round-only and wrong-image-always cells.
Only the Draft bank is replaced. Target image/question/pivot remain correct.
Wrong-image donors are selected by the same seeded rule in probes and rate cells;
donor prompt length cannot incorrectly mask its already-available visual bank.
Wrong-image end-to-end timing includes extra donor prefill and is diagnostic.
`off` removes evidence positions from attention (rather than zeroing only V).

Fixed-prefix probes walk anchors on one frozen Target-only reference sequence.
All variants use the same prefix and pivot, and no counterfactual proposals enter
Target state. Existing full-sequence causal hidden extraction is used in this
untimed probe; it is not claimed numerically identical to stepwise Target decode.
The first anchor is always probed, then every `PROBE_STRIDE` positions.
AR continuations too short to contain a complete block have no probes; explicit
anchor counts/denominators are reported. Later candidate agreement after the first
mismatch is only agreement with the reference at that position, not an actual
counterfactual Target verification result. Continuous-prefix length is meaningful.

Metrics: first candidate hit; per-position hits/counts/rates; repairs and breaks
relative to disabled; first-rejection repair opportunities/rate; prefix improvement,
worsening and delta. Summaries are anchor-weighted and split into first anchor,
later anchors and decode-position buckets. No significance claim is made from 16
samples. The first candidate excludes the already-known pending/anchor token.

## III: Cold start versus every-round bottleneck

`context_schedule`: `always`, `off`, `first_round`. The latter releases raw evidence,
encoded bank, recurrent controller state and per-layer projected K/V after the
first proposal. Subsequent commit cannot recreate these tensors. The next block
uses the same trained model with evidence disabled. Target cache is never dropped.
The CUDA caching allocator may retain freed blocks; nvidia-smi need not decrease.
Per-request outputs include final `retained_evidence_tokens` (zero after release).

`pivot_history_*` additionally supports `history_scope=prefill`. This keeps only
the prompt bank and never appends answer hidden; `full` keeps the original
committed-history control. Four cells: off/full, first_round/prefill,
always/prefill, always/full. These are interventions on a trained history reader,
not switches that magically give a pivot-only checkpoint a history interface.
This control remains DLite + context, not native DFlash.

Closed-loop first-candidate metrics are computed against the actual block Target
verification. They are split into first and subsequent rounds. Final output-cap
rounds with no candidate are excluded from the first-hit denominator. EOS/cap can
truncate committed lengths. Later-round prefixes can differ between cells, which
is why paired fixed-prefix probes are separately reported. Diagnostic boolean
arrays are transferred to CPU after the timed interval; small recording kernels
remain included. No CUDA-profiler pass is mixed into rates.

## Outputs

- `diagnostics.md`: closed-loop performance, fixed-prefix repairs/breaks, first vs later.
- `comparison.csv`: raw decode-only speed, throughput, prefill, peak allocated memory,
  and AR exact-match flags. It is offline static batching, not a serving benchmark.
- `fixed_prefix.csv`: condition/scope metrics plus per-position denominators.
- `first_candidate.csv`: actual closed-loop first-candidate rates.
- `*_probe_*/anchors.jsonl`: reference and proposals, hit masks, repair/break evidence.
- `*_b1.jsonl`: generated token IDs and per-round verification diagnostics.
- `settings.json`: immutable checkpoint targets and sample hashes; subprocess logs
  are written beside cell directories. Failures are surfaced rather than ignored.

## Training-state saving

On this branch `train_visual_kv.sh` defaults to `SAVE_TRAINING_STATE=0`, matching
`run_training_vlm.sh`. Model checkpoints and TensorBoard still save normally.
Existing state files are not deleted, and currently running processes are not
changed by pulling this branch. Exact optimizer/RNG resume requires an explicit
future run with `SAVE_TRAINING_STATE=1`.

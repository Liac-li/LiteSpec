# Paired legacy DLite vs DFlash first-candidate test

This entry point accepts the old `DLiteDraftModel` / `DFlashDraftModel` checkpoints
without `experiment_name`, including the paired run from 2026-10-22. It does not
require visual-kv model training or launch any training.

## Server command (existing diagnostics worktree)

```bash
cd "$DLITE_WORKSPACE/LiteSpec-visual-diagnostics"
git fetch origin research/qwen35-vlm-visual-diagnostics
git merge --ff-only FETCH_HEAD
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export PAIR_RUN="$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049"
export RESULT_DIR="$PAIR_RUN/first_candidate_pair_$(date +%Y%m%d_%H%M%S)"
EVAL_GPU=7 EVAL_SAMPLES=64 EVAL_MAX_NEW_TOKENS=256 \
TESTS="text-only charxiv_val textvqa" \
bash tests/compare_dlite_dflash_first.sh
cat "$RESULT_DIR/first_candidate_report.md"
```

Use an idle GPU. Change `EVAL_SAMPLES=16` for a quick check. Optional datasets:
`flickr30k hr_bench_4k hr_bench_8k`. Data paths use existing `TEXT_DATA`, `DATA_DIR`,
`IMAGE_ROOT` selectors. Defaults: image budget 1024, sample seed 42, SDPA,
per-token GDN state commits, B=1. No dependencies are installed.

## What is compared

Each dataset is sampled once and written to a frozen snapshot. One subprocess
loads the Target, runs AR, then loads DLite and DFlash sequentially. Checkpoint
`last` links are pinned to fixed step paths and config hashes are recorded before
starting. Checks enforce target path, block size, hidden width, draft depth,
Target depth and vocabulary equality. Different Target layer selections and
prediction heads are intentionally allowed and recorded in settings; these are
architecture comparisons, not an isolated causal KV ablation. Epoch/data/optimizer
matching cannot be proven from config.json; original run logs are authoritative.

The known pending/anchor token does not count as a draft candidate. We report the
first unknown candidate hit in every round, separately for the request's very first
round and all subsequent rounds. If the anchor is EOS there is no meaningful
candidate to score; that round is excluded. The final output-cap round without a
candidate is also excluded. Output-cap and EOS censoring are explicit denominators.

The first round shares the original input/prefill. Its per-request anchor token,
first draft proposal, and Target argmax at that anchor are exported. The paired
summary counts both-hit, only-DLite-hit, only-DFlash-hit and both-miss. It also
checks whether anchor and Target argmax agree between the two verification calls;
a same-Target-root subset is separately reported if numerical differences occur.
Later rounds can visit different prefixes and have different boundaries; their
aggregate rates are descriptive and not paired fixed-prefix evidence.

Acceptance includes the guaranteed anchor. First-round verifier-kept length and
actual EOS-truncated emitted length are separately available. Raw speedups use
normal decode-only timing; prefill, end-to-end rates, exact AR matches, memory and
full request outputs remain in the per-dataset benchmark cells. Small diagnostic
recording operations remain included, with GPU-to-CPU conversion after timing.
This is offline static B=1, not a production serving benchmark.

Outputs:
- `first_candidate_report.md`: model comparison and paired first-round counts.
- `comparison.csv`: full performance rows including AR.
- `first_candidate.csv`: all/first/later hit rates with opportunity counts.
- `paired_summary.json`: paired counts and same-root subset.
- `<dataset>/paired_requests.jsonl`: token-level first-round evidence.
- `<dataset>/{ar,dlite,dflash}_b1.jsonl`: generated sequences and round diagnostics.
- `settings.json`: complete model configs, checkpoint paths/hashes, dataset hashes.

No model parameters, training state files or running jobs are modified. These old
models are tested through their original inference interfaces; no visual evidence
disabling or first-round-only KV switching is silently applied.

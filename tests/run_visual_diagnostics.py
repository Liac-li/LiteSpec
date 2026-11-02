"""No training. Fixed-prefix probes + isolated closed-loop context-schedule cells."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()} for row in rows)


def resolve_models(group, names, budget):
    result = []
    for name in names:
        path = (group / name / 'last').resolve(strict=True)
        cfg = json.loads((path / 'config.json').read_text())
        mode = cfg.get('dlite_config', {}).get('context_mode')
        if mode not in {'pivot', 'pivot_visual', 'pivot_history', 'visual_static_slots', 'visual_pivot_conditioned', 'visual_recurrent'}:
            raise ValueError(f'{name}: requires a visual-kv branch checkpoint, got {mode!r}')
        if cfg.get('dlite_config', {}).get('experiment_name') != name:
            raise ValueError(f'{name}: checkpoint experiment name differs')
        if cfg.get('vlm_config', {}).get('image_max_tokens') != budget:
            raise ValueError(f'{name}: image budget differs from checkpoint')
        if not any((path / x).exists() for x in ('model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin', 'pytorch_model.bin.index.json')):
            raise ValueError(f'{name}: checkpoint weights not exported yet: {path}')
        result.append((name, path, mode, hashlib.sha256((path/'config.json').read_bytes()).hexdigest()))
    return result


def cells(mode, has_images):
    if mode == 'pivot':
        return [('correct', 'always', 'full')]
    if mode == 'pivot_history':
        return [('correct', 'off', 'full'), ('correct', 'first_round', 'prefill'),
                ('correct', 'always', 'prefill'), ('correct', 'always', 'full')]
    values = [('correct', 'always', 'full'), ('correct', 'off', 'full'), ('correct', 'first_round', 'full')]
    if has_images:
        values.append(('wrong_image', 'always', 'full'))
    return values


def report(root, performance, fixed, first):
    write_csv(root/'comparison.csv', performance)
    write_csv(root/'fixed_prefix.csv', fixed)
    write_csv(root/'first_candidate.csv', first)
    def number(v, fmt='.3f'):
        return '—' if v is None else format(v, fmt)
    lines = ['# Visual context diagnostics', '',
        'Decode-only raw ratios, offline B=1. AR match is recorded, not a rate gate.',
        'Fixed-prefix probes are untimed. Closed-loop later hits use candidate-conditioned Target predictions.',
        'First-round context is an inference intervention, not a separately trained strategy.', '',
        '## Closed-loop performance', '',
        '| Dataset | Model | Condition | Schedule | History scope | Accept | Speedup | tok/s | Peak GiB |',
        '|---|---|---|---|---|---:|---:|---:|---:|']
    for r in performance:
        lines.append(f"| {r['dataset_name']} | {r['model']} | {r.get('memory_condition')} | {r.get('context_schedule')} | {r.get('history_scope')} | {number(r['mean_committed_length'])} | {number(r['raw_throughput_speedup_vs_ar'])} | {number(r['output_tokens_per_second'])} | {number(r['peak_allocated_gib'])} |")
    lines += ['', '## Fixed-prefix visual repair / break', '',
        'Anchor-weighted; accepted length includes the guaranteed token. Repair/break values are per anchor.', '',
        '| Dataset | Model | Bank | Condition | Scope | N | First hit | Accept | First repair | First break | Prefix delta |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in fixed:
        if r['scope'] not in {'all', 'first_round', 'later_rounds'}:
            continue
        lines.append(f"| {r['dataset']} | {r['model']} | {r['history_scope']} | {r['condition']} | {r['scope']} | {r['anchors']} | {number(r['first_candidate_hit_rate'])} | {number(r['reference_committed_length'])} | {number(r['first_candidate_repaired_per_anchor'])} | {number(r['first_candidate_broken_per_anchor'])} | {number(r['prefix_delta_per_anchor'])} |")
    lines += ['', '## Closed-loop first candidate', '',
        '| Dataset | Model/cell | Scope | Opportunities | First hit |',
        '|---|---|---|---:|---:|']
    for r in first:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['scope']} | {r['first_candidate_opportunities']} | {number(r['first_candidate_hit_rate'])} |")
    (root/'diagnostics.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-group', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--experiments', nargs='+', default=['pivot_rnn', 'pivot_visual_rnn', 'pivot_history_rnn'])
    p.add_argument('--tests', nargs='+', default=['text-only', 'charxiv_val', 'textvqa'])
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-max-tokens', type=int, default=1024)
    p.add_argument('--target-model', required=True)
    p.add_argument('--state-mode', choices=['per_token', 'compact_replay', 'full_target_replay'], default='per_token')
    p.add_argument('--probe-only', action='store_true', help='still creates shared AR references; skips speculative timing')
    args = p.parse_args()
    if min(args.samples, args.max_new_tokens, args.stride, args.image_max_tokens) < 1 or len(set(args.experiments)) != len(args.experiments):
        p.error('positive limits and unique experiments required')
    group = args.run_group.expanduser().resolve()
    models = resolve_models(group, args.experiments, args.image_max_tokens)
    from run_vlm_matrix import select_data
    from evaluate_memory_ablation import run_logged
    from visual_diagnostic_metrics import summarize_closed_loop
    packets = select_data(SimpleNamespace(tests=args.tests, samples=args.samples, seed=args.seed))
    target = Path(args.target_model).expanduser().resolve()
    v = json.loads((target/'config.json').read_text())['vision_config']
    pixels = args.image_max_tokens * (v['patch_size'] * v['spatial_merge_size'])**2
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments=vars(args), checkpoint_paths=[dict(name=n, path=str(path), mode=m, config_sha256=h) for n,path,m,h in models],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), datasets={})
    performance, fixed, first = [], [], []
    env = dict(os.environ)
    for case, packet in packets.items():
        snapshot = root/f'{case}.jsonl'
        snapshot.write_text(packet['serialized'])
        records = [json.loads(l) for l in packet['serialized'].splitlines() if l.strip()]
        has_images = any(r.get('image') for r in records)
        manifest['datasets'][case] = dict(sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), sample_ids=[r['id'] for r in records], source=packet.get('source'))
        (root/'settings.json').write_text(json.dumps(manifest, default=str, indent=2))
        common = [sys.executable, str(HERE/'benchmark_vlm_static_batch.py'), '--target-model', str(target),
            '--data', str(snapshot), '--dataset-name', case, '--max-samples', str(packet['samples']),
            '--max-new-tokens', str(args.max_new_tokens), '--batch-sizes', '1', '--seed', str(args.seed),
            '--image-max-pixels', str(pixels), '--cpu-inputs', '--attention-backend', 'sdpa', '--state-mode', args.state_mode]
        ar = root/f'{case}_ar'
        run_logged([*common, '--output-dir', str(ar)], env, ar)
        performance.extend(json.loads((ar/'summary.json').read_text()))
        for name, path, mode, _ in models:
            if not args.probe_only:
                for condition, schedule, scope in cells(mode, has_images):
                    label = f'{name}_{condition}_{schedule}_{scope}'
                    dest = root/f'{case}_{label}'
                    run_logged([*common, '--checkpoint', label, str(path), '--ar-reference', str(ar),
                        '--output-dir', str(dest), '--memory-condition', condition,
                        '--context-schedule', schedule, '--history-scope', scope], env, dest)
                    performance.extend(r for r in json.loads((dest/'summary.json').read_text()) if r['model'] != 'ar')
                    entries = [json.loads(l)['result'] for l in (dest/f'{label}_b1.jsonl').read_text().splitlines() if l.strip()]
                    first.extend(dict(dataset=case, model=label, **r) for r in summarize_closed_loop(entries))
                    report(root, performance, fixed, first)
            for scope in (['full', 'prefill'] if mode == 'pivot_history' else ['full']):
                dest = root/f'{case}_{name}_probe_{scope}'
                run_logged([sys.executable, str(HERE/'probe_visual_kv.py'), '--target-model', str(target),
                    '--checkpoint', str(path), '--data', str(snapshot), '--output-dir', str(dest),
                    '--samples', str(packet['samples']), '--max-new-tokens', str(args.max_new_tokens),
                    '--image-max-tokens', str(args.image_max_tokens), '--stride', str(args.stride),
                    '--seed', str(args.seed), '--history-scope', scope, '--reference-file', str(ar/'ar_b1.jsonl')], env, dest)
                fixed.extend(dict(dataset=case, model=name, history_scope=scope, **r) for r in json.loads((dest/'summary.json').read_text()))
                report(root, performance, fixed, first)
    print(f'Results: {root}/diagnostics.md', flush=True)


if __name__ == '__main__':
    main()

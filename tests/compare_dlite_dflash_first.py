"""Matched legacy DLite/DFlash checkpoints: first-round and per-round first candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from run_visual_diagnostics import write_csv
from visual_diagnostic_metrics import summarize_closed_loop

HERE = Path(__file__).resolve().parent


def validate_pair(group, target):
    configs, models = {}, []
    for name, architecture in [('dlite', 'DLiteDraftModel'), ('dflash', 'DFlashDraftModel')]:
        path = (group/name/'last').resolve(strict=True)
        cfg = json.loads((path/'config.json').read_text())
        if cfg.get('architectures') != [architecture]:
            raise ValueError(f'{name}: expected {architecture}')
        if not any((path/f).is_file() for f in ('model.safetensors','model.safetensors.index.json','pytorch_model.bin','pytorch_model.bin.index.json')):
            raise ValueError(f'{name}: weights not exported: {path}')
        trained_target = cfg.get('vlm_config',{}).get('target_model')
        if not trained_target or Path(trained_target).expanduser().resolve() != target:
            raise ValueError(f'{name}: Target model path differs from training: {trained_target}')
        configs[name] = cfg
        models.append((name,path))
    for key in ('block_size','hidden_size','num_hidden_layers','num_target_layers','vocab_size'):
        if configs['dlite'].get(key) != configs['dflash'].get(key):
            raise ValueError(f'paired models differ in {key}')
    if configs['dlite']['dlite_config'].get('model_role') != 'student':
        raise ValueError('DLite must be a history-free student')
    return models, configs


def paired_first(dlite, dflash):
    if [r['sample_id'] for r in dlite] != [r['sample_id'] for r in dflash]:
        raise ValueError('paired sample IDs/order differ')
    rows=[]
    for a,b in zip(dlite,dflash):
        x=a['result'].get('round_diagnostics',[])
        y=b['result'].get('round_diagnostics',[])
        if not x or not y:
            raise ValueError('missing round diagnostics; update the diagnostics checkout')
        x,y=x[0],y[0]
        eligible=x['first_candidate_hit'] is not None and y['first_candidate_hit'] is not None
        root_equal=(x.get('anchor_token_id') == y.get('anchor_token_id') and
                    x.get('target_first_candidate_token_id') == y.get('target_first_candidate_token_id'))
        rows.append(dict(sample_id=a['sample_id'], eligible=eligible,
            same_anchor_and_target_prediction=root_equal,
            dlite_first_hit=x['first_candidate_hit'], dflash_first_hit=y['first_candidate_hit'],
            dlite_accept=x['committed_length'], dflash_accept=y['committed_length'],
            dlite_emitted=x.get('emitted_length'), dflash_emitted=y.get('emitted_length'),
            dlite_anchor=x.get('anchor_token_id'), dflash_anchor=y.get('anchor_token_id'),
            dlite_proposal=x.get('first_candidate_token_id'), dflash_proposal=y.get('first_candidate_token_id'),
            dlite_target_prediction=x.get('target_first_candidate_token_id'), dflash_target_prediction=y.get('target_first_candidate_token_id')))
    eligible=[r for r in rows if r['eligible']]
    def counts(selected):
        return dict(samples=len(selected),
            both_hit=sum(r['dlite_first_hit'] and r['dflash_first_hit'] for r in selected),
            only_dlite_hit=sum(r['dlite_first_hit'] and not r['dflash_first_hit'] for r in selected),
            only_dflash_hit=sum(r['dflash_first_hit'] and not r['dlite_first_hit'] for r in selected),
            both_miss=sum(not r['dlite_first_hit'] and not r['dflash_first_hit'] for r in selected))
    return rows, dict(all_eligible=counts(eligible),
        same_target_root=counts([r for r in eligible if r['same_anchor_and_target_prediction']]),
        target_root_mismatches=sum(not r['same_anchor_and_target_prediction'] for r in eligible),
        excluded_samples=len(rows)-len(eligible))


def make_report(root, performance, metrics, pairs):
    write_csv(root/'comparison.csv',performance)
    write_csv(root/'first_candidate.csv',metrics)
    (root/'paired_summary.json').write_text(json.dumps(pairs,indent=2))
    def fmt(v, percentage=False):
        return '—' if v is None else f'{100*v:.1f}%' if percentage else f'{v:.3f}'
    lines=['# DLite vs DFlash: first candidate', '',
        'Legacy paired checkpoints; greedy B=1; raw decode-only speedup. No training.',
        'First candidate excludes the already-known anchor. First-round rows share prompt/Target prefill;',
        'later rounds have model-dependent boundaries/prefixes and are descriptive, not paired interventions.',
        'EOS-before-candidate and output-cap rounds with no candidate are excluded from hit denominators.',
        'Different heads and Target feature layers prevent attributing differences solely to history KV.', '',
        '| Dataset | Model | First-round N | First-round hit | Later-round N | Later hit | First-round accept | Overall accept | tok/s | Speedup | AR match |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in performance:
        if r['model']=='ar':
            continue
        first=next(v for v in metrics if v['dataset']==r['dataset_name'] and v['model']==r['model'] and v['scope']=='first_round')
        later=next(v for v in metrics if v['dataset']==r['dataset_name'] and v['model']==r['model'] and v['scope']=='later_rounds')
        lines.append(f"| {r['dataset_name']} | {r['model']} | {first['first_candidate_opportunities']} | {fmt(first['first_candidate_hit_rate'],True)} | {later['first_candidate_opportunities']} | {fmt(later['first_candidate_hit_rate'],True)} | {fmt(first['mean_committed_length'])} | {fmt(r['mean_committed_length'])} | {fmt(r['output_tokens_per_second'])} | {fmt(r['raw_throughput_speedup_vs_ar'])}× | {r['matched']}/{r['requests']} |")
    lines += ['', '## Paired first-round outcomes', '',
        '| Dataset | Eligible | Both hit | Only DLite hit | Only DFlash hit | Both miss | Target-root mismatch | Excluded |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset,r in pairs.items():
        c=r['all_eligible']
        lines.append(f"| {dataset} | {c['samples']} | {c['both_hit']} | {c['only_dlite_hit']} | {c['only_dflash_hit']} | {c['both_miss']} | {r['target_root_mismatches']} | {r['excluded_samples']} |")
    lines += ['', 'Root mismatches flag numerical/implementation differences in Target verification; see paired_summary.json for the same-root subset.',
              'Acceptance includes the guaranteed anchor; first-round accept is verifier-kept length before EOS truncation, emitted lengths are in paired_requests.jsonl.',
              'No visual-memory ablations or first-round-only KV switches are applied to these old checkpoints.']
    (root/'first_candidate_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--target-model',type=Path,required=True)
    p.add_argument('--tests',nargs='+',default=['text-only','charxiv_val','textvqa'])
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--image-max-tokens',type=int,default=1024)
    p.add_argument('--state-mode',choices=['per_token','compact_replay','full_target_replay'],default='per_token')
    args=p.parse_args()
    if min(args.samples,args.max_new_tokens,args.image_max_tokens)<1:
        p.error('positive sample/output/image limits required')
    target=args.target_model.expanduser().resolve()
    group=args.pair_run.expanduser().resolve()
    models,configs=validate_pair(group,target)
    from run_vlm_matrix import select_data
    from evaluate_memory_ablation import run_logged
    packets=select_data(SimpleNamespace(tests=args.tests,samples=args.samples,seed=args.seed))
    cfg=json.loads((target/'config.json').read_text())['vision_config']
    unit=cfg['patch_size']*cfg['spatial_merge_size']
    root=args.output_dir.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    settings=dict(arguments=vars(args),checkpoints=[dict(model=name,path=str(path),
        config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest()) for name,path in models],
        configs=configs,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),datasets={},
        training_metadata_note='Checkpoint config does not prove epochs, training data, or optimizer matching; inspect original run logs.')
    performance,metrics,pairs=[],[],{}
    for case,packet in packets.items():
        snapshot=root/f'{case}.jsonl'
        snapshot.write_text(packet['serialized'])
        settings['datasets'][case]=dict(source=packet.get('source'),sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        (root/'settings.json').write_text(json.dumps(settings,default=str,indent=2))
        dest=root/case
        command=[sys.executable,str(HERE/'benchmark_vlm_static_batch.py'),'--target-model',str(target),
            '--data',str(snapshot),'--dataset-name',case,'--output-dir',str(dest),
            '--max-samples',str(packet['samples']),'--max-new-tokens',str(args.max_new_tokens),
            '--batch-sizes','1','--seed',str(args.seed),'--image-max-pixels',str(args.image_max_tokens*unit*unit),
            '--attention-backend','sdpa','--cpu-inputs','--state-mode',args.state_mode]
        for name,path in models:
            command += ['--checkpoint',name,str(path)]
        run_logged(command,dict(os.environ),dest)
        performance.extend(json.loads((dest/'summary.json').read_text()))
        results={}
        for name,_ in models:
            results[name]=[json.loads(l) for l in (dest/f'{name}_b1.jsonl').read_text().splitlines() if l.strip()]
            metrics.extend(dict(dataset=case,model=name,**row) for row in summarize_closed_loop([r['result'] for r in results[name]]))
        pair_rows,pairs[case]=paired_first(results['dlite'],results['dflash'])
        (dest/'paired_requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in pair_rows))
        make_report(root,performance,metrics,pairs)
    print(f'Report: {root}/first_candidate_report.md',flush=True)


if __name__=='__main__':
    main()

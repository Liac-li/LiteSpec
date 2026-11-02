import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compare_dlite_dflash_first import paired_first, validate_pair
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.runtime import PreparedExample, visual_positions
from litespec.vlm.baselines import make_draft
from test_vlm import tiny_qwen35, tiny_image_inputs
from test_vlm_aligned import fixture_model, Processor


def record(sample, hit, anchor=1, target=2):
    return dict(sample_id=sample,result=dict(round_diagnostics=[dict(first_candidate_hit=hit,
        committed_length=2,anchor_token_id=anchor,first_candidate_token_id=2,
        target_first_candidate_token_id=target,emitted_length=2)]))


def test_pair_counts_exclusions_and_root_mismatch():
    left=[record('a',True),record('b',False),record('c',False),record('d',None)]
    right=[record('a',False),record('b',True),record('c',False,target=3),record('d',True)]
    rows,summary=paired_first(left,right)
    assert summary['all_eligible']==dict(samples=3,both_hit=0,only_dlite_hit=1,only_dflash_hit=1,both_miss=1)
    assert summary['target_root_mismatches']==1
    assert summary['same_target_root']['samples']==2
    assert summary['excluded_samples']==1
    assert not rows[2]['same_anchor_and_target_prediction']
    with pytest.raises(ValueError,match='IDs'):
        paired_first(left,right[::-1])


@torch.no_grad()
def test_real_legacy_pair_cli_first_candidate_and_rates(tmp_path,monkeypatch):
    import benchmark_vlm_static_batch as benchmark
    import compare_dlite_dflash_first as cli
    import run_vlm_matrix
    import evaluate_memory_ablation
    torch.set_num_threads(1)
    target,dlite,_,_=fixture_model()
    target.eval();target.config.text_config.eos_token_id=-1
    processor=Processor()
    processor.image_processor=SimpleNamespace(size=dict(shortest_edge=4,longest_edge=4096))
    target_path=tmp_path/'target';target_path.mkdir()
    (target_path/'config.json').write_text(target.config.to_json_string())
    policy=dict(prefill_text='keep',prefill_vision='keep',decode_mode='all',window_size=0,implementation='compact')
    dflash=make_draft(target,processor,dict(algorithm='dflash',num_draft_layers=1,block_size=4,
        target_model=str(target_path),context_policy=policy))
    dlite.config.vlm_config=dict(target_model=str(target_path),context_policy=dict(prefill_text='drop',prefill_vision='drop',decode_mode='none'))
    group=tmp_path/'pair'
    for name,draft in [('dlite',dlite),('dflash',dflash)]:
        step=group/name/'step_1'
        draft.save_pretrained(step)
        (step.parent/'last').symlink_to(step,target_is_directory=True)
    models,configs=validate_pair(group,target_path)
    assert models[0][1].name=='step_1'
    monkeypatch.setattr(benchmark,'load_target',lambda *a,**kw:(target,processor))
    monkeypatch.setattr(run_vlm_matrix,'select_data',lambda a:{'charxiv_val':dict(source='tiny',samples=2,serialized=''.join(json.dumps(dict(id=str(i),question='describe'))+'\n' for i in range(2)))})
    commands=[]
    def run(command,env,dest):
        commands.append(command)
        monkeypatch.setattr(sys,'argv',[*command[1:],'--device','cpu'])
        benchmark.main()
    monkeypatch.setattr(evaluate_memory_ablation,'run_logged',run)
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['pair','--pair-run',str(group),'--target-model',str(target_path),'--output-dir',str(out),'--samples','2','--max-new-tokens','9'])
    cli.main()
    assert len(commands)==1  # same Target instance, one AR + both models
    assert commands[0].count('--checkpoint')==2
    paired=[json.loads(l) for l in (out/'charxiv_val/paired_requests.jsonl').read_text().splitlines()]
    assert all(r['same_anchor_and_target_prediction'] for r in paired)
    assert all(r['eligible'] for r in paired)
    assert (out/'first_candidate_report.md').is_file()
    assert 'dlite' in (out/'first_candidate_report.md').read_text()
    assert 'dflash' in (out/'first_candidate_report.md').read_text()
    metrics=[json.loads(l) for l in (out/'charxiv_val/dflash_b1.jsonl').read_text().splitlines()]
    assert all(r['result']['round_diagnostics'][0]['anchor_token_id'] is not None for r in metrics)
    assert not list(group.rglob('training_state_rank_*'))


@torch.no_grad()
def test_eos_anchor_not_counted_as_a_first_candidate():
    torch.set_num_threads(1)
    target,draft,_,_=fixture_model()
    target.eval();draft.eval()
    inputs=tiny_image_inputs()
    example=PreparedExample(inputs,inputs['input_ids'].size(1),visual_positions(inputs['input_ids'][0],target.config),'tiny')
    root=target(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1:].argmax(-1).item()
    target.config.text_config.eos_token_id=root
    result=generate_static_batch(target,draft,example,max_new_tokens=5,state_mode='per_token')
    row=result['round_diagnostics'][0]
    assert result['output_tokens']==1
    assert row['emitted_length']==1
    assert row['first_candidate_hit'] is None
    assert not row['first_candidate_eligible']

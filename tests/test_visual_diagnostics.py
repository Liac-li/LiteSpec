from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from visual_diagnostic_metrics import compare_conditions, summarize_probes, summarize_closed_loop
from run_visual_diagnostics import cells, resolve_models
from litespec.vlm.visual_kv import VisualKVCache
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.runtime import PreparedExample, selected_layers, visual_positions
from test_visual_kv import setup, model
from test_vlm import tiny_qwen35, tiny_image_inputs


def test_repair_break_and_first_rejection_denominators():
    conditions = compare_conditions([1, 2, 3], {
        'disabled': {'tokens': [9, 2, 3]},
        'correct': {'tokens': [1, 8, 3]},
        'wrong': {'tokens': [9, 2, 8]},
    })
    assert conditions['correct']['first_candidate_repaired']
    assert conditions['correct']['repairs'] == 1
    assert conditions['correct']['breaks'] == 1
    assert conditions['correct']['prefix_delta'] == 1
    assert conditions['wrong']['prefix_delta'] == 0  # suffix change has no prefix gain
    rows = [dict(decode_offset=0, conditions=conditions), dict(decode_offset=8, conditions=conditions)]
    summary = summarize_probes(rows)
    first = next(r for r in summary if r['condition']=='correct' and r['scope']=='first_round')
    assert first['first_candidate_hit_rate'] == 1
    assert first['per_position_hit_rates'] == [1, 0, 1]
    full = compare_conditions([1], {'disabled': {'tokens':[1]}, 'correct':{'tokens':[1]}})
    assert summarize_probes([dict(decode_offset=0, conditions=full)])[0]['first_rejection_repair_rate'] is None
    with pytest.raises(ValueError):
        compare_conditions([1, 2], {'disabled':{'tokens':[1]}})


def test_cap_censoring_in_closed_loop_first_metric():
    data = [dict(round_diagnostics=[dict(round_index=0, first_candidate_hit=True, committed_length=3),
            dict(round_index=1, first_candidate_hit=False, committed_length=1),
            dict(round_index=2, first_candidate_hit=None, committed_length=1)])]
    rows = summarize_closed_loop(data)
    assert rows[0]['first_candidate_opportunities']==2
    assert rows[0]['first_candidate_hit_rate']==.5
    assert rows[1]['first_candidate_hit_rate']==1
    assert rows[2]['first_candidate_hit_rate']==0


@pytest.mark.parametrize('name', ['pivot_visual_rnn', 'pivot_history_rnn', 'visual_recurrent'])
@torch.no_grad()
def test_first_round_is_always_then_off_and_releases_bank(name):
    target, draft, _, example, _ = setup(name)
    draft.eval(); target.eval()
    start=example.prefill_length
    output=target(**example.inputs, use_cache=False, output_hidden_states=True)
    prefix=SimpleNamespace(hidden_states=tuple(x[:,:start] for x in output.hidden_states))
    inputs={'input_ids': example.inputs['input_ids'][:,:start]}
    pivot=selected_layers(prefix.hidden_states, draft.target_layer_ids, 4)[:,-1:]
    pending=example.inputs['input_ids'][:,start:start+1]
    caches={s: VisualKVCache(draft, target, prefix, inputs, context_schedule=s) for s in ('always','first_round','off')}
    expected=caches['always'].propose(draft,target,pending,pivot,start)
    first=caches['first_round'].propose(draft,target,pending,pivot,start)
    torch.testing.assert_close(first[1],expected[1],rtol=0,atol=0)
    assert caches['first_round'].evidence is None
    caches['first_round'].commit(draft, prefix, 1, start)
    assert caches['first_round'].evidence is None  # commit cannot revive the bank
    off=caches['off'].propose(draft,target,pending,pivot,start)
    later=caches['first_round'].propose(draft,target,pending,pivot,start)
    torch.testing.assert_close(off[1],later[1],rtol=0,atol=0)


@torch.no_grad()
def test_prefill_bank_never_appends_answer_and_donor_positions_rebased():
    target,draft,_,example,_=setup('pivot_history_rnn')
    output=target(**example.inputs,use_cache=False,output_hidden_states=True)
    start=example.prefill_length
    prefix=SimpleNamespace(hidden_states=tuple(x[:,:start] for x in output.hidden_states))
    inputs={'input_ids':example.inputs['input_ids'][:,:start]}
    fixed=VisualKVCache(draft,target,prefix,inputs,history_scope='prefill')
    full=VisualKVCache(draft,target,prefix,inputs)
    before=[(k.clone(),v.clone()) for k,v in fixed.evidence.projected]
    fixed.commit(draft,output,2,start)
    full.commit(draft,output,2,start)
    assert fixed.evidence.values.size(1)==start
    assert full.evidence.values.size(1)==start+2
    for old,new in zip(before,fixed.evidence.projected):
        torch.testing.assert_close(old[0],new[0],rtol=0,atol=0)
    vd=model('pivot_visual_rnn')
    donor=VisualKVCache(vd,target,prefix,inputs,donor=(prefix,inputs['input_ids']))
    assert donor.evidence.positions.count_nonzero()==0
    with pytest.raises(ValueError):
        VisualKVCache(vd,target,prefix,inputs,history_scope='prefill')


@pytest.mark.parametrize('schedule,scope', [('always','full'),('off','full'),('first_round','full'),('always','prefill'),('first_round','prefill')])
@torch.no_grad()
def test_real_hybrid_decode_schedules_and_metrics(schedule,scope):
    torch.manual_seed(7); torch.set_num_threads(1)
    target=tiny_qwen35().eval()
    target.config.text_config.eos_token_id=-1
    draft=model('pivot_history_rnn').eval()
    inputs=tiny_image_inputs()
    example=PreparedExample(inputs, inputs['input_ids'].size(1),visual_positions(inputs['input_ids'][0],target.config),'tiny')
    before={k:v.clone() for k,v in draft.state_dict().items()}
    ar=generate_static_batch(target,None,example,max_new_tokens=9)
    result=generate_static_batch(target,draft,example,max_new_tokens=9,state_mode='per_token',context_schedule=schedule,history_scope=scope)
    assert result['token_ids']==ar['token_ids']
    assert len(result['round_diagnostics']) >= 2
    assert result['round_diagnostics'][0]['decode_offset']==0
    assert result['round_diagnostics'][0]['first_candidate_hit'] is not None
    assert result['context_schedule']==schedule
    for k,v in draft.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_cells_and_checkpoint_validation(tmp_path):
    assert len(cells('pivot_history',True))==4
    assert all(c[0]!='wrong_image' for c in cells('pivot_visual',False))
    d=tmp_path/'pivot_rnn'/'step_0001'; d.mkdir(parents=True)
    (d.parent/'last').symlink_to(d, target_is_directory=True)
    (d/'config.json').write_text(json.dumps(dict(dlite_config=dict(context_mode='pivot',experiment_name='pivot_rnn'),vlm_config=dict(image_max_tokens=1024))))
    with pytest.raises(ValueError,match='not exported'):
        resolve_models(tmp_path,['pivot_rnn'],1024)
    (d/'model.safetensors').touch()
    assert resolve_models(tmp_path,['pivot_rnn'],1024)[0][1]==d
    with pytest.raises(ValueError,match='budget'):
        resolve_models(tmp_path,['pivot_rnn'],512)


@torch.no_grad()
def test_fixed_prefix_probe_runs_real_model_and_writes_scope_metrics(tmp_path, monkeypatch):
    import sys
    import probe_visual_kv as cli
    from test_vlm_aligned import Processor
    target=tiny_qwen35().requires_grad_(False)
    target.config.text_config.eos_token_id=-1
    draft=model('pivot_visual_rnn').eval()
    processor=Processor()
    processor.image_processor=SimpleNamespace(size={"shortest_edge":4,"longest_edge":4096})
    monkeypatch.setattr(cli,'load_target',lambda *a,**kw:(target,processor))
    monkeypatch.setattr(cli,'load_draft',lambda *a,**kw:draft)
    data=tmp_path/'data.jsonl'
    data.write_text(json.dumps(dict(id='a',question='describe'))+'\n')
    refs=tmp_path/'ar.jsonl'
    refs.write_text(json.dumps(dict(sample_id='a',result=dict(token_ids=[4,5,6,7,8,9,10,11])))+'\n')
    out=tmp_path/'probe'
    monkeypatch.setattr(sys,'argv',['probe','--target-model','tiny','--checkpoint','tiny','--data',str(data),'--output-dir',str(out),
        '--samples','1','--device','cpu','--reference-file',str(refs),'--stride','2'])
    cli.main()
    summary=json.loads((out/'summary.json').read_text())
    assert any(r['scope']=='first_round' for r in summary)
    assert any(r['scope']=='later_rounds' for r in summary)
    assert {r['condition'] for r in summary} >= {'correct','disabled','schedule_first_round'}
    anchors=[json.loads(l) for l in (out/'anchors.jsonl').read_text().splitlines()]
    assert anchors[0]['decode_offset']==0
    for row in anchors[1:]:
        assert row['conditions']['schedule_first_round']['tokens']==row['conditions']['disabled']['tokens']


def test_orchestrator_routes_fixed_and_timed_jobs_without_training(tmp_path, monkeypatch):
    import sys
    import run_visual_diagnostics as cli
    import run_vlm_matrix
    import evaluate_memory_ablation
    group=tmp_path/'group'
    step=group/'pivot_history_rnn'/'step_1';step.mkdir(parents=True)
    (step.parent/'last').symlink_to(step,target_is_directory=True)
    (step/'config.json').write_text(json.dumps(dict(dlite_config=dict(context_mode='pivot_history',experiment_name='pivot_history_rnn'),vlm_config=dict(image_max_tokens=1024))))
    (step/'model.safetensors').touch()
    target=tmp_path/'target';target.mkdir()
    (target/'config.json').write_text(json.dumps(dict(vision_config=dict(patch_size=14,spatial_merge_size=2))))
    monkeypatch.setattr(run_vlm_matrix,'select_data',lambda a:{'gsm8k':dict(samples=1,source='source.jsonl',serialized=json.dumps(dict(id='a',question='q'))+'\n')})
    calls=[]
    def run(command,env,dest):
        calls.append(command)
        assert not any('train' in Path(x).name for x in command[:2])
        dest.mkdir()
        if command[1].endswith('probe_visual_kv.py'):
            (dest/'summary.json').write_text('[]')
            return
        label=command[command.index('--checkpoint')+1] if '--checkpoint' in command else 'ar'
        row=dict(model=label,dataset_name='gsm8k',memory_condition='correct',context_schedule='always',history_scope='full',
            mean_committed_length=2,raw_throughput_speedup_vs_ar=1,output_tokens_per_second=10,peak_allocated_gib=1)
        (dest/'summary.json').write_text(json.dumps([row]))
        (dest/f'{label}_b1.jsonl').write_text(json.dumps(dict(sample_id='a',result=dict(token_ids=[4,5,6,7],round_diagnostics=[])))+'\n')
    monkeypatch.setattr(evaluate_memory_ablation,'run_logged',run)
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['diag','--run-group',str(group),'--output-dir',str(out),'--target-model',str(target),'--experiments','pivot_history_rnn','--samples','1'])
    cli.main()
    timed=[c for c in calls if c[1].endswith('benchmark_vlm_static_batch.py')]
    assert len(timed)==5
    assert len([c for c in calls if c[1].endswith('probe_visual_kv.py')])==2
    assert any(c[c.index('--context-schedule')+1]=='first_round' and c[c.index('--history-scope')+1]=='prefill' for c in timed if '--context-schedule' in c)
    assert (out/'diagnostics.md').exists()
    assert json.loads((out/'settings.json').read_text())['checkpoint_paths'][0]['path']==str(step)

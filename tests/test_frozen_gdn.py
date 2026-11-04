"""Pretrained preservation, real optimizer updates and content interventions."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from litespec.vlm import train_aligned as aligned
from litespec.vlm.baselines import load_draft
from litespec.vlm.reader_adapt import frozen_digest, initialize_reader_adaptation
from test_gdn_reader import draft_for
from test_vlm_aligned import fixture_model, Processor
from diagnose_gdn import replace_content_keep_scale, reader_mode, paired_effects


def adaptation_fixture(tmp_path, feature='full'):
    target,_,_,example=fixture_model()
    base,cfg=draft_for(target,'dlite',False)
    source=tmp_path/'base';base.save_pretrained(source)
    cfg.update(pretrained_draft=str(source),reader_only=True,gdn_enabled=True,gdn_feature_mode=feature,
               device='cpu',use_muon=False,learning_rate=.0001,target_model='tiny')
    draft=initialize_reader_adaptation(target,cfg,attention='sdpa')
    return target,base,draft,cfg,example


@pytest.mark.parametrize('feature',['full','content_only'])
def test_frozen_optimizer_preserves_original_and_reloads(tmp_path,feature):
    target,base,draft,cfg,example=adaptation_fixture(tmp_path,feature)
    torch.manual_seed(7)
    inputs=aligned.prepare_forward(target,aligned.make_online(target,draft,cfg),example)
    online=aligned.make_online(target,draft,cfg)
    # Zero-init identity must compare the SAME Target execution/anchor features,
    # not numerical differences between two independently executed Target passes.
    base.eval().requires_grad_(False)
    with torch.no_grad():
        expected=aligned.make_online(target,base,cfg)(**{**inputs,'gdn_states':None}).loss
        torch.testing.assert_close(online(**inputs).loss,expected)
    parameters,specs=aligned.register_optimizer_parameters(draft)
    assert all('.gdn_reader.' in s.name and s.optimizer=='adamw' for s in specs)
    optimizer=aligned.BF16Optimizer(draft,.0001,weight_decay=0,max_grad_norm=1,total_steps=4,
                                   warmup_ratio=0,parameters=parameters,parameter_specs=specs)
    gradient={}
    for _ in range(3):
        optimizer.zero_grad()
        online(**inputs).loss.backward()
        gradient={n:float(p.grad.norm()) for n,p in draft.named_parameters() if p.requires_grad}
        optimizer.step()
    assert gradient['layers.0.gdn_reader.query.weight']>0
    assert gradient['layers.0.gdn_reader.value_down']>0
    assert frozen_digest(draft.state_dict())==cfg['frozen_backbone_sha256']==frozen_digest(base.state_dict())
    assert all(p.grad is None for n,p in draft.named_parameters() if '.gdn_reader.' not in n)
    draft.eval();path=tmp_path/'adapted';draft.save_pretrained(path)
    reloaded=load_draft(path,attn_implementation='sdpa').eval()
    torch.testing.assert_close(aligned.make_online(target,reloaded,cfg)(**inputs).loss,online(**inputs).loss)


def test_content_interventions_preserve_scale_and_restore(tmp_path):
    _,_,draft,_,_=adaptation_fixture(tmp_path)
    reader=draft.layers[0].gdn_reader
    torch.nn.init.normal_(reader.up.weight,std=.1)
    states=tuple(torch.randn(1,1,reader.heads,reader.dk,reader.dv) for _ in reader.layer_ids)
    donors=tuple(torch.randn_like(s)*5 for s in states)
    replaced=replace_content_keep_scale(states,donors)
    for own,donor,new in zip(states,donors,replaced):
        torch.testing.assert_close(own.square().mean((-2,-1)),new.square().mean((-2,-1)))
        assert not torch.allclose(own,new)
    hidden=torch.randn(1,4,draft.config.hidden_size)
    with reader_mode(reader,'scale_only'):a=reader(hidden,states)
    with reader_mode(reader,'scale_only'):b=reader(hidden,replaced)
    torch.testing.assert_close(a,b)
    with reader_mode(reader,'content_only'):c=reader(hidden,states)
    with reader_mode(reader,'content_only'):d=reader(hidden,replaced)
    assert not torch.allclose(c,d)
    with pytest.raises(RuntimeError):
        with reader_mode(reader,'content_only'):raise RuntimeError('abort')
    assert reader.probe_mode=='normal'
    row=dict(first_candidate_correct=True,fixed_reference_committed_length=3)
    paired_effects(row,dict(first_candidate_correct=False,fixed_reference_committed_length=1))
    assert row['first_candidate_repaired']==1 and row['first_candidate_broken']==0
    fixed=dict(first_candidate_correct=True,fixed_reference_committed_length=4,
               candidate_token_ids=[1,2,3],reference_candidate_token_ids=[1,2,3])
    off=dict(first_candidate_correct=True,fixed_reference_committed_length=2,candidate_token_ids=[1,9,3])
    paired_effects(fixed,off)
    assert fixed['baseline_first_rejection_repaired']==1
    fixed['candidate_token_ids']=[8,2,3]
    paired_effects(fixed,off)
    assert fixed['baseline_first_rejection_repaired']==0


def test_pretrained_architecture_wins_over_recipe_defaults(tmp_path):
    target,base,_,cfg,_=adaptation_fixture(tmp_path)
    cfg.update(block_size=8,num_draft_layers=5,chs_num_layers=12,sequential_rank=320)
    draft=initialize_reader_adaptation(target,cfg,attention='sdpa')
    assert cfg['block_size']==draft.block_size==base.block_size==4
    assert cfg['num_draft_layers']==1 and cfg['chs_num_layers']==2
    assert cfg['sequential_rank']==8 and draft.mask_token_id==base.mask_token_id
    assert frozen_digest(draft.state_dict())==frozen_digest(base.state_dict())


def test_load_rebuilds_nonpersistent_rope_without_touching_weights(monkeypatch,tmp_path):
    target,base,_,cfg,_=adaptation_fixture(tmp_path)
    cls=type(base)
    original=cls.from_pretrained.__func__
    def corrupt_buffers(model_class,*args,**kwargs):
        loaded=original(model_class,*args,**kwargs)
        model=loaded[0] if isinstance(loaded,tuple) else loaded
        # Emulate uninitialized nonpersistent storage left by meta loading.
        model.rotary_emb.inv_freq.fill_(float('nan'))
        model.rotary_emb.original_inv_freq.zero_()
        return loaded
    monkeypatch.setattr(cls,'from_pretrained',classmethod(corrupt_buffers))
    loaded,info=load_draft(tmp_path/'base',attn_implementation='sdpa',output_loading_info=True)
    torch.testing.assert_close(loaded.rotary_emb.inv_freq,base.rotary_emb.inv_freq,rtol=0,atol=0)
    torch.testing.assert_close(loaded.rotary_emb.original_inv_freq,base.rotary_emb.original_inv_freq,rtol=0,atol=0)
    assert frozen_digest(loaded.state_dict())==frozen_digest(base.state_dict())


def test_reader_only_actual_training_exports_integrity(monkeypatch,tmp_path):
    target,base,_,cfg,_=adaptation_fixture(tmp_path)
    monkeypatch.setattr(aligned,'load_target',lambda *a,**kw:(target,Processor()))
    data=tmp_path/'train.jsonl'
    data.write_text(''.join(json.dumps(dict(id=str(i),question='describe',answer='a square'))+'\n' for i in range(4)))
    cfg.update(upstream_config='scripts/config/dlite_sft.yaml',train_data=str(data),output_dir=str(tmp_path/'run'),
               report_to='none',log_interval=1,num_epochs=1,batch_size=2,accumulation_steps=1,
               save_interval=4000,num_anchors=3)
    path=tmp_path/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
    aligned.run(['--config',str(path)])
    checkpoint=tmp_path/'run/last'
    loaded=load_draft(checkpoint,attn_implementation='sdpa')
    assert frozen_digest(loaded.state_dict())==frozen_digest(base.state_dict())
    assert loaded.layers[0].gdn_reader.up.weight.count_nonzero()>0
    assert not list(checkpoint.glob('training_state_rank_*.pt'))
    metadata=json.loads((checkpoint/'vlm_run.json').read_text())
    assert all('.gdn_reader.' in n for n in metadata['trainable_parameter_names'])


def test_custom_checkpoint_cli_content_diagnostics(monkeypatch,tmp_path):
    import diagnose_gdn as diag
    from types import SimpleNamespace
    from PIL import Image
    target,_,draft,cfg,_=adaptation_fixture(tmp_path)
    target.config.text_config.eos_token_id=None
    processor=Processor()
    processor.image_processor=SimpleNamespace(size=dict(shortest_edge=16,longest_edge=65536))
    monkeypatch.setattr(diag,'load_target',lambda *a,**kw:(target,processor))
    torch.nn.init.normal_(draft.layers[0].gdn_reader.up.weight,std=.1)
    checkpoint=tmp_path/'checkpoint';draft.save_pretrained(checkpoint)
    (checkpoint/'vlm_run.json').write_text(json.dumps(cfg))
    inputs=tmp_path/'inputs';inputs.mkdir()
    rows=[]
    for i in range(2):
        path=tmp_path/f'{i}.png';Image.new('RGB',(4,4),(i*100,0,0)).save(path)
        rows.append(dict(id=str(i),image=str(path),question='describe'))
    (inputs/'vision.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output=tmp_path/'diagnostics'
    diag.main(['--run-group',str(tmp_path),'--checkpoint','custom',str(checkpoint),
               '--inputs-dir',str(inputs),'--output-dir',str(output),'--device','cpu',
               '--samples-per-dataset','2','--reference-tokens','6','--offsets','0','2',
               '--modes','normal','off','content_only','scale_only','wrong_content','shared_query'])
    summary=json.loads((output/'summary.json').read_text())
    assert len(summary)==6 and all(r['model']=='custom' for r in summary)
    assert all('first_candidate_repairs' in r and 'first_candidate_breaks' in r for r in summary)
    assert json.loads((output/'checkpoint_audit.json').read_text())['custom']['passed']

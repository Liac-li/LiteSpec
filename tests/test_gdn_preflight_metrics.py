"""Sparse BF16 outliers are bounded; systematic state/content corruption fails."""
import torch
from gdn_preflight_metrics import state_pair_metrics, read_pair_metrics
from litespec.vlm.gdn_reader import GDNStateReader


def test_sparse_small_outliers_do_not_mask_global_error():
    reference=torch.full((1,32,128,128),.1)
    actual=reference.clone()
    actual.reshape(-1)[:8]+=.0077
    result=state_pair_metrics(actual,reference)
    assert not result['strict_allclose_passed']
    assert result['old_tolerance_outlier_count']==8 and result['passed']
    actual=reference+.01
    assert not state_pair_metrics(actual,reference)['passed']


def test_isolated_large_error_and_small_signal_corruption_fail():
    reference=torch.full((1,32,128,128),.1)
    actual=reference.clone();actual.reshape(-1)[0]+=1.
    assert not state_pair_metrics(actual,reference)['passed']
    reference=torch.full((1,2,16,16),.00001)
    actual=reference+.0001
    result=state_pair_metrics(actual,reference)
    assert result['strict_allclose_passed'] and not result['passed']


def test_axis_or_head_corruption_and_nonfinite_fail():
    torch.manual_seed(9)
    reference=torch.randn(1,4,16,16)
    assert not state_pair_metrics(reference.transpose(-1,-2),reference)['passed']
    assert not state_pair_metrics(reference.roll(1,1),reference)['passed']
    bad=reference.clone();bad[0,0,0,0]=float('nan')
    assert not state_pair_metrics(bad,reference)['passed']


def test_reader_content_test_is_not_hidden_by_zero_output_projection():
    torch.manual_seed(8)
    reader=GDNStateReader(16,dict(layer_ids=[0],heads=2,key_dim=8,value_dim=8,
        rank=2,bottleneck=8,chunk_anchors=1))
    hidden=torch.randn(1,4,16)
    reference=(torch.randn(1,2,8,8),)
    actual=(reference[0].roll(1,-1),)
    # At initialization any state gives the SAME final Reader output.
    torch.testing.assert_close(reader(hidden,(reference[0][:,None],)),hidden)
    torch.testing.assert_close(reader(hidden,(actual[0][:,None],)),hidden)
    assert read_pair_metrics(reader,hidden,reference,reference)['passed']
    assert not read_pair_metrics(reader,hidden,actual,reference)['passed']


def test_preflight_control_flow_on_cpu_fixture(monkeypatch,tmp_path):
    """Run the complete CLI against a tiny real Target; mock CUDA selection only."""
    import json
    import sys
    from types import SimpleNamespace
    import check_frozen_gdn_cuda as check
    from test_vlm_aligned import fixture_model,Processor
    from test_gdn_reader import draft_for
    target,_,_,_=fixture_model()
    base,cfg=draft_for(target,'dlite',False)
    source=tmp_path/'base';base.save_pretrained(source)
    processor=Processor()
    processor.image_processor=SimpleNamespace(size=dict(shortest_edge=16,longest_edge=65536))
    monkeypatch.setattr(check,'load_target',lambda *a,**kw:(target,processor))
    monkeypatch.setattr(check.aligned,'load_config',lambda *a,**kw:dict(cfg,gdn_enabled=True))
    original_prepare=check.prepare_example
    def prepare(*a,**kw):
        kw['device']='cpu'
        return original_prepare(*a,**kw)
    monkeypatch.setattr(check,'prepare_example',prepare)
    original_tensor=torch.tensor
    def tensor(*a,**kw):
        if kw.get('device')=='cuda':kw['device']='cpu'
        return original_tensor(*a,**kw)
    monkeypatch.setattr(torch,'tensor',tensor)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    data=tmp_path/'data.jsonl';data.write_text(json.dumps(dict(id='0',question='describe',answer='a square'))+'\n')
    output=tmp_path/'preflight.json'
    monkeypatch.setattr(sys,'argv',['check','--target','tiny','--draft',str(source),
                                  '--data',str(data),'--output',str(output)])
    check.main()
    report=json.loads(output.read_text())
    assert report['passed'] and report['frozen_backbone_unchanged']
    assert all(row['same_prefix_layers'] and row['same_prefix_reader_content']['passed'] for row in report['state_parity'])
    # A failing gate must keep a readable report and must not reach training.
    monkeypatch.setattr(check,'state_pair_metrics',lambda *a:dict(passed=False,finite=True,relative_l2=1.))
    import pytest
    with pytest.raises(RuntimeError,match='state/content mismatch'):
        check.main()
    report=json.loads(output.read_text())
    assert report['passed'] is False and report['state_parity'][0]['passed'] is False

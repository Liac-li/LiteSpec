import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch


def test_video_custom_checkpoints_and_frozen_selection(tmp_path,monkeypatch):
    import benchmark_video as cli
    movie=tmp_path/'v.mp4';movie.touch();s=movie.stat()
    selected={'Video-MME':[dict(id='Video-MME:v',video=str(movie),question='describe',duration='short',video_bytes=s.st_size,video_mtime_ns=s.st_mtime_ns)]}
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(selected))
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['video','--checkpoint','old_20k',str(tmp_path/'ckpt'),'--datasets','Video-MME','--selection-file',str(selection),
        '--samples','1','--context-window-size','10','--inspect-only','--output-dir',str(out)])
    cli.main()
    settings=json.loads((out/'settings.json').read_text())
    assert settings['models']==['old_20k']
    assert settings['context_window_size']==10
    assert settings['selected']==selected
    movie.write_text('changed')
    monkeypatch.setattr(sys,'argv',['video','--checkpoint','old_20k',str(tmp_path/'ckpt'),'--datasets','Video-MME','--selection-file',str(selection),
        '--samples','1','--inspect-only','--output-dir',str(tmp_path/'bad')])
    with pytest.raises(ValueError,match='changed'):cli.main()


def test_actual_text_image_benchmark_override(tmp_path,monkeypatch):
    from test_vlm_aligned import fixture_model,Processor
    import benchmark_vlm_static_batch as cli
    torch.set_num_threads(1)
    target,draft,_,_=fixture_model();target.eval();target.config.text_config.eos_token_id=-1
    processor=Processor()
    monkeypatch.setattr(cli,'load_target',lambda *a,**kw:(target,processor))
    path=tmp_path/'checkpoint';draft.save_pretrained(path)
    data=tmp_path/'eval.jsonl';data.write_text(json.dumps(dict(id='a',question='describe'))+'\n')
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['bench','--target-model','tiny','--checkpoint','windowed',str(path),'--data',str(data),
        '--device','cpu','--attention-backend','sdpa','--state-mode','per_token','--max-samples','1','--max-new-tokens','9',
        '--batch-sizes','1','--context-window-size','10','--output-dir',str(out)])
    cli.main()
    rows=json.loads((out/'summary.json').read_text())
    assert rows[1]['context_window_size']==10
    assert rows[1]['context_slots']==20
    assert json.loads((path/'config.json').read_text())['dlite_config'].get('context_window_size',1)==1


def test_four_dataset_orchestrator_freezes_inputs_and_reuses_parents(tmp_path,monkeypatch):
    import evaluate_sw_token as cli
    import video_benchmark_data
    import run_vlm_matrix
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'model.safetensors').touch()
    (parent/'config.json').write_text(json.dumps(dict(architectures=['DLiteDraftModel'],dlite_config=dict(model_role='student'))))
    target=tmp_path/'target';target.mkdir();(target/'config.json').write_text(json.dumps(dict(vision_config=dict(patch_size=16,spatial_merge_size=2))))
    movies={case:[dict(id=case+':a',video='v.mp4',question='describe')] for case in ('Video-MME','VideoDetailCaption')}
    monkeypatch.setattr(video_benchmark_data,'select_data',lambda *a:movies)
    monkeypatch.setattr(run_vlm_matrix,'select_data',lambda a:{name:dict(source=name,samples=1,serialized=json.dumps(dict(id=name,question='q'))+'\n') for name in ('gsm8k','textvqa')})
    calls=[]
    def run(command,env,dest):
        calls.append(command);dest.mkdir()
        k=int(command[command.index('--context-window-size')+1]);video=command[1].endswith('benchmark_video.py')
        row=dict(model='old_20k',context_window_size=k,batch_size=1,mean_committed_length=2,
            prefill_seconds=1,peak_allocated_gib=1,matched=1,requests=1)
        row.update(dict(decode_tok_s=10,raw_decode_ratio_vs_ar=2) if video else dict(output_tokens_per_second=10,raw_throughput_speedup_vs_ar=2))
        (dest/'summary.json').write_text(json.dumps([row]))
    monkeypatch.setattr(cli,'run_logged',run)
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['scan','--workspace',str(tmp_path),'--target-model',str(target),'--output-dir',str(out),
        '--checkpoint','old_20k',str(parent),'--windows','1','2','--samples','1'])
    cli.main()
    assert len(calls)==8
    for command in calls:
        assert command[command.index('--checkpoint')+2]==str(parent)
    video_files=[c[c.index('--selection-file')+1] for c in calls if '--selection-file' in c]
    assert len(set(video_files))==2  # each same file used for both K
    settings=json.loads((out/'settings.json').read_text())
    assert len(settings['snapshots'])==4
    assert (out/'comparison.md').exists()

"""CPU checks independent of GPU/model downloads."""
import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
from video_benchmark_data import bounded_size, select_rows, CAPTION_PROMPT
from benchmark_video import report


class VideoDataTests(unittest.TestCase):
    def test_pixel_budget_and_alignment(self):
        for w,h in [(1920,1080),(1080,1920),(640,480),(3840,2160),(80,4000),(20,20)]:
            a,b=bounded_size(w,h,448*448,32)
            self.assertLessEqual(a*b,448*448)
            self.assertEqual(a%32,0); self.assertEqual(b%32,0)
        self.assertRaises(ValueError,bounded_size,640,480,100,32)

    def test_caption_mkv_and_no_answer_in_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'Test_Videos').mkdir()
            rows=[]
            for i in range(5):
                (root/'Test_Videos'/f'v{i}.mkv').touch()
                rows.append(dict(video_name=f'v{i}',question=f'describe {i}',answer='SECRET'))
            out=select_rows('VideoDetailCaption',rows,root,4,42)
            self.assertEqual(out,select_rows('VideoDetailCaption',rows,root,4,42))
            self.assertEqual(len(set(r['id'] for r in out)),4)
            self.assertNotIn('SECRET',json.dumps(out))
            self.assertTrue(all(r['video'].endswith('.mkv') for r in out))
            (root/'Test_Videos/v0.mp4').touch()
            self.assertRaises(ValueError,select_rows,'VideoDetailCaption',rows,root,4,42)

    def test_mme_unique_balanced_and_caption_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'data').mkdir(); rows=[]
            for duration in ['short','medium','long']:
                for i in range(6):
                    key=f'{duration}{i}'; (root/'data'/f'{key}.mp4').touch()
                    for j in range(3): rows.append(dict(videoID=key,duration=duration,question=f'Q{j}',answer='SECRET'))
            out=select_rows('Video-MME',rows,root,16,42)
            self.assertEqual(len(set(r['video'] for r in out)),16)
            counts=[sum(r['duration']==x for r in out) for x in ['short','medium','long']]
            self.assertLessEqual(max(counts)-min(counts),1)
            self.assertTrue(all(r['question']==CAPTION_PROMPT for r in out))
            self.assertNotIn('SECRET',json.dumps(out))
            self.assertRaises(ValueError,select_rows,'Video-MME',rows,root,19,42)

    def test_report_preserves_mismatch_and_all_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'cells').mkdir()
            row=dict(dataset='Video-MME',model='dlite_original',batch_size=16,
                mean_committed_length=2.4,mean_accepted_draft_tokens=1.4,decode_tok_s=400.,
                raw_decode_ratio_vs_ar=1.8,matched=16,requests=32,peak_allocated_gib=20.,
                valid_token_match=False)
            (root/'cells/x.json').write_text(json.dumps(dict(summary=row)))
            report(root)
            self.assertFalse(json.loads((root/'summary.json').read_text())[0]['valid_token_match'])
            with (root/'comparison.csv').open() as f: parsed=list(csv.DictReader(f))
            self.assertEqual(parsed[0]['matched'],'16')
            self.assertIn('16/32',(root/'comparison.md').read_text())

    def test_prepare_video_metadata_and_short_clip(self):
        import numpy as np
        from PIL import Image
        from video_runtime import prepare_video
        class Tensor:
            def __init__(self,a): self.a=np.array(a)
            def cpu(self): return self
            @property
            def shape(self): return self.a.shape
            def __getitem__(self,i): return Tensor(self.a[i])
            def __eq__(self,x): return Tensor(self.a==x)
            def prod(self,axis): return Tensor(self.a.prod(axis))
            def sum(self): return Tensor(self.a.sum())
            def __int__(self): return int(self.a)
            def tolist(self): return self.a.tolist()
        class Frame:
            def to_image(self): return Image.new('RGB',(640,480))
        class Container:
            # Unknown reported frame count, like some MKV streams.
            streams=SimpleNamespace(video=[SimpleNamespace(frames=0,average_rate=30,guessed_rate=30)])
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def decode(self,**kwargs): return iter([Frame() for _ in range(5)])
        class Processor:
            def __init__(self):
                self.video_processor=SimpleNamespace(patch_size=16,merge_size=2,temporal_patch_size=2,
                                                     fetch_videos=lambda *a:None)
            def apply_chat_template(self,messages,**kwargs):
                video,meta=self.video_processor.fetch_videos('clip.mkv')
                self.captured_video,self.meta=video,meta
                self.assertion=kwargs['processor_kwargs']['videos_kwargs']['do_resize']
                h,w=video.shape[1:3]; grid=[8,h//16,w//16]
                count=np.prod(grid)//4
                return dict(input_ids=Tensor([[9]*count+[1,2]]),video_grid_thw=Tensor([grid]))
        processor=Processor(); previous=processor.video_processor.fetch_videos
        class Prepared:
            def __init__(self,inputs,length,mask,key): self.inputs=inputs; self.prefill_length=length
        mocks={'av':SimpleNamespace(open=lambda p:Container()),
               'torch':SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor)),
               'transformers.video_utils':SimpleNamespace(VideoMetadata=lambda **kw:SimpleNamespace(**kw)),
               'litespec.vlm.runtime':SimpleNamespace(PreparedExample=Prepared)}
        with patch.dict(sys.modules,mocks):
            example,meta=prepare_video(processor,SimpleNamespace(video_token_id=9),
                                      dict(id='x',video='clip.mkv',question='describe'))
        self.assertIs(processor.video_processor.fetch_videos,previous)
        self.assertEqual(meta['actual_frames'],16)
        self.assertEqual(meta['frame_indices'][0],0)
        self.assertEqual(meta['frame_indices'][-1],4)
        self.assertEqual(meta['frame_timestamps_seconds'][-1],4/30)
        self.assertEqual(processor.meta.fps,30)
        self.assertFalse(processor.assertion)
        self.assertLessEqual(np.prod(meta['processed_wh']),200704)
        self.assertEqual(meta['video_tokens'],np.prod(meta['video_grid_thw'])//4)

    def test_actual_replication_video_grid_and_timestamps(self):
        # Execute the production helper without importing the heavy model stack.
        import numpy as np
        source=Path(__file__).parents[1]/'litespec/vlm/static_batch.py'
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replicate_inputs')
        class Tensor:
            def __init__(self,a): self.a=np.array(a)
            @property
            def ndim(self): return self.a.ndim
            @property
            def shape(self): return self.a.shape
            def repeat(self,*dims): return Tensor(np.tile(self.a,dims))
        ns={'torch':SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        example=SimpleNamespace(prefill_length=4,inputs={
            'input_ids':Tensor([[1,2,3,4]]),'pixel_values_videos':Tensor([[1,2],[3,4]]),
            'video_grid_thw':Tensor([[8,14,28]]),'second_per_grid_ts':[0.5],
            'position_ids':Tensor(np.zeros((3,1,4),dtype=int))})
        out=ns['replicate_inputs'](example,16)
        self.assertEqual(out['input_ids'].shape,(16,4))
        self.assertEqual(out['video_grid_thw'].shape,(16,3))
        self.assertEqual(out['pixel_values_videos'].shape,(32,2))
        self.assertEqual(out['position_ids'].shape,(3,16,4))
        self.assertEqual(out['second_per_grid_ts'],[0.5]*16)
        self.assertRaises(ValueError,ns['replicate_inputs'],example,0)


if __name__=='__main__': unittest.main()

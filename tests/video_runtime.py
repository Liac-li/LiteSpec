"""Native Qwen video processing with explicit PyAV decode, timestamps and size cap."""
from pathlib import Path
import time
from types import MethodType
from video_benchmark_data import bounded_size


def prepare_video(processor, target_config, record, frames=16, max_pixels=448*448):
    import av
    import numpy as np
    import torch
    from PIL import Image
    from transformers.video_utils import VideoMetadata
    from litespec.vlm.runtime import PreparedExample
    if frames < 2 or frames % 2: raise ValueError('use an even frame count >= 2')
    vp = processor.video_processor
    factor = int(vp.patch_size) * int(vp.merge_size)
    temporal = int(vp.temporal_patch_size)
    if frames % temporal: raise ValueError('frame count must divide temporal patch size')
    captured = {}
    started = time.perf_counter()

    def fetch(self, videos, sample_indices_fn=None):
        if isinstance(videos, (list, tuple)):
            return list(zip(*[fetch(self, v, sample_indices_fn) for v in videos]))
        path = str(videos)
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            total = stream.frames
            if total <= 0: total = sum(1 for _ in container.decode(video=0))
        if total < 1 or fps <= 0: raise ValueError(f'no valid frame count/FPS: {path}')
        indices = np.linspace(0, total - 1, frames, dtype=int).tolist()
        wanted, found = set(indices), {}
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    image = frame.to_image().convert('RGB')
                    original_size = image.size
                    size = bounded_size(*image.size, max_pixels, factor)
                    if image.size != size: image = image.resize(size, Image.Resampling.BICUBIC)
                    found[index] = np.asarray(image)
                if index >= indices[-1]: break
        if wanted != set(found): raise ValueError(f'incomplete decode: {path}')
        video = np.stack([found[i] for i in indices])
        metadata = VideoMetadata(total_num_frames=total, fps=fps, duration=total/fps,
                                 frames_indices=indices, height=video.shape[1], width=video.shape[2],
                                 video_backend='pyav')
        captured.update(frame_indices=indices, source_fps=fps, source_total_frames=total,
                        original_wh=list(original_size), processed_wh=[video.shape[2], video.shape[1]],
                        frame_timestamps_seconds=[i/fps for i in indices], actual_frames=len(indices))
        return video, metadata

    previous = vp.fetch_videos
    vp.fetch_videos = MethodType(fetch, vp)
    try:
        inputs = processor.apply_chat_template(
            [{'role':'user', 'content':[{'type':'video','path':record['video']},
                                       {'type':'text','text':record['question']}]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True,
            processor_kwargs={'videos_kwargs':dict(num_frames=frames, fps=None,
                do_sample_frames=True, do_resize=False, cap_pixels_per_frame=True),
                'return_tensors':'pt'})
    finally:
        vp.fetch_videos = previous
    # Metadata is consumed during processing, not a model-forward kwarg.
    inputs = dict(inputs)
    inputs.pop('video_metadata', None)
    inputs = {k:v.cpu() if torch.is_tensor(v) else v for k,v in inputs.items()}
    ids = inputs['input_ids']
    grid = inputs['video_grid_thw']
    expected = int(grid.prod(-1).sum()) // int(vp.merge_size)**2
    mask = ids[0] == target_config.video_token_id
    if int(mask.sum()) != expected or grid.shape != (1,3):
        raise ValueError('video grid/token mismatch')
    if int(grid[0,0]) * temporal != frames: raise ValueError('unexpected processed frame count')
    captured.update(id=record['id'], video=record['video'], input_tokens=ids.shape[1],
                    video_tokens=int(mask.sum()), video_grid_thw=grid.tolist(),
                    preprocess_seconds=time.perf_counter()-started)
    return PreparedExample(inputs, ids.shape[1], mask, record['id']), captured

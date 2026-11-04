"""Read-only local video benchmark selection; no answers enter model inputs."""
from pathlib import Path
import random
import math

CAPTION_PROMPT = ('Please provide a detailed description of the video, focusing on '
                  'the main subjects, their actions, and the background scenes.')
DATASETS = ('VideoDetailCaption', 'Video-MME')


def bounded_size(width, height, max_pixels, factor):
    if min(width, height, max_pixels, factor) < 1 or max_pixels < factor * factor:
        raise ValueError('invalid video size/budget')
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    w = max(factor, int(width * scale / factor) * factor)
    h = max(factor, int(height * scale / factor) * factor)
    while w * h > max_pixels:
        if w >= h and w > factor: w -= factor
        elif h > factor: h -= factor
        else: raise ValueError('budget cannot fit one spatial unit')
    return w, h


def select_rows(dataset, rows, root, samples=16, seed=42):
    if dataset not in DATASETS or samples < 1:
        raise ValueError('known dataset and positive sample count required')
    root = Path(root).expanduser().resolve()
    candidates, seen = [], set()
    for raw in rows:
        if dataset == 'VideoDetailCaption':
            key = str(raw['video_name'])
            if Path(key).name != key: raise ValueError(f'unsafe video_name: {key}')
            matches = [root / 'Test_Videos' / (key + suffix) for suffix in ('.mp4', '.mkv')]
            matches = [p for p in matches if p.is_file()]
            if len(matches) != 1: raise ValueError(f'{key}: expected one MP4/MKV, found {matches}')
            prompt = raw['question']
            duration = None
        else:
            key = str(raw['videoID'])
            if Path(key).name != key: raise ValueError(f'unsafe videoID: {key}')
            if key in seen: continue
            matches = [root / 'data' / (key + '.mp4')]
            if not matches[0].is_file(): raise FileNotFoundError(matches[0])
            prompt = CAPTION_PROMPT
            duration = str(raw['duration'])
        if key in seen: raise ValueError(f'duplicate video: {key}')
        if not isinstance(prompt, str) or not prompt.strip(): raise ValueError(f'empty prompt: {key}')
        seen.add(key)
        candidates.append(dict(id=f'{dataset}:{key}', video=str(matches[0]), question=prompt,
                               duration=duration))
    rng = random.Random(seed)
    if dataset == 'Video-MME':
        groups = {}
        for row in candidates: groups.setdefault(row['duration'], []).append(row)
        for values in groups.values(): rng.shuffle(values)
        candidates = []
        while any(groups.values()):
            for label in sorted(groups):
                if groups[label]: candidates.append(groups[label].pop())
    else:
        rng.shuffle(candidates)
    if len(candidates) < samples: raise ValueError(f'{dataset}: only {len(candidates)} unique videos for {samples}')
    selected = candidates[:samples]
    for row in selected:
        p = Path(row['video']); s = p.stat()
        row['video_bytes'], row['video_mtime_ns'] = s.st_size, s.st_mtime_ns
    return selected


def select_data(root, datasets, samples, seed):
    import pandas as pd
    result = {}
    for dataset in datasets:
        folder = Path(root).expanduser() / dataset
        annotation = folder / ('data/test-00000-of-00001.parquet' if dataset == DATASETS[0]
                               else 'videomme/test-00000-of-00001.parquet')
        result[dataset] = select_rows(dataset, pd.read_parquet(annotation).to_dict('records'),
                                     folder, samples, seed)
    return result

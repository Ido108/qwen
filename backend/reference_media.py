"""Bounded 24-fps media decoding for MiniMax H3 reference conditioning."""
from __future__ import annotations

from pathlib import Path


def read_frames(path: Path, max_frames: int, max_edge: int = 1024):
    import av
    import numpy as np
    import torch
    from PIL import Image

    with av.open(str(path)) as media:
        stream=next((s for s in media.streams if s.type=='video'),None)
        if stream is None:
            raise ValueError('The source file has no video stream.')
        rate=float(stream.average_rate) if stream.average_rate else 24.0
        if rate<=0 or rate>1000:
            rate=24.0
        frames=[]
        next_time=0.0
        previous_time=-1.0
        for index,frame in enumerate(media.decode(stream)):
            seconds=frame.time if frame.time is not None else index/rate
            if seconds<previous_time:
                seconds=previous_time+1/rate
            previous_time=seconds
            if seconds+0.5/rate<next_time:
                continue
            picture=frame.to_image().convert('RGB')
            if max(picture.size)>max_edge:
                picture.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
            array=np.asarray(picture,dtype=np.uint8)
            while next_time<=seconds+0.5/rate and len(frames)<max_frames:
                frames.append(array)
                next_time=len(frames)/24
            if len(frames)>=max_frames:
                break
        if len(frames)<5:
            raise ValueError('A reference video needs at least five frames.')
        # MiniMax H3 aligns reference-video latents to 17k+5.
        while len(frames)%17!=5 and len(frames)>5:
            frames.pop()
        return torch.from_numpy(np.stack(frames).astype(np.float32)/255.0)


def read_audio(path: Path, max_seconds: float):
    import av
    import numpy as np
    import torch

    with av.open(str(path)) as media:
        stream=next((s for s in media.streams if s.type=='audio'),None)
        if stream is None:
            return None
        resampler=av.AudioResampler(format='fltp',layout='stereo',rate=32000)
        chunks=[]
        length=0
        target=round(max_seconds*32000)
        for frame in media.decode(stream):
            for converted in resampler.resample(frame):
                chunk=np.asarray(converted.to_ndarray(),dtype=np.float32)
                if chunk.ndim==1:
                    chunk=chunk[None,:]
                chunks.append(chunk)
                length+=chunk.shape[-1]
                if length>=target:
                    break
            if length>=target:
                break
        if not chunks:
            return None
        waveform=np.concatenate(chunks,axis=-1)[:,:target]
        return {'waveform':torch.from_numpy(waveform.copy()).unsqueeze(0),
                'sample_rate':32000}

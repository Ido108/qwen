"""Bounded 24-fps media decoding for MiniMax H3 reference conditioning."""
from __future__ import annotations

from pathlib import Path


def _seek(media, stream, start_seconds: float) -> float:
    """Seek to the preceding keyframe and retain a small decode margin."""
    seek_seconds = max(0.0, start_seconds - 0.5)
    if seek_seconds and stream.time_base:
        timestamp = (stream.start_time or 0) + int(seek_seconds / float(stream.time_base))
        media.seek(timestamp, stream=stream, backward=True)
    return seek_seconds


def read_frames(path: Path, max_frames: int, max_edge: int = 1024,
                start_seconds: float = 0, end_seconds: float | None = None):
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
        seek_time=_seek(media,stream,start_seconds)
        origin=float(stream.start_time*stream.time_base) if stream.start_time is not None and stream.time_base else 0.0
        end_seconds=end_seconds if end_seconds is not None else start_seconds+max_frames/24
        selection_length=end_seconds-start_seconds
        frames=[]
        previous_time=-1.0
        for index,frame in enumerate(media.decode(stream)):
            seconds=(frame.time-origin) if frame.time is not None else seek_time+index/rate
            if seconds<previous_time:
                seconds=previous_time+1/rate
            previous_time=seconds
            target_time=start_seconds+(len(frames)+0.5)*selection_length/max_frames
            if seconds+0.5/rate<target_time:
                continue
            if seconds>end_seconds+0.5/rate:
                break
            picture=frame.to_image().convert('RGB')
            if max(picture.size)>max_edge:
                picture.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
            array=np.asarray(picture,dtype=np.uint8)
            while len(frames)<max_frames and start_seconds+(len(frames)+0.5)*selection_length/max_frames<=seconds+0.5/rate:
                frames.append(array)
            if len(frames)>=max_frames:
                break
        if not frames:
            raise ValueError('The selected video segment has no readable frames.')
        while len(frames)<max_frames:
            frames.append(frames[-1])
        return torch.from_numpy(np.stack(frames).astype(np.float32)/255.0)


def read_audio(path: Path, max_seconds: float, start_seconds: float = 0,
               end_seconds: float | None = None):
    import av
    import numpy as np
    import torch

    with av.open(str(path)) as media:
        stream=next((s for s in media.streams if s.type=='audio'),None)
        if stream is None:
            return None
        seek_time=_seek(media,stream,start_seconds)
        origin=float(stream.start_time*stream.time_base) if stream.start_time is not None and stream.time_base else 0.0
        end_seconds=end_seconds if end_seconds is not None else start_seconds+max_seconds
        resampler=av.AudioResampler(format='fltp',layout='stereo',rate=32000)
        chunks=[]
        length=0
        target=round(max_seconds*32000)
        source_target=round((end_seconds-start_seconds)*32000)
        for index,frame in enumerate(media.decode(stream)):
            frame_time=(frame.time-origin) if frame.time is not None else seek_time+index*frame.samples/frame.sample_rate
            for converted in resampler.resample(frame):
                chunk=np.asarray(converted.to_ndarray(),dtype=np.float32)
                if chunk.ndim==1:
                    chunk=chunk[None,:]
                chunk_time=(converted.time-origin) if converted.time is not None else frame_time
                first=max(0,min(chunk.shape[-1],round((start_seconds-chunk_time)*32000)))
                last=max(first,min(chunk.shape[-1],round((end_seconds-chunk_time)*32000)))
                if last>first:
                    chunks.append(chunk[:,first:last])
                    length+=last-first
                if length>=source_target:
                    break
            if length>=source_target or frame_time>end_seconds+0.1:
                break
        if not chunks:
            return None
        waveform=np.concatenate(chunks,axis=-1)[:,:source_target]
        if waveform.shape[-1]<source_target:
            waveform=np.pad(waveform,((0,0),(0,source_target-waveform.shape[-1])))
        if source_target!=target:
            source_positions=np.arange(source_target,dtype=np.float64)
            output_positions=np.linspace(0,max(0,source_target-1),target)
            waveform=np.stack([np.interp(output_positions,source_positions,channel)
                               for channel in waveform]).astype(np.float32)
        return {'waveform':torch.from_numpy(waveform.copy()).unsqueeze(0),
                'sample_rate':32000}

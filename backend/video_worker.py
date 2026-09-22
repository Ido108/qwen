"""Video job output: real H.264 MP4, optional AAC audio, thumbnail and provenance."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import time
from .common import resolve_image, write_json


def save_mp4(path, frames, audio, fps=24):
    import av
    import numpy as np
    frame_count, height, width, _ = frames.shape
    if frame_count < 1:
        raise RuntimeError('The model returned no video frames.')
    with av.open(str(path), 'w', options={'movflags': '+faststart'}) as output:
        stream = output.add_stream('libx264', rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        audio_stream = None
        if audio:
            waveform = audio['waveform'][0].numpy()
            if not np.isfinite(waveform).all():
                raise RuntimeError('The model returned invalid audio samples. Retry the video.')
            rate = int(audio['sample_rate'])
            target = round(frame_count / fps * rate)
            waveform = np.pad(waveform[:, :target], ((0, 0), (0, max(0, target - waveform.shape[1]))))
            layout = 'stereo' if waveform.shape[0] == 2 else 'mono'
            audio_stream = output.add_stream('aac', rate=rate)
            audio_stream.layout = layout
            audio_stream.bit_rate = 192000
        for index, pixels in enumerate(frames):
            array = pixels.numpy()
            if not np.isfinite(array).all():
                raise RuntimeError('The model returned invalid video pixels. Retry the video.')
            frame = av.VideoFrame.from_ndarray((array.clip(0, 1) * 255).round().astype(np.uint8), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        if audio_stream:
            for offset in range(0, waveform.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform[:, offset:offset + 1024]), format='fltp', layout=layout)
                frame.sample_rate = rate
                frame.pts, frame.time_base = offset, Fraction(1, rate)
                for packet in audio_stream.encode(frame):
                    output.mux(packet)
            for packet in audio_stream.encode():
                output.mux(packet)


def run(request, job, directory, update):
    import torch
    from PIL import Image
    from .models import entry
    from .video_engine import generate
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Check the graphics driver and restart Qwen.')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(8)
    images = {}
    for name in ('first_frame', 'last_frame'):
        if request.get(name):
            with Image.open(resolve_image(request[name])) as source:
                images[name] = source.convert('RGB')
                images[name].save(directory / f'{name}.png')
    prompt = request['prompt']
    if request['enhance']:
        from .prompt_enhancer import enhance_video
        prompt = enhance_video(prompt, images, request, update)
    frames, audio = generate(request, prompt, images, update)
    if len(frames) != request['frames']:
        raise RuntimeError(f'The decoder returned {len(frames)} frames; expected {request["frames"]}.')
    update(stage='saving', message='Saving MP4…')
    destination = Path(request['output_dir']) / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '.mp4')
    save_mp4(destination, frames, audio)
    poster = directory / 'poster.png'
    Image.fromarray((frames[0].numpy().clip(0, 1) * 255).round().astype('uint8')).save(poster)
    update(status='succeeded', stage='complete', message='Video saved.', output_path=str(destination),
        video_url=f'/api/jobs/{job["id"]}/video', poster_url=f'/api/jobs/{job["id"]}/poster',
        finished_at=time.time(), elapsed_seconds=round(time.time() - job['started_at'], 1),
        duration_seconds=round(len(frames) / 24, 3), fps=24, frames=len(frames),
        has_audio=audio is not None, audio_sample_rate=audio['sample_rate'] if audio else None,
        actual_prompt=prompt, model_revision=entry(request['model_id'])['revision'])
    write_json(destination.with_suffix('.json'), job)

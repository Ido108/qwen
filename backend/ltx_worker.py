"""LTX multi-reference media preparation and common MP4 output."""
from pathlib import Path
import time
from .common import resolve_image, resolve_audio, write_json


def run(request, job, directory, update):
    import torch
    import av
    from PIL import Image
    from .ltx_settings import MultiReferenceRequest, required_ltx
    from .ltx_engine import generate
    from .video_worker import save_mp4
    from .reference_media import read_audio
    from .models import entry
    settings = MultiReferenceRequest(**request)
    required_ltx(settings)
    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA GPU is required. Use RunPod on macOS.')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(8)
    images, voices = {}, {}
    update(stage='reading_references', message='Preparing reference images and voices...')
    slots = [(f'pic{i+1}', value) for i, value in enumerate(settings.reference_images)]
    if settings.background:
        slots.append(('background', settings.background))
    for name, value in slots:
        with Image.open(resolve_image(value)) as source:
            images[name] = source.convert('RGB')
        images[name].save(directory / (name + '.png'))
    for index, value in enumerate(settings.audio_references):
        if not value:
            continue
        path = resolve_audio(value)
        with av.open(str(path)) as media:
            stream = next(s for s in media.streams if s.type == 'audio')
            duration = float(stream.duration * stream.time_base) if stream.duration else float(media.duration or 5e6) / 1e6
        duration = min(5.0, duration)
        audio = read_audio(path, duration, end_seconds=duration)
        if audio is None:
            raise ValueError('This voice reference has no readable audio.')
        voices[f'audio_ref{index+1}'] = audio
    frames, audio = generate(settings.model_dump(), images, voices, update, directory)
    if len(frames) != settings.frames:
        raise RuntimeError(f'The decoder returned {len(frames)} frames; expected {settings.frames}.')
    update(stage='saving', message='Saving MP4...')
    destination = Path(settings.output_dir) / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '.mp4')
    save_mp4(destination, frames, audio, settings.fps)
    Image.fromarray((frames[0].numpy().clip(0, 1)*255).round().astype('uint8')).save(directory/'poster.png')
    update(status='succeeded', stage='complete', message='Video saved.', output_path=str(destination),
        video_url=f'/api/jobs/{job["id"]}/video', poster_url=f'/api/jobs/{job["id"]}/poster',
        finished_at=time.time(), elapsed_seconds=round(time.time()-job['started_at'], 1),
        duration_seconds=round(len(frames)/settings.fps, 3), fps=settings.fps, frames=len(frames),
        width=int(frames.shape[2]), height=int(frames.shape[1]), has_audio=audio is not None,
        actual_prompt=settings.prompt, model_revision=entry(settings.model_id)['revision'])
    write_json(destination.with_suffix('.json'), job)

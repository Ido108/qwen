"""Reference images media preparation (LTX 2.5 MSR or MiniMax H3 Ref2VA) and common MP4 output."""
from pathlib import Path
import time
from .common import resolve_image, resolve_audio, write_json


def run(request, job, directory, update):
    import torch
    import av
    from PIL import Image
    from .ltx_settings import MultiReferenceRequest, required_ltx, is_ref2va
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
    from .lora_prompts import internal_prompt
    settings.prompt, lora_words = internal_prompt(settings.prompt, settings.model_dump())
    update(lora_prompt_words=lora_words)
    if is_ref2va(entry(settings.model_id)):
        frames, audio, prompt = run_ref2va(settings, images, voices, update, directory)
    else:
        from .ltx_engine import generate
        frames, audio = generate(settings.model_dump(), images, voices, update, directory)
        prompt = settings.prompt
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
        actual_prompt=prompt, model_revision=entry(settings.model_id)['revision'])
    write_json(destination.with_suffix('.json'), job)


def run_ref2va(settings, images, voices, update, directory):
    """MiniMax H3 Ref2VA from reference images (and voices) only, as in the official r2v template."""
    from .ltx_settings import ref2va_prompt
    from .models import entry
    from .reference_sampling import reference_sampling
    from .video_engine import generate
    data = settings.model_dump()
    profile = reference_sampling(entry(settings.model_id), settings.quality, settings.steps)
    ordered = [*settings.reference_images, *([settings.background] if settings.background else [])]
    prompt = ref2va_prompt(data)
    work = {**data, 'kind': 'reference-video', 'workflow': 'reference', 'reference_images': ordered,
            'reference_image_size': 'match', 'quality': profile['quality'], 'steps': profile['steps']}
    pictures = [images[f'pic{i+1}'] for i in range(len(settings.reference_images))]
    if settings.background:
        pictures.append(images['background'])
    media = {f'ref_image_{i+1}': picture for i, picture in enumerate(pictures)}
    # H3 numbers standalone audio separately: <Audio 1>, <Audio 2> in slot order.
    media['ref_audios'] = {f'ref_audio_{i+1}': voice for i, (_, voice) in enumerate(sorted(voices.items()))}
    update(actual_prompt=prompt, sampling_profile=profile)
    frames, audio = generate(work, prompt, media, update, preview_directory=directory)
    return frames, (audio if settings.audio else None), prompt

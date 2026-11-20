"""Animate image, Lip sync, Extract audio and Face restore jobs in the single-use worker (see portrait_engine)."""
from __future__ import annotations

import time
from pathlib import Path

from .common import resolve_audio, resolve_image, resolve_video, write_json
from . import portrait_engine
from .portrait_settings import AUDIO_KINDS

SUFFIX = {'animate': 'animated', 'lipsync': 'lipsync', 'face-restore': 'face-restored', 'extract-audio': 'audio'}


def audio_input(engine, request):
    mode = request.get('audio_mode')
    if mode == 'file' and request.get('audio_file'):
        return engine.upload('audio', resolve_audio(request['audio_file']))
    if mode == 'video' and request.get('audio_video'):
        return engine.upload('audio', resolve_video(request['audio_video']))
    if mode in ('file', 'video'):
        raise ValueError('Select the audio for this job.')
    return None


def fields(kind, request, engine, update):
    update(stage='uploading', message='Preparing the inputs…')
    if kind == 'animate':
        values = {**request.get('options', {}), 'action': 'preview' if request.get('preview') else 'render',
                  'source': engine.upload('source', resolve_image(request['source_image'])),
                  'driving': engine.upload('driving', resolve_video(request['driving_video']))}
        audio = audio_input(engine, request)
        if audio:
            values['audio'] = audio
        return values
    driving = engine.upload('driving', resolve_video(request['source_video']))
    if kind == 'lipsync':
        values = {**request.get('options', {}), 'action': 'sync', 'driving': driving}
        audio = audio_input(engine, request)
        if audio:
            values['audio'] = audio
        return values
    clip = {key: request[key] for key in ('start', 'duration') if request.get(key) is not None}
    if kind == 'extract-audio':
        return {'action': 'extract', 'driving': driving, 'audio_format': request['audio_format'], **clip}
    return {'action': 'enhance', 'driving': driving, 'enhance_blend': request['enhance_blend'],
            'enhance_scale': request['enhance_scale'], **clip}


def run(request, job, directory, update):
    kind = request['kind']
    portrait_engine.ensure(update)
    with portrait_engine.Engine(directory) as engine:
        result = engine.run(fields(kind, request, engine, update), update)
        output = engine.output(result)
        update(stage='saving', message='Saving the result…', step=0, total_steps=0)
        name = time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '_' + SUFFIX[kind] + output.suffix
        destination = portrait_engine.move_output(output, Path(request['output_dir']) / name)
    details = {'engine_stats': result.get('stats') or {}, 'engine_version': portrait_engine.ENGINE_COMMIT[:12]}
    if kind in AUDIO_KINDS:
        finish_audio(job, directory, destination, update, details)
    else:
        finish_video(job, directory, destination, update, details)
    write_json(destination.with_suffix('.json'), job)


def finish_audio(job, directory, destination, update, details):
    import av
    with av.open(str(destination)) as media:
        stream = media.streams.audio[0]
        duration = media.duration / 1_000_000 if media.duration else float(stream.duration * stream.time_base or 0)
        sample_rate, channels = stream.rate, stream.channels
    update(status='succeeded', stage='complete', message='Audio saved.', output_path=str(destination),
           audio_url=f'/api/jobs/{job["id"]}/audio', finished_at=time.time(),
           elapsed_seconds=round(time.time() - job['started_at'], 1), duration_seconds=round(duration, 3),
           sample_rate=sample_rate, channels=channels, audio_format=destination.suffix[1:], **details)


def finish_video(job, directory, destination, update, details):
    import av
    from PIL import Image
    with av.open(str(destination)) as media:
        video = media.streams.video[0]
        frame = next(media.decode(video))
        width, height = frame.width, frame.height
        fps = float(video.average_rate) if video.average_rate else 25.0
        frames = int(video.frames or 0)
        duration = media.duration / 1_000_000 if media.duration else (frames / fps if frames else 0)
        has_audio = bool(media.streams.audio)
        poster = frame.to_image().convert('RGB')
        poster.thumbnail((960, 540), Image.Resampling.LANCZOS)
        poster.save(directory / 'poster.png')
    (directory / 'live-preview.webp').unlink(missing_ok=True)
    update(status='succeeded', stage='complete', message='Video saved.', output_path=str(destination),
           video_url=f'/api/jobs/{job["id"]}/video', poster_url=f'/api/jobs/{job["id"]}/poster', preview_url=None,
           finished_at=time.time(), elapsed_seconds=round(time.time() - job['started_at'], 1),
           width=width, height=height, fps=fps, frames=frames, duration_seconds=round(duration, 3),
           has_audio=has_audio, **details)

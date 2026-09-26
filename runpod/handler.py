"""RunPod Serverless worker for the exact Qwen image and video engines."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('QWEN_DATA_DIR', '/runpod-volume/qwen-image-studio')
from backend.common import APP_DIR, JOBS_DIR, UPLOADS_DIR, VIDEO_UPLOADS_DIR, ReferenceVideoRequest, UpscaleRequest, read_json, write_json  # noqa: E402
from backend import models  # noqa: E402
from backend.common import AUDIO_UPLOADS_DIR
from backend.ltx_settings import MultiReferenceRequest, required_ltx, media_inputs as ltx_media_inputs
from backend.resident_worker import ImageWorker  # noqa: E402

VOLUME = Path('/runpod-volume')
IMAGE_WORKER = ImageWorker()


def checked_path(key: str, prefix: str) -> Path:
    if not isinstance(key, str) or not key.startswith(prefix) or '..' in Path(key).parts:
        raise ValueError('Invalid network volume file key.')
    result = (VOLUME / key).resolve()
    if not result.is_relative_to(VOLUME.resolve()):
        raise ValueError('File is outside the network volume.')
    return result


def ensure_models(request: dict, kind: str, progress=lambda _: None) -> None:
    if kind == 'image':
        needed = models.required(request['model_id'], request.get('text_encoder', 'auto'))
    elif kind == 'video':
        needed = models.required_video(type('VideoSettings', (), request))
    elif kind == 'reference-video':
        needed = models.required_reference(ReferenceVideoRequest(**request))
    elif kind == 'multi-reference':
        needed = required_ltx(MultiReferenceRequest(**request))
    elif kind == 'upscale':
        needed = models.required_upscale(UpscaleRequest(**request))
    else:
        raise ValueError('Invalid generation type.')
    lock_path = VOLUME / 'qwen-image-studio' / '.model-download.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            last_update = 0.0
            started = time.time()
            def downloading(state):
                nonlocal last_update
                now = time.monotonic()
                if now-last_update < 1:
                    return
                last_update = now
                stage = state.get('stage')
                message = ('Checking model access' if stage == 'checking_access' else
                           'Checking model files' if stage == 'verifying' else 'Downloading required model files')
                progress({'stage': 'downloading_models', 'stage_started_at': started,
                          'message': message, 'download_completed_bytes': state.get('completed_bytes', 0),
                          'download_total_bytes': state.get('total_bytes', 0)})
            models.ensure_installed(needed, owner='runpod', progress=downloading)
            if not all(models.is_installed(item) for item in needed):
                raise RuntimeError('The requested model or a supporting file is missing after download.')
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def handler(job: dict) -> dict:
    payload = job.get('input') or {}
    job_id = payload.get('job_id', '')
    kind = payload.get('kind')
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Invalid job ID.')
    if kind not in ('image', 'video', 'reference-video', 'multi-reference', 'upscale'):
        raise ValueError('Invalid generation type.')
    prefix = f'qwen-transfer/{job_id}/'
    if payload.get('output_prefix') != prefix:
        raise ValueError('Invalid output folder.')
    request = dict(payload.get('request') or {})
    if request.get('model_id') is None or request.get('enhance'):
        raise ValueError('The model or prompt preparation is invalid.')
    directory = JOBS_DIR / job_id
    directory.mkdir(parents=True, exist_ok=True)
    output_dir = checked_path(prefix, prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    request['output_dir'] = str(output_dir)
    input_keys = payload.get('input_keys') or []
    if kind == 'image':
        expected = [('image', value) for value in request.get('images', [])]
    elif kind == 'video':
        expected = [('image', value) for value in (request.get('first_frame'), request.get('last_frame')) if value]
    elif kind == 'multi-reference':
        expected = ltx_media_inputs(request)
    elif kind == 'reference-video':
        expected = [('video', request['source_video'])]
        if request.get('warp_video'):
            expected.append(('video', request['warp_video']))
        expected.extend(('image', value) for value in request.get('reference_images', []))
    else:
        expected = [('video', request['source_video'])]
    if len(input_keys) != len(expected) or len(input_keys) > 12:
        raise ValueError('Input file count does not match the request.')
    for index, key in enumerate(input_keys):
        source = checked_path(key, prefix)
        media, name = expected[index]
        root = VIDEO_UPLOADS_DIR if media == 'video' else AUDIO_UPLOADS_DIR if media == 'audio' else UPLOADS_DIR
        target = (root / name).resolve()
        suffixes = {'.mp4', '.mov', '.mkv', '.webm'} if media == 'video' else {'.png'}
        if media == 'audio':
            suffixes = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        if not target.is_relative_to(root.resolve()) or target.name != name or target.suffix.lower() not in suffixes:
            raise ValueError('Invalid input file name.')
        shutil.copy2(source, target)
    import runpod
    ensure_models(request, kind, lambda message: runpod.serverless.progress_update(job, message))
    status = {'id': job_id, 'kind': kind, 'status': 'queued', 'stage': 'queued',
              'started_at': time.time(), 'request': request, 'seed': request['seed']}
    write_json(directory / 'request.json', request)
    write_json(directory / 'status.json', status)
    if kind == 'image':
        process = IMAGE_WORKER.start(directory, request)
    else:
        IMAGE_WORKER.close()
        log = (directory / 'worker.log').open('a', encoding='utf-8')
        try:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'backend.worker', str(directory)],
                                       cwd=APP_DIR, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
    last_progress = None
    while process.poll() is None:
        try:
            current = read_json(directory / 'status.json')
        except (OSError, ValueError):
            current = {}
        progress = (current.get('stage'), current.get('step'),current.get('preview_updated_at'),current.get('input_frames_done'))
        if progress != last_progress and progress[0]:
            from backend.cloud import progress_details
            details=progress_details({'output':current})
            details['message']=current.get('message',progress[0])
            preview=directory/'live-preview.webp'
            if current.get('preview_url') and preview.is_file():
                try:
                    temporary=output_dir/'preview.tmp'
                    shutil.copyfile(preview,temporary)
                    temporary.replace(output_dir/'preview.webp')
                    details['preview_key']=prefix+'preview.webp'
                except OSError:
                    pass
            runpod.serverless.progress_update(job, details)
            last_progress = progress
        if kind == 'image' and (directory / 'worker.done.json').exists():
            break
        time.sleep(1)
    code = IMAGE_WORKER.wait(directory, process) if kind == 'image' else process.wait()
    current = read_json(directory / 'status.json')
    if code or current.get('status') != 'succeeded':
        raise RuntimeError(current.get('message', f'Generation failed (code {code}).'))
    output = Path(current['output_path'])
    if not output.is_file() or not output.resolve().is_relative_to(output_dir.resolve()):
        raise RuntimeError('Generated output is missing from the network volume.')
    final_output = output_dir / ('result.mp4' if kind != 'image' else 'result.png')
    if output != final_output:
        output.replace(final_output)
    result = {'output_key': prefix + final_output.name,
              'metadata': {k: v for k, v in current.items() if k not in ('id', 'request', 'output_path', 'status')}}
    if kind != 'image':
        poster = directory / 'poster.png'
        if poster.is_file():
            shutil.copy2(poster, output_dir / 'poster.png')
            result['poster_key'] = prefix + 'poster.png'
    if kind == 'reference-video':
        intermediate = directory / 'subject-replacement.mp4'
        if intermediate.is_file():
            shutil.copy2(intermediate, output_dir / 'subject-replacement.mp4')
            result['subject_video_key'] = prefix + 'subject-replacement.mp4'
    return result


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from collections import deque
import hashlib
import httpx
import io
import json
import logging
import os
from pathlib import Path
from typing import Literal
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field

from .common import APP_DIR, DATA_DIR, EXAMPLES_DIR, JOBS_DIR, MODELS_DIR, OUTPUTS_DIR, VIDEOS_DIR, UPLOADS_DIR, VIDEO_UPLOADS_DIR, GenerationRequest, VideoRequest, ReferenceVideoRequest, UpscaleRequest, model_status, resolve_image, resolve_video, read_json, write_json
from .reference_timing import probe_video_duration, resolve_reference_clip
from . import models, cloud, civitai, people, training, moise_remote
from .resident_worker import ImageWorker

from .common import AUDIO_UPLOADS_DIR, resolve_audio
from .ltx_settings import MultiReferenceRequest, required_ltx, is_ref2va, media_inputs as ltx_media_inputs
from . import image_upscale
from .image_upscale import ImageUpscaleRequest

SESSION_TOKEN = secrets.token_urlsafe(32)
register_heif_opener(thumbnails=False)
active_lock = threading.Lock()
active_job: str | None = None
active_process: subprocess.Popen | None = None
active_remote: tuple[str, str] | None = None
cancelled_jobs: set[str] = set()
pending_jobs: list[str] = []
recent_jobs: deque[str] = deque(maxlen=20)
queue_paused = False
shutting_down = False
# Do not reload yesterday's model just because Windows starts the server.
# A new local image job enables reuse and the existing post-video warm-up.
image_warmup_suspended = True
IDLE_MODEL_RELEASE_SECONDS = 10 * 60
image_idle_since = time.monotonic()
image_worker = ImageWorker()
VIDEO_KINDS={'video','reference-video','multi-reference','upscale'}
# Image results: generated images and upscaled images. An image upscale runs in its own process, like a video job.
IMAGE_KINDS={'image','image-upscale'}
# Completed job IDs never change their output. Keep private images in this browser,
# not in shared proxies. Mutable API responses and failed reads remain no-store.
RESULT_CACHE = {'Cache-Control': 'private, max-age=604800, immutable'}
display_locks = [threading.Lock() for _ in range(16)]


def start_next_locked():
    """Caller holds active_lock. Advance only after the previous job finishes."""
    global active_job
    if active_job or queue_paused or shutting_down:
        return
    while pending_jobs:
        job_id = pending_jobs.pop(0)
        if read_job(job_id)['status'] != 'queued':
            continue
        active_job = job_id
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
        return
    warm_image_worker_locked()


def warm_image_worker_locked():
    """Caller holds active_lock. Restore the recent image model after a video job."""
    if image_warmup_suspended or active_job or pending_jobs or queue_paused or shutting_down or cloud.settings()['mode'] != 'local':
        return
    if time.monotonic() >= image_idle_since + IDLE_MODEL_RELEASE_SECONDS:
        return
    if os.environ.get('QWEN_NO_WARMUP') or image_worker.status()['loaded']:
        return
    last = next((job for job in sorted(all_jobs(), key=lambda j: j['created_at'], reverse=True)
                 if job.get('kind', 'image') == 'image' and job.get('status') == 'succeeded' and job.get('request', {}).get('execution') != 'runpod'), None)
    if not last:
        return
    try:
        request = json.loads((JOBS_DIR / last['id'] / 'request.json').read_text(encoding='utf-8'))
        if all(models.is_installed(model_id) for model_id in models.required_image(request)[:3]):
            image_worker.warm(request, DATA_DIR / 'temp')
    except (OSError, ValueError, KeyError):
        pass  # the next job loads normally


def release_idle_image_worker():
    """Use the scheduler lock so expiry and a new job cannot overlap.

    Closing the process releases CPU models, conditioning caches and its CUDA
    context. Moving weights to CPU or emptying the CUDA cache would retain RAM.
    Polling the UI and queue controls never extend the idle deadline.
    """
    global image_warmup_suspended
    with active_lock:
        if shutting_down or active_job or pending_jobs or active_process is not None:
            return False
        if time.monotonic() < image_idle_since + IDLE_MODEL_RELEASE_SECONDS:
            return False
        loaded = image_worker.status()['loaded']
        # Also suspend when there is no worker: resuming an empty queue must
        # not reload an old model after the deadline.
        image_warmup_suspended = True
        if not loaded:
            return False
        image_worker.close()
        print('[Memory] Image model released after 10 minutes without jobs. '
              'The next image job will load it again.', flush=True)
        return True


def watch_idle_image_worker(stopped):
    """One low-cost timer per server. No model imports or disk scans."""
    while not stopped.wait(1):
        try:
            release_idle_image_worker()
        except Exception:
            logging.exception('Could not release the idle image model; retrying in 10 seconds.')
            if stopped.wait(10):
                break


def save_queue_locked():
    write_json(DATA_DIR / 'queue-state.json', {'paused': queue_paused, 'pending': pending_jobs})


def enqueue_locked(job, data):
    from .lora_preferences import snapshot
    snapshot(data)
    directory = JOBS_DIR / job['id']
    directory.mkdir()
    write_json(directory / 'request.json', data)
    write_json(directory / 'status.json', job)
    pending_jobs.append(job['id'])
    save_queue_locked()
    start_next_locked()


def all_jobs():
    result = []
    for path in JOBS_DIR.glob('*/status.json'):
        try:
            result.append(read_json(path))
        except (ValueError, OSError):
            continue
    return result

def stop_worker(process):
    """Windows venv launchers spawn a child interpreter; stop the owned tree."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW,check=False)
    else:
        process.terminate()

def read_job(job_id):
    if not job_id or any(c not in '0123456789abcdef' for c in job_id) or len(job_id) != 32:
        raise HTTPException(404, 'Generation not found.')
    try:
        return read_json(JOBS_DIR / job_id / 'status.json')
    except FileNotFoundError:
        raise HTTPException(404, 'Generation not found.')

def public_job(job):
    public = {k: v for k, v in job.items() if k not in ('output_path',)}
    if job.get('status') == 'succeeded' and job.get('kind') not in ('training', 'captioning'):
        public['thumbnail_url'] = f'/api/jobs/{job["id"]}/thumbnail'
    return public


def run_cloud_job(job_id: str, request: dict, kind: str) -> None:
    """Keep queue, gallery and output files local while the GPU runs on RunPod."""
    global active_remote
    directory = JOBS_DIR / job_id
    job = read_job(job_id)
    remote_id = job.get('remote_id')
    work = dict(request)
    task_kind = request.get('kind', kind)
    if remote_id:
        try:
            work = read_json(directory / 'remote-request.json')
        except (OSError, ValueError):
            pass
    if not remote_id:
        job.update(status='running', stage='uploading', message='Sending inputs to RunPod', started_at=time.time())
        write_json(directory / 'status.json', job)
        if task_kind == 'image':
            input_ids = [('image', value) for value in request.get('images', [])]
        elif task_kind == 'video':
            input_ids = [('image', value) for value in (request.get('first_frame'), request.get('last_frame')) if value]
        elif task_kind == 'multi-reference':
            input_ids = ltx_media_inputs(request)
        elif task_kind == 'reference-video':
            input_ids = [('video', request['source_video'])]
            if request.get('warp_video'):
                input_ids.append(('video', request['warp_video']))
            input_ids.extend(('image', value) for value in request.get('reference_images', []))
        elif task_kind == 'upscale':
            input_ids = [('video', request['source_video'])]
        else:
            raise ValueError('Unknown generation type.')
        source_paths = [resolve_video(value) if media == 'video' else resolve_audio(value) if media == 'audio' else resolve_image(value)
                        for media, value in input_ids]
        if request.get('enhance'):
            pictures = []
            if task_kind in ('image', 'video'):
                for path in source_paths:
                    with Image.open(path) as opened:
                        pictures.append(opened.convert('RGBA' if kind == 'image' else 'RGB'))
            def enhancement_progress(**values):
                current = read_job(job_id)
                current.update(values)
                write_json(directory / 'status.json', current)
            if kind == 'image':
                # RunPod jobs rewrite here, on this computer, before upload; the text encoder route needs ComfyUI, which
                # this process does not load, so it uses the Qwen PE rewriter with the same official prompt instead.
                from .enhancer_settings import route
                from .prompt_enhancer import enhance_local
                item = models.entry(request['model_id'])
                chosen = route(item.get('family', 'qwen-image-2.1'), pictures, models.is_zimage(item))
                rewrite = enhance_local(request['prompt'], pictures, enhancement_progress, family=item.get('family', 'qwen-image-2.1'),
                                        redraw=models.is_zimage(item), seed=request.get('seed', 0), spec=chosen['spec'],
                                        runner='llama', fast=chosen.get('fast', False)) if chosen['mode'] == 'local' else {}
                work['prompt'] = rewrite.get('positive_prompt') or request['prompt']
                work['enhancement_result'] = rewrite
            elif task_kind == 'video':
                from .prompt_enhancer import enhance_video
                if models.is_ltx_video(models.entry(request['model_id'])):
                    from .video_worker import enhance_ltx_video as enhance_video
                named = {}
                if request.get('first_frame'):
                    named['first_frame'] = pictures[0]
                if request.get('last_frame'):
                    named['last_frame'] = pictures[-1]
                work['prompt'] = enhance_video(request['prompt'], named, request, enhancement_progress)
            elif task_kind == 'reference-video':
                from .prompt_enhancer import enhance_reference
                from .reference_prompt import reference_instructions
                named = {}
                for index, value in enumerate(request.get('reference_images', []), 1):
                    with Image.open(resolve_image(value)) as opened:
                        named[f'ref_image_{index}'] = opened.convert('RGB')
                camera_request = {**request, 'reference_images': []} if request['workflow'] == 'crossview' else request
                tagged_prompt = reference_instructions(request)
                work['prompt'] = enhance_reference(tagged_prompt, named, camera_request, enhancement_progress) if tagged_prompt else ''
                if request['workflow'] == 'crossview' and request.get('replace_subjects_first'):
                    replacement_request = {**request, 'workflow': 'faceswap'}
                    tagged_replacement = reference_instructions(request,replacement=True)
                    work['replacement_prompt'] = enhance_reference(tagged_replacement, named,
                        replacement_request, enhancement_progress) if tagged_replacement else ''
            work['enhance'] = False
        keys = []
        for index, path in enumerate(source_paths):
            key = f'qwen-transfer/{job_id}/input-{index}{path.suffix.lower()}'
            cloud.upload(path, key)
            keys.append(key)
        if task_kind == 'image':
            work['images'] = [f'cloud-{index}.png' for index in range(len(keys))]
        elif task_kind == 'video':
            iterator = iter(f'cloud-{index}.png' for index in range(len(keys)))
            work['first_frame'] = next(iterator) if request.get('first_frame') else None
            work['last_frame'] = next(iterator) if request.get('last_frame') else None
        elif task_kind == 'reference-video':
            old_images = request.get('reference_images', [])
            work['reference_images'] = [f'cloud-ref-{index}.png' for index in range(len(old_images))]
            image_map=dict(zip(old_images,work['reference_images']))
            if work.get('reference_people'):
                work['reference_people']=[{**person,'images':[image_map[image] for image in person['images']]} for person in work['reference_people']]
            if work.get('person_assignments'):
                work['person_assignments']={person:(image_map.get(image,image) if image else None)
                                            for person,image in work['person_assignments'].items()}
        work['output_dir'] = '/runpod-volume/qwen-transfer/' + job_id
        write_json(directory / 'remote-request.json', work)
        payload = {'job_id': job_id, 'kind': task_kind, 'request': work, 'input_keys': keys,
                   'output_prefix': f'qwen-transfer/{job_id}/'}
        if job_id in cancelled_jobs:
            current = read_job(job_id)
            current.update(status='cancelled', stage='cancelled', message='Generation cancelled.', finished_at=time.time())
            write_json(directory / 'status.json', current)
            return
        remote_id = cloud.submit(kind, payload)
    with active_lock:
        active_remote = (job_id, remote_id)
    job = read_job(job_id)
    job.update(status='running', stage='waiting', message='Waiting for RunPod GPU', remote_id=remote_id)
    write_json(directory / 'status.json', job)
    def update_remote(state, result):
        current = read_job(job_id)
        if current.get('status') == 'cancelled':
            return
        current.update(stage='processing' if state == 'IN_PROGRESS' else 'waiting',
                       message=cloud.progress_message(result) or ('Generating on RunPod' if state == 'IN_PROGRESS' else 'Waiting for RunPod GPU'))
        details=cloud.progress_details(result)
        preview_key=details.pop('preview_key',None)
        revision=details.pop('preview_updated_at',None)
        if state=='IN_PROGRESS':
            current.update(details)
        if (preview_key==f'qwen-transfer/{job_id}/preview.webp' and revision
                and revision!=current.get('preview_updated_at')):
            temporary=directory/'remote-preview.tmp'
            try:
                cloud.download(preview_key,temporary)
                with Image.open(temporary) as preview:
                    if preview.format!='WEBP' or max(preview.size)>1024:
                        raise ValueError('Invalid preview')
                temporary.replace(directory/'live-preview.webp')
                current.update(preview_url=f'/api/jobs/{job_id}/live-preview',preview_updated_at=revision)
            except Exception:
                temporary.unlink(missing_ok=True)
                # Preview transport must never fail the paid generation.
                pass
        write_json(directory / 'status.json', current)
    try:
        result = cloud.poll(kind, remote_id, update_remote, lambda: job_id in cancelled_jobs)
    except RuntimeError as exc:
        if '(404)' not in str(exc):
            raise
        result = {'output_key': f'qwen-transfer/{job_id}/result.' + ('mp4' if kind == 'video' else 'png')}
    output_key = result.get('output_key', '')
    expected_prefix = f'qwen-transfer/{job_id}/'
    if not output_key.startswith(expected_prefix) or '/' in output_key[len(expected_prefix):]:
        raise RuntimeError('RunPod returned an unexpected output file.')
    suffix = '.mp4' if kind == 'video' else '.png'
    if not output_key.endswith(suffix):
        raise RuntimeError('RunPod returned the wrong media format.')
    destination = Path(request['output_dir']) / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job_id[:8] + suffix)
    current = read_job(job_id)
    current.update(stage='downloading', message='Saving result on this computer')
    write_json(directory / 'status.json', current)
    cloud.download(output_key, destination)
    if kind == 'video' and result.get('poster_key'):
        poster_key = result['poster_key']
        if not poster_key.startswith(expected_prefix):
            raise RuntimeError('RunPod returned an unexpected poster file.')
        cloud.download(poster_key, directory / 'poster.png')
    if task_kind == 'reference-video' and result.get('subject_video_key'):
        subject_key = result['subject_video_key']
        if not subject_key.startswith(expected_prefix) or '/' in subject_key[len(expected_prefix):]:
            raise RuntimeError('RunPod returned an unexpected intermediate video file.')
        cloud.download(subject_key, directory / 'subject-replacement.mp4')
    current = read_job(job_id)
    metadata = result.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    safe = {k: v for k, v in metadata.items() if k not in ('id', 'request', 'output_path', 'status', 'remote_id')}
    current.update(safe)
    current.update(status='succeeded', stage='complete', message='Video saved.' if kind == 'video' else 'Image saved.',
                   output_path=str(destination), finished_at=time.time(),
                   elapsed_seconds=round(time.time() - current['started_at'], 1),
                   video_url=f'/api/jobs/{job_id}/video' if kind == 'video' else None,
                   poster_url=f'/api/jobs/{job_id}/poster' if kind == 'video' else None,
                   image_url=f'/api/jobs/{job_id}/image' if kind == 'image' else None,
                   rewritten_prompt=current.get('rewritten_prompt') or (
                       work.get('prompt') if work.get('prompt') != request.get('prompt') else ''))
    write_json(directory / 'status.json', current)

def run_job(job_id):
    global active_job, active_process, active_remote, image_warmup_suspended, image_idle_since
    directory = JOBS_DIR / job_id
    try:
        request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
        is_video = request.get('kind') in VIDEO_KINDS
        separate = is_video or request.get('kind') == 'image-upscale'
        if request.get('kind') in ('training', 'captioning'):
            run_training(job_id, request, directory)
            return
        if request.get('execution') == 'runpod':
            run_cloud_job(job_id, request, 'video' if is_video else 'image')
            return
        prepare_job_models(job_id,request)
        with (directory / 'worker.log').open('a', encoding='utf-8') as log:
            with active_lock:
                if job_id in cancelled_jobs:
                    job = read_job(job_id)
                    job.update(status='cancelled', message='Generation cancelled.', finished_at=time.time())
                    write_json(directory / 'status.json', job)
                    return
                if separate:
                    image_worker.close()
                    active_process = subprocess.Popen(
                        [sys.executable, '-X', 'faulthandler', '-u', '-m', 'backend.worker', str(directory)],
                        cwd=APP_DIR, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    )
                else:
                    active_process = image_worker.start(directory, request)
                    # A new local image explicitly requests the model again.
                    image_warmup_suspended = False
                process = active_process
            return_code = process.wait() if separate else image_worker.wait(directory, process)
        with active_lock:
            job = read_job(job_id)
            if job_id in cancelled_jobs:
                job.update(status='cancelled', message='Generation cancelled.', finished_at=time.time())
                write_json(directory / 'status.json', job)
            elif job.get('status') not in ('succeeded', 'failed'):
                job.update(status='failed', message=f'The generation process stopped (code {return_code}). You can retry this job from the queue.', finished_at=time.time())
                write_json(directory / 'status.json', job)
    except Exception as exc:
        job = read_job(job_id)
        if isinstance(exc,models.DownloadCancelled) or job_id in cancelled_jobs:
            job.update(status='cancelled',stage='cancelled',message=str(exc) or 'Generation cancelled.',finished_at=time.time())
            write_json(directory/'status.json',job)
        elif job.get('status') != 'cancelled':
            job.update(status='failed', message=str(exc), finished_at=time.time())
            write_json(directory / 'status.json', job)
    finally:
        with active_lock:
            active_job = None
            active_process = None
            active_remote = None
            image_idle_since = time.monotonic()
            cancelled_jobs.discard(job_id)
            recent_jobs.appendleft(job_id)
            save_queue_locked()
            start_next_locked()


def run_training(job_id, request, directory):
    """A LoRA training: the GPU is free of image models first, and Cancel stops the trainer process."""
    global active_process
    with active_lock:
        if job_id in cancelled_jobs:
            job = read_job(job_id)
            job.update(status='cancelled', message='Captions cancelled.' if request.get('kind') == 'captioning' else 'Training cancelled.', finished_at=time.time())
            write_json(directory / 'status.json', job)
            return
        image_worker.close()

    def started(process):
        global active_process
        with active_lock:
            active_process = process
            if job_id in cancelled_jobs:
                stop_worker(process)
    cancelled = lambda: job_id in cancelled_jobs or shutting_down
    if request.get('kind') == 'captioning':
        from . import training_captions
        def update(**values):
            job = read_job(job_id)
            job.update(values)
            write_json(directory / 'status.json', job)
        began = time.time()
        update(status='running', started_at=began, stage='captioning', message='Preparing captions...')
        results = training_captions.run(request, directory, started, cancelled, update)
        if not cancelled():
            failed = sum(not results.get(i['id'], {}).get('caption') for i in request['images'])
            update(status='failed' if failed else 'succeeded', stage='failed' if failed else 'complete',
                   message=f'{failed} captions need a retry.' if failed else 'Captions ready.',
                   finished_at=time.time(), elapsed_seconds=round(time.time()-began, 1),
                   step=len(request['images'])-failed, total_steps=len(request['images']))
    else:
        training.run(job_id, request, directory, started, cancelled)
    with active_lock:
        if job_id in cancelled_jobs:
            job = read_job(job_id)
            job.update(status='cancelled', stage='cancelled', message='Captions stopped. Completed captions remain available.' if request.get('kind') == 'captioning' else 'Training stopped. Start it again with the same name to continue from the last saved step.',
                       finished_at=time.time())
            write_json(directory / 'status.json', job)


def prepare_job_models(job_id,request):
    kind=request.get('kind','image')
    if kind=='video':
        needed=models.required_video(VideoRequest(**request))
    elif kind=='multi-reference':
        needed=required_ltx(MultiReferenceRequest(**request))
    elif kind=='reference-video':
        needed=models.required_reference(ReferenceVideoRequest(**request))
    elif kind=='upscale':
        needed=models.required_upscale(UpscaleRequest(**request))
    elif kind=='image-upscale':
        needed=image_upscale.required(request)
    else:
        needed=models.required_image(request)
    def cancelled():
        return job_id in cancelled_jobs or shutting_down
    def progress(state):
        if cancelled():
            return
        job=read_job(job_id)
        stage=state.get('stage','preparing')
        title=state.get('title') or models.entry(state.get('model_id',request['model_id']))['title']
        message=f'Checking model access: {title}' if stage=='checking_access' else state.get('message') if stage=='waiting_for_download' else (
            f'Checking model files: {title}' if stage=='verifying' else f'Downloading model files: {title}')
        if job.get('stage')!='downloading_models':
            job['stage_started_at']=time.time()
        job.update(status='running',stage='downloading_models',message=message,
                   download_completed_bytes=state.get('completed_bytes',0),download_total_bytes=state.get('total_bytes',0))
        job.setdefault('started_at',time.time())
        write_json(JOBS_DIR/job_id/'status.json',job)
    models.ensure_installed(needed,owner=job_id,cancelled=cancelled,progress=progress)

@asynccontextmanager
async def lifespan(app):
    global queue_paused, shutting_down, active_job, active_process, image_warmup_suspended, image_idle_since
    with active_lock:
        shutting_down = False
        image_warmup_suspended = True
        image_idle_since = time.monotonic()
        active_job = None
        active_process = None
        pending_jobs.clear()
        recent_jobs.clear()
        try:
            saved = json.loads((DATA_DIR / 'queue-state.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            saved = {}
        queue_paused = bool(saved.get('paused', False))
        for job in sorted(all_jobs(), key=lambda j: j['created_at']):
            if job.get('status') == 'running':
                if job.get('request', {}).get('execution') == 'runpod' and job.get('remote_id'):
                    job.update(status='queued', stage='queued', message='Reconnecting to RunPod')
                else:
                    job.update(status='failed', message='The app closed before this generation finished. Use Retry to run it again.', finished_at=time.time())
                write_json(JOBS_DIR / job['id'] / 'status.json', job)
            if job.get('status') == 'queued':
                pending_jobs.append(job['id'])
            else:
                recent_jobs.appendleft(job['id'])
        order = saved.get('pending', [])
        pending_jobs.sort(key=lambda job_id: order.index(job_id) if job_id in order else len(order))
        start_next_locked()
    # Fix the settings of Civitai models saved by older versions (for example a Base model saved as Turbo).
    threading.Thread(target=civitai.repair_entries, name='civitai-repair', daemon=True).start()
    from . import model_library
    # Models the user already keeps in the shared ComfyUI library (D:\models) join the model lists.
    threading.Thread(target=lambda: (model_library.scan(), model_library.merge_scan()), name='library-scan', daemon=True).start()
    recover_civitai_downloads()
    threading.Thread(target=civitai.warm_indexes, name='civitai-index-warm', daemon=True).start()
    # Female or male for every person in People with no label yet (once, on the CPU, never over a hand-set label).
    def preset_people_labels():
        try:
            people.preset_labels()
        except Exception:
            logging.exception('Could not preset People labels')
    threading.Thread(target=preset_people_labels, name='people-labels', daemon=True).start()
    # Remote use from the Moise dashboard (also on the phone), when this PC has the Moise connection settings.
    remote = moise_remote.start(int(os.environ['QWEN_SERVER_PORT']),
                                lambda: {'busy': active_job is not None, 'queue': len(pending_jobs)}) if os.environ.get('QWEN_SERVER_PORT', '').isdigit() else None
    idle_stop = threading.Event()
    idle_thread = threading.Thread(target=watch_idle_image_worker, args=(idle_stop,),
                                   name='image-idle-release', daemon=True)
    idle_thread.start()
    # BF16, FP16 and FP32 Qwen-Image-2.1 models get an INT8 ConvRot copy, made one at a time while the queue is idle.
    from . import int8_copies
    int8_copies.start(lambda: bool(shutting_down or active_job or pending_jobs or active_process is not None or models.transfer.get('status') == 'running'), idle_stop)
    try:
        yield
    finally:
        idle_stop.set()
        if remote:
            remote.set()
        with active_lock:
            shutting_down = True
            save_queue_locked()
            if active_process and active_process.poll() is None:
                stop_worker(active_process)
                active_process.wait(timeout=30)
            image_worker.close()
        idle_thread.join(timeout=5)

app = FastAPI(title='Qwen', lifespan=lifespan)

@app.middleware('http')
async def local_only(request: Request, call_next):
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('localhost', '127.0.0.1', 'testserver'):
        return JSONResponse({'detail': 'Local access only.'}, status_code=403)
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        origin = request.headers.get('origin')
        if origin and origin not in (f'http://{request.headers.get("host")}', 'http://127.0.0.1:5173'):
            return JSONResponse({'detail': 'Local access only.'}, status_code=403)
        if request.headers.get('x-studio-token') != SESSION_TOKEN:
            return JSONResponse({'detail': 'Refresh the app and try again.', 'code': 'session_expired'}, status_code=403, headers={'Cache-Control': 'no-store'})
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    if request.url.path.startswith('/api'):
        response.headers.setdefault('Cache-Control', 'no-store')
    elif request.url.path in ('/', '/index.html'):
        # Always check for a newer build; the hashed assets can stay cached.
        response.headers['Cache-Control'] = 'no-cache'
    return response

@app.get('/api/health')
def health():
    return {'app': 'image-studio-qwen21', 'version': '1.0.0', 'app_dir':str(APP_DIR), 'data_dir':str(DATA_DIR)}

@app.get('/api/this-pc')
def this_pc():
    from .moise_remote import gpu_info, worker_name
    info = gpu_info()
    return {'name': worker_name(), 'gpu': info.get('gpu', ''), 'vram_total_mb': info.get('vramTotalMb', 0)}

@app.get('/api/gpu-memory')
def gpu_memory_status():
    from .gpu_memory import dedicated_memory_snapshot
    return {**dedicated_memory_snapshot(), 'resident_image': image_worker.status(), 'job_active': bool(active_job),
            'idle_release_seconds': IDLE_MODEL_RELEASE_SECONDS}

@app.get('/api/config')
def config():
    return {'token': SESSION_TOKEN, 'models': model_status(), 'catalog': catalog_state(), 'enhancement': None, 'cloud': cloud.status(), 'output_dir': str(OUTPUTS_DIR), 'video_output_dir':str(VIDEOS_DIR), 'active_job': active_job, 'model': 'Qwen-Image-2.1', 'device': 'RunPod GPU' if cloud.settings()['mode'] == 'runpod' else 'Local NVIDIA GPU', 'data_dir': str(DATA_DIR)}


@app.get('/api/session')
def session():
    return {'token': SESSION_TOKEN}


class CloudInput(BaseModel):
    mode: Literal['local', 'runpod']
    image_endpoint: str = ''
    video_endpoint: str = ''
    datacenter: str = ''
    volume_id: str = ''
    api_key: str = ''
    s3_access_key: str = ''
    s3_secret_key: str = ''


@app.get('/api/cloud')
def cloud_status():
    return cloud.status()


@app.post('/api/cloud')
def cloud_configure(body: CloudInput):
    try:
        with active_lock:
            if active_job or pending_jobs:
                raise ValueError('Finish or cancel queued jobs before changing the processing location.')
            return cloud.configure(body.model_dump())
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@app.post('/api/cloud/test')
def cloud_test():
    try:
        return cloud.test_connection()
    except Exception as exc:
        raise HTTPException(424, str(exc)[:500])


def catalog_state():
    state = models.state()
    if cloud.settings()['mode'] == 'runpod':
        for item in state['models']:
            if item['kind'] == 'reference-video' and item.get('family') != 'minimax-h3-ref2va':
                item['ready'] = False
            elif item['kind'] in ('image', 'video', 'reference-video', 'multi-reference', 'upscale'):
                item['ready'] = cloud.ready('image' if item['kind'] == 'image' else 'video')
        state['remote'] = True
    return state

class CivitaiKey(BaseModel):
    key: str = Field(min_length=10, max_length=200)


class CivitaiLink(BaseModel):
    link: str = Field(min_length=10, max_length=500)


class CivitaiInstall(BaseModel):
    version_id: int = Field(ge=1)
    file_id: int | None = None
    turbo: bool = False
    title: str = Field(default='Civitai model', max_length=200)


def civitai_call(function, *args):
    try:
        return function(*args)
    except (ValueError, ConnectionError, PermissionError) as exc:
        raise HTTPException(400, str(exc))
    except httpx.HTTPError:
        raise HTTPException(502, 'Civitai is not reachable. Check the internet connection and try again.')


@app.get('/api/civitai/status')
def civitai_status():
    return civitai.status()


@app.post('/api/civitai/login')
def civitai_login():
    return civitai_call(civitai.start_login)


@app.get('/api/civitai/login/{state}')
def civitai_login_poll(state: str):
    return civitai_call(civitai.poll_login, state)


@app.post('/api/civitai/key')
def civitai_key(body: CivitaiKey):
    return civitai_call(civitai.save_api_key, body.key)


@app.post('/api/civitai/disconnect')
def civitai_disconnect():
    return civitai.disconnect()


@app.get('/api/civitai/search')
def civitai_search(query: str = '', kind: Literal['LORA', 'Checkpoint'] = 'LORA',
                   sort: Literal['Most Downloaded', 'Highest Rated', 'Newest'] = 'Most Downloaded', cursor: str | None = None,
                   area: Literal['image', 'video'] = 'image', variant: Literal['all', 'standard', 'turbo'] = 'all',
                   arch: Literal['all', 'qwen-image-2.1', 'krea-2', 'z-image', 'flux3-klein', 'moss-image', 'minimax-h3', 'ltx25'] = 'all'):
    return civitai_call(civitai.search, query, kind, sort, cursor, True, area, variant, arch)


@app.get('/api/civitai/models/{model_id}')
def civitai_details(model_id: int):
    return civitai_call(civitai.details, model_id)


@app.post('/api/civitai/resolve')
def civitai_resolve(body: CivitaiLink):
    return civitai_call(civitai.resolve_link, body.link)


# Each Civitai file has its own target, progress, recovery request and Cancel switch.
civitai_downloads: dict[str, dict] = {}
civitai_cancels: dict[str, threading.Event] = {}
CIVITAI_PENDING_PATH = DATA_DIR / 'civitai-pending-downloads.json'


def pending_civitai_downloads():
    try:
        saved=read_json(CIVITAI_PENDING_PATH)
        return saved if isinstance(saved,dict) else {}
    except (FileNotFoundError,ValueError):
        return {}


def recover_civitai_downloads():
    for values in pending_civitai_downloads().values():
        try:
            civitai_install(CivitaiInstall(**values))
        except Exception as exc:
            print(f'[Civitai] Could not resume a saved download: {exc}',flush=True)


def civitai_download_list() -> list[dict]:
    with models.lock:
        old = [key for key, item in civitai_downloads.items() if item['status'] != 'running' and time.time() - item.get('ended_at', 0) > 600]
        for key in old:
            civitai_downloads.pop(key, None)
            civitai_cancels.pop(key, None)
        return [dict(item) for item in civitai_downloads.values()]


@app.post('/api/civitai/install')
def civitai_install(body: CivitaiInstall):
    from .civitai_files import download_key
    selected = civitai_call(civitai.download_file, body.version_id, body.file_id)
    body = body.model_copy(update={'file_id': selected['id']})
    key = download_key(body.version_id, body.file_id)
    with models.lock:
        if civitai_downloads.get(key, {}).get('status') == 'running':
            raise HTTPException(409, 'This file is already downloading.')
        label = selected['precision'] or selected['format']
        record = civitai_downloads[key] = {
            'status': 'running', 'stage': 'preparing', 'requested_id': f'civitai:{key}',
            'version_id': body.version_id, 'file_id': body.file_id, 'file_name': selected['name'],
            'precision': selected['precision'], 'format': selected['format'],
            'title': f'{body.title} · {label}', 'completed_bytes': 0, 'total_bytes': selected['bytes'],
            'started_at': time.time()}
        cancel = civitai_cancels[key] = threading.Event()
        pending=pending_civitai_downloads()
        # Replace the old version-only recovery key only for this exact request.
        legacy = pending.get(str(body.version_id))
        if legacy and legacy.get('file_id') in (None, body.file_id):
            pending.pop(str(body.version_id))
        pending[key]=body.model_dump()
        write_json(CIVITAI_PENDING_PATH,pending)

    def run():
        started, printed = time.monotonic(), [0.0]

        def progress(done, total, stage):
            if cancel.is_set():
                raise models.DownloadCancelled()
            with models.lock:
                record.update(completed_bytes=done, total_bytes=total, stage=stage)
            now = time.monotonic()
            if now - printed[0] > 5 or stage != 'downloading':
                printed[0] = now
                speed = done / max(now - started, .1) / 1e6
                print(f'[Civitai] {stage}: {done / 1e9:.2f} of {total / 1e9:.2f} GB · {speed:.0f} MB/s', flush=True)
        try:
            entry = civitai.install(body.version_id, body.file_id, body.turbo, progress)
            with models.lock:
                record.update(status='succeeded', stage='complete', title=entry['title'], model_id=entry['id'], installed_id=entry['id'], ended_at=time.time())
        except models.DownloadCancelled:
            with models.lock:
                record.update(status='cancelled', message='Download cancelled. Add it again to resume.', ended_at=time.time())
        except Exception as exc:
            with models.lock:
                record.update(status='failed', message=str(exc)[:400], ended_at=time.time())
        finally:
            with models.lock:
                pending=pending_civitai_downloads()
                pending.pop(key,None)
                write_json(CIVITAI_PENDING_PATH,pending)
    threading.Thread(target=run, name=f'civitai-{key}', daemon=True).start()
    return {'ok': True, 'download_id': f'civitai:{key}', 'file_id': body.file_id, 'download': dict(record)}


@app.post('/api/civitai/install/{version_id}/cancel')
def civitai_install_cancel(version_id: int, file_id: int | None = None):
    from .civitai_files import download_key
    with models.lock:
        keys = [key for key, item in civitai_downloads.items()
                if item.get('version_id') == version_id and item['status'] == 'running']
        if file_id is None and len(keys) > 1:
            raise HTTPException(409, 'Choose the file to cancel.')
        key = download_key(version_id, file_id) if file_id is not None else next(iter(keys), str(version_id))
        cancel = civitai_cancels.get(key)
    if cancel:
        cancel.set()
    return {'ok': True}


@app.get('/api/civitai/previews/{entry_id}')
def civitai_preview(entry_id: str):
    path = (civitai.PREVIEWS / f'{entry_id}.jpg').resolve()
    if not path.is_relative_to(civitai.PREVIEWS.resolve()) or not path.is_file():
        raise HTTPException(404, 'Preview not found.')
    return FileResponse(path, headers={'Cache-Control': 'private, max-age=86400'})


# Hugging Face and ModelScope: the same browsing and download flow as Civitai. Their downloads join the
# Civitai list, so the download indicator and the details dialog follow them the same way.
HubSource = Literal['huggingface', 'modelscope']


class HubInstall(BaseModel):
    repo: str = Field(min_length=3, max_length=200, pattern=r'^[\w.-]+/[\w.-]+$')
    path: str = Field(min_length=1, max_length=500)
    turbo: bool = False
    title: str = Field(default='Model', max_length=200)


class HubCancel(BaseModel):
    download_id: str = Field(min_length=3, max_length=800)


class HubToken(BaseModel):
    token: str = Field(min_length=10, max_length=200)


def hub_call(source, function, *args):
    try:
        return function(*args)
    except (ValueError, ConnectionError, PermissionError) as exc:
        raise HTTPException(400, str(exc))
    except httpx.HTTPError:
        raise HTTPException(502, f"{'Hugging Face' if source == 'huggingface' else 'ModelScope'} is not reachable. Check the internet connection and try again.")


@app.get('/api/hub/huggingface/status')
def hub_hf_status():
    from . import hf_hub
    return hf_hub.hf_status()


@app.post('/api/hub/huggingface/token')
def hub_hf_token(body: HubToken):
    from . import hf_hub
    return hub_call('huggingface', hf_hub.save_hf_token, body.token)


@app.post('/api/hub/huggingface/disconnect')
def hub_hf_disconnect():
    from . import hf_hub
    return hf_hub.forget_hf_token()


@app.get('/api/hub/{source}/search')
def hub_search(source: HubSource, query: str = '', kind: Literal['LORA', 'Checkpoint'] = 'LORA',
               sort: Literal['Most Downloaded', 'Highest Rated', 'Newest'] = 'Most Downloaded', cursor: str | None = None,
               area: Literal['image', 'video'] = 'image', variant: Literal['all', 'standard', 'turbo'] = 'all',
               arch: Literal['all', 'qwen-image-2.1', 'krea-2', 'z-image', 'flux3-klein', 'moss-image', 'minimax-h3', 'ltx25'] = 'all'):
    from . import hf_hub
    return hub_call(source, hf_hub.search, source, query, kind, sort, cursor, area, variant, arch)


@app.get('/api/hub/{source}/model')
def hub_details(source: HubSource, repo: str):
    from . import hf_hub
    return hub_call(source, hf_hub.details, source, repo)


@app.post('/api/hub/resolve')
def hub_resolve(body: CivitaiLink):
    from . import hf_hub
    return hub_call('huggingface', hf_hub.resolve_link, body.link)


@app.post('/api/hub/{source}/install')
def hub_install(source: HubSource, body: HubInstall):
    from . import hf_hub
    selected = hub_call(source, hf_hub.selected_file, source, body.repo, body.path)['file']
    key = f'{source}:{body.repo}:{body.path}'
    with models.lock:
        if civitai_downloads.get(key, {}).get('status') == 'running':
            raise HTTPException(409, 'This file is already downloading.')
        record = civitai_downloads[key] = {
            # A negative version id lets the download indicator's Civitai cancel route find this record too.
            'status': 'running', 'stage': 'preparing', 'requested_id': key, 'source': source, 'repo': body.repo, 'path': body.path,
            'version_id': -1 - uuid.uuid4().int % 10**9,
            'file_name': body.path.rsplit('/', 1)[-1], 'precision': selected['precision'], 'format': selected['format'],
            'title': f"{body.title} · {selected['precision'] or selected['format']}", 'completed_bytes': 0, 'total_bytes': selected['bytes'],
            'started_at': time.time()}
        cancel = civitai_cancels[key] = threading.Event()

    def run():
        def progress(done, total, stage):
            if cancel.is_set():
                raise models.DownloadCancelled()
            with models.lock:
                record.update(completed_bytes=done, total_bytes=total, stage=stage)
        try:
            entry = hf_hub.install(source, body.repo, body.path, body.turbo, progress)
            with models.lock:
                record.update(status='succeeded', stage='complete', title=entry['title'], model_id=entry['id'], installed_id=entry['id'], ended_at=time.time())
        except models.DownloadCancelled:
            with models.lock:
                record.update(status='cancelled', message='Download cancelled. Add it again to resume.', ended_at=time.time())
        except Exception as exc:
            with models.lock:
                record.update(status='failed', message=str(exc)[:400], ended_at=time.time())
    threading.Thread(target=run, name=f'hub-{source}', daemon=True).start()
    return {'ok': True, 'download_id': key, 'download': dict(record)}


@app.post('/api/hub/cancel')
def hub_cancel(body: HubCancel):
    with models.lock:
        cancel = civitai_cancels.get(body.download_id) if body.download_id.startswith(('huggingface:', 'modelscope:')) else None
    if cancel:
        cancel.set()
    return {'ok': True}


class PersonBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    image_ids: list[str] = Field(min_length=1, max_length=20)


class PersonUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    image_ids: list[str] | None = Field(default=None, max_length=20)
    label: str | None = Field(default=None, max_length=48)


class MoiseImport(BaseModel):
    password: str | None = Field(default=None, max_length=200)


def people_call(function, *args):
    try:
        for image_id in (args[1] if len(args) > 1 and isinstance(args[1], list) else []):
            resolve_image(image_id)
        return function(*args)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(400, str(exc))


@app.get('/api/people')
def list_people():
    return people.listing()


@app.post('/api/people')
def add_person(body: PersonBody):
    return people_call(people.upsert, body.name, body.image_ids)


@app.patch('/api/people/{person_id}')
def edit_person(person_id: str, body: PersonUpdate):
    if body.image_ids:
        for image_id in body.image_ids:
            people_call(resolve_image, image_id)
    return people_call(people.update, person_id, body.name, body.image_ids, body.label)


@app.delete('/api/people/{person_id}')
def remove_person(person_id: str):
    people.delete(person_id)
    return {'ok': True}


@app.get('/api/people/{person_id}/avatar')
def person_avatar(person_id: str):
    try:
        return FileResponse(people.avatar(person_id), media_type='image/webp', headers={'Cache-Control': 'private, max-age=86400'})
    except (ValueError, OSError):
        raise HTTPException(404, 'No avatar.')


@app.post('/api/people/import/moise')
def import_moise(body: MoiseImport):
    if people.status.get('state') == 'running':
        raise HTTPException(409, 'An import is already running.')
    threading.Thread(target=people.import_moise, args=(body.password,), daemon=True).start()
    return {'ok': True}


@app.get('/api/people/import/status')
def import_status():
    return dict(people.status)


@app.get('/api/models')
def get_models():
    return catalog_state() | {'resident': image_worker.status() if cloud.settings()['mode'] == 'local' else None, 'downloads': civitai_download_list()}


class LoraPromptSettings(BaseModel):
    prompt_addition: str = Field(default='', max_length=4000)


class ModelDefaultSetting(BaseModel):
    scope: str
    field: str
    value: dict | None


@app.get('/api/model-defaults')
def get_model_defaults():
    from .model_defaults import read
    return read()


@app.put('/api/model-defaults/{model_id}')
def put_model_default(model_id: str, body: ModelDefaultSetting):
    from .model_defaults import save
    try:
        return save(model_id, body.scope, body.field, body.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# The Create settings in use (model, steps, LoRAs, size...), sent by the page when they change. Moise starts its jobs
# from them, so a picture from WhatsApp is made like one from the Create button.
CREATE_SETTINGS_PATH = DATA_DIR / 'create-settings.json'
# Not taken from the page: the job's own content, and the output folder (a page draft from another PC carried its E:\
# folder here, and every Moise job failed; each PC saves to its own folder).
CREATE_SETTINGS_SKIPPED = ('prompt', 'images', 'image_labels', 'seed', 'source', 'output_dir')


@app.get('/api/create-settings')
def get_create_settings():
    saved = read_json(CREATE_SETTINGS_PATH) if CREATE_SETTINGS_PATH.is_file() else {}
    return {key: value for key, value in saved.items() if key not in CREATE_SETTINGS_SKIPPED}


@app.put('/api/create-settings')
def put_create_settings(body: dict):
    # A restored job puts its own fields (source_prompt...) in the page settings; only job settings are kept.
    keep = {key: value for key, value in body.items()
            if key in GenerationRequest.model_fields and key not in CREATE_SETTINGS_SKIPPED}
    if not isinstance(keep.get('model_id'), str) or not keep['model_id']:
        raise HTTPException(400, 'Choose a model.')
    write_json(CREATE_SETTINGS_PATH, keep)
    return {'ok': True}


@app.get('/api/loras/{model_id}/settings')
def get_lora_settings(model_id: str):
    try:
        item = models.entry(model_id)
        if 'lora' not in item.get('kind', ''):
            raise ValueError('Select a LoRA.')
        return {'prompt_addition': item.get('prompt_addition', '')}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.put('/api/loras/{model_id}/settings')
def save_lora_settings(model_id: str, body: LoraPromptSettings):
    from .lora_preferences import save
    try:
        return save(model_id, body.prompt_addition)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@app.post('/api/models/release')
def release_models():
    global image_warmup_suspended
    with active_lock:
        if active_job or pending_jobs:
            raise HTTPException(409, 'Finish or cancel queued jobs before releasing models.')
        image_worker.close()
        # Queue controls and later video jobs must not undo a manual unload.
        image_warmup_suspended = True
        return image_worker.status()

@app.post('/api/models/{model_id}/download')
def download_model(model_id: str):
    try:
        return models.start_download(model_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))

class Int8Choice(BaseModel):
    use: str = Field(pattern='^(int8|original)$')


@app.put('/api/models/{model_id}/int8')
def choose_int8(model_id: str, body: Int8Choice):
    """Run this model from its INT8 ConvRot copy (the default) or from its own file."""
    from . import int8_copies
    try:
        return int8_copies.set_choice(model_id, body.use)
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@app.post('/api/models/download/cancel')
def cancel_download():
    models.cancel_event.set()
    return {'ok':True}

@app.delete('/api/models/{model_id}')
def remove_model(model_id: str):
    with active_lock:
        if active_job or pending_jobs:
            raise HTTPException(409, 'Finish or cancel queued jobs before removing models.')
        try:
            image_worker.close()
            return models.remove(model_id)
        except (ValueError,OSError) as exc:
            raise HTTPException(409, str(exc))

@app.get('/api/examples')
def examples():
    path = APP_DIR / 'public' / 'examples.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

@app.post('/api/uploads')
async def upload(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 10:
        raise HTTPException(400, 'Upload between 1 and 10 images.')
    prepared = []
    for file in files:
        raw = await file.read(32 * 1024 * 1024 + 1)
        await file.close()
        if len(raw) > 32 * 1024 * 1024:
            raise HTTPException(413, 'Each image must be under 32 MB.')
        try:
            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > 64_000_000:
                    raise HTTPException(413, 'Each image must be under 64 megapixels.')
                picture = ImageOps.exif_transpose(source).convert('RGBA')
                picture.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(400, 'Could not read this image. Choose a JPG, PNG, WebP or HEIC photo.')
        prepared.append((picture, file.filename or 'Image'))
    return [save_upload(picture, name) for picture, name in prepared]


@app.get('/api/web-search/status')
def web_search_status():
    from . import web_search
    return {'ready': bool(web_search.key())}


class WebSearchKey(BaseModel):
    key: str = Field(default='', max_length=200)


@app.put('/api/web-search/key')
def web_search_key(body: WebSearchKey):
    from . import web_search
    web_search.save_key(body.key)
    return {'ready': bool(web_search.key())}


@app.get('/api/web-search/images')
def web_search_images(q: str, count: int = 24):
    from . import web_search
    if not q.strip():
        raise HTTPException(400, 'Type what to search for.')
    try:
        return {'results': web_search.search(q.strip()[:200], max(1, min(count, 60)))}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except httpx.HTTPError:
        raise HTTPException(502, 'The web search did not answer. Try again.')


class WebImport(BaseModel):
    url: str = Field(max_length=4000)
    # The search's own small copy, used when the site blocks the full-size file.
    fallback: str | None = Field(default=None, max_length=4000)
    person_name: str | None = Field(default=None, max_length=80)


@app.post('/api/web-search/import')
def web_search_import(body: WebImport):
    """Download one web picture into Uploads; with a name, also make it a new person in People."""
    from . import web_search
    try:
        raw = web_search.download_any([body.url, body.fallback])
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 64_000_000:
                raise ValueError('The picture is over 64 megapixels.')
            picture = ImageOps.exif_transpose(source).convert('RGBA')
            picture.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, 'Could not read this picture. Choose another one.')
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except httpx.HTTPError:
        raise HTTPException(502, 'Could not download this picture. Choose another one.')
    name = (body.person_name or '').strip()
    image = save_upload(picture, name or Path(urlparse(body.url).path).name or 'Web picture')
    if name:
        image['person'] = people.upsert(name, [image['id']], source='web')
        # Female or male for the new person, in the background like at start.
        threading.Thread(target=people.preset_labels, name='people-labels', daemon=True).start()
    return image


def save_upload(picture: Image.Image, name: str) -> dict:
    # Identical pixels share one file, so Recent inputs lists each image once.
    image_id = hashlib.sha256(picture.tobytes() + f'{picture.size}'.encode()).hexdigest()[:32] + '.png'
    path = UPLOADS_DIR / image_id
    if path.is_file():
        os.utime(path)
    else:
        temporary = UPLOADS_DIR / f'.{uuid.uuid4().hex}.tmp'
        picture.save(temporary, format='PNG')
        temporary.replace(path)
    return {'id': image_id, 'name': name, 'url': f'/api/uploads/{image_id}', 'thumb_url': f'/api/uploads/{image_id}/thumb',
            'width': picture.width, 'height': picture.height}


def upload_fingerprints(files: list[Path], index_name: str = 'fingerprints.json') -> dict[str, str]:
    """Pixel hash per upload, cached by size and time, so copies saved under old random names group together."""
    index_path = UPLOADS_DIR / '.thumbs' / index_name
    try:
        index = json.loads(index_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        index = {}
    changed = False
    result = {}
    for path in files:
        stat = path.stat()
        key = f'{stat.st_size}:{int(stat.st_mtime)}'
        cached = index.get(path.name)
        if not cached or cached[0] != key:
            try:
                with Image.open(path) as picture:
                    picture = picture.convert('RGBA')
                    digest = hashlib.sha256(picture.tobytes() + f'{picture.size}'.encode()).hexdigest()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                digest = path.name
            cached = index[path.name] = [key, digest]
            changed = True
        result[path.name] = cached[1]
    if changed:
        index_path.parent.mkdir(exist_ok=True)
        write_json(index_path, {name: value for name, value in index.items() if name in result})
    return result


@app.get('/api/uploads')
def recent_uploads(offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=200)):
    files = [path for path in UPLOADS_DIR.iterdir() if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    people_photos = {image_id for person in people.load() for image_id in person['image_ids']}
    files = [path for path in files if path.name not in people_photos]  # shown under People
    fingerprints = upload_fingerprints(files)
    # Results used as inputs are shown under Results, not again under Uploads: the links made for them (job-*.png)
    # and the full copies made before 2026-10-22 (same pixels as a result).
    outputs = [Path(job['output_path']) for job in all_jobs() if job.get('status') == 'succeeded'
               and job.get('kind', 'image') in IMAGE_KINDS and Path(job.get('output_path') or '').is_file()]
    result_pixels = set(upload_fingerprints(outputs, 'result-fingerprints.json').values())
    files = [path for path in files if not path.name.startswith('job-') and fingerprints[path.name] not in result_pixels]
    seen, unique = set(), []
    for path in files:  # newest copy of each image only
        if fingerprints[path.name] not in seen:
            seen.add(fingerprints[path.name])
            unique.append(path)
    return {'total': len(unique), 'items': [{'id': path.name, 'name': path.name, 'url': f'/api/uploads/{path.name}',
                                             'thumb_url': f'/api/uploads/{path.name}/thumb', 'modified': path.stat().st_mtime}
                                            for path in unique[offset:offset + limit]]}


class UploadDelete(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=5000)


@app.post('/api/uploads/delete')
def delete_uploads(body: UploadDelete):
    """Delete uploaded pictures from this computer, every saved copy of each (the list shows one per picture).
    People photos and pictures that a waiting or running job uses stay."""
    files = [path for path in UPLOADS_DIR.iterdir() if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
    fingerprints = upload_fingerprints(files)
    people_photos = {image_id for person in people.load() for image_id in person['image_ids']}
    with active_lock:
        busy = json.dumps([read_job(job_id).get('request') for job_id in [active_job, *pending_jobs] if job_id], default=str)
    wanted = {fingerprints[name] for name in body.ids if name in fingerprints}
    deleted, kept = [], []
    for path in files:
        if fingerprints[path.name] not in wanted:
            continue
        if path.name in people_photos or path.name in busy:
            kept.append(path.name)
            continue
        try:
            path.unlink()
            (UPLOADS_DIR / '.thumbs' / (path.stem + '.webp')).unlink(missing_ok=True)
            deleted.append(path.name)
        except OSError:
            kept.append(path.name)
    return {'deleted': len(deleted), 'kept': len(kept), 'deleted_ids': deleted}


@app.get('/api/uploads/{image_id}')
def uploaded_image(image_id: str):
    try:
        path = resolve_image(image_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return FileResponse(path)


@app.get('/api/uploads/{image_id}/thumb')
def uploaded_thumbnail(image_id: str):
    try:
        path = resolve_image(image_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    folder = UPLOADS_DIR / '.thumbs'
    folder.mkdir(exist_ok=True)
    cached = folder / (path.stem + '.webp')
    if not cached.is_file() or cached.stat().st_mtime < path.stat().st_mtime - 1:
        temporary = folder / f'{uuid.uuid4().hex}.tmp'
        try:
            with Image.open(path) as picture:
                picture.thumbnail((320, 320), Image.Resampling.LANCZOS)
                picture.save(temporary, format='WEBP', quality=82)
            try:
                temporary.replace(cached)
            except PermissionError:
                # Windows refuses while the old thumbnail is being sent; that one is shown until the next request.
                if not cached.is_file():
                    raise
        finally:
            temporary.unlink(missing_ok=True)
    return FileResponse(cached, media_type='image/webp', headers={'Cache-Control': 'private, max-age=86400'})


@app.put('/api/video-uploads')
async def upload_video(request: Request):
    """Stream large media directly to the model drive instead of the OS temp drive."""
    name=unquote(request.headers.get('x-file-name','video.mp4'))
    suffix=Path(name).suffix.lower()
    if suffix not in {'.mp4','.mov','.mkv','.webm'}:
        raise HTTPException(400,'Select an MP4, MOV, MKV or WebM video.')
    video_id=uuid.uuid4().hex+suffix
    destination=VIDEO_UPLOADS_DIR/video_id
    temporary=VIDEO_UPLOADS_DIR/(video_id+'.part')
    total=0
    limit=20*1024**3
    try:
        with temporary.open('wb') as handle:
            async for chunk in request.stream():
                total+=len(chunk)
                if total>limit:
                    raise HTTPException(413,'Videos must be under 20 GB.')
                handle.write(chunk)
        if total==0:
            raise HTTPException(400,'The video file is empty.')
        import av
        with av.open(str(temporary)) as media:
            stream=next((s for s in media.streams if s.type=='video'),None)
            if stream is None:
                raise HTTPException(400,'This file has no readable video stream.')
            width,height=stream.width,stream.height
            if not width or not height or width*height>80_000_000:
                raise HTTPException(400,'Video dimensions are unsupported.')
            frame=next(media.decode(stream),None)
            if frame is None:
                raise HTTPException(400,'This video has no readable frames.')
            fps=float(stream.average_rate) if stream.average_rate else None
            duration=(media.duration/1_000_000) if media.duration else (
                float(stream.duration*stream.time_base) if stream.duration and stream.time_base else (
                    stream.frames/fps if stream.frames and fps else None))
            poster=frame.to_image()
            poster.thumbnail((960,540),Image.Resampling.LANCZOS)
            poster.save(VIDEO_UPLOADS_DIR/(video_id+'.jpg'),quality=85)
            has_audio=any(s.type=='audio' for s in media.streams)
        temporary.replace(destination)
    except HTTPException:
        temporary.unlink(missing_ok=True)
        (VIDEO_UPLOADS_DIR/(video_id+'.jpg')).unlink(missing_ok=True)
        raise
    except (OSError,ValueError,StopIteration) as exc:
        temporary.unlink(missing_ok=True)
        (VIDEO_UPLOADS_DIR/(video_id+'.jpg')).unlink(missing_ok=True)
        raise HTTPException(400,f'Could not read this video: {exc}')
    result={'id':video_id,'name':Path(name).name,'url':f'/api/video-uploads/{video_id}',
            'poster_url':f'/api/video-uploads/{video_id}/poster','bytes':total,
            'width':width,'height':height,'fps':fps,'duration_seconds':duration,'has_audio':has_audio}
    write_json(VIDEO_UPLOADS_DIR/(video_id+'.json'),result)
    return result


@app.get('/api/video-uploads/{video_id}/metadata')
def video_upload_metadata(video_id: str):
    try:
        resolve_video(video_id)
        return read_json(VIDEO_UPLOADS_DIR/(video_id+'.json'))
    except (ValueError,FileNotFoundError) as exc:
        raise HTTPException(404,str(exc))


@app.get('/api/video-uploads/{video_id}/poster')
def video_upload_poster(video_id: str):
    try:
        resolve_video(video_id)
    except ValueError as exc:
        raise HTTPException(404,str(exc))
    return FileResponse(VIDEO_UPLOADS_DIR/(video_id+'.jpg'),media_type='image/jpeg')


@app.get('/api/video-uploads/{video_id}/timeline')
def video_upload_timeline(video_id: str):
    try:
        source = resolve_video(video_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    target = VIDEO_UPLOADS_DIR/(video_id+'.timeline-v2.webp')
    if not target.is_file():
        import av
        duration = probe_video_duration(source)
        count, tile_width, tile_height = 12, 112, 68
        strip = Image.new('RGB', (count*tile_width, tile_height), '#24212c')
        with av.open(str(source)) as media:
            stream = next(item for item in media.streams if item.type == 'video')
            origin = float(stream.start_time*stream.time_base) if stream.start_time is not None and stream.time_base else 0.0
            for index in range(count):
                second = min(max(0, duration-0.1), (index+0.5)*duration/count)
                if stream.time_base:
                    media.seek((stream.start_time or 0)+int(second/float(stream.time_base)),
                               stream=stream, backward=True)
                frame = None
                for candidate in media.decode(stream):
                    frame = candidate
                    if candidate.time is None or candidate.time-origin >= second-1/24:
                        break
                if frame is not None:
                    tile = ImageOps.contain(frame.to_image().convert('RGB'), (tile_width,tile_height))
                    strip.paste(tile, (index*tile_width+(tile_width-tile.width)//2,
                                       (tile_height-tile.height)//2))
        temporary = target.with_suffix('.webp.part')
        strip.save(temporary, format='WEBP', quality=78)
        temporary.replace(target)
    return FileResponse(target,media_type='image/webp')


class PeopleAnalysisRequest(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)


@app.get('/api/image-people')
def image_people(image_id: str = Query(max_length=200)):
    try:
        from .image_people import analyze_image
        return analyze_image(resolve_image(image_id))
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc))


@app.post('/api/video-uploads/{video_id}/people')
def video_upload_people(video_id: str, body: PeopleAnalysisRequest):
    try:
        source = resolve_video(video_id)
        duration = probe_video_duration(source)
        if body.end_seconds <= body.start_seconds or body.end_seconds > duration + 1/24:
            raise ValueError('Select a valid part of the source video.')
        from .reference_people import analyze_people
        people = analyze_people(source, body.start_seconds, min(duration,body.end_seconds))
        return {'people':people,'start_seconds':body.start_seconds,'end_seconds':body.end_seconds}
    except (ValueError,OSError) as exc:
        raise HTTPException(400,str(exc))


@app.get('/api/video-uploads/{video_id}')
def uploaded_video(video_id: str):
    try:
        return FileResponse(resolve_video(video_id))
    except ValueError as exc:
        raise HTTPException(404,str(exc))

OUTPAINT_DIRECTIONS = {'all', 'horizontal', 'vertical', 'left', 'right', 'top', 'bottom'}


class PromptEdit(BaseModel):
    text: str = Field(min_length=1, max_length=40000)


@app.get('/api/prompts')
def list_prompts():
    """Every internal prompt the app sends to a model, its official default and the user's edit."""
    from . import prompt_registry
    return {'prompts': prompt_registry.listing()}


@app.put('/api/prompts/{key}')
def edit_prompt(key: str, body: PromptEdit):
    from . import prompt_registry
    try:
        prompt_registry.save(key, body.text)
    except KeyError:
        raise HTTPException(400, 'This prompt cannot be edited.')
    return {'prompts': prompt_registry.listing()}


@app.delete('/api/prompts/{key}')
def restore_prompt(key: str):
    from . import prompt_registry
    prompt_registry.restore(key)
    return {'prompts': prompt_registry.listing()}


class EnhancerChoice(BaseModel):
    choice: str = Field(min_length=1, max_length=80)


@app.get('/api/enhancer-settings')
def enhancer_settings_list():
    """The prompt enhancer of each image architecture: its official default and the user's choice."""
    from . import enhancer_settings
    return {'architectures': enhancer_settings.listing()}


@app.put('/api/enhancer-settings/{arch}')
def enhancer_settings_save(arch: str, body: EnhancerChoice):
    from . import enhancer_settings
    try:
        enhancer_settings.save(arch, body.choice)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {'architectures': enhancer_settings.listing()}


class SwapPromptEdit(BaseModel):
    text: str = Field(default='', max_length=6000)
    expected: str = Field(max_length=80)
    restore_id: str | None = Field(default=None, max_length=80)


@app.get('/api/swap-prompts/{tool_id}')
def get_swap_prompt(tool_id: str):
    from . import swap_prompts
    try:
        tool = models.image_tool(tool_id)
        if not tool.get('swap'):
            raise ValueError('Choose a Swap tool.')
        return swap_prompts.get(tool)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.put('/api/swap-prompts/{tool_id}')
def edit_swap_prompt(tool_id: str, body: SwapPromptEdit):
    from . import swap_prompts
    try:
        return swap_prompts.save(models.image_tool(tool_id), body.text, body.expected, body.restore_id)
    except swap_prompts.PromptConflict as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post('/api/swap-prompts/{tool_id}/preview')
def preview_swap_prompt(tool_id: str, body: GenerationRequest):
    from .image_people import build_swap_prompt
    try:
        if not body.swap_selection:
            raise ValueError('Select the faces to preview the complete prompt.')
        return {'prompt': build_swap_prompt(models.image_tool(tool_id), body.swap_selection.model_dump(),
            body.images, resolve_image, body.swap_extra, body.swap_pose_source, body.swap_style_source, body.swap_prompt_template, body.image_labels, body.swap_style_mode)}
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc))


def validate_image_tool(body: GenerationRequest) -> dict:
    tool = models.image_tool(body.tool_id)
    limits = tool['inputs']
    if not limits['min'] <= len(body.images) <= limits['max']:
        labels = ', '.join(f'{i + 1}. {label}' for i, label in enumerate(limits['labels']))
        raise ValueError(f"{tool['title']} needs {limits['min']} input image{'s' if limits['min'] > 1 else ''}: {labels}.")
    if tool.get('prepare') == 'outpaint' and (body.tool_options.get('direction') not in OUTPAINT_DIRECTIONS
                                             or body.tool_options.get('amount') not in ('25', '50', '100')):
        raise ValueError('Choose where and how much to expand the image.')
    return tool


@app.post('/api/job-plan')
def job_plan_preview(body: dict):
    """What an image job with these choices will really run: steps, CFG, sampler, schedule and every LoRA."""
    from .job_plan import plan
    try:
        return plan(body)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@app.post('/api/generate', status_code=202)
def generate(body: GenerationRequest):
    source_prompt, source_negative_prompt = body.prompt, body.negative_prompt
    if not body.prompt.strip() and body.images:
        body.enhance = False
    try:
        from .image_sampling import validate as validate_sampling
        validate_sampling(body.model_dump(), models.entry(body.model_id))
        needed = models.required(body.model_id, body.text_encoder, body.tool_id, body.vae_id, [lora.id for lora in body.loras], len(body.images), body.krea_reference_mode)
        if body.tool_id and models.image_tool(body.tool_id).get('krea_edit') and not body.prompt.strip():
            raise ValueError(f"{models.image_tool(body.tool_id)['title']} needs an instruction in the prompt, for example: remove the person on the left.")
        if (body.images and not body.tool_id and body.krea_reference_mode == 'identity' and models.is_krea(models.entry(body.model_id))
                and not body.prompt.strip() and not body.enhance):
            # The identity LoRA restages from "an image and a plain-language instruction" (its card); with no instruction
            # it copies the photo, and the beside placement turned that into a two-panel image.
            raise ValueError('Krea 2 identity edit needs an instruction. Describe the new scene and how much of the person shows, '
                             'for example: full body, standing on a beach at sunset.')
        tool = validate_image_tool(body) if body.tool_id else None
        if tool and tool.get('swap'):
            if not body.swap_selection:
                raise ValueError('Select the faces to replace before creating a swap.')
            from .image_people import build_swap_prompt
            from . import swap_prompts
            if body.swap_prompt_template is None:
                saved = swap_prompts.get(tool)
                body.swap_prompt_template, body.swap_prompt_revision = saved['text'], saved['current']
            body.prompt = build_swap_prompt(tool, body.swap_selection.model_dump(), body.images, resolve_image, body.swap_extra, body.swap_pose_source, body.swap_style_source, body.swap_prompt_template, body.image_labels, body.swap_style_mode)
            from .prompt_mentions import image_mentions
            body.negative_prompt = image_mentions(body.negative_prompt, len(body.images), labels=body.image_labels)
        else:
            # Face picks left in the settings from a restored Qwen swap belong only to the Qwen swaps; other jobs
            # (Create, the Krea 2 swaps) ignore them instead of failing.
            body.swap_selection = None
            from .prompt_mentions import image_mentions
            notation = 'qwen' if models.entry(body.model_id).get('family', '').startswith('qwen-image') else 'plain'
            body.prompt = image_mentions(body.prompt, len(body.images), notation, body.image_labels)
            body.negative_prompt = image_mentions(body.negative_prompt, len(body.images), notation, body.image_labels)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    remote = cloud.settings()['mode'] == 'runpod'
    if remote and not cloud.ready('image'):
        raise HTTPException(503, 'Connect a RunPod image endpoint and network volume in Settings.')
    if models.is_flux3(models.entry(body.model_id)):
        body.enhance=False  # FLUX.3 runs its required local prompt model in its worker.
    if tool:
        # Tool prompts carry the LoRA trigger wording; enhancement would rewrite it.
        body.enhance = False
    try:
        for image_id in body.images:
            resolve_image(image_id)
        destination = Path(body.output_dir.strip() or OUTPUTS_DIR).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        probe = destination / f'.write-test-{uuid.uuid4().hex}'
        probe.write_bytes(b'')
        probe.unlink()
        if shutil.disk_usage(destination).free < 512 * 1024 * 1024:
            raise ValueError('The output drive needs at least 512 MB of free space.')
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc))
    with active_lock:
        # Removal uses the same lock; recheck after acquiring it so another tab
        # cannot remove a dependency between validation and starting the worker.
        remote = cloud.settings()['mode'] == 'runpod'
        if remote and not cloud.ready('image'):
            raise HTTPException(503, 'Connect a RunPod image endpoint and network volume in Settings.')
        job_id = uuid.uuid4().hex
        data = body.model_dump()
        data.update(source_prompt=source_prompt, source_negative_prompt=source_negative_prompt)
        data['execution'] = 'runpod' if remote else 'local'
        defaults = models.entry(body.model_id)['generation_defaults']
        # A tool's card can name its own step count (BFS Krea 2 body swap: 10 steps on Turbo).
        from .job_plan import recipe_for, speed_lora
        recipe = dict(recipe_for(tool, models.entry(body.model_id), data))
        # The author's 8 steps, CFG and sampler belong to the speed LoRA; without it the model's own settings apply.
        if recipe.get('speed_lora') and not speed_lora(tool, models.entry(body.model_id), data):
            recipe = {}
        data['steps'] = body.steps if body.steps is not None else (tool or {}).get('steps') or recipe.get('steps') or defaults['steps']
        data['steps_mode'] = 'custom' if body.steps is not None else 'model'
        if tool:
            data['enhance'] = False
            data['lora_strength'] = body.lora_strength if body.lora_strength is not None else tool['strength']
            data['reference_resolution'] = tool['reference_resolution']
            # The tool author's sampler, schedule and CFG (their workflow) unless the user set one; a model's own
            # defaults (for example a Civitai merge's euler ancestral / beta) do not replace them.
        for key in ('sampler', 'scheduler', 'cfg'):
            if data.get(key) is None and recipe.get(key) is not None:
                data[key] = recipe[key]
        data['output_dir'] = str(destination)
        if body.randomize_seed:
            data['seed'] = secrets.randbelow(2147483648)
        job = {'id': job_id, 'kind': 'image', 'status': 'queued', 'message': 'Waiting in queue', 'stage': 'queued', 'created_at': time.time(), 'seed': data['seed'], 'request': data, 'rewritten_prompt': '', 'step': 0, 'total_steps': data['steps']}
        enqueue_locked(job, data)
    if body.source:
        loras = ', '.join(f"{lora.id} {lora.strength:g}" for lora in body.loras) or 'none'
        pictures = ', '.join(f"{image_id} ({label or 'picture'})" for image_id, label in zip(body.images, list(body.image_labels) + [''] * len(body.images))) or 'none'
        print(f"[Moise] {body.source} | job {job_id[:8]} | model {models.entry(body.model_id).get('title', body.model_id)} | steps {data['steps']} | "
              f"cfg {data.get('cfg') or 'model'} | LoRAs {loras} | size {'%dx%d' % (data['width'], data['height']) if data['custom_size'] else 'auto %d' % data['auto_resolution']} | "
              f"pictures {pictures} | prompt: {body.prompt}", flush=True)
    return public_job(job)

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return public_job(read_job(job_id))

from .orbit import OrbitRequest


@app.post('/api/orbit/generate', status_code=202)
def generate_orbit(body: OrbitRequest):
    from .orbit import video_request
    try:
        request=video_request(body)
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    return generate_video(request)


@app.post('/api/video/generate', status_code=202)
def generate_video(body: VideoRequest):
    try:
        from .prompt_mentions import frame_mentions
        body.prompt = frame_mentions(body.prompt, body.mode == 'image' and bool(body.first_frame), body.mode == 'image' and bool(body.last_frame))
        needed=models.required_video(body)
        for image_id in (body.first_frame,body.last_frame):
            if image_id:
                resolve_image(image_id)
        destination=Path(body.output_dir.strip() or VIDEOS_DIR).expanduser().resolve()
        destination.mkdir(parents=True,exist_ok=True)
        probe=destination/f'.write-test-{uuid.uuid4().hex}'
        probe.write_bytes(b'');probe.unlink()
        if shutil.disk_usage(destination).free<1024**3:
            raise ValueError('The output drive needs at least 1 GB of free space.')
    except (ValueError,OSError) as exc:
        raise HTTPException(400,str(exc))
    with active_lock:
        remote = cloud.settings()['mode'] == 'runpod'
        if remote and not cloud.ready('video'):
            raise HTTPException(503, 'Connect a RunPod video endpoint and network volume in Settings.')
        data=body.model_dump()
        data.update(kind='video',execution='runpod' if remote else 'local',output_dir=str(destination))
        if body.randomize_seed:
            data['seed']=secrets.randbelow(2147483648)
        selected=models.entry(body.model_id)
        from .video_sampling import video_sampling
        sampling=video_sampling(selected,body.quality,body.steps,bool(body.last_frame))
        steps=sampling['total_steps']
        data.update(steps=sampling['steps'],quality=sampling['quality'],sampling=sampling,
                    steps_mode='custom' if body.steps is not None and not sampling.get('fixed_steps') else 'model')
        job_id=uuid.uuid4().hex
        job={'id':job_id,'kind':'video','status':'queued','message':'Waiting in queue','stage':'queued',
             'created_at':time.time(),'seed':data['seed'],'request':data,'rewritten_prompt':'','step':0,'total_steps':steps}
        enqueue_locked(job, data)
    return public_job(job)


@app.post('/api/reference-video/generate',status_code=202)
def generate_reference_video(body: ReferenceVideoRequest):
    try:
        from .reference_sampling import reference_sampling
        body.quality=reference_sampling(models.entry(body.model_id),body.quality,body.steps)['quality']
        needed=models.required_reference(body)
        source=resolve_video(body.source_video)
        metadata_path=VIDEO_UPLOADS_DIR/(body.source_video+'.json')
        metadata=read_json(metadata_path) if metadata_path.is_file() else {}
        duration=metadata.get('duration_seconds') or probe_video_duration(source)
        maximum=models.entry(body.model_id).get('generation_defaults',{}).get('max_frames',362)
        resolve_reference_clip(body,duration,maximum)
        if body.person_assignments and body.workflow=='crossview' and not body.replace_subjects_first:
            raise ValueError('Enable Replace subjects first to use person assignments in Camera view.')
        for person in body.source_people:
            if not body.source_start_seconds <= person.time <= body.source_end_seconds:
                raise ValueError('Scan people in the selected source segment again.')
        from .reference_prompt import build_reference_prompt
        prompt_request=body.model_dump()
        build_reference_prompt(prompt_request)
        if body.workflow=='crossview' and body.replace_subjects_first:
            build_reference_prompt(prompt_request,replacement=True)
        if not body.prompt.strip() and not (body.workflow=='crossview' and body.replace_subjects_first and body.replacement_prompt.strip()):
            body.enhance=False
        if body.warp_video:
            resolve_video(body.warp_video)
        for image_id in body.reference_images:
            resolve_image(image_id)
        destination=Path(body.output_dir.strip() or VIDEOS_DIR).expanduser().resolve()
        destination.mkdir(parents=True,exist_ok=True)
        probe=destination/f'.write-test-{uuid.uuid4().hex}'
        probe.write_bytes(b'');probe.unlink()
        if shutil.disk_usage(destination).free<1024**3:
            raise ValueError('The output drive needs at least 1 GB of free space.')
    except (ValueError,OSError) as exc:
        raise HTTPException(400,str(exc))
    with active_lock:
        remote = cloud.settings()['mode'] == 'runpod'
        if remote and not cloud.ready('video'):
            raise HTTPException(503, 'Connect a RunPod video endpoint and network volume in Settings.')
        selected=models.entry(body.model_id)
        from .reference_sampling import reference_sampling
        sampling=reference_sampling(selected,body.quality,body.steps)
        steps=sampling['steps']
        data=body.model_dump()
        data.update(kind='reference-video',execution='runpod' if remote else 'local',steps=steps,output_dir=str(destination),
                    sampling_profile=sampling,
                    steps_mode='custom' if body.steps is not None else 'model')
        if body.randomize_seed:
            data['seed']=secrets.randbelow(2147483648)
        job_id=uuid.uuid4().hex
        job={'id':job_id,'kind':'reference-video','status':'queued','message':'Waiting in queue','stage':'queued',
             'created_at':time.time(),'seed':data['seed'],'request':data,'rewritten_prompt':'','step':0,
             'total_steps':steps*(2 if body.workflow=='crossview' and body.replace_subjects_first else 1)}
        enqueue_locked(job,data)
    return public_job(job)


@app.post('/api/upscale/generate',status_code=202)
def generate_upscale(body: UpscaleRequest):
    try:
        needed=models.required_upscale(body)
        source=resolve_video(body.source_video)
        if source.stat().st_size>20*1024**3:
            raise ValueError('Videos must be under 20 GB.')
        destination=Path(body.output_dir.strip() or VIDEOS_DIR).expanduser().resolve()
        destination.mkdir(parents=True,exist_ok=True)
        probe=destination/f'.write-test-{uuid.uuid4().hex}'
        probe.write_bytes(b'');probe.unlink()
        if shutil.disk_usage(destination).free<max(source.stat().st_size*2,2*1024**3):
            raise ValueError('The output drive needs more free space for this video.')
    except (ValueError,OSError) as exc:
        raise HTTPException(400,str(exc))
    with active_lock:
        remote = cloud.settings()['mode'] == 'runpod'
        if remote and not cloud.ready('video'):
            raise HTTPException(503, 'Connect a RunPod video endpoint and network volume in Settings.')
        data=body.model_dump()
        data.update(kind='upscale',execution='runpod' if remote else 'local',output_dir=str(destination))
        job_id=uuid.uuid4().hex
        job={'id':job_id,'kind':'upscale','status':'queued','message':'Waiting in queue','stage':'queued',
             'created_at':time.time(),'seed':body.seed,'request':data,'rewritten_prompt':'','step':0,'total_steps':0}
        enqueue_locked(job,data)
    return public_job(job)


def image_upscale_source(source_image: str, source_job: str) -> Path:
    if source_job:
        return image_upscale.job_image(source_job)
    if not source_image:
        raise ValueError('Choose an image to upscale.')
    return resolve_image(source_image)


@app.get('/api/image-upscale/plan')
def image_upscale_plan(source_image: str = Query('', max_length=200), source_job: str = Query('', max_length=32),
                       size_mode: Literal['scale', 'long_side'] = 'scale', scale: int = Query(2, ge=2, le=4, multiple_of=2),
                       long_side: int = Query(4096, ge=256, le=image_upscale.MAX_SIDE), model_id: str = Query('', max_length=200)):
    """Source size and result size before a job starts; nothing is loaded on the GPU."""
    try:
        path = image_upscale_source(source_image, source_job)
        return image_upscale.plan(path, {'size_mode': size_mode, 'scale': scale, 'long_side': long_side, 'model_id': model_id}, strict=False)
    except (ValueError, OSError, UnidentifiedImageError) as exc:
        raise HTTPException(400, str(exc))


@app.post('/api/image-upscale/generate', status_code=202)
def generate_image_upscale(body: ImageUpscaleRequest):
    try:
        image_upscale.required(body.model_dump())
        path = image_upscale_source(body.source_image, body.source_job)
        sizes = image_upscale.plan(path, body.model_dump())
        if body.source_job:
            # A gallery image becomes an upload, so the result can show and compare it as its input.
            with Image.open(path) as picture:
                body.source_image = save_upload(ImageOps.exif_transpose(picture).convert('RGBA'), path.name)['id']
        destination = Path(body.output_dir.strip() or OUTPUTS_DIR).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        probe = destination / f'.write-test-{uuid.uuid4().hex}'
        probe.write_bytes(b''); probe.unlink()
        if shutil.disk_usage(destination).free < 1024**3:
            raise ValueError('The output drive needs at least 1 GB of free space.')
    except (ValueError, OSError, UnidentifiedImageError) as exc:
        raise HTTPException(400, str(exc))
    with active_lock:
        data = body.model_dump()
        # Always on this GPU: the RunPod workers have no image upscaler. images lets the gallery show the input.
        data.update(kind='image-upscale', execution='local', output_dir=str(destination), images=[body.source_image],
                    prompt='', source_width=sizes['source_width'], source_height=sizes['source_height'])
        job_id = uuid.uuid4().hex
        job = {'id': job_id, 'kind': 'image-upscale', 'status': 'queued', 'message': 'Waiting in queue', 'stage': 'queued',
               'created_at': time.time(), 'seed': body.seed, 'request': data, 'rewritten_prompt': '', 'step': 0, 'total_steps': 0,
               'width': sizes['width'], 'height': sizes['height'],
               'source_width': sizes['source_width'], 'source_height': sizes['source_height']}
        enqueue_locked(job, data)
    return public_job(job)


@app.post('/api/multi-reference/generate', status_code=202)
def generate_multi_reference(body: MultiReferenceRequest):
    source_prompt = body.prompt
    if not body.image_labels:
        body.image_labels = ['person'] * len(body.reference_images) + (['background'] if body.background else [])
    try:
        from .prompt_mentions import image_mentions
        body.prompt = image_mentions(body.prompt, len(body.reference_images) + bool(body.background), 'plain', body.image_labels)
        required_ltx(body)
        for media, value in ltx_media_inputs(body.model_dump()):
            (resolve_audio if media == 'audio' else resolve_image)(value)
        destination = Path(body.output_dir.strip() or VIDEOS_DIR).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        probe = destination / ('.write-test-' + uuid.uuid4().hex)
        probe.write_bytes(b''); probe.unlink()
        if shutil.disk_usage(destination).free < 1024**3:
            raise ValueError('The output drive needs at least 1 GB of free space.')
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc))
    with active_lock:
        remote = cloud.settings()['mode'] == 'runpod'
        if remote and not cloud.ready('video'):
            raise HTTPException(503, 'Connect a RunPod video endpoint and network volume in Settings.')
        data = body.model_dump()
        data.update(kind='multi-reference', execution='runpod' if remote else 'local', output_dir=str(destination), source_prompt=source_prompt)
        if body.randomize_seed:
            data['seed'] = secrets.randbelow(2147483648)
        selected = models.entry(body.model_id)
        total_steps = 11 if body.refine else 8
        if is_ref2va(selected):
            from .reference_sampling import reference_sampling
            sampling = reference_sampling(selected, body.quality, body.steps)
            total_steps = sampling['steps']
            data.update(quality=sampling['quality'], steps=sampling['steps'], sampling_profile=sampling, steps_mode='custom' if body.steps is not None else 'model')
        job = {'id': uuid.uuid4().hex, 'kind': 'multi-reference', 'status': 'queued',
               'stage': 'queued', 'message': 'Waiting in queue', 'created_at': time.time(),
               'seed': data['seed'], 'request': data, 'step': 0, 'total_steps': total_steps}
        enqueue_locked(job, data)
    return public_job(job)


@app.post('/api/audio-uploads')
async def upload_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}:
        raise HTTPException(400, 'Choose an MP3, WAV, M4A, FLAC or OGG file.')
    audio_id = uuid.uuid4().hex + suffix
    destination = AUDIO_UPLOADS_DIR / audio_id
    try:
        total = 0
        with destination.open('wb') as target:
            while chunk := await file.read(1024*1024):
                total += len(chunk)
                if total > 100*1024**2:
                    raise ValueError('Voice references must be under 100 MB.')
                target.write(chunk)
        import av
        with av.open(str(destination)) as media:
            stream = next(s for s in media.streams if s.type == 'audio')
            next(media.decode(stream))
        return {'id': audio_id, 'name': Path(file.filename).name, 'url': f'/api/audio-uploads/{audio_id}'}
    except (OSError, ValueError, StopIteration):
        destination.unlink(missing_ok=True)
        raise HTTPException(400, 'Could not read this audio. Choose a valid audio file under 100 MB.')


@app.get('/api/audio-uploads/{audio_id}')
def audio_upload(audio_id: str):
    try:
        return FileResponse(resolve_audio(audio_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc))

@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str):
    with active_lock:
        if job_id in pending_jobs:
            pending_jobs.remove(job_id)
            job = read_job(job_id)
            job.update(status='cancelled', stage='cancelled', message='Removed from queue.', finished_at=time.time())
            write_json(JOBS_DIR / job_id / 'status.json', job)
            recent_jobs.appendleft(job_id)
            save_queue_locked()
            return {'ok': True}
        if active_job != job_id:
            raise HTTPException(409, 'This generation is no longer running.')
        if read_job(job_id)['status'] in ('succeeded', 'failed', 'cancelled'):
            return {'ok': True}
        cancelled_jobs.add(job_id)
        remote_to_cancel = active_remote[1] if active_remote and active_remote[0] == job_id else None
        if active_process and active_process.poll() is None:
            stop_worker(active_process)
    if remote_to_cancel:
        try:
            cloud.cancel('video' if read_job(job_id).get('kind') in VIDEO_KINDS else 'image', remote_to_cancel)
        except Exception:
            pass
    return {'ok': True}

@app.get('/api/queue')
def get_queue():
    with active_lock:
        return {'paused': queue_paused,
                'active': public_job(read_job(active_job)) if active_job else None,
                'pending': [public_job(read_job(job_id)) | {'position': i + 1} for i, job_id in enumerate(pending_jobs)],
                'recent': [public_job(read_job(job_id)) for job_id in recent_jobs]}


@app.get('/api/activity')
def get_activity():
    """Small live snapshot for the activity terminal; no catalog or file scan."""
    with active_lock:
        current = read_job(active_job) if active_job else None
        result = {'active': current, 'pending_count': len(pending_jobs), 'paused': queue_paused}
    with models.lock:
        result.update(transfer=dict(models.transfer), downloads=civitai_download_list())
    return result


class QueueControl(BaseModel):
    paused: bool


@app.post('/api/queue/control')
def control_queue(body: QueueControl):
    global queue_paused
    with active_lock:
        queue_paused = body.paused
        save_queue_locked()
        start_next_locked()
    return get_queue()


class QueuePosition(BaseModel):
    position: int = Field(ge=1)


@app.post('/api/jobs/{job_id}/move')
def move_job(job_id: str, body: QueuePosition):
    with active_lock:
        if job_id not in pending_jobs:
            raise HTTPException(409, 'Only waiting jobs can be moved.')
        pending_jobs.remove(job_id)
        pending_jobs.insert(min(body.position - 1, len(pending_jobs)), job_id)
        save_queue_locked()
    return get_queue()


@app.post('/api/jobs/{job_id}/retry', status_code=202)
def retry_job(job_id: str):
    previous = read_job(job_id)
    data = dict(previous['request'])
    data.update(seed=previous['seed'], randomize_seed=False)
    kind=previous.get('kind','image')
    if kind=='video':
        return generate_video(VideoRequest(**data))
    if kind=='multi-reference':
        return generate_multi_reference(MultiReferenceRequest(**data))
    if kind=='reference-video':
        return generate_reference_video(ReferenceVideoRequest(**data))
    if kind=='upscale':
        return generate_upscale(UpscaleRequest(**data))
    if kind=='image-upscale':
        return generate_image_upscale(ImageUpscaleRequest(**data))
    if kind=='captioning':
        return training_captions(CaptionRequest(**{**data, 'images': [i['id'] for i in data['images']]}))
    if kind=='training':
        return start_training(TrainingRequest(**data))
    return generate(GenerationRequest(**data))


def library_jobs(kind: str, search: str) -> list[dict]:
    query = search.casefold().strip()
    titles = {item['id']: item['title'] for item in models.catalog()}
    jobs = [j for j in all_jobs() if j.get('status') == 'succeeded' and j.get('kind') not in ('training', 'captioning')
            and (kind == 'all' or (kind=='video' and j.get('kind') in VIDEO_KINDS)
                 or j.get('kind', 'image') == kind or (kind == 'image' and j.get('kind') == 'image-upscale'))
            and (not query or query in (j.get('request', {}).get('prompt', '') + ' ' + j.get('request', {}).get('model_id', '')
                 + ' ' + titles.get(j.get('request', {}).get('model_id', ''), '')).casefold())]
    jobs.sort(key=lambda j: j['created_at'], reverse=True)
    return jobs


@app.get('/api/library')
def library(kind: Literal['all', 'image', 'video'] = 'all', search: str = '',
            offset: int = Query(0, ge=0), limit: int = Query(48, ge=1, le=100)):
    jobs = library_jobs(kind, search)
    return {'items': [public_job(j) for j in jobs[offset:offset + limit]], 'total': len(jobs)}


class LibraryDelete(BaseModel):
    ids: list[str] = Field(default_factory=list, max_length=100000)
    all: bool = False
    kind: Literal['all', 'image', 'video'] = 'all'
    search: str = Field(default='', max_length=2000)


MEDIA_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.webm', '.mov', '.wav'}


@app.post('/api/library/delete')
def library_delete(body: LibraryDelete):
    """Delete finished results: the saved file, its settings file next to it, and the job folder."""
    if body.all:
        targets = library_jobs(body.kind, body.search)
    else:
        wanted = set(body.ids)
        targets = [job for job in all_jobs() if job['id'] in wanted]
    deleted, failed = 0, []
    for job in targets:
        with active_lock:
            if job['id'] == active_job or job['id'] in pending_jobs or job.get('status') != 'succeeded':
                continue  # never touch running or waiting jobs
            if job['id'] in recent_jobs:
                recent_jobs.remove(job['id'])
        output = Path(job.get('output_path') or '')
        try:
            if output.suffix.lower() in MEDIA_SUFFIXES and output.is_file():
                output.unlink()
            sidecar = output.with_suffix('.json')
            if output.suffix.lower() in MEDIA_SUFFIXES and sidecar.is_file():
                sidecar.unlink()
            directory = JOBS_DIR / job['id']
            if directory.is_dir() and directory.parent == JOBS_DIR:
                def keep_open_log(function, path, error, directory=directory):
                    # the model worker keeps the log of its last job open; the folder goes when it is free
                    if Path(path).name != 'worker.log' and Path(path) != directory:
                        raise error[1]
                shutil.rmtree(directory, onerror=keep_open_log)
            deleted += 1
        except OSError as exc:
            failed.append(f"{job['id']}: {exc}")
    return {'deleted': deleted, 'failed': failed[:20]}


class TrainingImage(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    caption: str = Field(default='', max_length=8000)


class TrainingRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    arch: Literal['flux3-klein-base', 'qwen-image-2.1', 'qwen-image-2.1-turbo', 'krea-2', 'krea-2-turbo', 'z-image-turbo', 'z-image', 'moss-image'] = 'z-image-turbo'
    subject: Literal['character', 'style'] = 'character'
    trigger_word: str = Field(default='', max_length=40)
    images: list[TrainingImage] = Field(min_length=1, max_length=300)
    steps: int = Field(default=2000, ge=50, le=20000)
    rank: int = Field(default=32, ge=4, le=128)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-3)
    resolutions: list[Literal[512, 768, 1024, 1280, 1536]] = Field(default_factory=lambda: [512, 768, 1024], min_length=1)
    save_every: int = Field(default=250, ge=25, le=5000)
    sample_prompts: list[str] = Field(default_factory=list, max_length=6)
    auto_caption: bool = True
    settings_mode: Literal['recommended', 'custom'] = 'custom'
    profile_version: str = Field(default='', max_length=80)


class CaptionRequest(BaseModel):
    images: list[str] = Field(min_length=1, max_length=300)
    name: str = Field(default='', max_length=60)
    trigger_word: str = Field(default='', max_length=40)
    subject: Literal['character', 'style'] = 'character'
    # The training's base model: some architectures have their own caption form (reference/caption-prompts.json).
    arch: str = Field(default='', max_length=40)


@app.get('/api/training')
def training_state():
    def kept(job):
        # A finished training whose folder was deleted has no samples or LoRA left to show.
        if job.get('status') in ('queued', 'running'): return True
        try: return training.output_dir(job['request']).is_dir()
        except (KeyError, TypeError): return True
    jobs = sorted((j for j in all_jobs() if j.get('kind') == 'training' and kept(j)), key=lambda j: j['created_at'], reverse=True)
    return {'setup': training.setup_status(),
            'archs': [{'id': key, 'label': info['label'], 'gate': info.get('gate'), 'families': info['families'],
                       'turbo': 'assistant_lora_path' in info} for key, info in training.ARCHS.items()],
            'jobs': [public_job(job) for job in jobs[:50]]}


@app.post('/api/training/setup')
def training_setup():
    if cloud.settings()['mode'] == 'runpod':
        raise HTTPException(409, 'Training runs on this computer. Switch the GPU to Local in Settings.')
    return training.start_setup()


@app.post('/api/training/captions', status_code=202)
def training_captions(body: CaptionRequest):
    try:
        for image_id in body.images: resolve_image(image_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    with active_lock:
        job_id = uuid.uuid4().hex
        data = body.model_dump() | {'kind': 'captioning', 'execution': 'local', 'prompt': body.name or 'Training captions',
                                    'images': [{'id': i, 'caption': ''} for i in dict.fromkeys(body.images)]}
        job = {'id': job_id, 'kind': 'captioning', 'status': 'queued', 'message': 'Waiting in queue', 'stage': 'queued',
               'created_at': time.time(), 'seed': 0, 'request': data, 'step': 0, 'total_steps': len(data['images'])}
        enqueue_locked(job, data)
    return public_job(job)


@app.get('/api/training/captions/{job_id}')
def caption_results(job_id: str):
    job = read_job(job_id)
    if job.get('kind') not in ('training', 'captioning'): raise HTTPException(404, 'Captions not found.')
    path = JOBS_DIR / job_id / 'captions.json'
    return {'job': public_job(job), 'captions': read_json(path) if path.exists() else {}}


class DatasetFiles(BaseModel):
    class Item(BaseModel):
        name: str = Field(min_length=1, max_length=600)
        size: int = Field(ge=0, le=1024**3)
    files: list[Item] = Field(min_length=1, max_length=1500)


@app.post('/api/training/imports')
def create_training_import(body: DatasetFiles):
    from . import training_data
    try: return training_data.create([f.model_dump() for f in body.files])
    except ValueError as exc: raise HTTPException(400, str(exc))


@app.put('/api/training/imports/{import_id}/files/{index}')
async def upload_training_part(import_id: str, index: int, request: Request, offset: int = Query(ge=0)):
    from . import training_data
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > training_data.MAX_CHUNK: raise HTTPException(413, 'Upload part is too large.')
    try: return training_data.append(import_id, index, offset, raw)
    except (ValueError, FileNotFoundError) as exc: raise HTTPException(400, str(exc))


@app.post('/api/training/imports/{import_id}/finish')
def finish_training_import(import_id: str):
    from . import training_data
    import zipfile
    try: return training_data.finish(import_id, save_upload)
    except (ValueError, OSError, zipfile.BadZipFile, Image.DecompressionBombError) as exc:
        raise HTTPException(400, str(exc) if isinstance(exc, ValueError) else 'Cannot read the dataset. Check the ZIP and image files.')


@app.delete('/api/training/imports/{import_id}')
def delete_training_import(import_id: str):
    from . import training_data
    try: training_data.discard(import_id)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}


class TrainingAdvice(BaseModel):
    arch: str
    subject: Literal['character', 'style'] = 'character'
    images: list[dict] = Field(default_factory=list, max_length=300)


@app.post('/api/training/recommendations')
def training_recommendations(body: TrainingAdvice):
    from .training_profiles import recommend
    if body.arch not in training.ARCHS: raise HTTPException(400, 'Choose a supported base model.')
    images = []
    for image in body.images:
        try:
            with Image.open(resolve_image(str(image.get('id', '')))) as photo:
                from .training_profiles import image_facts
                images.append(image_facts(photo))
        except (ValueError, OSError): raise HTTPException(400, 'A training image is missing. Add it again.')
    return recommend(body.arch, body.subject, images)


@app.post('/api/training/start', status_code=202)
def start_training(body: TrainingRequest):
    if cloud.settings()['mode'] == 'runpod':
        raise HTTPException(409, 'Training runs on this computer. Switch the GPU to Local in Settings.')
    if not training.setup_status()['ready']:
        raise HTTPException(409, 'Install the trainer first.')
    try:
        for image in body.images:
            resolve_image(image.id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    with active_lock:
        job_id = uuid.uuid4().hex
        data = body.model_dump() | {'kind': 'training', 'execution': 'local', 'prompt': body.name, 'model_id': f'training:{body.arch}'}
        if body.settings_mode == 'recommended':
            profile = training_recommendations(TrainingAdvice(arch=body.arch, subject=body.subject, images=[{'id': i.id} for i in body.images]))
            data.update({k: profile[k] for k in ('steps', 'rank', 'learning_rate', 'resolutions', 'save_every')})
            data['profile_version'] = profile['version']
        job = {'id': job_id, 'kind': 'training', 'status': 'queued', 'message': 'Waiting in queue', 'stage': 'queued', 'created_at': time.time(),
               'seed': 0, 'request': data, 'step': 0, 'total_steps': data['steps'], 'title': body.name}
        enqueue_locked(job, data)
    return public_job(job)


@app.get('/api/training/{job_id}/samples/{name}')
def training_sample(job_id: str, name: str):
    job = read_job(job_id)
    if job.get('kind') != 'training' or name not in job.get('samples', []):
        raise HTTPException(404, 'Sample not found.')
    folder = training.slug(job['request']['name'])
    # Trainings made before the per-name output folder kept their output in the job folder.
    path = next((base / folder / 'samples' / name for base in (training.output_dir(job['request']), JOBS_DIR / job_id / 'output')
                 if (base / folder / 'samples' / name).is_file()), None)
    if not path:
        raise HTTPException(404, 'Sample not found.')
    return FileResponse(path, media_type='image/jpeg')


@app.get('/api/history')
def history(kind: str = 'image'):
    jobs = []
    for path in JOBS_DIR.glob('*/status.json'):
        try:
            job = read_json(path)
            if job.get('status') == 'succeeded' and (job.get('kind','image') == kind or (kind == 'image' and job.get('kind') == 'image-upscale')):
                jobs.append(public_job(job))
        except (ValueError, OSError):
            pass
    return sorted(jobs, key=lambda j: j['created_at'], reverse=True)[:100]

@app.get('/api/jobs/{job_id}/image')
def result(job_id: str, download: bool = False):
    job = read_job(job_id)
    if job.get('kind','image') not in IMAGE_KINDS or job['status'] != 'succeeded' or not Path(job.get('output_path', '')).is_file():
        raise HTTPException(404, 'Saved image not found.')
    path = Path(job['output_path'])
    return FileResponse(path, media_type='image/png', filename=path.name if download else None, headers=RESULT_CACHE)


@app.get('/api/jobs/{job_id}/display')
def display_image(job_id: str):
    """A bounded, alpha-preserving viewing copy. /image remains the original PNG."""
    job = read_job(job_id)
    source = Path(job.get('output_path', ''))
    if job.get('kind', 'image') not in IMAGE_KINDS or job['status'] != 'succeeded' or not source.is_file():
        raise HTTPException(404, 'Saved image not found.')
    directory = JOBS_DIR / job_id
    cached = directory / 'display-v1.webp'
    # Multiple browser panels can request the same uncached result together.
    with display_locks[int(job_id[:2], 16) % len(display_locks)]:
        if not cached.is_file():
            temporary = directory / f'display-{uuid.uuid4().hex}.tmp'
            try:
                with Image.open(source) as picture:
                    picture = ImageOps.exif_transpose(picture)
                    picture.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                    if picture.mode not in ('RGB', 'RGBA'):
                        picture = picture.convert('RGBA' if 'A' in picture.getbands() or 'transparency' in picture.info else 'RGB')
                    picture.save(temporary, format='WEBP', quality=90, method=4,
                                 **({'icc_profile': picture.info['icc_profile']} if picture.info.get('icc_profile') else {}))
                temporary.replace(cached)
            finally:
                temporary.unlink(missing_ok=True)
    return FileResponse(cached, media_type='image/webp', headers=RESULT_CACHE)

@app.get('/api/jobs/{job_id}/video')
def video_result(job_id: str, download: bool = False):
    job=read_job(job_id)
    path=Path(job.get('output_path',''))
    if job.get('kind') not in VIDEO_KINDS or job['status']!='succeeded' or not path.is_file():
        raise HTTPException(404,'Saved video not found.')
    return FileResponse(path,media_type='video/mp4',filename=path.name if download else None)

@app.get('/api/jobs/{job_id}/live-preview')
def live_video_preview(job_id: str):
    job=read_job(job_id)
    path=JOBS_DIR/job_id/'live-preview.webp'
    # Videos and images (each sampling step's estimate) share the same preview file.
    if not path.is_file():
        raise HTTPException(404,'The live preview is not available yet.')
    return Response(path.read_bytes(),media_type='image/webp',headers={'Cache-Control':'no-store'})


@app.get('/api/jobs/{job_id}/poster')
def video_poster(job_id: str):
    job=read_job(job_id)
    path=JOBS_DIR/job_id/'poster.png'
    if job.get('kind') not in VIDEO_KINDS or job['status']!='succeeded' or not path.is_file():
        raise HTTPException(404,'Video preview not found.')
    return FileResponse(path,media_type='image/png')


@app.get('/api/jobs/{job_id}/subject-video')
def subject_replacement_video(job_id: str, download: bool = False):
    job=read_job(job_id)
    path=JOBS_DIR/job_id/'subject-replacement.mp4'
    if job.get('kind')!='reference-video' or not job.get('subject_replacement_ready') or not path.is_file():
        raise HTTPException(404,'Subject replacement video not found.')
    return FileResponse(path,media_type='video/mp4',filename=path.name if download else None)


@app.get('/api/jobs/{job_id}/thumbnail')
def thumbnail(job_id: str):
    job = read_job(job_id)
    if job['status'] != 'succeeded':
        raise HTTPException(404, 'Preview not found.')
    directory = JOBS_DIR / job_id
    cached = directory / 'thumbnail.webp'
    # The gallery can request the same new thumbnail twice at once; Windows refuses to replace a file that is being sent.
    with display_locks[int(job_id[:2], 16) % len(display_locks)]:
        if not cached.exists():
            source = directory / 'poster.png' if job.get('kind') in VIDEO_KINDS else Path(job.get('output_path', ''))
            if not source.is_file():
                raise HTTPException(404, 'Saved result not found.')
            temporary = directory / f'thumbnail-{uuid.uuid4().hex}.tmp'
            try:
                with Image.open(source) as picture:
                    picture.thumbnail((480, 360), Image.Resampling.LANCZOS)
                    picture.save(temporary, format='WEBP', quality=85)
                temporary.replace(cached)
            finally:
                temporary.unlink(missing_ok=True)
    return FileResponse(cached, media_type='image/webp', headers=RESULT_CACHE)

@app.get('/api/jobs/{job_id}/metadata')
def metadata(job_id: str):
    job = read_job(job_id)
    return JSONResponse(job, headers={'Content-Disposition': f'attachment; filename="{job_id}.json"'})

@app.post('/api/jobs/{job_id}/reuse')
def reuse(job_id: str):
    job = read_job(job_id)
    path = Path(job.get('output_path', ''))
    if job.get('kind','image') not in IMAGE_KINDS or job['status'] != 'succeeded' or not path.is_file():
        raise HTTPException(404, 'Saved image not found.')
    # The input points at the saved result itself (resolve_image reads job-<id>.png from the outputs): no copy.
    image_id = f'job-{job["id"]}.png'
    with Image.open(path) as picture:
        width, height = picture.size
    return {'id': image_id, 'name': path.name, 'url': f'/api/uploads/{image_id}', 'thumb_url': f'/api/uploads/{image_id}/thumb',
            'width': width, 'height': height}

@app.post('/api/jobs/{job_id}/reuse-video')
def reuse_video(job_id: str, part: Literal['final', 'subject'] = 'final'):
    job = read_job(job_id)
    path = Path(job.get('output_path', ''))
    if part == 'subject':
        path = JOBS_DIR / job_id / 'subject-replacement.mp4'
        available = job.get('kind') == 'reference-video' and job.get('subject_replacement_ready')
    else:
        available = job.get('kind') in VIDEO_KINDS and job.get('status') == 'succeeded'
    if not available or not path.is_file():
        raise HTTPException(404, 'Saved video not found.')
    from .video_reuse import reuse_video_file
    try:
        return reuse_video_file(path, VIDEO_UPLOADS_DIR, JOBS_DIR / job_id, part)
    except (OSError, ValueError, StopIteration):
        raise HTTPException(422, 'Could not prepare this video. Check that the saved file is available and try again.')


@app.post('/api/open-output')
def open_output(kind: str = 'image'):
    destination = VIDEOS_DIR if kind in VIDEO_KINDS else OUTPUTS_DIR
    if os.name == 'nt':
        os.startfile(destination)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', str(destination)])
    return {'ok': True}

if EXAMPLES_DIR.exists():
    app.mount('/examples', StaticFiles(directory=EXAMPLES_DIR), name='examples')
if (APP_DIR / 'dist').exists():
    app.mount('/', StaticFiles(directory=APP_DIR / 'dist', html=True), name='frontend')

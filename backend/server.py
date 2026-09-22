from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from collections import deque
import io
import json
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

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .common import APP_DIR, DATA_DIR, EXAMPLES_DIR, JOBS_DIR, MODELS_DIR, OUTPUTS_DIR, VIDEOS_DIR, UPLOADS_DIR, GenerationRequest, VideoRequest, model_status, resolve_image, read_json, write_json
from . import models, antigravity
from .resident_worker import ImageWorker

SESSION_TOKEN = secrets.token_urlsafe(32)
active_lock = threading.Lock()
active_job: str | None = None
active_process: subprocess.Popen | None = None
cancelled_jobs: set[str] = set()
pending_jobs: list[str] = []
recent_jobs: deque[str] = deque(maxlen=20)
queue_paused = False
shutting_down = False
image_worker = ImageWorker()


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


def save_queue_locked():
    write_json(DATA_DIR / 'queue-state.json', {'paused': queue_paused, 'pending': pending_jobs})


def enqueue_locked(job, data):
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
    if job.get('status') == 'succeeded':
        public['thumbnail_url'] = f'/api/jobs/{job["id"]}/thumbnail'
    return public

def run_job(job_id):
    global active_job, active_process
    directory = JOBS_DIR / job_id
    try:
        request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
        is_video = request.get('kind') == 'video'
        with (directory / 'worker.log').open('a', encoding='utf-8') as log:
            with active_lock:
                if job_id in cancelled_jobs:
                    job = read_job(job_id)
                    job.update(status='cancelled', message='Generation cancelled.', finished_at=time.time())
                    write_json(directory / 'status.json', job)
                    return
                if is_video:
                    image_worker.close()
                    active_process = subprocess.Popen(
                        [sys.executable, '-X', 'faulthandler', '-u', '-m', 'backend.worker', str(directory)],
                        cwd=APP_DIR, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    )
                else:
                    active_process = image_worker.start(directory, request)
                process = active_process
            return_code = process.wait() if is_video else image_worker.wait(directory, process)
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
        job.update(status='failed', message=str(exc), finished_at=time.time())
        write_json(directory / 'status.json', job)
    finally:
        with active_lock:
            active_job = None
            active_process = None
            cancelled_jobs.discard(job_id)
            recent_jobs.appendleft(job_id)
            save_queue_locked()
            start_next_locked()

@asynccontextmanager
async def lifespan(app):
    global queue_paused, shutting_down, active_job, active_process
    with active_lock:
        shutting_down = False
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
                job.update(status='failed', message='The app closed before this generation finished. Use Retry to run it again.', finished_at=time.time())
                write_json(JOBS_DIR / job['id'] / 'status.json', job)
            if job.get('status') == 'queued':
                pending_jobs.append(job['id'])
            else:
                recent_jobs.appendleft(job['id'])
        order = saved.get('pending', [])
        pending_jobs.sort(key=lambda job_id: order.index(job_id) if job_id in order else len(order))
        start_next_locked()
    yield
    with active_lock:
        shutting_down = True
        save_queue_locked()
        if active_process and active_process.poll() is None:
            stop_worker(active_process)
            active_process.wait(timeout=30)
        image_worker.close()

app = FastAPI(title='Qwen', lifespan=lifespan)

@app.middleware('http')
async def local_only(request: Request, call_next):
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('localhost', '127.0.0.1', 'testserver'):
        return JSONResponse({'detail': 'Local access only.'}, status_code=403)
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        if request.headers.get('x-studio-token') != SESSION_TOKEN:
            return JSONResponse({'detail': 'Refresh the app and try again.'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin not in (f'http://{request.headers.get("host")}', 'http://127.0.0.1:5173'):
            return JSONResponse({'detail': 'Local access only.'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    if request.url.path.startswith('/api'):
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.get('/api/health')
def health():
    return {'app': 'image-studio-qwen21', 'version': '1.0.0', 'app_dir':str(APP_DIR), 'data_dir':str(DATA_DIR)}

@app.get('/api/config')
def config():
    return {'token': SESSION_TOKEN, 'models': model_status(), 'catalog': models.state(), 'enhancement': antigravity.status(), 'output_dir': str(OUTPUTS_DIR), 'video_output_dir':str(VIDEOS_DIR), 'active_job': active_job, 'model': 'Qwen-Image-2.1', 'device': 'NVIDIA GeForce RTX 4090', 'data_dir': str(DATA_DIR)}

class CallbackInput(BaseModel):
    callbackInput: str = Field(min_length=1, max_length=16000)

class EnhancementModel(BaseModel):
    model: str = Field(min_length=1, max_length=100)

@app.get('/api/oauth/antigravity/status')
def connection_status():
    return antigravity.status()

@app.post('/api/oauth/antigravity/start')
def connection_start():
    return antigravity.start()

@app.get('/api/oauth/antigravity/poll/{state}')
def connection_poll(state: str):
    return antigravity.poll(state)

@app.post('/api/oauth/antigravity/exchange')
def connection_exchange(body: CallbackInput):
    try:
        return antigravity.exchange(body.callbackInput)
    except (ValueError, antigravity.ConnectionError) as exc:
        raise HTTPException(400, str(exc))

@app.post('/api/oauth/antigravity/refresh')
def connection_refresh():
    try:
        antigravity.ensure_tokens(force=True)
        return antigravity.status()
    except antigravity.ConnectionError as exc:
        raise HTTPException(424, str(exc))

@app.post('/api/oauth/antigravity/disconnect')
def connection_disconnect():
    with active_lock:
        if active_job or pending_jobs:
            raise HTTPException(409, 'Finish or cancel queued jobs before disconnecting.')
        return antigravity.disconnect()

@app.post('/api/enhancement/model')
def enhancement_model(body: EnhancementModel):
    try:
        return antigravity.set_model(body.model)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@app.post('/api/oauth/antigravity/test')
def connection_test():
    try:
        text = antigravity.generate_text(antigravity.selected_model(), 'Return only the word OK.', [{'text': 'Connection test. Reply OK.'}])
        if 'OK' not in text:
            raise antigravity.ConnectionError('The selected model did not return the expected test response.')
        return {'success': True, 'model': antigravity.selected_model(), 'connection': antigravity.status()}
    except antigravity.ConnectionError as exc:
        raise HTTPException(424, str(exc))

@app.get('/api/models')
def get_models():
    return models.state() | {'resident': image_worker.status()}

@app.post('/api/models/release')
def release_models():
    with active_lock:
        if active_job or pending_jobs:
            raise HTTPException(409, 'Finish or cancel queued jobs before releasing models.')
        image_worker.close()
        return image_worker.status()

@app.post('/api/models/{model_id}/download')
def download_model(model_id: str):
    try:
        return models.start_download(model_id)
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
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, 'This file is not a supported image.')
        prepared.append((picture, file.filename or 'Image'))
    results = []
    for picture, name in prepared:
        image_id = uuid.uuid4().hex + '.png'
        picture.save(UPLOADS_DIR / image_id)
        results.append({'id': image_id, 'name': name, 'url': f'/api/uploads/{image_id}', 'width': picture.width, 'height': picture.height})
    return results

@app.get('/api/uploads/{image_id}')
def uploaded_image(image_id: str):
    try:
        path = resolve_image(image_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return FileResponse(path)

@app.post('/api/generate', status_code=202)
def generate(body: GenerationRequest):
    try:
        needed = models.required(body.model_id, body.text_encoder)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not all(models.is_installed(name) for name in needed):
        raise HTTPException(503, 'A required model is not installed. Open Settings to download it.')
    if body.enhance and not antigravity.status()['connected']:
        raise HTTPException(409, 'Connect Google Account in Settings → Prompt enhancement, or turn off Enhance prompt.')
    try:
        for image_id in body.images:
            resolve_image(image_id)
        destination = Path(body.output_dir).expanduser().resolve()
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
        if not all(models.is_installed(name) for name in needed):
            raise HTTPException(503, 'A required model is not installed. Open Settings to download it.')
        job_id = uuid.uuid4().hex
        data = body.model_dump()
        default_steps = models.entry(body.model_id)['generation_defaults']['steps']
        data['steps'] = body.steps if body.steps is not None else default_steps
        data['steps_mode'] = 'custom' if body.steps is not None else 'model'
        # Store only a model selection. The worker resolves current tokens at execution.
        data['enhancement_model'] = antigravity.selected_model()
        data['output_dir'] = str(destination)
        if body.randomize_seed:
            data['seed'] = secrets.randbelow(2147483648)
        job = {'id': job_id, 'kind': 'image', 'status': 'queued', 'message': 'Waiting in queue', 'stage': 'queued', 'created_at': time.time(), 'seed': data['seed'], 'request': data, 'rewritten_prompt': '', 'step': 0, 'total_steps': data['steps']}
        enqueue_locked(job, data)
    return public_job(job)

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return public_job(read_job(job_id))

@app.post('/api/video/generate', status_code=202)
def generate_video(body: VideoRequest):
    try:
        needed=models.required_video(body)
        for image_id in (body.first_frame,body.last_frame):
            if image_id:
                resolve_image(image_id)
        destination=Path(body.output_dir).expanduser().resolve()
        destination.mkdir(parents=True,exist_ok=True)
        probe=destination/f'.write-test-{uuid.uuid4().hex}'
        probe.write_bytes(b'');probe.unlink()
        if shutil.disk_usage(destination).free<1024**3:
            raise ValueError('The output drive needs at least 1 GB of free space.')
    except (ValueError,OSError) as exc:
        raise HTTPException(400,str(exc))
    if body.enhance and not antigravity.status()['connected']:
        raise HTTPException(409,'Connect Google Account in Settings, or turn off Enhance prompt.')
    with active_lock:
        if not all(models.is_installed(name) for name in needed):
            raise HTTPException(503,'A required video model is not installed. Open Settings to download it.')
        data=body.model_dump()
        data.update(kind='video',output_dir=str(destination),enhancement_model=antigravity.selected_model())
        if body.randomize_seed:
            data['seed']=secrets.randbelow(2147483648)
        fast=models.entry(body.model_id)['family']=='fasth3'
        steps=8 if fast or body.quality=='turbo' else body.steps
        job_id=uuid.uuid4().hex
        job={'id':job_id,'kind':'video','status':'queued','message':'Waiting in queue','stage':'queued',
             'created_at':time.time(),'seed':data['seed'],'request':data,'rewritten_prompt':'','step':0,'total_steps':steps}
        enqueue_locked(job, data)
    return public_job(job)

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
        if active_process and active_process.poll() is None:
            stop_worker(active_process)
    return {'ok': True}

@app.get('/api/queue')
def get_queue():
    with active_lock:
        return {'paused': queue_paused,
                'active': public_job(read_job(active_job)) if active_job else None,
                'pending': [public_job(read_job(job_id)) | {'position': i + 1} for i, job_id in enumerate(pending_jobs)],
                'recent': [public_job(read_job(job_id)) for job_id in recent_jobs]}


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
    return generate_video(VideoRequest(**data)) if previous.get('kind') == 'video' else generate(GenerationRequest(**data))


@app.get('/api/library')
def library(kind: Literal['all', 'image', 'video'] = 'all', search: str = '',
            offset: int = Query(0, ge=0), limit: int = Query(48, ge=1, le=100)):
    query = search.casefold().strip()
    titles = {item['id']: item['title'] for item in models.catalog()}
    jobs = [j for j in all_jobs() if j.get('status') == 'succeeded'
            and (kind == 'all' or j.get('kind', 'image') == kind)
            and (not query or query in (j.get('request', {}).get('prompt', '') + ' ' + j.get('request', {}).get('model_id', '')
                 + ' ' + titles.get(j.get('request', {}).get('model_id', ''), '')).casefold())]
    jobs.sort(key=lambda j: j['created_at'], reverse=True)
    return {'items': [public_job(j) for j in jobs[offset:offset + limit]], 'total': len(jobs)}


@app.get('/api/history')
def history(kind: str = 'image'):
    jobs = []
    for path in JOBS_DIR.glob('*/status.json'):
        try:
            job = read_json(path)
            if job.get('status') == 'succeeded' and job.get('kind','image') == kind:
                jobs.append(public_job(job))
        except (ValueError, OSError):
            pass
    return sorted(jobs, key=lambda j: j['created_at'], reverse=True)[:100]

@app.get('/api/jobs/{job_id}/image')
def result(job_id: str, download: bool = False):
    job = read_job(job_id)
    if job.get('kind','image') != 'image' or job['status'] != 'succeeded' or not Path(job.get('output_path', '')).is_file():
        raise HTTPException(404, 'Saved image not found.')
    path = Path(job['output_path'])
    return FileResponse(path, media_type='image/png', filename=path.name if download else None)

@app.get('/api/jobs/{job_id}/video')
def video_result(job_id: str, download: bool = False):
    job=read_job(job_id)
    path=Path(job.get('output_path',''))
    if job.get('kind')!='video' or job['status']!='succeeded' or not path.is_file():
        raise HTTPException(404,'Saved video not found.')
    return FileResponse(path,media_type='video/mp4',filename=path.name if download else None)

@app.get('/api/jobs/{job_id}/poster')
def video_poster(job_id: str):
    job=read_job(job_id)
    path=JOBS_DIR/job_id/'poster.png'
    if job.get('kind')!='video' or job['status']!='succeeded' or not path.is_file():
        raise HTTPException(404,'Video preview not found.')
    return FileResponse(path,media_type='image/png')


@app.get('/api/jobs/{job_id}/thumbnail')
def thumbnail(job_id: str):
    job = read_job(job_id)
    if job['status'] != 'succeeded':
        raise HTTPException(404, 'Preview not found.')
    directory = JOBS_DIR / job_id
    cached = directory / 'thumbnail.webp'
    if not cached.exists():
        source = directory / 'poster.png' if job.get('kind') == 'video' else Path(job.get('output_path', ''))
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
    return FileResponse(cached, media_type='image/webp')

@app.get('/api/jobs/{job_id}/metadata')
def metadata(job_id: str):
    job = read_job(job_id)
    return JSONResponse(job, headers={'Content-Disposition': f'attachment; filename="{job_id}.json"'})

@app.post('/api/jobs/{job_id}/reuse')
def reuse(job_id: str):
    job = read_job(job_id)
    path = Path(job.get('output_path', ''))
    if job.get('kind','image') != 'image' or job['status'] != 'succeeded' or not path.is_file():
        raise HTTPException(404, 'Saved image not found.')
    image_id = uuid.uuid4().hex + '.png'
    shutil.copy2(path, UPLOADS_DIR / image_id)
    return {'id': image_id, 'name': path.name, 'url': f'/api/uploads/{image_id}'}

@app.post('/api/open-output')
def open_output(kind: str = 'image'):
    if os.name == 'nt':
        os.startfile(VIDEOS_DIR if kind=='video' else OUTPUTS_DIR)
    return {'ok': True}

if EXAMPLES_DIR.exists():
    app.mount('/examples', StaticFiles(directory=EXAMPLES_DIR), name='examples')
if (APP_DIR / 'dist').exists():
    app.mount('/', StaticFiles(directory=APP_DIR / 'dist', html=True), name='frontend')

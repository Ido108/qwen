from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .common import APP_DIR, DATA_DIR, EXAMPLES_DIR, JOBS_DIR, MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR, GenerationRequest, model_status, resolve_image, write_json
from . import models, antigravity

SESSION_TOKEN = secrets.token_urlsafe(32)
active_lock = threading.Lock()
active_job: str | None = None
active_process: subprocess.Popen | None = None
cancelled_jobs: set[str] = set()

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
        raise HTTPException(404, 'Image job not found.')
    try:
        return json.loads((JOBS_DIR / job_id / 'status.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise HTTPException(404, 'Image job not found.')

def public_job(job):
    return {k: v for k, v in job.items() if k not in ('output_path',)}

def run_job(job_id):
    global active_job, active_process
    directory = JOBS_DIR / job_id
    try:
        with (directory / 'worker.log').open('w', encoding='utf-8') as log:
            with active_lock:
                if job_id in cancelled_jobs:
                    job = read_job(job_id)
                    job.update(status='cancelled', message='Generation cancelled.', finished_at=time.time())
                    write_json(directory / 'status.json', job)
                    return
                active_process = subprocess.Popen(
                    [sys.executable, '-X', 'faulthandler', '-u', '-m', 'backend.worker', str(directory)],
                    cwd=APP_DIR, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
                process = active_process
            return_code = process.wait()
        with active_lock:
            job = read_job(job_id)
            if job_id in cancelled_jobs:
                job.update(status='cancelled', message='Generation cancelled.', finished_at=time.time())
                write_json(directory / 'status.json', job)
            elif job.get('status') not in ('succeeded', 'failed'):
                job.update(status='failed', message=f'The image process stopped (code {return_code}). Retry, or use a smaller image size.', finished_at=time.time())
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

@asynccontextmanager
async def lifespan(app):
    for path in JOBS_DIR.glob('*/status.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if job.get('status') in ('queued', 'running'):
                job.update(status='failed', message='The app closed before this image finished. You can retry.', finished_at=time.time())
                write_json(path, job)
        except (ValueError, OSError):
            pass
    yield
    with active_lock:
        if active_process and active_process.poll() is None:
            stop_worker(active_process)

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
    return {'token': SESSION_TOKEN, 'models': model_status(), 'catalog': models.state(), 'enhancement': antigravity.status(), 'output_dir': str(OUTPUTS_DIR), 'active_job': active_job, 'model': 'Qwen-Image-2.1', 'device': 'NVIDIA GeForce RTX 4090', 'data_dir': str(DATA_DIR)}

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
        if active_job:
            raise HTTPException(409, 'Wait for generation to finish before disconnecting.')
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
    return models.state()

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
        if active_job:
            raise HTTPException(409, 'Wait for generation to finish before removing models.')
        try:
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
    global active_job
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
        if active_job:
            raise HTTPException(409, 'An image is already being generated. Wait or cancel it first.')
        # Removal uses the same lock; recheck after acquiring it so another tab
        # cannot remove a dependency between validation and starting the worker.
        if not all(models.is_installed(name) for name in needed):
            raise HTTPException(503, 'A required model is not installed. Open Settings to download it.')
        job_id = uuid.uuid4().hex
        directory = JOBS_DIR / job_id
        directory.mkdir()
        data = body.model_dump()
        # Store only a model selection. The worker resolves current tokens at execution.
        data['enhancement_model'] = antigravity.selected_model()
        data['output_dir'] = str(destination)
        if body.randomize_seed:
            data['seed'] = secrets.randbelow(2147483648)
        job = {'id': job_id, 'status': 'queued', 'message': 'Starting…', 'stage': 'starting', 'created_at': time.time(), 'seed': data['seed'], 'request': data, 'rewritten_prompt': '', 'step': 0, 'total_steps': body.steps}
        write_json(directory / 'request.json', data)
        write_json(directory / 'status.json', job)
        active_job = job_id
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return public_job(job)

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return public_job(read_job(job_id))

@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str):
    with active_lock:
        if active_job != job_id:
            raise HTTPException(409, 'This image is no longer running.')
        cancelled_jobs.add(job_id)
        if active_process and active_process.poll() is None:
            stop_worker(active_process)
    return {'ok': True}

@app.get('/api/history')
def history():
    jobs = []
    for path in JOBS_DIR.glob('*/status.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if job.get('status') == 'succeeded':
                jobs.append(public_job(job))
        except (ValueError, OSError):
            pass
    return sorted(jobs, key=lambda j: j['created_at'], reverse=True)[:100]

@app.get('/api/jobs/{job_id}/image')
def result(job_id: str, download: bool = False):
    job = read_job(job_id)
    if job['status'] != 'succeeded' or not Path(job.get('output_path', '')).is_file():
        raise HTTPException(404, 'Saved image not found.')
    path = Path(job['output_path'])
    return FileResponse(path, media_type='image/png', filename=path.name if download else None)

@app.get('/api/jobs/{job_id}/metadata')
def metadata(job_id: str):
    job = read_job(job_id)
    return JSONResponse(job, headers={'Content-Disposition': f'attachment; filename="{job_id}.json"'})

@app.post('/api/jobs/{job_id}/reuse')
def reuse(job_id: str):
    job = read_job(job_id)
    path = Path(job.get('output_path', ''))
    if job['status'] != 'succeeded' or not path.is_file():
        raise HTTPException(404, 'Saved image not found.')
    image_id = uuid.uuid4().hex + '.png'
    shutil.copy2(path, UPLOADS_DIR / image_id)
    return {'id': image_id, 'name': path.name, 'url': f'/api/uploads/{image_id}'}

@app.post('/api/open-output')
def open_output():
    if os.name == 'nt':
        os.startfile(OUTPUTS_DIR)
    return {'ok': True}

if EXAMPLES_DIR.exists():
    app.mount('/examples', StaticFiles(directory=EXAMPLES_DIR), name='examples')
if (APP_DIR / 'dist').exists():
    app.mount('/', StaticFiles(directory=APP_DIR / 'dist', html=True), name='frontend')

"""Image animation, lip sync and face restoration run in the portrait engine (the Multifck app).

It keeps its own pinned Python 3.10 environment, which its installer builds for the graphics card of this computer,
so its versions of torch, ONNX Runtime and the face models never meet the app's own. A job starts the engine's local
server on a free port, sends the inputs and settings, follows the job and copies the result; the server stops after
the job, so no engine model stays in GPU memory between jobs.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

from .common import APP_DIR, DATA_DIR, LIBRARY_DIR, MODELS_DIR

ENGINE_REPOSITORY = 'https://github.com/Ido108/Multifck.git'
# The tested engine version. A newer commit is used only after it is tested with this app and pinned here.
ENGINE_COMMIT = 'ea42e91ddcbf6523594b21db7269219f53094351'
RUNTIME_DIR = Path(os.environ.get('QWEN_RUNTIME_DIR') or DATA_DIR).resolve()
ENGINE_DIR = RUNTIME_DIR / 'engines' / 'multifck'
ENGINE_PYTHON = ENGINE_DIR / ('.runtime/Scripts/python.exe' if os.name == 'nt' else '.runtime/bin/python')
ENGINE_DATA = DATA_DIR / 'portrait-engine'
ENGINE_MODELS = ((LIBRARY_DIR if LIBRARY_DIR is not None and LIBRARY_DIR.is_dir() else MODELS_DIR) / 'portrait').resolve()
INSTALL_LOG = RUNTIME_DIR / 'engines' / 'multifck-install.log'
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
# Weight files are written to a new name and renamed, so a hard link to a copy in another Multifck folder stays safe.
LINKABLE = {'.pth', '.pt', '.safetensors', '.onnx', '.bin', '.ckpt', '.tar', '.task', '.npy', '.pkl', '.model'}


class EngineError(RuntimeError):
    pass


def engine_env() -> dict:
    """The worker sets the app's offline Hugging Face cache; the engine downloads its optional models itself."""
    # The app launcher sets UV_PYTHON_PREFERENCE for its own interpreter.
    # The engine's pinned uv selects its private interpreter with --managed-python.
    environment = {key: value for key, value in os.environ.items()
                   if key not in {'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HOME', 'PYTHONPATH', 'PYTHONHOME',
                                  'VIRTUAL_ENV', 'UV_PYTHON_PREFERENCE'}}
    environment.update(MULTI_DATA_DIR=str(ENGINE_DATA), MULTI_MODELS_DIR=str(ENGINE_MODELS), PYTHONUTF8='1',
                       PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(ENGINE_DIR / '.runtime-cache' / 'numba'),
                       MULTI_DEVICE=installed_backend())
    return environment


def installed_backend() -> str:
    try:
        return json.loads((ENGINE_DIR / '.runtime' / 'install-state.json').read_text(encoding='utf-8'))['backend']
    except (OSError, ValueError, KeyError):
        return 'auto'


def git(*arguments, check=True, capture=True):
    return subprocess.run(['git', '-C', str(ENGINE_DIR), *arguments], check=check, capture_output=capture, text=True,
                          creationflags=NO_WINDOW, timeout=900)


def head() -> str:
    if not (ENGINE_DIR / '.git').exists():
        return ''
    result = git('rev-parse', 'HEAD', check=False)
    return result.stdout.strip() if result.returncode == 0 else ''


def local_source() -> Path | None:
    """A Multifck checkout next to this app (also from a worktree inside it): no download and no sign-in."""
    for parent in list(APP_DIR.parents)[:3]:
        candidate = parent / 'Multifck'
        if (candidate / '.git').exists() and (candidate / 'portrait_app').is_dir():
            return candidate
    return None


def quick_check() -> tuple[bool, str]:
    """The engine's own start check: right version, Python, dependencies, base models, and a build for this GPU."""
    if not ENGINE_PYTHON.is_file() or head() != ENGINE_COMMIT:
        return False, 'not installed'
    result = subprocess.run([str(ENGINE_PYTHON), 'scripts/verify_install.py', '--quick'], cwd=ENGINE_DIR, env=engine_env(),
                            capture_output=True, text=True, timeout=300, creationflags=NO_WINDOW)
    return result.returncode == 0, (result.stderr or result.stdout).strip()[-400:]


def status() -> dict:
    ready, detail = quick_check()
    return {'installed': ENGINE_PYTHON.is_file(), 'ready': ready, 'version': head()[:12], 'expected': ENGINE_COMMIT[:12],
            'detail': '' if ready else detail, 'models_dir': str(ENGINE_MODELS)}


def checkout(update) -> None:
    sources = [str(path) for path in (local_source(),) if path] + [ENGINE_REPOSITORY]
    if not (ENGINE_DIR / '.git').exists():
        ENGINE_DIR.parent.mkdir(parents=True, exist_ok=True)
        update(message='Getting the animation and lip sync tools…')
        subprocess.run(['git', 'clone', '--no-checkout', sources[0], str(ENGINE_DIR)], check=True, capture_output=True,
                       text=True, creationflags=NO_WINDOW, timeout=1800)
        git('remote', 'set-url', 'origin', ENGINE_REPOSITORY)
    if git('cat-file', '-e', ENGINE_COMMIT + '^{commit}', check=False).returncode != 0:
        for source in sources:
            git('fetch', '--quiet', source, '+refs/heads/*:refs/remotes/source/*', check=False)
            if git('cat-file', '-e', ENGINE_COMMIT + '^{commit}', check=False).returncode == 0:
                break
        else:
            raise EngineError('The animation tools could not be downloaded. Check the internet connection and retry.')
    # A clone with --no-checkout has HEAD on the default branch but no files yet, so HEAD can already be the pinned commit
    empty = {path.name for path in ENGINE_DIR.iterdir()} == {'.git'}
    if empty:
        git('checkout', '--quiet', '--force', '--detach', ENGINE_COMMIT)
    elif head() != ENGINE_COMMIT:
        changed = git('status', '--porcelain', '--untracked-files=no').stdout.strip()
        if changed:
            raise EngineError(f'The engine folder {ENGINE_DIR} has changed files. Move them away and retry.')
        git('checkout', '--quiet', '--detach', ENGINE_COMMIT)


def copy_needed(relative: Path, request: dict) -> bool:
    """On separate drives, copy only base weights and the selected variants."""
    folder, name = relative.parts[0], relative.name
    if folder in {'insightface', 'liveportrait', 'musetalk', 'rvc', 'gfpgan'} or relative.as_posix() == 'tpsmm/vox.pth.tar':
        return True
    options = request.get('options', {})
    kind = request.get('kind')
    animation = options.get('animation_engine', 'imtalker') if kind == 'animate' else None
    lip = options.get('sync_model', 'musetalk') if kind == 'lipsync' or (animation and options.get('lipsync')) else None
    if folder == 'face_mesh':
        return bool(animation and options.get('face_analysis') == 'mediapipe')
    if folder == 'imtalker' and name == 's3fd.pth':
        return (lip in ('wav2lip', 'wav2lip_gan') and options.get('wav2lip_face_box', 'detector') == 'detector' or
                animation == 'imtalker' and options.get('imtalker_face_fit') == 'box')
    if folder == 'tpsmm':
        return animation == folder and name == options.get('tps_checkpoint', 'vox') + '.pth.tar'
    if folder == 'tbdub':
        return lip in ('tbdub', 'tbdub_teacher') and (not name.startswith('tbdub_') or
                name == ('tbdub_teacher.safetensors' if lip == 'tbdub_teacher' else 'tbdub_student.safetensors'))
    if folder == 'wav2lip':
        return name == f'{lip}.pth'
    if folder == 'mrfa':
        return animation == folder and name == options.get('mrfa_checkpoint', 'vox') + '.safetensors'
    return folder in (animation, lip)


def seed_models(request=None) -> int:
    """Hard links to the weights of a Multifck install next to this app, so the same files are not downloaded twice."""
    source = local_source()
    if source is None or not (source / 'models').is_dir():
        return 0
    linked = 0
    for path in (source / 'models').rglob('*'):
        config = path.name in {'config.json', 'preprocessor_config.json', 'model_index.json'}
        if not path.is_file() or not (path.suffix.lower() in LINKABLE or config):
            continue
        relative = path.relative_to(source / 'models')
        target = ENGINE_MODELS / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not config:
                os.link(path, target)
                linked += 1
                continue
        except FileExistsError:
            continue
        except OSError as error:
            if error.errno != errno.EXDEV and getattr(error, 'winerror', None) != 17:
                continue
        if not copy_needed(relative, request or {}) and (not config or path.stat().st_dev != target.parent.stat().st_dev):
            continue
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
        try:
            shutil.copyfile(path, temporary)
            temporary.replace(target)
            linked += 1
        finally:
            temporary.unlink(missing_ok=True)
    return linked


def install(update, request=None) -> None:
    """First use, or a new pinned version: the engine's own installer, which detects the GPU and tests every model."""
    update(stage='installing', message='Installing the animation and lip sync tools…', step=0, total_steps=0)
    checkout(update)
    ENGINE_MODELS.mkdir(parents=True, exist_ok=True)
    ENGINE_DATA.mkdir(parents=True, exist_ok=True)
    seed_models(request)
    if os.name == 'nt':
        command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ENGINE_DIR / 'Install.ps1'), '-NoShortcuts']
    else:
        command = ['bash', str(ENGINE_DIR / 'Install.command')]
    installer_error = ''
    with INSTALL_LOG.open('a', encoding='utf-8') as log:
        log.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} engine {ENGINE_COMMIT}\n')
        process = subprocess.Popen(command, cwd=ENGINE_DIR, env=engine_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   encoding='utf-8', errors='replace', bufsize=1, creationflags=NO_WINDOW)
        for line in process.stdout:
            clean = line.strip()
            if not clean:
                continue
            log.write(clean + '\n')
            log.flush()
            print(clean, flush=True)
            if clean.lower().startswith('error:') or (not installer_error and
                    clean.lower().startswith(('installation failed:', 'runtimeerror:'))):
                installer_error = clean
            if clean.startswith(('Downloading', 'Installing for')):
                update(message='Installing the animation and lip sync tools: downloading…')
            elif clean.startswith('Testing '):
                update(message='Installing the animation and lip sync tools: testing on this GPU…')
        code = process.wait()
    if code:
        raise EngineError('The animation tools did not install. ' + (installer_error or f'Exit code {code}. See {INSTALL_LOG}.'))
    ready, detail = quick_check()
    if not ready:
        raise EngineError('The animation tools did not install. ' + (detail or f'See {INSTALL_LOG}.'))


def ensure(update, request=None) -> None:
    if not quick_check()[0]:
        install(update, request)
    else:
        seed_models(request)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def stop_tree(pid: int) -> None:
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], capture_output=True, creationflags=NO_WINDOW)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, 9)


class Engine:
    """One engine server for one job, started in the worker process, so Cancel (which ends the worker tree) ends it too."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.port = free_port()
        self.url = f'http://127.0.0.1:{self.port}'
        self.process = None
        self.client = None

    def __enter__(self):
        import httpx
        ENGINE_DATA.mkdir(parents=True, exist_ok=True)
        # The engine restores its waiting jobs on start; a job of a cancelled run must not run before this one.
        (ENGINE_DATA / 'queue-state.json').unlink(missing_ok=True)
        self.log = (self.directory / 'engine.log').open('a', encoding='utf-8')
        self.process = subprocess.Popen([str(ENGINE_PYTHON), '-u', '-m', 'portrait_app', '--no-browser', '--port', str(self.port)],
                                        cwd=ENGINE_DIR, env=engine_env(), stdout=self.log, stderr=subprocess.STDOUT,
                                        creationflags=NO_WINDOW)
        self.client = httpx.Client(base_url=self.url, timeout=httpx.Timeout(60, connect=5))
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EngineError(f'The animation tools stopped while starting. See {self.directory / "engine.log"}.')
            with contextlib.suppress(httpx.HTTPError, ValueError):
                health = self.client.get('/api/health', timeout=2).json()
                if health.get('ready') and Path(health.get('root', '')).resolve() == ENGINE_DIR.resolve():
                    return self
            time.sleep(.25)
        raise EngineError('The animation tools did not start within 3 minutes.')

    def __exit__(self, *exc):
        control = ENGINE_DIR / '.runtime' / f'server-{self.port}.json'
        try:
            if self.process and self.process.poll() is None:
                with contextlib.suppress(Exception):
                    token = json.loads(control.read_text())['token']
                    self.client.post('/api/restart', json={'token': token}, timeout=5)
                    self.process.wait(timeout=20)
        finally:
            if self.process and self.process.poll() is None:
                stop_tree(self.process.pid)
            if self.client:
                self.client.close()
            control.unlink(missing_ok=True)
            self.log.close()

    def check(self, response):
        if response.status_code >= 400:
            try:
                detail = response.json().get('detail')
            except ValueError:
                detail = response.text
            if isinstance(detail, list):
                detail = '; '.join(str(item.get('msg', item)) for item in detail)
            raise EngineError(str(detail or f'Request failed ({response.status_code}).'))
        return response.json()

    def upload(self, kind: str, path: Path) -> str:
        path = Path(path).resolve()
        stat = path.stat()
        identity = json.dumps([kind, str(path), stat.st_size, stat.st_mtime_ns]).encode()
        directory = ENGINE_DATA / 'input-references'
        directory.mkdir(exist_ok=True, parents=True)
        entry = directory / (hashlib.sha256(identity).hexdigest() + '.json')
        try:
            identifier = json.loads(entry.read_text())['id']
            if isinstance(identifier, str) and len(identifier) == 64 and all(c in '0123456789abcdef' for c in identifier):
                upload = ENGINE_DATA / 'uploads' / identifier
                if upload.with_suffix('.json').is_file() and upload.with_suffix(path.suffix.lower()).is_file():
                    return identifier
        except (OSError, ValueError, KeyError):
            pass
        with path.open('rb') as stream:
            identifier = self.check(self.client.post(f'/api/upload/{kind}', files={'file': (path.name, stream)}, timeout=600))['id']
        temporary = entry.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps({'id': identifier}))
            temporary.replace(entry)
        finally:
            temporary.unlink(missing_ok=True)
        return identifier

    def run(self, fields: dict, update, cancelled=lambda: False) -> dict:
        created = self.check(self.client.post('/api/jobs', json=fields))
        identifier = created['id']
        preview_stamp = None
        framing_saved = False
        while True:
            job = self.check(self.client.get(f'/api/jobs/{identifier}'))
            state = job.get('status')
            if not framing_saved and (job.get('analysis') or {}).get('alignment'):
                saved = ENGINE_DATA / 'jobs' / identifier / 'alignment.jpg'
                if saved.is_file():
                    temporary = self.directory / 'framing.partial.jpg'
                    shutil.copyfile(saved, temporary)
                    temporary.replace(self.directory / 'framing.jpg')
                    update(framing_url=f'/api/jobs/{self.directory.name}/framing')
                    framing_saved = True
            if state == 'completed':
                count = int((job.get('stats') or {}).get('frames') or 0)
                if count:
                    update(step=count, total_steps=count, engine_progress=1.)
                return job
            if state in ('failed', 'cancelled'):
                raise EngineError(job.get('error') or job.get('message') or 'The job did not finish.')
            if cancelled():
                with contextlib.suppress(Exception):
                    self.client.post(f'/api/jobs/{identifier}/cancel')
                raise EngineError('Cancelled.')
            stage = 'processing'
            values = {'stage': stage, 'message': job.get('message') or 'Working…',
                      'engine_progress': round(float(job.get('progress') or 0), 4), 'engine_stage': job.get('stage') or ''}
            total, current = int(job.get('total') or 0), int(job.get('current') or 0)
            frames = job.get('progress_unit') == 'frames'
            values.update(step=current if frames else 0, total_steps=total if frames else 0)
            update(**values)
            preview_stamp = self.save_preview(identifier, preview_stamp, update)
            time.sleep(.5)

    def save_preview(self, identifier: str, previous, update):
        from PIL import Image
        source = ENGINE_DATA / 'jobs' / identifier / 'preview.jpg'
        try:
            stamp = source.stat().st_mtime_ns
            if stamp == previous:
                return previous
            target = self.directory / 'live-preview.webp'
            temporary = target.with_suffix('.webp.part')
            with Image.open(source) as frame:
                frame.thumbnail((960, 540))
                frame.convert('RGB').save(temporary, format='WEBP', quality=80)
            temporary.replace(target)
            update(preview_url=f'/api/jobs/{self.directory.name}/live-preview', preview_updated_at=time.time())
            return stamp
        except OSError:
            return previous

    def output(self, job: dict) -> Path:
        path = ENGINE_DATA / 'outputs' / job['output']
        if not path.is_file():
            raise EngineError('The result file is missing.')
        return path


def models() -> list[dict]:
    """The engine's model list with install state and whether this GPU can run each one (no server needed)."""
    if not ENGINE_PYTHON.is_file():
        return []
    script = ('import json\nfrom portrait_app.model_catalog import inventory\nfrom portrait_app.capabilities import model_support\n'
              'items=inventory(False)["models"]\nfor m in items: m.update(model_support(m["id"]))\n'
              'print(json.dumps([{k:m.get(k) for k in ("id","name","task","description","installed","supported","reason","files")} for m in items]))')
    result = subprocess.run([str(ENGINE_PYTHON), '-c', script], cwd=ENGINE_DIR, env=engine_env(), capture_output=True,
                            text=True, timeout=120, creationflags=NO_WINDOW)
    if result.returncode:
        return []
    items = json.loads(result.stdout.strip().splitlines()[-1])
    for item in items:
        files = item.pop('files', None) or []
        item['bytes'] = sum(int(f.get('expected_bytes') or f.get('bytes') or 0) for f in files)
    return items


def remove_engine_output(path: Path) -> None:
    """The engine's copy of a result is moved out; the engine's own sidecar files go with it."""
    path.unlink(missing_ok=True)
    for sidecar in path.parent.glob(path.stem + '.*'):
        sidecar.unlink(missing_ok=True)


def move_output(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.replace(destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, 'winerror', None) != 17:
            raise
    else:
        remove_engine_output(source)
        return destination
    temporary = destination.with_name(destination.stem + '.partial' + destination.suffix)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    remove_engine_output(source)
    return destination

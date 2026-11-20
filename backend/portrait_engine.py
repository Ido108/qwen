"""Image animation, lip sync, face restore and audio extraction run in the portrait engine (the Multifck app).

It keeps its own pinned Python 3.10 environment, which its installer builds for the graphics card of this computer,
so its versions of torch, ONNX Runtime and the face models never meet the app's own. A job starts the engine's local
server on a free port, sends the inputs and settings, follows the job and copies the result; the server stops after
the job, so no engine model stays in GPU memory between jobs.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from .common import APP_DIR, DATA_DIR, LIBRARY_DIR, MODELS_DIR

ENGINE_REPOSITORY = 'https://github.com/Ido108/Multifck.git'
# The tested engine version. A newer commit is used only after it is tested with this app and pinned here.
ENGINE_COMMIT = 'd5eef5d39ac530ca13d5e7436641a963636c6d3c'
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
    environment = {key: value for key, value in os.environ.items()
                   if key not in {'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HOME', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}}
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


def seed_models() -> int:
    """Hard links to the weights of a Multifck install next to this app, so the same files are not downloaded twice."""
    source = local_source()
    if source is None or not (source / 'models').is_dir():
        return 0
    linked = 0
    for path in (source / 'models').rglob('*'):
        if not path.is_file() or path.suffix.lower() not in LINKABLE or path.name.endswith('.partial'):
            continue
        target = ENGINE_MODELS / path.relative_to(source / 'models')
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
            linked += 1
        except OSError:
            return linked  # another drive: the installer downloads what is missing
    return linked


def install(update) -> None:
    """First use, or a new pinned version: the engine's own installer, which detects the GPU and tests every model."""
    update(stage='installing', message='Installing the animation and lip sync tools…', step=0, total_steps=0)
    checkout(update)
    ENGINE_MODELS.mkdir(parents=True, exist_ok=True)
    ENGINE_DATA.mkdir(parents=True, exist_ok=True)
    seed_models()
    if os.name == 'nt':
        command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ENGINE_DIR / 'Install.ps1'), '-NoShortcuts']
    else:
        command = ['bash', str(ENGINE_DIR / 'Install.command')]
    tail = []
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
            tail = (tail + [clean])[-12:]
            if clean.startswith(('Downloading', 'Installing for')):
                update(message='Installing the animation and lip sync tools: downloading…')
            elif clean.startswith('Testing '):
                update(message='Installing the animation and lip sync tools: testing on this GPU…')
        code = process.wait()
    ready, detail = quick_check()
    if code or not ready:
        raise EngineError('The animation tools did not install. ' + (next((line for line in reversed(tail) if 'failed' in line.lower()), detail) or
                          f'See {INSTALL_LOG}.'))


def ensure(update) -> None:
    if not quick_check()[0]:
        install(update)


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
        with Path(path).open('rb') as stream:
            return self.check(self.client.post(f'/api/upload/{kind}', files={'file': (Path(path).name, stream)}, timeout=600))['id']

    def run(self, fields: dict, update, cancelled=lambda: False) -> dict:
        created = self.check(self.client.post('/api/jobs', json=fields))
        identifier = created['id']
        preview_at = 0.0
        while True:
            job = self.check(self.client.get(f'/api/jobs/{identifier}'))
            state = job.get('status')
            if state == 'completed':
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
            values.update(step=current if total else 0, total_steps=total)
            update(**values)
            if time.monotonic() - preview_at > 1.5 and stage == 'processing':
                preview_at = time.monotonic()
                self.save_preview(identifier, update)
            time.sleep(.5)

    def save_preview(self, identifier: str, update) -> None:
        from io import BytesIO
        from PIL import Image
        with contextlib.suppress(Exception):
            response = self.client.get(f'/api/jobs/{identifier}/preview', timeout=5)
            if response.status_code != 200 or not response.content:
                return
            target = self.directory / 'live-preview.webp'
            temporary = target.with_suffix('.webp.part')
            Image.open(BytesIO(response.content)).convert('RGB').save(temporary, format='WEBP', quality=80)
            temporary.replace(target)
            update(preview_url=f'/api/jobs/{self.directory.name}/live-preview', preview_updated_at=time.time())

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
    temporary = destination.with_name(destination.stem + '.partial' + destination.suffix)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    remove_engine_output(source)
    return destination

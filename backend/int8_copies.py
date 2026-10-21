"""INT8 ConvRot copies of the user's BF16, FP16 and FP32 Qwen-Image-2.1 and MOSS Image models, made once on this computer and used automatically.

On an RTX 5090 the official INT8 ConvRot file runs a Qwen-Image-2.1 step about 6x faster than BF16 (measured 2026-10-22,
edit with 2 photos at 896x1152: 0.35 against 2.04 s/step at CFG 2). backend.int8_convert writes the official Comfy-Org
layout; for the two official BF16 models its output is byte for byte the official INT8 file, so those use the official
file when it is installed and need no copy.

int8-copies.json (data folder): {'copies': {model id: {...}}, 'original': [model ids that run their own file],
'failed': {model id: {'message', 'source'}}}. The original file is never changed or removed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import APP_DIR, DATA_DIR, LIBRARY_DIR, MODELS_DIR, write_json
from . import int8_convert

REGISTRY = DATA_DIR / 'int8-copies.json'
LOG = DATA_DIR / 'logs' / 'int8-convert.log'
RESERVE_BYTES = 3 * 1024**3  # free space kept on the drive after the copy
IDLE_SECONDS = 60  # the queue stays empty this long before a conversion starts
# BF16 models whose INT8 conversion is the official INT8 file: byte for byte for Qwen (int8_convert.compare, 2026-10-22);
# for MOSS the planned header is the official header byte for byte and every layer checked matched bit for bit.
OFFICIAL = {'official-bf16': 'int8-convrot', 'turbo-bf16': 'turbo-int8-convrot', 'moss-sft-bf16': 'moss-sft-int8', 'moss-base-bf16': 'moss-base-int8'}
FAMILIES = ('qwen-image-2.1', 'moss-image')
LABEL_MADE = 'INT8 (made on this computer)'
LABEL_OFFICIAL = 'INT8 (official file)'
_lock = threading.RLock()
_plans = {}
_running = {}  # model id -> {'process', 'target', 'status', 'flag', 'source', 'started'}
SCAN_SECONDS = 60
_scanned = [0.0]


def read():
    try:
        data = json.loads(REGISTRY.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    return {'copies': data.get('copies', {}), 'original': data.get('original', []), 'failed': data.get('failed', {})}


def save(data):
    write_json(REGISTRY, data)


def source_paths(item):
    """The transformer files of a model: one ComfyUI file, or the Diffusers shards. None for GGUF and other layouts."""
    from .models import local_path
    files = item.get('files') or []
    if item.get('format') == 'GGUF' or any(f['path'].endswith('.gguf') for f in files):
        return None
    if item.get('layout') == 'comfy':
        weights = [f for f in files if f['path'].endswith(('.safetensors', '.sft'))]
        return [local_path(item, weights[0])] if len(weights) == 1 else None
    shards = [f for f in files if f['path'].startswith('transformer/') and f['path'].endswith('.safetensors')]
    return [local_path(item, f) for f in sorted(shards, key=lambda f: f['path'])] or None


def signature(paths):
    return [{'path': str(p), 'bytes': p.stat().st_size, 'mtime': int(p.stat().st_mtime)} for p in paths]


def candidate(item):
    """(source paths, planned output bytes) for an installed float Qwen-Image-2.1 or MOSS Image model, else None. Headers are read once per file."""
    if item.get('kind') != 'image' or not str(item.get('family', '')).startswith(FAMILIES):
        return None
    try:
        paths = source_paths(item)
        if not paths or not all(p.is_file() for p in paths):
            return None
        key = tuple((str(p), p.stat().st_size, p.stat().st_mtime) for p in paths)
    except (OSError, ValueError, KeyError):
        return None
    with _lock:
        if key not in _plans:
            try:
                planned = int8_convert.plan(int8_convert.source_files(paths), turbo(item))
                _plans[key] = int8_convert.output_bytes(planned)
            except (OSError, ValueError, KeyError):
                _plans[key] = None
        size = _plans[key]
    return (paths, size) if size else None


def turbo(item):
    return item.get('family') == 'qwen-image-2.1-turbo'


def _official_path(model_id):
    from .models import entry, is_installed, local_path
    official = OFFICIAL.get(model_id)
    try:
        if official and is_installed(official):
            item = entry(official)
            return local_path(item, item['files'][0])
    except (ValueError, OSError):
        pass
    return None


def _ready_copy(model_id, item, data):
    """This model's copy, or the copy of another entry of the same file (MOSS Serenity v2 and its Turbo entry share one)."""
    try:
        paths = source_paths(item)
        current = signature(paths) if paths else None
    except (OSError, ValueError, KeyError):
        return None
    if not current:
        return None
    for copy in [data['copies'].get(model_id), *data['copies'].values()]:
        if not copy or copy['source'] != current:
            continue
        path = Path(copy['path'])
        try:
            if path.is_file() and path.stat().st_size == copy['bytes']:
                return path
        except OSError:
            pass
    return None


def active(item):
    """The INT8 file to load in place of this model's own file, or None (no copy yet, or the user chose the original)."""
    try:
        data = read()
        if item['id'] in data['original']:
            return None
        return _official_path(item['id']) or _ready_copy(item['id'], item, data)
    except Exception:  # never stop a job over the copy
        return None


def active_for_id(model_id):
    try:
        from .models import entry
        path = active(entry(model_id))
        return str(path) if path else None
    except Exception:
        return None


def info(item):
    """What the model list shows about this model's INT8 copy, or None when the model has none."""
    model_id = item['id']
    if model_id in OFFICIAL and _official_path(model_id):
        data = read()
        return {'status': 'original' if model_id in data['original'] else 'ready', 'official': True, 'label': LABEL_OFFICIAL,
                'path': str(_official_path(model_id)), 'bytes': _official_path(model_id).stat().st_size}
    found = candidate(item)
    if not found:
        return None
    data = read()
    result = {'official': False, 'label': LABEL_MADE, 'bytes': found[1]}
    ready = _ready_copy(model_id, item, data)
    with _lock:
        running = _running.get(model_id)
    if ready:
        result.update(status='original' if model_id in data['original'] else 'ready', path=str(ready))
    elif running:
        progress = {}
        try:
            progress = json.loads(Path(running['status']).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
        result.update(status='converting', done=progress.get('done', 0), total=progress.get('total', 0), path=str(running['target']))
    elif model_id in data['failed']:
        result.update(status='failed', message=data['failed'][model_id].get('message', 'The conversion failed.'))
    elif model_id in data['original']:
        result.update(status='original')
    else:
        target, reason = target_path(item, found[0], found[1])
        result.update(status='waiting' if target else 'no_space', message=reason)
    return result


def title(item, base):
    """The model list title: the model's own title, plus how it runs when the INT8 copy is in use."""
    state = info(item)
    if state and state['status'] == 'ready':
        return f"{base} · runs as {state['label']}"
    return base


def set_choice(model_id, use):
    """use: 'int8' (the default: the INT8 copy, made again after a failure) or 'original' (the model's own file)."""
    if use not in ('int8', 'original'):
        raise ValueError('Choose int8 or original.')
    from .models import entry
    item = entry(model_id)
    if not (model_id in OFFICIAL or candidate(item)):
        raise ValueError('This model has no INT8 copy.')
    with _lock:
        data = read()
        data['original'] = [m for m in data['original'] if m != model_id] + ([model_id] if use == 'original' else [])
        if use == 'int8':
            data['failed'].pop(model_id, None)
            _scanned[0] = 0.0  # look for work at the next idle tick
        save(data)
    return info(item)


def target_path(item, paths, size):
    """Where the copy goes: next to the shared model library when its drive keeps 3 GB free after it, else the data folder.
    (None, reason) when no drive has room."""
    name = (paths[0].stem if len(paths) == 1 else item['id']) + '_int8_convrot.safetensors'
    folders = ([LIBRARY_DIR / 'diffusion_models' / 'int8-local'] if LIBRARY_DIR is not None and LIBRARY_DIR.is_dir() else []) + [MODELS_DIR / 'int8-local']
    for folder in folders:
        try:
            anchor = next(p for p in [folder, *folder.parents] if p.exists())
            free = shutil.disk_usage(anchor).free
        except (OSError, StopIteration):
            continue
        if free >= size + RESERVE_BYTES:
            return folder / name, ''
    return None, f'Needs {size / 1e9:.1f} GB plus 3 GB free on the model drive or the data drive.'


def paths():
    """Every INT8 copy file (the library scan skips them: they belong to their source model)."""
    return [copy['path'] for copy in read()['copies'].values()]


def remove(model_id):
    """The source model was removed: delete the copy the app made for it (never an official or user file)."""
    with _lock:
        run = _running.pop(model_id, None)
        if run and run['process'].poll() is None:
            run['process'].terminate()
            run['process'].wait(timeout=30)
        data = read()
        copy = data['copies'].pop(model_id, None)
        data['failed'].pop(model_id, None)
        data['original'] = [m for m in data['original'] if m != model_id]
        save(data)
    if copy and Path(copy['path']).name.endswith('_int8_convrot.safetensors') and 'int8-local' in Path(copy['path']).parts:
        Path(copy['path']).unlink(missing_ok=True)


def pending():
    """Installed models that still need a copy, in catalog order."""
    from .models import catalog, is_installed
    data = read()
    found = []
    for item in catalog():
        model_id = item['id']
        if model_id in OFFICIAL and _official_path(model_id):
            continue
        if model_id in data['failed'] or model_id in data['original'] or model_id in _running:
            continue
        try:
            if not is_installed(model_id):
                continue
        except Exception:
            continue
        plan = candidate(item)
        if plan and not _ready_copy(model_id, item, data) and not any(run['source'] == signature(plan[0]) for run in _running.values()):
            found.append((item, plan))
    return found


def _start(item, plan):
    paths, size = plan
    target, reason = target_path(item, paths, size)
    if not target:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    status, flag = target.with_name(target.name + '.status.json'), target.with_name(target.name + '.use-cpu')
    flag.unlink(missing_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', 'backend.int8_convert', *map(str, paths), '--target', str(target),
               '--status', str(status), '--cpu-flag', str(flag), *(['--turbo'] if turbo(item) else [])]
    with LOG.open('a', encoding='utf-8') as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {item['id']}: {' '.join(command)}\n")
        log.flush()
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == 'nt' else 0
        process = subprocess.Popen(command, cwd=APP_DIR, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    with _lock:
        _running[item['id']] = {'process': process, 'target': target, 'status': status, 'flag': flag,
                                'source': signature(paths), 'started': time.time(), 'bytes': size}
    return True


def _finish(model_id, run):
    code = run['process'].returncode
    target = Path(run['target'])
    with _lock:
        data = read()
        ok = code == 0 and target.is_file() and target.stat().st_size == run['bytes']
        if ok:
            try:
                int8_convert.read_header(target)
            except (OSError, ValueError):
                ok = False
        if ok:
            data['copies'][model_id] = {'path': str(target), 'bytes': run['bytes'], 'source': run['source'], 'made_at': time.time(),
                                        'seconds': round(time.time() - run['started'], 1), 'method': int8_convert.METHOD,
                                        'version': int8_convert.VERSION}
            data['failed'].pop(model_id, None)
        else:
            data['failed'][model_id] = {'message': f'The INT8 conversion stopped (exit code {code}). See logs/int8-convert.log.',
                                        'source': run['source']}
        save(data)
        _running.pop(model_id, None)
    for leftover in (run['status'], run['flag'], target.with_name(target.name + '.part')):
        Path(leftover).unlink(missing_ok=True)


def tick(busy):
    """One scheduler step: finish ended conversions, move a running one to the CPU while a job runs, start the next when idle."""
    with _lock:
        running = list(_running.items())
    for model_id, run in running:
        if run['process'].poll() is not None:
            _finish(model_id, run)
        elif busy():
            Path(run['flag']).touch()
    with _lock:
        if _running:
            return
    if busy() or time.monotonic() - _scanned[0] < SCAN_SECONDS:
        return
    _scanned[0] = time.monotonic()
    for item, plan in pending():
        if _start(item, plan):
            return


def start(busy, stopped):
    """Background thread: converts one model at a time, only after the queue and downloads are idle for a minute."""
    def loop():
        idle_since = None
        while not stopped.wait(5):
            try:
                now = time.monotonic()
                is_busy = busy()
                idle_since = None if is_busy else (idle_since or now)
                tick(lambda: busy() or idle_since is None or now - idle_since < IDLE_SECONDS)
            except Exception as exc:  # the next tick tries again
                print(f'[INT8] {type(exc).__name__}: {exc}', flush=True)
        with _lock:
            for run in _running.values():
                if run['process'].poll() is None:
                    run['process'].terminate()
    thread = threading.Thread(target=loop, name='int8-copies', daemon=True)
    thread.start()
    return thread

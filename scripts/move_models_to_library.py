"""Move the app's model files from the data folder (E:\\AI\\qwen-image-studio\\models) into the shared ComfyUI library
(models_library_dir in local-settings.json, for example D:\\models), sorted by type: diffusion_models, text_encoders, vae, loras, ...

models.local_path() reads the library copy first and the data folder second, so a file works from either drive at every step.
Files that do not fit stay in the data folder. A file the library already has (same size and content, anywhere in it) is not
copied again: the app records it in .qwen-library.json and uses it in place.

    D:\\AI\\qwen-image-studio\\venv\\Scripts\\python.exe scripts\\move_models_to_library.py                    plan only
    ... --execute                  copy and verify (resumable; the data folder copies stay)
    ... --execute --delete-source  also delete each data folder copy after its library copy is verified
    ... --scan                     classify the user's own library models (library-scan.json) without moving anything
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
# Same folders as Start Qwen.bat (scripts/paths.ps1) when run on its own.
settings = json.loads((APP / 'local-settings.json').read_text(encoding='utf-8-sig')) if (APP / 'local-settings.json').is_file() else {}
for name, key in (('QWEN_DATA_DIR', 'data_dir'), ('QWEN_RUNTIME_DIR', 'runtime_dir'), ('QWEN_MODELS_LIBRARY', 'models_library_dir')):
    if not os.environ.get(name) and settings.get(key):
        os.environ[name] = settings[key]
sys.path.insert(0, str(APP))
from backend.common import DATA_DIR, LIBRARY_DIR, MODELS_DIR  # noqa: E402
from backend import model_library, models  # noqa: E402

CHUNK = 64 * 1024**2
SAMPLE = 16 * 1024**2
LOG = DATA_DIR / 'logs' / 'library-move.log'
USE_KEYS = ('selected_model', 'model_id', 'text_encoder_id', 'vae_id', 'id', 'tool_id', 'requested_id')
# The models each tab starts with (src/main.tsx, VideoTab, ReferenceTab, OrbitTab, MultiReferenceTab, upscaleSettings).
DEFAULTS = ('gguf-q8_0', 'int8-convrot', 'turbo-int8-convrot', 'fasth3-q4_k_m', 'h3-int8', 'h3-ref-turbo-int8', 'ltx25-msr-q4_k_m', 'seedvr2-3b-fp16')


def log(text):
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')


def gb(value):
    return f'{value / 1024**3:.1f} GB'


def samples(path, size):
    """SHA-256 of the head, middle and tail (16 MB each): a fast identity check for multi-GB files."""
    digest = hashlib.sha256(str(size).encode())
    with open(path, 'rb') as handle:
        for offset in sorted({0, max(0, size // 2 - SAMPLE // 2), max(0, size - SAMPLE)}):
            handle.seek(offset)
            digest.update(handle.read(SAMPLE))
    return digest.hexdigest()


def full_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def usage():
    """{model id or title: (uses, last use time)} from job logs, saved results and the queue."""
    found = {}
    def walk(value, when):
        if isinstance(value, dict):
            for name, inner in value.items():
                if name in USE_KEYS and isinstance(inner, str):
                    uses, last = found.get(inner, (0, 0))
                    found[inner] = (uses + 1, max(last, when))
                elif isinstance(inner, (dict, list)):
                    walk(inner, when)
        elif isinstance(value, list):
            for inner in value:
                walk(inner, when)
    sources = [*(DATA_DIR / 'jobs').glob('*/worker.log'), *(DATA_DIR / 'jobs').glob('*/*.json'), *(DATA_DIR / 'images').glob('*.json'),
               *(DATA_DIR / 'videos').glob('*.json'), DATA_DIR / 'queue-state.json']
    for path in sources:
        try:
            when, text = path.stat().st_mtime, path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for line in text.splitlines() if path.suffix == '.log' else [text]:
            try:
                walk(json.loads(line), when)
            except ValueError:
                pass
    return found


def plan():
    """Every data-folder file to move, in priority order, with what fits the library drive."""
    entries = models.catalog()
    by_id = {item['id']: item for item in entries}
    used = usage()
    score = {item['id']: used.get(item['id'], used.get(item['title'], (0, 0))) for item in entries}
    # A used model brings its encoder, VAE and other dependencies with it.
    first = {model_id for model_id in DEFAULTS if model_id in by_id}
    for item in sorted(entries, key=lambda item: score[item['id']], reverse=True):
        for dependency in item.get('dependencies', []):
            if dependency in score and score[item['id']][0]:
                score[dependency] = max(score[dependency], score[item['id']])
            if dependency in score and item['id'] in first:
                first.add(dependency)
    def tier(item):
        if score[item['id']][0] or item['id'] in first:
            return 0
        if item['kind'] in ('encoder', 'vae') or item['kind'].endswith('-support'):
            return 1
        return 2 if item['kind'].endswith('lora') else 3
    # Upscalers load companion files from one folder; a Diffusers model is one folder too. Each group moves whole or not at all.
    groups = {}
    for item in entries:
        group = ('folder', item['directory']) if item['kind'] in model_library.TOGETHER else ('entry', item['id'])
        files = []
        # Only the files the app loads; leftovers of older layouts (removal_files) stay until the model is removed.
        for file in item['files']:
            source = models.store_path(item, file)
            try:
                size = source.stat().st_size
            except OSError:
                continue
            if file.get('bytes') and size != file['bytes']:
                continue  # an incomplete download stays where it is
            files.append((item, file, source, size))
        if files:
            groups.setdefault(group, []).extend(files)
    def last_use(files):
        return max(max(score[item['id']][1], source.stat().st_mtime) for item, _, source, _ in files)
    order = sorted(groups.values(), key=lambda files: (min(tier(item) for item, *_ in files),
                                                       -max(score[item['id']][0] for item, *_ in files), -last_use(files)))
    return order, by_id


def library_index():
    """{size: [paths]} of every file already in the library (outside the app's staging folder)."""
    index = {}
    for path in LIBRARY_DIR.rglob('*'):
        if path.is_file() and '.downloads' not in path.parts and '.qwen-partial' not in path.parts and not path.name.endswith('.qwen-partial') and path.name != '.qwen-library.json':
            index.setdefault(path.stat().st_size, []).append(path)
    return index


def existing_copy(file, source, size, index, target):
    """A library file with the same content (catalog SHA-256 when known, else head, middle and tail), or None."""
    for candidate in index.get(size, []):
        if candidate.resolve() == target.resolve():
            continue
        if file.get('sha256') and size > 64 * 1024**2:
            if full_sha256(candidate) == file['sha256']:
                return candidate
        elif samples(candidate, size) == samples(source, size):
            return candidate
    return None


def copy(source, target, size):
    # robocopy /J: unbuffered I/O. Buffered 64 MB copies filled the Windows file cache and RAM ran low
    # while jobs loaded models (2026-10-21, 0.3 GB/s). A staging folder keeps a half-copied file out of place.
    staging = target.parent / '.qwen-partial'
    partial = staging / source.name
    staging.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if os.name == 'nt':
        code = subprocess.run(['robocopy', str(source.parent), str(staging), source.name, '/J', '/NP', '/NJH', '/NJS', '/NFL', '/NDL', '/R:2', '/W:5'],
                              stdout=subprocess.DEVNULL).returncode
        if code >= 8:
            raise RuntimeError(f'robocopy failed ({code}) for {source}')
    else:
        with open(source, 'rb') as reader, open(partial, 'wb') as writer:
            while chunk := reader.read(CHUNK):
                writer.write(chunk)
    if partial.stat().st_size != size or samples(partial, size) != samples(source, size):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'Verification failed for {target}')
    os.replace(partial, target)
    try:
        staging.rmdir()
    except OSError:
        pass
    return size / max(time.perf_counter() - started, 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true', help='copy and verify the planned files')
    parser.add_argument('--delete-source', action='store_true', help='with --execute: delete each verified data folder copy')
    parser.add_argument('--scan', action='store_true', help='only classify the user\'s own library models')
    parser.add_argument('--reserve-gb', type=float, default=model_library.RESERVE_BYTES / 1024**3)
    args = parser.parse_args()
    if not model_library.enabled():
        sys.exit('Set models_library_dir in local-settings.json (for example "D:\\\\models") to a folder that exists.')
    if args.scan:
        result = model_library.scan()
        print(f"{len(result['entries'])} library files can join the app, {len(result['skipped'])} skipped (see {model_library.SCAN_PATH})")
        return
    if args.delete_source and not args.execute:
        sys.exit('--delete-source needs --execute.')
    order, _ = plan()
    index = library_index()
    budget = shutil.disk_usage(LIBRARY_DIR).free - int(args.reserve_gb * 1024**3)
    moved = reused = staying = 0
    rows = []
    for files in order:
        work = []
        for item, file, source, size in files:
            target = model_library.library_path(item, file)
            done = target.is_file() and target.stat().st_size == size
            same = None if done else existing_copy(file, source, size, index, target)
            work.append((item, file, source, size, target, done, same))
        need = sum(size for _, _, _, size, _, done, same in work if not done and not same)
        fits = need <= budget
        if fits:
            budget -= need
        for item, file, source, size, target, done, same in work:
            if not fits:
                staying += size
                rows.append(('stays', item['id'], source, None, size))
            elif same:
                reused += size
                rows.append(('reuse', item['id'], source, same, size))
            else:
                moved += size
                rows.append(('done' if done else 'copy', item['id'], source, target, size))
        if not args.execute or not fits:
            continue
        for item, file, source, size, target, done, same in work:
            if same:
                model_library.record(item, file, same, reused=True)
                log(f'reuse  {item["id"]}  {source} = {same}')
            elif not done:
                speed = copy(source, target, size)
                log(f'copied {item["id"]}  {source} -> {target}  {gb(size)} at {speed / 1024**3:.2f} GB/s')
            elif samples(target, size) != samples(source, size):
                target.unlink()
                speed = copy(source, target, size)
                log(f'recopied {item["id"]}  {target} (the earlier copy differed)')
            else:
                log(f'verified {item["id"]}  {target}')
            if args.delete_source:
                kept = same or target
                if kept.stat().st_size == size and samples(kept, size) == samples(source, size):
                    try:
                        source.unlink()
                        log(f'deleted {source}')
                    except OSError as exc:  # a running model worker can hold the file open
                        log(f'kept   {source} ({exc}); delete it after the app is closed')
    for action, model_id, source, target, size in rows:
        print(f'{action:6} {gb(size):>9}  {model_id:40}  {source}' + (f'  ->  {target}' if target else ''))
    left = sum(path.stat().st_size for path in MODELS_DIR.rglob('*') if path.is_file()) - moved - reused
    print(f'\nTo the library: {gb(moved)} copied, {gb(reused)} already there (used in place).'
          f' Staying in {MODELS_DIR}: {gb(staying)} of loaded catalog files, {gb(left)} in all (old layouts, partial and unlisted files).')
    print(f'Library: {LIBRARY_DIR}, free after the move about {gb(budget + int(args.reserve_gb * 1024**3))} (reserve {args.reserve_gb:.0f} GB).')


if __name__ == '__main__':
    main()

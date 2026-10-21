"""The shared ComfyUI model library (LIBRARY_DIR, for example D:\\models): the app's files sorted by type next to the user's own models.

A catalog file goes to <library>/<type>/<folder>/<name>, where type is a ComfyUI model folder (diffusion_models, text_encoders,
vae, loras, ...) and folder is the entry's own directory, which keeps same-named files (ae.safetensors) apart. ComfyUI lists
model folders recursively, so it sees every file. Files that already were in the library (the user's own copy) are recorded
in .qwen-library.json and used in place: the app never deletes them.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import struct
import threading
import time
from pathlib import Path
from .common import DATA_DIR, LIBRARY_DIR, MODELS_DIR, write_json

RESERVE_BYTES = 30 * 1024**3
# Path segments that name a ComfyUI model folder (Comfy-Org repos keep files under them).
SEGMENTS = {'diffusion_models': 'diffusion_models', 'unet': 'diffusion_models', 'text_encoders': 'text_encoders', 'clip': 'text_encoders',
            'vae': 'vae', 'loras': 'loras', 'latent_upscale_models': 'latent_upscale_models', 'geometry_estimation': 'geometry_estimation',
            'checkpoints': 'checkpoints'}
KINDS = {'image': 'diffusion_models', 'video': 'diffusion_models', 'reference-video': 'diffusion_models', 'multi-reference': 'diffusion_models',
         'training-base': 'diffusion_models', 'encoder': 'text_encoders', 'vae': 'vae', 'image-lora': 'loras', 'video-lora': 'loras',
         'reference-lora': 'loras', 'upscale': 'upscale_models', 'upscale-support': 'upscale_models', 'image-upscale': 'upscale_models', 'enhancer': 'diffusers', 'diffusers': 'diffusers'}
# Folder names that only repeat the type ('civitai/loras', 'video/Comfy-Org-MiniMax-H3', 'upscale/SeedVR2').
GENERIC = {'video', 'upscale', 'civitai', 'loras', 'models', 'vae'}
# Upscalers load their companion files by name from one folder (SeedVR2 --model_dir, the FlashVSR folder): keep each folder together.
TOGETHER = ('upscale', 'upscale-support')
_lock = threading.RLock()
_manifest = {'mtime': None, 'files': {}}


def enabled():
    return LIBRARY_DIR is not None and LIBRARY_DIR.is_dir()


def key(item, file):
    return f"{item['directory']}/{file['path']}"


def is_diffusers(item):
    """A model stored as a Diffusers folder (several weight shards), which keeps its inner tree."""
    return item.get('kind') in ('enhancer', 'diffusers') or sum(f['path'].endswith('.safetensors') for f in item.get('files', [])) > 1 and item.get('kind') == 'image'


def place(item, file):
    """(ComfyUI folder, path inside it) for one catalog file."""
    parts = Path(file['path']).parts
    keep = is_diffusers(item) or item.get('kind') in TOGETHER
    kind = 'diffusers' if is_diffusers(item) else None
    found = next((i for i, part in enumerate(parts[:-1]) if part in SEGMENTS), None)
    if kind is None and found is not None and not keep:
        kind, rest = SEGMENTS[parts[found]], Path(*parts[found + 1:])
    else:
        # A catalog folder named after a ComfyUI type ('text_encoders/qwen-image-pe') names the type itself.
        first = Path(item['directory']).parts[0]
        kind = kind or KINDS.get(item.get('kind')) or SEGMENTS.get(first) or ('loras' if item.get('format') == 'LoRA' else 'other')
        rest = Path(*parts) if keep else Path(parts[-1])
    folders = [part for part in Path(item['directory']).parts if part not in GENERIC and part not in SEGMENTS] or [Path(item['directory']).parts[0]]
    return kind, Path(*folders) / rest


def library_path(item, file):
    kind, rest = place(item, file)
    path = (LIBRARY_DIR / kind / rest).resolve()
    if not path.is_relative_to(LIBRARY_DIR.resolve()):
        raise ValueError('Invalid model path.')
    return path


def manifest_path():
    return LIBRARY_DIR / '.qwen-library.json'


def recorded():
    """{catalog key: {'path','bytes','reused'}}, re-read only when the file changes."""
    with _lock:
        try:
            mtime = manifest_path().stat().st_mtime
        except (OSError, TypeError, AttributeError):
            return {}
        if mtime != _manifest['mtime']:
            try:
                _manifest.update(mtime=mtime, files=json.loads(manifest_path().read_text(encoding='utf-8')).get('files', {}))
            except (OSError, ValueError):
                _manifest.update(mtime=mtime, files={})
        return _manifest['files']


def record(item, file, path, reused):
    with _lock:
        files = dict(recorded())
        files[key(item, file)] = {'path': str(path), 'bytes': file.get('bytes'), 'reused': bool(reused)}
        write_json(manifest_path(), {'version': 1, 'files': files})
        _manifest['mtime'] = None


def forget(item, file):
    with _lock:
        files = dict(recorded())
        if files.pop(key(item, file), None) is not None:
            write_json(manifest_path(), {'version': 1, 'files': files})
            _manifest['mtime'] = None


def _complete(path, size):
    try:
        info = path.stat()
    except OSError:
        return False
    return path.is_file() and (not size or info.st_size == size)


def find(item, file):
    """The library copy of a catalog file when it is complete, else None."""
    if not enabled():
        return None
    if item.get('library_path') and _complete(Path(item['library_path']), file.get('bytes')):
        return Path(item['library_path'])
    saved = recorded().get(key(item, file))
    if saved and _complete(Path(saved['path']), file.get('bytes')):
        return Path(saved['path'])
    path = library_path(item, file)
    return path if _complete(path, file.get('bytes')) else None


def download_path(item, file, legacy):
    """Where a new download goes: the library while its drive keeps the reserve free, else the data folder.
    An upscaler folder stays where its other files already are."""
    if not enabled():
        return legacy
    if item.get('kind') in TOGETHER:
        legacy_dir, library_dir = (MODELS_DIR / item['directory']), library_path(item, file).parent
        has = lambda folder: folder.is_dir() and any(child.is_file() for child in folder.iterdir())
        if has(legacy_dir) and not has(library_dir):
            return legacy
    try:
        free = shutil.disk_usage(LIBRARY_DIR).free
    except OSError:
        return legacy
    needed = file.get('bytes') or 0
    if item.get('kind') in TOGETHER:
        # One drive for the whole group: per file, the small support files fit the library and the large model did
        # not, so SwiftVR's config.json and its weights ended up on different drives (2026-10-21).
        from . import models
        group = [(other, f) for other in models.catalog() if other.get('directory') == item['directory'] for f in other.get('files', [])]
        needed = sum(f.get('bytes') or 0 for other, f in group if not find(other, f))
    return library_path(item, file) if free > needed + RESERVE_BYTES else legacy


def staging_dir(item):
    """Hugging Face downloads land here on the library drive, then move into place with one rename."""
    return LIBRARY_DIR / '.downloads' / item['directory']


def settle(downloaded, target):
    """Move a finished download from the staging folder to its library path (same drive: a rename)."""
    downloaded, target = Path(downloaded), Path(target)
    if downloaded.resolve() == target.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded, target)
    return target


def remove(item, file):
    """Delete the app's library copy; a reused file of the user's own library stays."""
    if not enabled():
        return
    saved = recorded().get(key(item, file))
    if saved:
        if not saved.get('reused'):
            Path(saved['path']).unlink(missing_ok=True)
        forget(item, file)
    path = library_path(item, file)
    if path.is_file() and not (saved and saved.get('reused') and Path(saved['path']).resolve() == path.resolve()):
        path.unlink()


# ---- the user's own models already in the library --------------------------

SCAN_FOLDERS = ('diffusion_models', 'unet', 'unet_gguf', 'checkpoints', 'text_encoders', 'clip', 'vae', 'loras')
SCAN_PATH = DATA_DIR / 'library-scan.json'
# Adapters that need their own conditioning nodes: loaded as plain LoRAs they do not do their job.
SPECIAL = re.compile(r'identity[_-]?edit|general[_-]?edit|reid', re.I)
OLD_LTX = re.compile(r'ltx[-_ ]?2[._-]?3|ltx23', re.I)
DTYPES = {'BF16': 'BF16', 'F16': 'FP16', 'F32': 'FP32', 'F8_E4M3': 'FP8', 'F8_E5M2': 'FP8', 'I8': 'INT8', 'U8': 'NVFP4'}


def _header(path):
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        return json.loads(handle.read(length))


def _precision(path):
    """The precision from the file name, else the most common weight dtype in the header."""
    match = re.search(r'(?:^|[_. -])(nvfp4|mxfp8|fp8_scaled|bf16|fp16|fp32|fp8|int8|int4|q[2-8](?:_[a-z0-9]+)*)(?:[_. -]|$)', path.stem, re.I)
    if match:
        return match[1].upper()
    if path.suffix.lower() == '.gguf':
        return 'GGUF'
    counts = {}
    for name, value in _header(path).items():
        if name != '__metadata__' and name.endswith('.weight'):
            counts[value.get('dtype')] = counts.get(value.get('dtype'), 0) + 1
    dtype = max(counts, key=counts.get) if counts else ''
    return DTYPES.get(dtype, dtype or 'SafeTensor')


def classify(path):
    """{'arch','kind'} from the file's tensor names and shapes (the Civitai import rules), or {'skip': reason}."""
    from . import civitai
    if path.suffix.lower() != '.gguf':
        with path.open('rb') as handle:
            head = handle.read(8)
        if len(head) < 8 or struct.unpack('<Q', head)[0] + 8 > path.stat().st_size:
            return {'skip': 'Not a model file (a saved web page or a broken download).'}
    try:
        info = civitai.inspect(path)
    except Exception as exc:  # unreadable, or not a model file
        return {'skip': f'Unreadable header ({type(exc).__name__}).'}
    if info['arch'] in ('unknown', 'qwen-image-1'):
        return {'skip': 'A Qwen-Image 1 or FLUX.1 file (3072-wide layers).' if info['arch'] == 'qwen-image-1' else 'Not an architecture this app runs.', **info}
    if SPECIAL.search(path.stem):
        return {'skip': 'Needs its own edit mode (special conditioning nodes), not supported yet.', **info}
    if info['arch'] == 'ltx25' and info['kind'] == 'model' and OLD_LTX.search(path.stem):
        return {'skip': 'An LTX-2.3 model; the app runs LTX 2.5 models.', **info}
    return info


LORA_FAMILIES = {'qwen-image-2.1': ['qwen-image-2.1', 'qwen-image-2.1-turbo'], 'krea-2': ['krea-2', 'krea-2-turbo'], 'z-image': ['z-image', 'z-image-turbo'],
                 'flux3-klein': ['flux3-klein', 'flux3-klein-base'], 'minimax-h3': ['minimax-h3'], 'ltx25': ['ltx25', 'ltx25-msr'],
                 'moss-image': ['moss-image', 'moss-image-base', 'moss-image-turbo']}
OWN_ARCH = {'krea-2': ('krea-2-turbo', 'krea-2', ['krea2-encoder-bf16', 'krea2-vae'], 'krea2-encoder-bf16', 'krea2-vae'),
            'z-image': ('z-image-turbo', 'z-image', ['zimage-encoder-bf16', 'zimage-vae'], 'zimage-encoder-bf16', 'zimage-vae'),
            'flux3-klein': ('flux3-klein', 'flux3-klein-base', ['flux3-encoder-int8', 'flux3-vae', 'flux3-prompt'], 'flux3-encoder-int8', 'flux3-vae'),
            'moss-image': ('moss-image-turbo', 'moss-image', ['moss-encoder-int8', 'moss-vae-small'], 'moss-encoder-int8', 'moss-vae-small')}


def entry_for(path, rel, info):
    """A local entry like a Civitai import, for one file of the library (None when its variant is unclear)."""
    from . import civitai, models
    size = path.stat().st_size
    entry = {'id': 'library-' + hashlib.sha1(rel.lower().encode('utf-8')).hexdigest()[:12], 'title': path.stem, 'repo': 'library',
             'source': 'library', 'revision': str(size), 'directory': 'library', 'files': [{'path': rel, 'bytes': size, 'sha256': None}],
             'bytes': size, 'dependencies': [], 'library_path': str(path), 'summary': f'From your model library: {path}'}
    arch = info['arch']
    if info['kind'] == 'lora':
        entry.update(kind='video-lora' if arch in ('minimax-h3', 'ltx25') else 'image-lora', format='LoRA', families=list(LORA_FAMILIES[arch]),
                     default_strength=1.0, trained_words=[])
        if arch == 'ltx25' and not re.search(r'ltx[-_ ]?2[._-]?5', path.stem, re.I):
            entry['made_for'] = ('LTX-2.3' if OLD_LTX.search(path.stem) else 'LTX 2.x') + ' (its layers match LTX 2.5)'
        try:
            from .lora_author import describe
            describe(entry, path)
        except Exception:  # the trained-on notice is optional
            pass
        return entry
    gguf = path.suffix.lower() == '.gguf'
    precision = _precision(path)
    turbo = bool(re.search(r'turbo|lightning|distill|[_-]\d{1,2}steps?\b', path.stem, re.I))
    if arch in OWN_ARCH:
        turbo = info.get('variant') == 'distilled' if arch == 'flux3-klein' else turbo
        family, base, dependencies, encoder, vae = OWN_ARCH[arch]
        entry.update(kind='image', family=civitai.moss_family(turbo, path.stem) if arch == 'moss-image' else family if turbo else base,
                     dependencies=dependencies, text_encoder=encoder, recommended_vae=vae, generation_defaults=civitai.image_defaults(turbo, {}, '', arch))
    elif arch == 'qwen-image-2.1':
        entry.update(kind='image', family='qwen-image-2.1-turbo' if turbo else 'qwen-image-2.1', dependencies=['encoder-int8', 'vae'],
                     generation_defaults=civitai.image_defaults(turbo, {}, ''))
    elif arch == 'minimax-h3':
        template = models.entry('h3-ref-base-int8' if re.search(r'ref2v(?:a)?|\br2v\b', path.stem, re.I) else 'h3-int8')
        entry.update(kind=template['kind'], family=template['family'], dependencies=list(template['dependencies']),
                     generation_defaults=dict(template.get('generation_defaults') or {}))
        if template.get('modes'):
            entry['modes'] = list(template['modes'])
    elif arch == 'ltx25':
        dev = not re.search(r'distill', path.stem, re.I)
        if dev and not re.search(r'(?:^|[_. -])dev(?:[_. -]|$)', path.stem, re.I):
            return None
        entry.update(kind='video', family='ltx25', modes=['text', 'image'],
                     dependencies=['ltx25-gemma-int8', 'ltx25-video-vae', 'ltx25-audio-vae', *(['ltx25-distill-lora'] if dev else [])],
                     generation_defaults=dict(models.entry('ltx25-distilled-int8').get('generation_defaults') or {}))
    else:
        return None
    entry.update(format='GGUF' if gguf else precision, quantization=precision, title=f'{path.stem} · {precision}')
    if not gguf:
        entry['layout'] = 'comfy'
    return entry


def catalog_paths():
    """Lower-case resolved library paths of every file of the app's catalog and user entries (except scanned library entries)."""
    from . import models
    paths = set()
    for item in models.catalog():
        if item.get('source') == 'library':
            continue
        for file in item.get('files', []):
            try:
                paths.add(str(library_path(item, file)).lower())
            except (ValueError, KeyError):
                pass
    return paths


def scan(reused=()):
    """Classify every model file in the library folders that is not one of the app's own copies.
    Writes library-scan.json: {'entries': [...], 'skipped': [{'path','reason'}], 'removed': [...]}."""
    if not enabled():
        return {'entries': [], 'skipped': []}
    own = {str(Path(saved['path']).resolve()).lower() for saved in recorded().values()} | {str(Path(p).resolve()).lower() for p in reused}
    # The app's own copies (move_models_to_library.py) sit at library_path() and are not recorded; they are not user files.
    own |= catalog_paths()
    entries, skipped = [], []
    for folder in SCAN_FOLDERS:
        root = LIBRARY_DIR / folder
        for path in sorted(root.rglob('*')) if root.is_dir() else ():
            if path.suffix.lower() not in ('.safetensors', '.gguf', '.sft') or not path.is_file() or '.qwen-partial' in path.parts or str(path.resolve()).lower() in own:
                continue
            rel = path.relative_to(LIBRARY_DIR).as_posix()
            if folder in ('text_encoders', 'clip', 'vae'):
                skipped.append({'path': rel, 'reason': 'Encoders and VAEs are used only as the exact file a model needs.'})
                continue
            info = classify(path)
            found = None if 'skip' in info else entry_for(path, rel, info)
            if found is None:
                skipped.append({'path': rel, 'reason': info.get('skip') or 'An LTX 2.5 file without "distilled" or "dev" in its name.'})
            else:
                entries.append(found)
    try:
        removed = json.loads(SCAN_PATH.read_text(encoding='utf-8')).get('removed', [])
    except (OSError, ValueError):
        removed = []
    result = {'scanned_at': time.time(), 'library': str(LIBRARY_DIR), 'entries': entries, 'skipped': skipped, 'removed': removed}
    write_json(SCAN_PATH, result)
    return result


def merge_scan():
    """Add the scanned library files to the user's model list (once each; one the user removed stays removed)."""
    from . import civitai, models
    try:
        found = json.loads(SCAN_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    known = {item['id'] for item in civitai.user_catalog()} | set(found.get('removed', []))
    added = [entry for entry in found.get('entries', []) if entry['id'] not in known and Path(entry['library_path']).is_file()]
    for entry in added:
        record(entry, entry['files'][0], entry['library_path'], reused=True)
        civitai._register(entry)
    if added:
        with models.lock:
            saved = models.installed_manifest()
            saved.update({entry['id']: {'revision': entry['revision'], 'verified': True} for entry in added})
            models.write_json(models.INSTALL_PATH, saved)
    return [entry['id'] for entry in added]


def dismiss(model_id):
    """A library entry the user removed from the app: its file stays, and the next merge does not add it again."""
    try:
        found = json.loads(SCAN_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        found = {'entries': [], 'skipped': []}
    found['removed'] = sorted(set(found.get('removed', [])) | {model_id})
    write_json(SCAN_PATH, found)

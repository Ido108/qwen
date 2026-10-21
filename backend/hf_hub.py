"""Hugging Face Hub and ModelScope browsing and installation for the architectures this app runs.

Only repos that declare one of the app's exact base models are listed: Hugging Face's
base_model:{adapter|finetune|quantized|merge}:<repo> relations, ModelScope's foundation tags and BaseModel field.
A declared base can be wrong (a FLUX.2 Klein or Qwen-Image 1 LoRA labelled as the newer model), so before a file
is offered its safetensors or GGUF header is read with HTTP range requests and its tensor names and shapes must
match the architecture (civitai.inspect_shapes). Downloads, checksums and catalog entries work as for Civitai.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import re
import shutil
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx

from . import civitai
from .antigravity import AUTH_DIR, _crypt
from .common import DATA_DIR, write_json

HUB = 'https://huggingface.co'
HEADERS = {'User-Agent': 'QwenStudio/1.0'}
TOKEN_FILE = AUTH_DIR / 'huggingface.dpapi'
PAGE = 24
# Base repo -> (Civitai base model name, Turbo). The Civitai name carries the architecture and area
# (civitai.BASE_MODELS) and the labels the page already shows. Comfy-Org repos hold both types: Turbo is unknown.
BASE_REPOS = {
    'Qwen/Qwen-Image-2.1': ('Qwen 2.1', False), 'Qwen/Qwen-Image-2.1-Turbo': ('Qwen 2.1', True), 'Comfy-Org/Qwen-Image-2.1': ('Qwen 2.1', None),
    'krea/Krea-2-Raw': ('Krea 2', False), 'krea/Krea-2-Turbo': ('Krea 2 Turbo', True), 'Comfy-Org/Krea-2': ('Krea 2', None),
    'Tongyi-MAI/Z-Image': ('ZImageBase', False), 'Tongyi-MAI/Z-Image-Turbo': ('ZImageTurbo', True),
    'Comfy-Org/z_image': ('ZImageBase', False), 'Comfy-Org/z_image_turbo': ('ZImageTurbo', True),
    'black-forest-labs/flux-3-image-klein': ('Flux.3 Klein', True), 'black-forest-labs/flux-3-image-klein-base': ('Flux.3 Klein-base', False),
    'black-forest-labs/flux-3-image-klein-comfy': ('Flux.3 Klein', None),
    'OpenMOSS-Team/MOSS-Image-SFT': ('MOSS Image', False), 'OpenMOSS-Team/MOSS-Image-Base': ('MOSS Image', False), 'Comfy-Org/MOSS-Image': ('MOSS Image', None),
    'MiniMaxAI/MiniMax-H3': ('MiniMax H3', None), 'Comfy-Org/MiniMax-H3': ('MiniMax H3', None), 'Lightricks/LTX-2.5': ('LTXV 2.5', None),
}
RELATIONS = {'LORA': ('adapter',), 'Checkpoint': ('finetune', 'quantized', 'merge')}
# Edit and identity adapters that need their own conditioning nodes. Their keys look like an ordinary
# LoRA, but loaded as one they do not do their job.
SPECIAL_REPOS = {'conradlocke/krea2-identity-edit', 'yijunwang2/krea2-reid', 'aliveai/krea-2-edit-outfit-transfer'}
SPECIAL_TEXT = re.compile(r'comfyui-krea2edit|krea2ostrisedit|krea2edit|both nodes are required|requires? (?:the |these |our )?(?:custom )?nodes?\b'
                          r'|custom nodes? (?:are |is )?required|reference latent mode|ref(?:erence)?[ _-]?boost', re.I)
SPECIAL_PROBLEM = 'Needs its own edit mode, not supported yet.'
VIDEO_MODEL_PROBLEM = 'Video models from this source are not supported yet. Video LoRAs are.'
_cache: dict = {}
_lock = threading.RLock()


def _cached(key, seconds, load):
    """Short reuse of API answers: Hugging Face allows 500 calls per 5 minutes."""
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < seconds:
            return hit[1]
    value = load()
    with _lock:
        _cache[key] = (time.time(), value)
        if len(_cache) > 400:
            for old in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
                _cache.pop(old, None)
    return value


# ---- Hugging Face token (only gated repos need one) ------------------------

def hf_token() -> str | None:
    try:
        saved = json.loads(_crypt(TOKEN_FILE.read_bytes(), decrypt=True)).get('token')
    except Exception:
        saved = None
    if saved or os.environ.get('HF_TOKEN'):
        return saved or os.environ.get('HF_TOKEN')
    default = Path.home() / '.cache' / 'huggingface' / 'token'
    return (default.read_text(encoding='utf-8').strip() or None) if default.is_file() else None


def hf_status() -> dict:
    return {'connected': TOKEN_FILE.exists(), 'external': not TOKEN_FILE.exists() and bool(hf_token())}


def save_hf_token(token: str) -> dict:
    token = token.strip()
    if not re.fullmatch(r'hf_[A-Za-z0-9]{20,80}', token):
        raise ValueError('This does not look like a Hugging Face token (it starts with hf_).')
    response = httpx.get(HUB + '/api/whoami-v2', headers={**HEADERS, 'Authorization': f'Bearer {token}'}, timeout=20)
    if response.status_code != 200:
        raise PermissionError('Hugging Face did not accept this token.')
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_FILE.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temporary.write_bytes(_crypt(json.dumps({'token': token}).encode()))
    temporary.replace(TOKEN_FILE)
    return hf_status()


def forget_hf_token() -> dict:
    TOKEN_FILE.unlink(missing_ok=True)
    return hf_status()


def _hf_headers() -> dict:
    token = hf_token()
    return {**HEADERS, **({'Authorization': f'Bearer {token}'} if token else {})}


def _hf_get(path: str, params=None):
    response = httpx.get(HUB + path, params=params, headers=_hf_headers(), timeout=40, follow_redirects=True)
    if response.status_code == 429:
        raise ConnectionError('Hugging Face asks to wait a moment (too many requests). Try again in a minute.')
    if response.status_code in (401, 403):
        raise PermissionError('This Hugging Face repo needs access. Accept its terms on Hugging Face and add a token.')
    if response.status_code != 200:
        raise ConnectionError(f'Hugging Face answered {response.status_code}. Try again in a moment.')
    return response


# ---- Hugging Face source ---------------------------------------------------

HF_SORT = {'Most Downloaded': 'downloads', 'Highest Rated': 'likes', 'Newest': 'createdAt'}
EXPAND = ['cardData', 'siblings', 'likes', 'downloads', 'author', 'sha', 'gated', 'tags', 'createdAt']


def _hf_preview(repo: str, sha: str, card: dict, siblings: list) -> list[str]:
    urls = [(w.get('output') or {}).get('url') for w in card.get('widget') or [] if isinstance(w, dict)]
    urls = [u if u.startswith('http') else f'{HUB}/{repo}/resolve/{sha}/{u.lstrip("/")}' for u in urls if isinstance(u, str) and u]
    images = [s['rfilename'] for s in siblings if re.search(r'\.(png|jpe?g|webp|mp4)$', s.get('rfilename', ''), re.I)]
    return list(dict.fromkeys(urls + [f'{HUB}/{repo}/resolve/{sha}/{name}' for name in images]))[:8]


def _hf_item(raw: dict) -> dict:
    card = raw.get('cardData') or {}
    declared = card.get('base_model')
    tags = [t.split(':', 2)[-1] for t in raw.get('tags', []) if t.startswith('base_model:') and t.count(':') == 2]
    bases = list(dict.fromkeys(tags + ([declared] if isinstance(declared, str) else list(declared or []))))
    preview = _hf_preview(raw['id'], raw.get('sha') or 'main', card, raw.get('siblings') or [])
    return {'repo': raw['id'], 'name': raw['id'].split('/', 1)[-1], 'creator': raw.get('author') or raw['id'].split('/')[0],
            'downloads': raw.get('downloads', 0), 'likes': raw.get('likes', 0), 'bases': bases, 'preview': preview[0] if preview else None,
            'relation': card.get('base_model_relation')}


def _hf_listing(bases: list[str], kind: str, sort: str, query: str, cursor: str | None) -> tuple[list[dict], str | None, int | None]:
    """One request per base repo and relation (Hugging Face filters combine with AND). The cursor keeps each
    stream's own next link, so Load more continues every stream."""
    streams = [f'base_model:{relation}:{repo}' for repo in bases for relation in RELATIONS[kind]]
    state = json.loads(cursor) if cursor else {stream: None for stream in streams}
    limit = max(6, -(-PAGE // max(1, len(state))))
    items, following = [], {}

    def page(stream):
        params = [('filter', stream), ('sort', HF_SORT.get(sort, 'downloads')), ('direction', '-1'), ('limit', str(limit)),
                  *[('expand[]', name) for name in EXPAND], *([('search', query)] if query else [])]
        link = state[stream]
        response = httpx.get(link, headers=_hf_headers(), timeout=40) if link else _hf_get('/api/models', params)
        if response.status_code != 200:
            raise ConnectionError(f'Hugging Face answered {response.status_code}. Try again in a moment.')
        return response.json(), response.links.get('next', {}).get('url')

    with ThreadPoolExecutor(8) as pool:
        for stream, (data, link) in zip(state, pool.map(lambda s: _cached(('hf-page', s, sort, query, state[s], limit), 120, lambda: page(s)), state)):
            items += [_hf_item(raw) for raw in data]
            if link and len(data) >= limit:
                following[stream] = link
    unique = list({item['repo']: item for item in items}.values())
    if sort != 'Newest':
        unique.sort(key=lambda item: -(item['likes'] if sort == 'Highest Rated' else item['downloads']))
    return unique, json.dumps(following) if following else None, None


def _hf_card(repo: str) -> dict:
    raw = _hf_get(f'/api/models/{repo}', [('expand[]', name) for name in EXPAND]).json()
    item = _hf_item(raw)
    sha = raw.get('sha') or 'main'
    tree = _hf_get(f'/api/models/{repo}/tree/{sha}', {'recursive': 'true', 'expand': 'true'}).json()
    files = [{'path': f['path'], 'bytes': int((f.get('lfs') or {}).get('size') or f.get('size') or 0),
              'sha256': ((f.get('lfs') or {}).get('oid') or '').lower()} for f in tree if f.get('type') == 'file']
    readme = ''
    if any(f['path'].lower() == 'readme.md' for f in files):
        try:
            readme = re.sub(r'^---\n.*?\n---\n', '', _hf_get(f'/{repo}/raw/{sha}/README.md').text, flags=re.S)
        except (ConnectionError, PermissionError, httpx.HTTPError):
            readme = ''
    card = raw.get('cardData') or {}
    prompt = card.get('instance_prompt')
    words = [prompt] if isinstance(prompt, str) else [w for w in prompt or [] if isinstance(w, str)]
    return {**item, 'source': 'huggingface', 'url': f'{HUB}/{repo}', 'revision': sha, 'gated': bool(raw.get('gated')),
            'text': readme[:20000], 'trained_words': [w.strip() for w in words if w.strip() and w.strip().lower() not in ('none', 'null')],
            'images': _hf_preview(repo, sha, card, raw.get('siblings') or []), 'files': files}


def _hf_file_url(repo: str, revision: str, path: str) -> str:
    return f'{HUB}/{repo}/resolve/{revision}/{path}'


HF = SimpleNamespace(name='Hugging Face', listing=_hf_listing, card=_hf_card, file_url=_hf_file_url, headers=_hf_headers)


def _source(name: str):
    if name == 'huggingface':
        return HF
    if name == 'modelscope':
        from . import modelscope
        return modelscope.SOURCE
    raise ValueError('Unknown model source.')


# ---- shared browsing -------------------------------------------------------

def _base(bases: list[str]) -> tuple[str, str, bool | None] | None:
    """The first declared base this app runs: (repo, Civitai base name, Turbo)."""
    lowered = {repo.lower(): repo for repo in BASE_REPOS}
    for base in bases:
        repo = lowered.get(str(base).strip().lower())
        if repo:
            return (repo, *BASE_REPOS[repo])
    return None


def _wanted(area: str, arch: str, variant: str, kind: str) -> list[str]:
    repos = []
    for repo, (label, turbo) in BASE_REPOS.items():
        info = civitai.BASE_MODELS[label]
        if info['area'] != area or arch not in ('all', info['arch']):
            continue
        if kind == 'Checkpoint' and variant != 'all' and turbo is not (variant == 'turbo'):
            continue
        repos.append(repo)
    return repos


def _installed(source: str) -> set[str]:
    return {item['hub']['repo'] for item in civitai.user_catalog() if (item.get('hub') or {}).get('source') == source}


def _summary(item: dict, kind: str, installed: set[str]) -> dict | None:
    base = _base(item['bases'])
    if not base:
        return None
    _, label, turbo = base
    special = item['repo'].lower() in SPECIAL_REPOS
    return {'id': item['repo'], 'name': item['name'], 'type': kind, 'creator': item['creator'], 'downloads': item['downloads'],
            'likes': item['likes'], 'preview': {'url': item['preview'], 'type': 'video' if (item['preview'] or '').endswith('.mp4') else 'image', 'nsfw_level': 1} if item['preview'] else None,
            'versions': [{'id': 1, 'name': 'Latest', 'base_model': label, 'variant': 'unknown' if turbo is None else 'turbo' if turbo else 'standard'}],
            'installed': item['repo'] in installed, 'special': special}


def search(source: str, query: str = '', kind: str = 'LORA', sort: str = 'Most Downloaded', cursor: str | None = None,
           area: str = 'image', variant: str = 'all', arch: str = 'all') -> dict:
    bases = _wanted(area, arch, variant, kind)
    if not bases:
        return {'items': [], 'total': 0, 'next': None, 'indexing': False}
    items, following, total = _source(source).listing(bases, kind, sort, query.strip(), cursor)
    installed = _installed(source)
    cards = [card for card in (_summary(item, kind, installed) for item in items) if card]
    if kind == 'Checkpoint' and variant != 'all':
        cards = [card for card in cards if card['versions'][0]['variant'] == variant]
    return {'items': cards, 'total': total, 'next': following, 'indexing': False}


def resolve_link(link: str) -> dict:
    match = re.search(r'huggingface\.co/(?!api/|datasets/|spaces/)([\w.-]+/[\w.-]+)', link)
    if match:
        return {'source': 'huggingface', 'repo': match.group(1)}
    match = re.search(r'modelscope\.(?:cn|ai)/models/([\w.-]+/[\w.-]+)', link)
    if match:
        return {'source': 'modelscope', 'repo': match.group(1)}
    raise ValueError('Paste a Hugging Face or ModelScope model link, for example https://huggingface.co/owner/name.')


# ---- remote header check ---------------------------------------------------

def _range(url: str, headers: dict, start: int, end: int) -> bytes:
    response = httpx.get(url, headers={**headers, 'Range': f'bytes={start}-{end}'}, timeout=40, follow_redirects=True)
    if response.status_code not in (200, 206):
        raise ConnectionError(f'The file header could not be read ({response.status_code}).')
    return response.content[:end - start + 1]


def remote_shapes(url: str, headers: dict) -> dict:
    """Tensor names and shapes of a remote .safetensors or .gguf file, read without downloading the weights."""
    if url.split('?')[0].lower().endswith('.gguf'):
        data = _range(url, headers, 0, 16 * 1024 * 1024 - 1)
        partial = DATA_DIR / 'temp' / f'header-{uuid.uuid4().hex}.gguf'
        partial.parent.mkdir(parents=True, exist_ok=True)
        try:
            partial.write_bytes(data)
            return civitai._gguf_shapes(partial)
        finally:
            partial.unlink(missing_ok=True)
    length = struct.unpack('<Q', _range(url, headers, 0, 7))[0]
    if not 0 < length <= 64 * 1024 * 1024:
        raise ValueError('This is not a safetensors file.')
    header = json.loads(_range(url, headers, 8, 7 + length))
    return {name: value['shape'] for name, value in header.items() if name != '__metadata__'}


def _check(source, card: dict, file: dict) -> dict:
    """Architecture and kind of one remote file from its header, cached per checksum."""
    key = ('hdr', card['source'], card['repo'], card['revision'], file['path'], file['sha256'])
    try:
        return _cached(key, 24 * 3600, lambda: civitai.inspect_shapes(remote_shapes(source.file_url(card['repo'], card['revision'], file['path']), source.headers())))
    except (ValueError, ConnectionError, PermissionError, struct.error, KeyError, httpx.HTTPError) as exc:
        return {'kind': None, 'arch': 'unknown', 'error': str(exc)[:200]}


def _precision(name: str) -> str:
    return civitai._file_precision({'name': name.rsplit('/', 1)[-1]})


def _special(card: dict) -> bool:
    return card['repo'].lower() in SPECIAL_REPOS or bool(SPECIAL_TEXT.search(card.get('text') or ''))


def details(source: str, repo: str) -> dict:
    """One repo in the shape the Civitai details dialog shows: one version (the latest revision) whose
    files are the weight files with a matching header."""
    adapter = _source(source)
    card = _cached(('card', source, repo), 300, lambda: adapter.card(repo))
    base = _base(card['bases'])
    lora_repo = card.get('relation') == 'adapter' or card.get('type') == 'LORA'
    weights = [f for f in card['files'] if re.search(r'\.(safetensors|gguf)$', f['path'], re.I) and not re.search(r'(^|/)(_?archive|checkpoints?)/', f['path'], re.I)]
    weights = weights or [f for f in card['files'] if re.search(r'\.(safetensors|gguf)$', f['path'], re.I)]
    weights = sorted(weights, key=lambda f: (f['path'].count('/'), f['path']))[:40]
    problem, files, label, turbo = None, [], None, None
    if not base:
        problem = f'This repo does not declare a base model that this app runs. Declared: {", ".join(card["bases"]) or "none"}.'
    else:
        _, label, turbo = base
        info = civitai.BASE_MODELS[label]
        with ThreadPoolExecutor(8) as pool:
            checks = list(pool.map(lambda f: _check(adapter, card, f), weights))
        text = civitai._text(card.get('text'))
        settings = civitai.author_settings(text, arch=info['arch'])
        for index, (f, check) in enumerate(zip(weights, checks)):
            lora = check.get('kind') == 'lora'
            if check['arch'] != info['arch']:
                continue  # a near match (Qwen-Image 1, FLUX.2 Klein) or another component: hidden
            name = f['path']
            entry = {'id': index + 1, 'name': name, 'type': 'Model', 'bytes': f['bytes'], 'format': 'GGUF' if name.lower().endswith('.gguf') else 'SafeTensor',
                     'precision': _precision(name), 'primary': False, 'sha256': f['sha256'], 'kind': check['kind'], 'supported': True, 'problem': None}
            if not lora and info['area'] == 'video':
                entry.update(supported=False, problem=VIDEO_MODEL_PROBLEM)
            if not lora and info['area'] == 'image':
                # The file name first (official repos hold both types), then the declared base, then the page settings.
                named = civitai.detect_turbo(name.rsplit('/', 1)[-1].replace('_', ' '), {})
                file_turbo = named if named is not None else turbo if turbo is not None else civitai.detect_turbo('', settings)
                if info['arch'] == 'flux3-klein' and check.get('variant'):
                    file_turbo = check['variant'] == 'distilled'
                # A page about both types (8 steps for Turbo) must not give a Standard file Turbo settings.
                fit = settings if file_turbo is None or civitai.detect_turbo('', settings) in (None, file_turbo) else {}
                entry.update(turbo_hint=bool(file_turbo), turbo_known=file_turbo is not None, author_settings=fit or None,
                             family=civitai.moss_family(file_turbo, card['repo'], name) if info['arch'] == 'moss-image' else
                                    info['arch'] + ('-base' if info['arch'] == 'flux3-klein' and file_turbo is False else '-turbo' if info['arch'] != 'flux3-klein' and file_turbo else ''),
                             generation_defaults=_defaults(bool(file_turbo), fit, card['url'], info['arch']))
            if _special(card):
                entry.update(supported=False, problem=SPECIAL_PROBLEM)
            files.append(entry)
        if not files:
            problem = f'No file in this repo matches {civitai.BASE_MODELS[label]["arch"]} by its tensor shapes.' if weights else 'This repo has no safetensors or GGUF weights.'
        elif _special(card):
            problem = SPECIAL_PROBLEM
        # A declared adapter is a LoRA repo; otherwise model files win (official repos also carry helper LoRAs).
        lora_repo = lora_repo or not any(f['kind'] == 'model' for f in files)
    kind = 'LORA' if lora_repo else 'Checkpoint'
    files = [f for f in files if (f['kind'] == 'lora') == lora_repo] or files
    for f in files:
        same = _existing(f['sha256'])
        if same:
            f['installed_as'] = same['id']  # the same file from another source or repo: not downloaded twice
    usable = [f for f in files if f['supported']]
    by_size = sorted(usable, key=lambda f: (civitai.CHECKPOINT_ORDER.get(f['precision'].lower(), 8) if kind == 'Checkpoint' else 0, -f['bytes'] if kind == 'Checkpoint' else f['name'].count('/')))
    if by_size:
        by_size[0]['primary'] = True
    text = civitai._text(card.get('text'))
    words = card['trained_words'] or _words(text, label)
    turbo_known = turbo is not None
    version = {'id': 1, 'name': card['revision'][:7] if re.fullmatch(r'[0-9a-f]{40}', card['revision']) else card['revision'],
               'base_model': label or '', 'area': civitai.BASE_MODELS[label]['area'] if label else None, 'problem': problem if not usable else None,
               'supported': bool(usable), 'trained_words': words, 'description': text, 'files': files,
               'default_file': by_size[0]['id'] if by_size else None, 'suggested_strength': civitai.suggested_strength(text),
               'turbo_hint': bool(turbo), 'turbo_known': turbo_known, 'turbo_source': 'tag' if turbo_known else None,
               'author_settings': None, 'generation_defaults': None, 'modes': [], 'ltx_variant': None,
               'vae_hint': civitai.recommended_vae(text) if label == 'Qwen 2.1' else None,
               'images': [{'url': url, 'type': 'video' if url.endswith('.mp4') else 'image', 'nsfw_level': 1} for url in card['images']]}
    if problem and not usable:
        version['problem'] = problem
    installed = _installed(source)
    return {'id': card['repo'], 'name': card['name'], 'type': kind, 'creator': card['creator'], 'downloads': card['downloads'], 'likes': card['likes'],
            'preview': version['images'][0] if version['images'] else None, 'versions': [version], 'installed': card['repo'] in installed,
            'description': text, 'url': card['url'], 'source': source, 'gated': card.get('gated', False), 'installed_versions': []}


def _defaults(turbo: bool, settings: dict, url: str, arch: str) -> dict:
    defaults = civitai.image_defaults(turbo, settings, url, arch)
    if defaults.get('note'):
        defaults['note'] = defaults['note'].replace('From the Civitai page', 'From the model page')
    return defaults


def _words(text: str, label: str | None) -> list[str]:
    from .lora_author import profile
    return profile({'baseModel': label or '', 'description': text, 'name': '', 'id': 0, 'images': []}, {'description': '', 'id': 0})['trained_words']


# ---- installation ----------------------------------------------------------

def _existing(sha256: str) -> dict | None:
    """An installed catalog entry with this exact file (same sha256), from any source."""
    if not sha256:
        return None
    from . import models
    for item in civitai.user_catalog():
        if any((f.get('sha256') or '').lower() == sha256 for f in item.get('files', [])) and models.local_path(item, item['files'][0]).is_file():
            return item
    return None


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('_')[:80]


def entry_id(source: str, repo: str, path: str, lora: bool) -> str:
    short = {'huggingface': 'hf', 'modelscope': 'ms'}[source]
    digest = hashlib.sha1(f'{repo}/{path}'.encode()).hexdigest()[:8]
    return f"{short}-{'lora' if lora else 'model'}-{_slug(repo.replace('/', '__'))}-{_slug(Path(path).stem)[:40]}-{digest}"


def download(url: str, target: Path, total: int, expected: str, headers: dict, progress, name: str) -> None:
    """Resolve the storage address with the source's own headers, then the Civitai ranged downloader (no
    Civitai token is sent here) and a checksum check."""
    with httpx.stream('GET', url, headers={**headers, 'Range': 'bytes=0-0'}, timeout=60, follow_redirects=True) as probe:
        if probe.status_code in (401, 403):
            raise PermissionError(f'{name} refused this file. Accept the repo terms on {name}' + (' and add a token.' if name == 'Hugging Face' else '.'))
        if probe.status_code not in (200, 206):
            raise ConnectionError(f'{name} download failed ({probe.status_code}).')
        final, ranged = str(probe.url), probe.status_code == 206
        if ranged:
            total = int(probe.headers.get('content-range', '/0').rsplit('/', 1)[-1] or total)
    if shutil.disk_usage(target.parent).free < total + 2 * 1024**3:
        raise RuntimeError('Not enough free space on the model drive.')
    partial = target.with_name(target.name + '.part')
    try:
        if not ranged or total < 64 * 1024 * 1024:
            civitai._single(final, partial, total, progress)
        else:
            civitai._parallel(final, partial, total, progress, 8)
    except ConnectionError as exc:
        raise ConnectionError(str(exc).replace('Civitai', name)) from exc
    progress(total, total, 'verifying')
    if expected:
        digest = hashlib.sha256()
        with partial.open('rb') as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            partial.unlink(missing_ok=True)
            raise ValueError('The download did not match its checksum. Try again.')
    partial.replace(target)


def selected_file(source: str, repo: str, path: str) -> dict:
    """The file to install, checked before a download starts."""
    model = details(source, repo)
    version = model['versions'][0]
    file = next((f for f in version['files'] if f['name'] == path), None)
    if not file:
        raise ValueError(version['problem'] or 'This file does not match an architecture that this app runs.')
    if not file['supported']:
        raise ValueError(file['problem'])
    return {'model': model, 'version': version, 'file': file}


def install(source: str, repo: str, path: str, turbo: bool, progress) -> dict:
    """Download, verify and register one file. `progress(done, total, stage)` reports bytes."""
    from . import models
    adapter = _source(source)
    chosen = selected_file(source, repo, path)
    model, version, file = chosen['model'], chosen['version'], chosen['file']
    same = _existing(file['sha256'])
    if same:
        return same
    card = _cached(('card', source, repo), 300, lambda: adapter.card(repo))
    label = version['base_model']
    info = civitai.BASE_MODELS[label]
    lora = file['kind'] == 'lora'
    new_id = entry_id(source, repo, path, lora)
    directory = f"{source}/{'loras' if lora else 'models'}/{_slug(repo.replace('/', '__'))}"
    if '..' in Path(path).parts or Path(path).is_absolute():
        raise ValueError('Invalid file path in this repo.')
    # The same place a Civitai file goes: the shared model library when its drive has room, else the data folder.
    kind = ('video-lora' if info['area'] == 'video' else 'image-lora') if lora else 'image'
    target = models.download_target({'directory': directory, 'kind': kind}, {'path': path, 'bytes': file['bytes']})
    target.parent.mkdir(parents=True, exist_ok=True)
    download(adapter.file_url(repo, card['revision'], path), target, file['bytes'], file['sha256'], adapter.headers(), progress, adapter.name)
    found = civitai.inspect(target)
    if found['arch'] != info['arch'] or found['kind'] != ('lora' if lora else 'model'):
        target.unlink(missing_ok=True)
        raise ValueError(f'The downloaded file is not a {label} {"LoRA" if lora else "model"}, so this app cannot use it. It was removed.')
    size = target.stat().st_size
    text = model['description']
    if not lora:
        turbo = file['turbo_hint'] if file.get('turbo_known') else turbo
    entry = {'id': new_id, 'title': model['name'] if len(version['files']) < 2 else f"{model['name']} · {Path(path).stem}",
             'repo': repo if source == 'huggingface' else f'modelscope:{repo}', 'source': source, 'revision': card['revision'],
             'directory': directory, 'files': [{'path': path, 'bytes': size, 'sha256': file['sha256'] or None}], 'bytes': size, 'dependencies': [],
             'hub': {'source': source, 'repo': repo, 'revision': card['revision'], 'path': path, 'sha256': file['sha256'], 'url': card['url'],
                     'creator': model['creator'], 'download_url': adapter.file_url(repo, card['revision'], path)}}
    entry.update(_entry_fields(info['arch'], lora, bool(turbo), target.suffix.lower() == '.gguf', file, card, text, path))
    if lora:
        from .lora_author import profile
        advice = profile({'baseModel': label, 'description': text, 'trainedWords': version['trained_words'], 'name': version['name'], 'id': 0, 'images': []},
                         {'description': '', 'id': 0})
        advice['source'] = card['url']
        entry.update(lora_author=advice, trained_words=advice['trained_words'], default_strength=advice['strength'] if advice['strength'] is not None else 1.0,
                     summary=advice['description'][:400])
    else:
        precision = file['precision'] or entry['format']
        entry['quantization'] = precision
        entry['title'] += ' · ' + precision
    _save_preview(new_id, card['images'])
    entry['preview'] = f'/api/civitai/previews/{new_id}' if (civitai.PREVIEWS / f'{new_id}.jpg').exists() else None
    civitai._register(entry)
    with models.lock:
        saved = models.installed_manifest()
        saved[new_id] = {'revision': entry['revision'], 'verified': True}
        write_json(models.INSTALL_PATH, saved)
    return entry


def _entry_fields(arch: str, lora: bool, turbo: bool, gguf: bool, file: dict, card: dict, text: str, path: str) -> dict:
    """Kind, families, dependencies and settings, as civitai.install writes them for the same architecture."""
    precision = 'GGUF' if gguf else (file['precision'] or 'SafeTensor').upper()
    if lora and arch == 'minimax-h3':
        from .civitai_settings import h3_capabilities
        ref2va, _ = h3_capabilities({'name': card['name'], 'description': text}, {'name': '', 'description': ''}, {'name': path})
        return {'kind': 'video-lora', 'format': 'LoRA', 'families': ['minimax-h3-ref2va'] if ref2va else ['minimax-h3'], 'made_for': 'Ref2VA' if ref2va else 'FL2VA'}
    if lora and arch == 'ltx25':
        return {'kind': 'video-lora', 'format': 'LoRA', 'families': ['ltx25', 'ltx25-msr'], 'made_for': 'LTX 2.5'}
    if lora:
        families = {'flux3-klein': ['flux3-klein', 'flux3-klein-base'], 'krea-2': list(civitai.KREA2_FAMILIES), 'z-image': list(civitai.ZIMAGE_FAMILIES),
                    'moss-image': list(civitai.MOSS_FAMILIES)}
        return {'kind': 'image-lora', 'format': 'LoRA', 'families': families.get(arch, ['qwen-image-2.1', 'qwen-image-2.1-turbo'])}
    settings = file.get('author_settings') or {}
    defaults = file.get('generation_defaults') or _defaults(turbo, settings, card['url'], arch)
    common = {'kind': 'image', 'format': precision, 'generation_defaults': defaults, 'settings_version': civitai.SETTINGS_VERSION}
    if arch == 'flux3-klein':
        distilled = file.get('family') != 'flux3-klein-base'
        return {**common, 'family': 'flux3-klein' if distilled else 'flux3-klein-base', 'dependencies': ['flux3-encoder-int8', 'flux3-vae', 'flux3-prompt'],
                'text_encoder': 'flux3-encoder-int8', 'recommended_vae': 'flux3-vae', 'layout': 'comfy'}
    if arch == 'moss-image':
        return {**common, 'family': file.get('family') or civitai.moss_family(turbo, card['repo'], path), 'dependencies': ['moss-encoder-int8', 'moss-vae-small'],
                'text_encoder': 'moss-encoder-int8', 'recommended_vae': 'moss-vae-small', **({} if gguf else {'layout': 'comfy'})}
    if arch == 'krea-2':
        return {**common, 'family': 'krea-2-turbo' if turbo else 'krea-2', 'dependencies': ['krea2-encoder-bf16', 'krea2-vae'],
                'text_encoder': 'krea2-encoder-bf16', 'recommended_vae': 'krea2-vae', **({} if gguf else {'layout': 'comfy'})}
    if arch == 'z-image':
        return {**common, 'family': 'z-image-turbo' if turbo else 'z-image', 'dependencies': ['zimage-encoder-bf16', 'zimage-vae'],
                'text_encoder': 'zimage-encoder-bf16', 'recommended_vae': 'zimage-vae', **({} if gguf else {'layout': 'comfy'})}
    vae = civitai.recommended_vae(text)
    return {**common, 'family': 'qwen-image-2.1-turbo' if turbo else 'qwen-image-2.1', 'dependencies': ['encoder-int8' if gguf else 'encoder-bf16', 'vae'],
            **({'recommended_vae': vae} if vae else {}), **({} if gguf else {'layout': 'comfy'})}


def _save_preview(new_id: str, images: list[str]) -> None:
    for url in images:
        if url.endswith('.mp4'):
            continue
        try:
            response = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
            if response.status_code != 200:
                continue
            from PIL import Image
            picture = Image.open(io.BytesIO(response.content)).convert('RGB')
            picture.thumbnail((320, 320 * 4))
            civitai.PREVIEWS.mkdir(parents=True, exist_ok=True)
            picture.save(civitai.PREVIEWS / f'{new_id}.jpg', quality=88)
            return
        except Exception:
            continue

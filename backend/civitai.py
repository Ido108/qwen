"""Civitai (civitai.red) browsing, account connection and model installation.

Images run on Qwen-Image-2.1 and Krea 2 files. Civitai calls those base models
"Qwen 2.1", "Krea 2" and "Krea 2 Turbo"; "Qwen" is Qwen-Image 1, a different
architecture. Every file is also checked after download: its tensor names and
shapes must match the architecture of its base model.
"""
from __future__ import annotations
import hashlib
import html
import json
import re
import shutil
import struct
import threading
import time
import uuid
from pathlib import Path

import httpx

from .antigravity import AUTH_DIR, _crypt
from .common import DATA_DIR, MODELS_DIR, write_json

SITE = 'https://civitai.red'
AUTH = 'https://auth.civitai.com'
CLIENT_ID = 'civitai-cli'  # Civitai's public device-flow client (no secret)
SCOPE = '1'                # UserRead: identity; downloads only need a signed-in token
# Civitai base model -> what this app runs it with. Each file is also checked by its tensor shapes.
BASE_MODELS = {
    'Qwen 2.1': {'arch': 'qwen-image-2.1', 'area': 'image', 'lora_kind': 'image-lora', 'model_kind': 'image'},
    'Krea 2': {'arch': 'krea-2', 'area': 'image', 'lora_kind': 'image-lora', 'model_kind': 'image'},
    'Krea 2 Turbo': {'arch': 'krea-2', 'area': 'image', 'lora_kind': 'image-lora', 'model_kind': 'image', 'turbo': True},
    'MiniMax H3':{'arch': 'minimax-h3', 'area': 'video', 'lora_kind': 'video-lora', 'model_kind': 'video'},
    'LTXV 2.5': {'arch': 'ltx25', 'area': 'video', 'lora_kind': 'video-lora', 'model_kind': 'video'},
}
TOKEN_FILE = AUTH_DIR / 'civitai.dpapi'
USER_CATALOG = DATA_DIR / 'user-models.json'
PREVIEWS = DATA_DIR / 'civitai-previews'
HEADERS = {'User-Agent': 'QwenStudio/1.0'}
QWEN21_HIDDEN = 4096   # Qwen-Image-2.1 transformer width
QWEN1_HIDDEN = 3072    # Qwen-Image 1 (Civitai base model "Qwen")
H3_HIDDEN = 5376       # MiniMax H3 (blocks.N.attn.qkv_proj, audio_patch_proj)
KREA2_HIDDEN = 6144    # Krea 2 single-stream DiT (blocks.N.attn.wq / transformer_blocks.N.attn.to_q)
KREA2_FAMILIES = ['krea-2', 'krea-2-turbo']
LTX_AUDIO_HIDDEN = 2048  # LTX 2.5 audio stream (audio_attn1.to_q, audio_adaln_single)

_lock = threading.RLock()
_flows: dict[str, dict] = {}


# ---- account ---------------------------------------------------------------

def _load() -> dict:
    if not TOKEN_FILE.exists():
        return {}
    try:
        return json.loads(_crypt(TOKEN_FILE.read_bytes(), decrypt=True))
    except Exception:
        return {}


def _save(tokens: dict) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_FILE.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temporary.write_bytes(_crypt(json.dumps(tokens).encode()))
    temporary.replace(TOKEN_FILE)


def _auth_post(path: str, form: dict) -> httpx.Response:
    # The auth host only accepts form posts with a same-origin Origin header.
    return httpx.post(AUTH + path, data=form, headers={**HEADERS, 'Origin': AUTH}, timeout=30)


def start_login() -> dict:
    response = _auth_post('/api/auth/oauth/device', {'client_id': CLIENT_ID, 'scope': SCOPE})
    if response.status_code != 200:
        raise ConnectionError(f'Civitai did not start the sign-in ({response.status_code}).')
    data = response.json()
    state = uuid.uuid4().hex
    flow = {'status': 'waiting', 'user_code': data['user_code'], 'url': data.get('verification_uri_complete') or data['verification_uri'],
            'expires_at': time.time() + int(data.get('expires_in', 900))}
    with _lock:
        _flows[state] = flow
    threading.Thread(target=_poll, args=(state, data['device_code'], int(data.get('interval', 5))), daemon=True).start()
    return {'state': state, **{k: flow[k] for k in ('user_code', 'url', 'expires_at')}}


def _poll(state: str, device_code: str, interval: int) -> None:
    flow = _flows[state]
    while time.time() < flow['expires_at'] and flow['status'] == 'waiting':
        time.sleep(interval)
        try:
            response = _auth_post('/api/auth/oauth/device-token', {'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                                                                  'device_code': device_code, 'client_id': CLIENT_ID})
            data = response.json()
        except Exception:
            continue
        if response.status_code == 200 and data.get('access_token'):
            tokens = {'access_token': data['access_token'], 'refresh_token': data.get('refresh_token'),
                      'expires_at': time.time() + int(data.get('expires_in', 3600)) - 120}
            tokens['username'] = _whoami(tokens['access_token'])
            _save(tokens)
            flow['status'] = 'connected'
            return
        error = data.get('error')
        if error == 'slow_down':
            interval += 5
        elif error != 'authorization_pending':
            flow['status'] = 'denied' if error == 'access_denied' else 'failed'
            return
    if flow['status'] == 'waiting':
        flow['status'] = 'expired'


def poll_login(state: str) -> dict:
    flow = _flows.get(state)
    if not flow:
        raise ValueError('This sign-in is no longer active. Start again.')
    return {'status': flow['status'], **status()}


def _whoami(token: str) -> str:
    try:
        response = httpx.get(AUTH + '/api/auth/oauth/userinfo', headers={**HEADERS, 'Authorization': f'Bearer {token}'}, timeout=20)
        return response.json().get('username', '') if response.status_code == 200 else ''
    except Exception:
        return ''


def access_token() -> str | None:
    """A valid bearer token, refreshed when it is about to expire."""
    with _lock:
        tokens = _load()
        if not tokens:
            return None
        if tokens.get('api_key'):
            return tokens['api_key']
        if time.time() < tokens.get('expires_at', 0):
            return tokens['access_token']
        if not tokens.get('refresh_token'):
            return None
        response = _auth_post('/api/auth/oauth/token', {'grant_type': 'refresh_token', 'refresh_token': tokens['refresh_token'], 'client_id': CLIENT_ID})
        if response.status_code in (400, 401, 403):
            _forget()  # Civitai ended the sign-in: show Connect, not a connection that does not work
            return None
        if response.status_code != 200:
            return None
        data = response.json()
        tokens.update(access_token=data['access_token'], refresh_token=data.get('refresh_token', tokens['refresh_token']),
                      expires_at=time.time() + int(data.get('expires_in', 3600)) - 120)
        _save(tokens)
        return tokens['access_token']


def save_api_key(key: str) -> dict:
    key = key.strip()
    if not re.fullmatch(r'[A-Za-z0-9_\-]{20,128}', key):
        raise ValueError('This does not look like a Civitai API key.')
    _save({'api_key': key, 'username': _whoami(key)})
    return status()


def _forget() -> None:
    TOKEN_FILE.unlink(missing_ok=True)
    _checked['at'] = 0


_checked = {'at': 0.0}
CHECK_EVERY = 600


def token_rejected(token: str) -> bool:
    """True only when Civitai answers that the token is not valid (not when Civitai is unreachable)."""
    try:
        response = httpx.get(AUTH + '/api/auth/oauth/userinfo', headers={**HEADERS, 'Authorization': f'Bearer {token}'}, timeout=20)
    except Exception:
        return False
    return response.status_code in (401, 403)


def status() -> dict:
    """Connected only while Civitai accepts the saved sign-in or key; checked again every 10 minutes."""
    if _load() and time.time() - _checked['at'] > CHECK_EVERY:
        token = access_token()
        if token and token_rejected(token):
            _forget()
        elif token:
            _checked['at'] = time.time()
    tokens = _load()
    return {'connected': bool(tokens), 'username': tokens.get('username', ''), 'method': 'key' if tokens.get('api_key') else 'account' if tokens else ''}


def disconnect() -> dict:
    TOKEN_FILE.unlink(missing_ok=True)
    return status()


# ---- browsing --------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict:
    token = access_token()
    headers = {**HEADERS, **({'Authorization': f'Bearer {token}'} if token else {})}
    response = httpx.get(SITE + path, params=params, headers=headers, timeout=40, follow_redirects=True)
    if response.status_code != 200:
        raise ConnectionError(f'Civitai answered {response.status_code}. Try again in a moment.')
    return response.json()


def _image(image: dict, width: int = 450) -> dict:
    url = image.get('url', '')
    # Civitai image URLs accept a width segment; small previews load much faster.
    url = re.sub(r'/(original=true|width=\d+)/', f'/width={width}/', url) if '/xG1nkqKTMzGDvpLrqFT7WA/' in url else url
    return {'url': url, 'type': image.get('type', 'image'), 'nsfw_level': image.get('nsfwLevel', 1), 'width': image.get('width'), 'height': image.get('height')}


def _installed_versions() -> set[int]:
    return {item['civitai']['version_id'] for item in user_catalog()}


def _summary(model: dict, bases=tuple(BASE_MODELS)) -> dict:
    installed = _installed_versions()
    versions = [v for v in model.get('modelVersions', []) if v.get('baseModel') in bases]
    images = [image for version in versions for image in version.get('images', [])]
    return {'id': model['id'], 'name': model['name'], 'type': model['type'], 'nsfw': model.get('nsfw', False),
            'creator': (model.get('creator') or {}).get('username', ''), 'downloads': (model.get('stats') or {}).get('downloadCount', 0),
            'likes': (model.get('stats') or {}).get('thumbsUpCount', 0), 'preview': _image(images[0]) if images else None,
            'versions': [{'id': v['id'], 'name': v['name'], 'base_model': v.get('baseModel'), 'variant': _variant(model, v)} for v in versions],
            'installed': any(v['id'] in installed for v in versions)}


_listings: dict = {}
_building: set = set()
PAGE = 24
INDEX_DIR = DATA_DIR / 'civitai-index'
INDEX_TTL = 6 * 3600


def _slim(model: dict) -> dict:
    """Only what the cards and the search need, so the saved index stays small."""
    return {'id': model['id'], 'name': model['name'], 'type': model['type'], 'nsfw': model.get('nsfw', False),
            'creator': {'username': (model.get('creator') or {}).get('username', '')}, 'tags': model.get('tags', []),
            'stats': {key: (model.get('stats') or {}).get(key, 0) for key in ('downloadCount', 'thumbsUpCount')},
            'modelVersions': [{'id': v['id'], 'name': v['name'], 'baseModel': v.get('baseModel'), 'images': v.get('images', [])[:1],
                               'variant': _variant(model, v)} for v in model.get('modelVersions', [])]}


def _index_path(key) -> Path:
    return INDEX_DIR / ('-'.join(str(part) for part in key).replace(' ', '_') + '-v3.json')


def _build_index(key) -> None:
    kind, sort, area, nsfw = key
    bases = [name for name, info in BASE_MODELS.items() if info['area'] == area]
    params = {'limit': 100, 'types': kind, 'baseModels': bases, 'sort': sort, 'period': 'AllTime', 'nsfw': str(nsfw).lower()}
    models, seen = [], set()
    try:
        for _ in range(400):  # up to 40,000 models; Civitai ends the cursor long before that
            data = _get('/api/v1/models', params)
            for model in data.get('items', []):
                if model['id'] not in seen:
                    seen.add(model['id'])
                    models.append(_slim(model))
            current = _listings.get(key)
            if not current or not current['done']:
                _listings[key] = {'at': time.time(), 'models': list(models), 'done': False}
            cursor = (data.get('metadata') or {}).get('nextCursor')
            if not cursor or not data.get('items'):
                break
            params['cursor'] = cursor
        _listings[key] = {'at': time.time(), 'models': models, 'done': True}
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        write_json(_index_path(key), {'at': time.time(), 'models': models})
    except Exception:
        pass  # keep what is cached; the next search tries again
    finally:
        _building.discard(key)


def _listing(kind: str, sort: str, area: str, nsfw: bool) -> dict:
    """Every model of the area's base models, in Civitai's order, for searching here: Civitai's own text search
    returns empty pages with a base model filter and misses many matches without one. The index builds in the
    background, is kept on disk for 6 hours, and is refreshed in the background after that."""
    key = (kind, sort, area, nsfw)
    current = _listings.get(key)
    if current is None:
        try:
            saved = json.loads(_index_path(key).read_text(encoding='utf-8'))
            current = _listings[key] = {'at': saved['at'], 'models': saved['models'], 'done': True}
        except (OSError, ValueError, KeyError):
            current = {'at': 0, 'models': [], 'done': False}
    stale = not current['done'] or time.time() - current['at'] > INDEX_TTL
    with _lock:
        if stale and key not in _building:
            _building.add(key)
            threading.Thread(target=_build_index, args=(key,), name='civitai-index', daemon=True).start()
    return current


def warm_indexes() -> None:
    """Build the default indexes after the app starts, so the first search is instant."""
    for area in ('image', 'video'):
        for kind in ('LORA', 'Checkpoint'):
            _listing(kind, 'Most Downloaded', area, True)


def version_turbo(model: dict, version: dict, settings: dict | None = None) -> bool | None:
    """Turbo or not for one version: the base model tag, then the version name, then the page settings
    (8 steps at CFG 1 is a Turbo model), then the model name. None when nothing tells."""
    if (BASE_MODELS.get(version.get('baseModel')) or {}).get('turbo'):
        return True
    if settings is None:
        settings = author_settings(_text(version.get('description')), _text(model.get('description'))) or example_settings(version)
    turbo = detect_turbo(version['name'], settings)
    return detect_turbo(model['name'], {}) if turbo is None else turbo


def _variant(model: dict, version: dict) -> str:
    """'turbo', 'standard', or 'unknown' when neither the page nor the names tell (the app does not guess)."""
    if 'variant' in version:
        return version['variant']  # worked out from the full page when the index was built
    turbo = version_turbo(model, version)
    return 'unknown' if turbo is None else 'turbo' if turbo else 'standard'


def search(query: str = '', kind: str = 'LORA', sort: str = 'Most Downloaded', cursor: str | None = None, nsfw: bool = True,
           area: str = 'image', variant: str = 'all', arch: str = 'all') -> dict:
    bases = tuple(name for name, info in BASE_MODELS.items() if info['area'] == area and arch in ('all', info['arch']))
    words = [word for word in query.casefold().split() if word]
    if not bases:
        return {'items': [], 'total': 0, 'next': None, 'indexing': False}
    if not words and (variant == 'all' or kind != 'Checkpoint') and arch == 'all':
        # Plain browsing: Civitai's own pages are complete and fast.
        params = {'limit': PAGE, 'types': kind, 'baseModels': list(bases), 'sort': sort, 'period': 'AllTime', 'nsfw': str(nsfw).lower()}
        if cursor:
            params['cursor'] = cursor
        data = _get('/api/v1/models', params)
        items = [_summary(model, bases) for model in data.get('items', [])]
        total = (_listings.get((kind, sort, area, nsfw)) or {}).get('models')
        _listing(kind, sort, area, nsfw)  # build the search index in the background
        done = (_listings.get((kind, sort, area, nsfw)) or {}).get('done')
        return {'items': [item for item in items if item['versions']], 'total': len(total) if done and total is not None else None,
                'next': (data.get('metadata') or {}).get('nextCursor'), 'indexing': False}
    index = _listing(kind, sort, area, nsfw)
    found = []
    for model in index['models']:
        versions = [v for v in model.get('modelVersions', []) if v.get('baseModel') in bases]
        if variant != 'all' and kind == 'Checkpoint':
            versions = [v for v in versions if _variant(model, v) == variant]
        if not versions:
            continue
        text = ' '.join([model['name'], (model.get('creator') or {}).get('username', ''), *model.get('tags', []),
                         *(v['name'] for v in versions)]).casefold()
        if all(word in text for word in words):
            found.append(model)
    start = int(cursor or 0)
    page = [_summary(model, bases) for model in found[start:start + PAGE]]
    return {'items': [item for item in page if item['versions']], 'total': len(found),
            'next': str(start + PAGE) if start + PAGE < len(found) else None,
            'indexing': not index['done'], 'indexed': len(index['models'])}


def _text(description: str | None) -> str:
    text = re.sub(r'<\s*(br|/p|/li|/h\d)\s*/?>', '\n', description or '', flags=re.I)
    text = re.sub(r'<li[^>]*>', '• ', text, flags=re.I)
    return re.sub(r'\n{3,}', '\n\n', html.unescape(re.sub(r'<[^>]+>', '', text))).strip()


def suggested_strength(description: str) -> float | None:
    """A strength written in the description, such as "strength 0.8" or "weight: 0.7"."""
    match = re.search(r'(?:strength|weight|scale)\D{0,24}?(\d(?:\.\d+)?)', description, re.I)
    if match and 0.1 <= float(match.group(1)) <= 2:
        return float(match.group(1))
    return None


SAMPLERS = ('euler_ancestral_cfg_pp', 'euler_ancestral', 'euler_cfg_pp', 'euler', 'res_multistep', 'dpmpp_2m_sde', 'dpmpp_2m',
            'dpmpp_sde', 'dpmpp_3m_sde', 'uni_pc', 'lcm', 'heun', 'ddim', 'deis', 'er_sde', 'sa_solver')
SCHEDULERS = ('simple', 'normal', 'beta', 'karras', 'sgm_uniform', 'exponential', 'ddim_uniform', 'linear_quadratic', 'kl_optimal')


# Names authors write for ComfyUI samplers, longest first ("Euler A" is euler_ancestral, not euler).
SAMPLER_ALIASES = (('euler ancestral cfg++', 'euler_ancestral_cfg_pp'), ('euler a cfg++', 'euler_ancestral_cfg_pp'), ('euler cfg++', 'euler_cfg_pp'),
                   ('euler ancestral', 'euler_ancestral'), ('euler_a', 'euler_ancestral'), ('euler a', 'euler_ancestral'),
                   ('dpm++ 2m sde', 'dpmpp_2m_sde'), ('dpm++ 3m sde', 'dpmpp_3m_sde'), ('dpm++ 2m', 'dpmpp_2m'), ('dpm++ sde', 'dpmpp_sde'),
                   ('res multistep', 'res_multistep'), ('unipc', 'uni_pc'))


def find_sampler(text: str) -> str | None:
    lower = text.lower()
    for alias, name in SAMPLER_ALIASES + tuple((name, name) for name in SAMPLERS):
        if re.search(rf'(?<![a-z_+]){re.escape(alias)}(?![a-z_+])', lower):
            return name
    return None


def find_scheduler(text: str) -> str | None:
    lower = text.lower()
    return next((name for name in SCHEDULERS if re.search(rf'(?<![a-z_]){name}(?![a-z_])', lower)), None)


def _labeled_settings(text: str) -> dict:
    """A settings list such as "Sampler: Euler A + BETA" and "Steps: 8" on their own lines. Among several lists,
    the first one after a "recommended settings" line wins, else the first list."""
    lines = [line.strip(' \t•*-·>') for line in (text or '').split('\n')]
    label = re.compile(r'(sampler|sampling method|scheduler|schedule type|sampling steps|steps?|cfg(?:\s*scale)?|guidance(?:\s*scale)?)\s*[:：=]\s*(.+)', re.I)
    groups, current, last, recommended = [], None, -10, False
    for index, line in enumerate(lines):
        # "Steps: 10, Sampler: Euler a, CFG scale: 2" holds three settings on one line.
        matches = [label.match(part) for part in re.split(r'\s*[,;|]\s*(?=[A-Za-z][A-Za-z ]{1,20}[:：=])', line)]
        matches = [match for match in matches if match]
        if re.search(r'recommend|setting', line, re.I) and not matches:
            recommended = True
        if not matches:
            continue
        if current is None or index - last > 3:
            current = {'recommended': recommended, 'lines': []}
            groups.append(current)
            recommended = False
        current['lines'] += [match.groups() for match in matches]
        last = index
    for group in sorted(groups, key=lambda g: not g['recommended']):
        found = {}
        for key, value in group['lines']:
            key = key.lower()
            if key.startswith(('sampler', 'sampling method')):
                if find_sampler(value):
                    found.setdefault('sampler', find_sampler(value))
                if find_scheduler(value):
                    found.setdefault('scheduler', find_scheduler(value))
            elif key.startswith('schedule'):
                if find_scheduler(value):
                    found.setdefault('scheduler', find_scheduler(value))
            elif key.startswith(('cfg', 'guidance')):
                number = re.match(r'\s*(\d+(?:\.\d+)?)', value)
                if number and 0.5 <= float(number.group(1)) <= 15:
                    found.setdefault('cfg', float(number.group(1)))
            else:
                number = re.match(r'\s*(\d{1,3})\b', value)
                if number and 1 <= int(number.group(1)) <= 150:
                    found.setdefault('steps', int(number.group(1)))
        if 'steps' in found:
            found['quote'] = '; '.join(f'{key}: {value.strip()}' for key, value in group['lines'])[:300]
            return found
    return {}


def author_settings(*texts: str) -> dict:
    """Steps, CFG, sampler and scheduler that the page recommends. The version text wins over the model text,
    a labeled settings list wins over prose, and a sentence that names the author or a recommendation wins
    over example-image captions."""
    for text in texts:
        labeled = _labeled_settings(text)
        if labeled:
            return labeled
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text or '') if s.strip()]
        rank = lambda s: 0 if re.search(r'author|recommend|\bbest\b|suggested|default', s, re.I) else 1 if re.search(r'setting|use', s, re.I) else 2
        for sentence in sorted(sentences, key=rank):
            match = re.search(r'\b(\d{1,3})\s*(?:[-\u2013~]\s*\d{1,3}\s*)?steps?\b', sentence, re.I) or re.search(r'\bsteps?\s*[:=]?\s*(\d{1,3})\b', sentence, re.I)
            if not match or not 1 <= int(match.group(1)) <= 150:
                continue
            found = {'steps': int(match.group(1)), 'quote': sentence[:300]}
            cfg = re.search(r'\bcfg\s*(?:scale)?\s*[:=]?\s*(\d+(?:\.\d+)?)', sentence, re.I)
            if cfg and 0.5 <= float(cfg.group(1)) <= 15:
                found['cfg'] = float(cfg.group(1))
            sampler, scheduler = find_sampler(sentence), find_scheduler(sentence)
            if sampler:
                found['sampler'] = sampler
            if scheduler:
                found['scheduler'] = scheduler
            return found
    return {}


def detect_turbo(version_name: str, settings: dict) -> bool | None:
    """True for distilled few-step models, False for base models, None when the page does not say."""
    # "Raw" is the Krea 2 base only as a type word ("krea2_raw_int8", "Raw v2"), not in a theme name like "Raw Studio Shots".
    if re.search(r'(?<![a-z])(base|standard|full)(?![a-z])', version_name, re.I) or \
            re.search(r'(?<![a-z])raw(?=$|[ _.-]*(?:int|fp|bf|nv|mx|q\d|gguf|v?\d))', version_name, re.I):
        return False
    if re.search(r'turbo|lightning|distill|hyper|\b[48][ -]?steps?\b', version_name, re.I):
        return True
    if settings.get('cfg') is not None and settings['cfg'] <= 1.5:
        return True  # guidance off (CFG 1) is a distilled model; a base model needs CFG 2.5 or more
    if settings.get('steps'):
        return settings['steps'] <= 10
    return None


SAMPLER_NAMES = {'euler a': 'euler_ancestral', 'dpm++ 2m': 'dpmpp_2m', 'dpm++ 2m sde': 'dpmpp_2m_sde', 'dpm++ sde': 'dpmpp_sde',
                 'dpm++ 3m sde': 'dpmpp_3m_sde', 'unipc': 'uni_pc', 'res multistep': 'res_multistep'}


def example_settings(version: dict) -> dict:
    """Steps, CFG, sampler and scheduler that most of the version's example images were made with
    (Civitai keeps the generation data of each example image). Used when the page text names none."""
    metas = [image.get('meta') or {} for image in version.get('images', [])]
    metas = [meta for meta in metas if isinstance(meta.get('steps'), int) and 1 <= meta['steps'] <= 150]
    if not metas:
        return {}
    common = lambda values: max(set(values), key=values.count) if values else None
    steps = common([meta['steps'] for meta in metas])
    same = [meta for meta in metas if meta['steps'] == steps]
    found = {'steps': steps}
    cfg = common([float(meta['cfgScale']) for meta in same if isinstance(meta.get('cfgScale'), (int, float))])
    if cfg is not None and 0.5 <= cfg <= 15:
        found['cfg'] = cfg
    sampler = common([SAMPLER_NAMES.get(str(meta.get('sampler', '')).lower(), str(meta.get('sampler', '')).lower().replace(' ', '_')) for meta in same])
    if sampler in SAMPLERS:
        found['sampler'] = sampler
    scheduler = common([str(meta.get('scheduler', '')).lower() for meta in same])
    if scheduler in SCHEDULERS:
        found['scheduler'] = scheduler
    found['quote'] = (f"{len(same)} of {len(metas)} example images: {steps} steps" + (f", CFG {found['cfg']:g}" if 'cfg' in found else '')
                      + (f", {found['sampler']}" if 'sampler' in found else '') + (f" / {found['scheduler']}" if 'scheduler' in found else '') + '.')
    return found


def ltx_variant(model_name: str, version_name: str, settings: dict, text: str) -> str | None:
    """'distilled' or 'dev' for an LTX 2.5 checkpoint, None when the page does not say. Create video runs the
    official distilled 8 + 3 step schedule, which breaks a dev checkpoint."""
    for name in (version_name, model_name):
        # "dev ... bring your own distilled LoRA" is a dev model, so dev wins within one name.
        if re.search(r'\bdev\b', name, re.I):
            return 'dev'
        if re.search(r'distill', name, re.I):
            return 'distilled'
    if settings.get('steps'):
        if settings['steps'] <= 10:
            return 'distilled'
        if settings['steps'] >= 15:
            return 'dev'
    distilled, dev = re.search(r'distill', text, re.I), re.search(r'\bdev\b', text, re.I)
    return 'distilled' if distilled and not dev else 'dev' if dev and not distilled else None


def ltx_problem(model: dict, version: dict, settings: dict, text: str) -> str | None:
    if model.get('type') != 'Checkpoint' or version.get('baseModel') != 'LTXV 2.5':
        return None
    variant = ltx_variant(model['name'], version['name'], settings, text)
    if variant is None:
        return 'The page does not say that this LTX 2.5 checkpoint is distilled. The app runs distilled LTX 2.5 checkpoints only.'
    return None


def recommended_vae(text: str) -> str | None:
    if re.search(r'texture[ _-]?fix|checkerboard|madebyollin', text, re.I):
        return 'vae-texture-fix'
    return None


def image_defaults(turbo: bool, settings: dict, url: str, arch: str = 'qwen-image-2.1') -> dict:
    from . import models
    if arch == 'krea-2':
        # The official Krea 2 settings (Turbo: 8 steps, CFG 1, mu 1.15; Raw: 52 steps, CFG 4.5, size-based mu).
        # Krea 2 has no fixed sigma list, so the page's steps, CFG, sampler and scheduler all apply.
        defaults = dict(models.entry('krea2-turbo-fp8' if turbo else 'krea2-raw-fp8')['generation_defaults'], source=url)
        if settings:
            defaults.update({key: settings[key] for key in ('steps', 'cfg', 'sampler', 'scheduler') if key in settings})
            defaults['note'] = 'From the Civitai page: ' + settings['quote']
        return defaults
    defaults = dict(models.entry('turbo-gguf-q8_0' if turbo else 'gguf-q8_0')['generation_defaults'], source=url)
    if settings and not turbo:
        defaults.update({key: settings[key] for key in ('steps', 'cfg', 'sampler', 'scheduler') if key in settings})
        defaults['note'] = 'From the Civitai page: ' + settings['quote']
    elif settings:
        # A turbo merge can ask for other steps or CFG; the trained schedule is stretched to that length.
        defaults.update({key: settings[key] for key in ('steps', 'cfg') if key in settings})
        defaults.pop('fixed_steps', None)
        defaults['note'] = 'From the Civitai page: ' + settings['quote']
    return defaults


def details(model_id: int) -> dict:
    model = _get(f'/api/v1/models/{int(model_id)}')
    summary = _summary(model)
    description = _text(model.get('description'))
    versions = []
    for version in model.get('modelVersions', []):
        version_text = _text(version.get('description'))
        settings = author_settings(version_text, description) or example_settings(version)
        info = BASE_MODELS.get(version.get('baseModel'))
        turbo = version_turbo(model, version, settings)
        files = [{'id': f['id'], 'name': f['name'], 'type': f.get('type', 'Model'), 'bytes': int(f.get('sizeKB', 0) * 1024),
                  'format': _file_format(f), 'precision': (f.get('metadata') or {}).get('fp', ''),
                  'primary': bool(f.get('primary')), 'sha256': ((f.get('hashes') or {}).get('SHA256') or '').lower()}
                 for f in version.get('files', [])]
        usable = [f for f in files if _usable_file(model['type'], f)]
        rank = lambda f: _file_rank(f, _best_quality(model['type'], info))
        problem = ltx_problem(model, version, settings, version_text + '\n' + description)
        versions.append({'id': version['id'], 'name': version['name'], 'base_model': version.get('baseModel', ''),
                         'area': info['area'] if info else None, 'problem': problem,
                         'ltx_variant': ltx_variant(model['name'], version['name'], settings, version_text + '\n' + description) if version.get('baseModel') == 'LTXV 2.5' and model['type'] == 'Checkpoint' else None,
                         'supported': bool(info) and bool(usable) and not problem,
                         'trained_words': [w.strip() for w in version.get('trainedWords') or [] if w.strip()],
                         'description': version_text, 'files': files, 'default_file': (sorted(usable, key=rank)[0]['id'] if usable else None),
                         'suggested_strength': suggested_strength(version_text + '\n' + description),
                         'turbo_hint': bool(turbo), 'turbo_known': turbo is not None,
                         'author_settings': settings or None,
                         'vae_hint': _known_vae(next((f['name'] for f in files if f['type'] == 'VAE'), '')) or recommended_vae(version_text + '\n' + description)
                                     or ('file' if any(f['type'] == 'VAE' for f in files) else None),
                         'images': [_image(i, 700) for i in version.get('images', [])[:8]]})
    return {**summary, 'description': description, 'url': f"{SITE}/models/{model['id']}", 'versions': versions,
            'installed_versions': sorted(_installed_versions())}


def _file_format(file: dict) -> str:
    """The file format from Civitai, or from the file name when the uploader left it empty."""
    found = (file.get('metadata') or {}).get('format') or ''
    name = file.get('name', '').lower()
    return found or ('SafeTensor' if name.endswith('.safetensors') else 'GGUF' if name.endswith('.gguf') else '')


def _usable_file(model_type: str, file: dict) -> bool:
    if model_type == 'LORA':
        return file['format'] == 'SafeTensor' and file['type'] == 'Model'
    return file['type'] in ('Model', 'Diffusion Model') and file['format'] in ('SafeTensor', 'GGUF')


CHECKPOINT_ORDER = {'fp8': 0, 'fp8_scaled': 0, 'mxfp8': 1, 'nvfp4': 2, 'bf16': 3, 'fp16': 4, 'fp32': 5, 'int8': 6, 'int4': 7}


def _best_quality(model_type: str, info: dict | None) -> bool:
    """Qwen-Image-2.1 and Krea 2 checkpoints: choose by quality, not by the page's primary file."""
    return model_type != 'LORA' and bool(info) and info['arch'] in ('qwen-image-2.1', 'krea-2')


def _file_rank(file: dict, best_quality: bool = False):
    if best_quality:
        # The best file for a 32 GB RTX 5090: FP8 (near BF16 quality, native and fast, fits next to the
        # text encoder), then NVFP4, BF16, INT8, INT4; GGUF last. The primary file is often INT4.
        gguf = file.get('format') == 'GGUF' or file.get('name', '').lower().endswith('.gguf')
        # Within one precision the larger file keeps more weights at full precision (a mislabeled small file is 4-bit).
        return (gguf, CHECKPOINT_ORDER.get((file['precision'] or '').lower(), 8), not file['primary'], -file['bytes'])
    # Prefer the primary file, then full precision, then smaller quantized files.
    order = {'bf16': 0, 'fp16': 1, 'fp8': 2, 'int8': 3}
    return (not file['primary'], order.get(file['precision'], 4), file['bytes'])


def resolve_link(link: str) -> dict:
    match = re.search(r'civitai\.(?:red|com)/models/(\d+)', link)
    if not match:
        raise ValueError('Paste a Civitai model link, for example https://civitai.red/models/12345.')
    version = re.search(r'modelVersionId=(\d+)', link)
    return {'model_id': int(match.group(1)), 'version_id': int(version.group(1)) if version else None}


# ---- inspection ------------------------------------------------------------

def inspect(path: Path) -> dict:
    """Classify a downloaded file from its tensor names and shapes."""
    shapes = _gguf_shapes(path) if path.suffix.lower() == '.gguf' else _safetensors_shapes(path)
    names = list(shapes)
    lora = any(re.search(r'lora_(?:A|B|up|down)\b|\.lora\.|\.alpha$', name) for name in names)
    arch = 'unknown'
    for name, shape in shapes.items():
        if not shape:
            continue
        if lora and re.search(r'attn\.qkv_proj\.(?:lora_A|lora_down)', name) and max(shape) == H3_HIDDEN:
            arch = 'minimax-h3'; break
        if not lora and re.search(r'(^|\.)audio_patch_proj\.weight$', name) and max(shape) == H3_HIDDEN:
            arch = 'minimax-h3'; break
        if lora and re.search(r'audio_attn1\.to_q\.(?:lora_A|lora_down)', name) and max(shape) == LTX_AUDIO_HIDDEN:
            arch = 'ltx25'; break
        if not lora and re.search(r'(^|\.)audio_adaln_single\.linear\.weight$', name) and min(shape) == LTX_AUDIO_HIDDEN:
            arch = 'ltx25'; break
        # Packed 4-bit files halve the weight shapes; this audio-video cross-attention gate exists only in LTX 2.5.
        if not lora and re.search(r'(^|\.)av_ca_a2v_gate_adaln_single\.', name):
            arch = 'ltx25'; break
        # Krea 2: the txtfusion adapter (ComfyUI and diffusers names), or its 6144-wide single-stream blocks.
        if not lora and re.search(r'(^|\.)txtfusion\.projector\.weight$', name):
            arch = 'krea-2'; break
        if lora and (re.search(r'(txtfusion|text_fusion)[._](layerwise|refiner)_blocks', name)
                     or (re.search(r'blocks[._]\d+[._]attn[._](wq|gate|to_gate)[._]', name) and max(shape) == KREA2_HIDDEN)):
            arch = 'krea-2'; break
        width = max(shape)
        if (lora and re.search(r'attn\.to_q\.(?:lora_A|lora_down)', name)) or (not lora and re.search(r'(^|\.)img_in\.weight$', name)):
            arch = ('qwen-image-2.1' if width == QWEN21_HIDDEN else 'qwen-image-1' if width == QWEN1_HIDDEN
                    else 'krea-2' if lora and width == KREA2_HIDDEN else 'unknown')
            break
    return {'kind': 'lora' if lora else 'model', 'arch': arch, 'tensors': len(names)}


def _safetensors_shapes(path: Path) -> dict:
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
    return {name: value['shape'] for name, value in header.items() if name != '__metadata__'}


def _gguf_shapes(path: Path) -> dict:
    with path.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise ValueError('This is not a GGUF file.')
        read = lambda fmt: struct.unpack('<' + fmt, handle.read(struct.calcsize('<' + fmt)))[0]
        read('I'); tensors = read('Q'); kv = read('Q')
        sizes = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        def string(): return handle.read(read('Q')).decode('utf-8', 'replace')
        def value(kind):
            if kind == 8: return string()
            if kind == 9:
                inner, count = read('I'), read('Q')
                return [value(inner) for _ in range(count)]
            return read(sizes[kind])
        for _ in range(kv):
            string(); value(read('I'))
        shapes = {}
        for _ in range(tensors):
            name = string(); dims = [read('Q') for _ in range(read('I'))]; read('I'); read('Q')
            shapes[name] = dims
        return shapes


# ---- installation ----------------------------------------------------------

def user_catalog() -> list[dict]:
    try:
        return json.loads(USER_CATALOG.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return []


def _register(entry: dict) -> None:
    with _lock:
        items = [item for item in user_catalog() if item['id'] != entry['id']]
        write_json(USER_CATALOG, items + [entry])


def unregister(model_id: str) -> None:
    with _lock:
        write_json(USER_CATALOG, [item for item in user_catalog() if item['id'] != model_id])
    (PREVIEWS / f'{model_id}.jpg').unlink(missing_ok=True)


def install(version_id: int, file_id: int | None, turbo: bool, progress) -> dict:
    """Download, verify and register one Civitai file. `progress(done, total, stage)` reports bytes."""
    from . import models
    version = _get(f'/api/v1/model-versions/{int(version_id)}')
    model = _get(f"/api/v1/models/{version['modelId']}")
    base = BASE_MODELS.get(version.get('baseModel'))
    if not base:
        raise ValueError(f"This version is for {version.get('baseModel')}, which this app cannot run yet. Supported: {', '.join(BASE_MODELS)}.")
    page_text = _text(version.get('description')) + '\n' + _text(model.get('description'))
    problem = ltx_problem(model, version, author_settings(_text(version.get('description')), _text(model.get('description'))), page_text)
    if problem:
        raise ValueError(problem)
    files = [f for f in version.get('files', []) if file_id is None or f['id'] == file_id]
    candidates = [f for f in files if _usable_file(model['type'], {'type': f.get('type', 'Model'), 'format': _file_format(f)})]
    if not candidates:
        raise ValueError('This version has no LoRA or diffusion model file that this app can load.')
    file = sorted(candidates, key=lambda f: _file_rank({'primary': bool(f.get('primary')), 'precision': (f.get('metadata') or {}).get('fp', ''), 'bytes': f.get('sizeKB', 0),
                                                        'format': _file_format(f), 'name': f['name']},
                                                       _best_quality(model['type'], base)))[0]
    lora = model['type'] == 'LORA'
    entry_id = f"civitai-{'lora' if lora else 'model'}-{version['id']}"
    directory = f"civitai/{'loras' if lora else 'models'}"
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', file['name'])
    relative = f"{version['id']}_{name}"
    target = MODELS_DIR / directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    total = int(file.get('sizeKB', 0) * 1024)
    expected = ((file.get('hashes') or {}).get('SHA256') or '').lower()
    _download(file['downloadUrl'], target, total, expected, progress)
    info = inspect(target)
    if info['arch'] != base['arch'] or info['kind'] != ('lora' if lora else 'model'):
        target.unlink(missing_ok=True)
        reason = 'a Qwen-Image 1 file' if info['arch'] == 'qwen-image-1' else f"not a {version.get('baseModel')} {'LoRA' if lora else 'model'}"
        raise ValueError(f'The downloaded file is {reason}, so this app cannot use it. It was removed.')
    description = _text(version.get('description')) + '\n' + _text(model.get('description'))
    # Civitai's "Krea 2" holds both Raw and Turbo merges; without page settings the example images tell which.
    settings = author_settings(_text(version.get('description')), _text(model.get('description'))) or example_settings(version)
    detected = version_turbo(model, version, settings)
    turbo = turbo if detected is None else detected  # the type the user chose when the page does not say
    size = target.stat().st_size
    entry = {'id': entry_id, 'title': model['name'] if len(model.get('modelVersions', [])) < 2 else f"{model['name']} · {version['name']}",
             'repo': 'civitai', 'source': 'civitai', 'revision': str(version['id']), 'directory': directory,
             'files': [{'path': relative, 'bytes': size, 'sha256': expected or None}], 'bytes': size, 'dependencies': [],
             'civitai': {'model_id': model['id'], 'version_id': version['id'], 'url': f"{SITE}/models/{model['id']}?modelVersionId={version['id']}",
                         'creator': (model.get('creator') or {}).get('username', ''), 'nsfw': bool(model.get('nsfw')),
                         'download_url': file['downloadUrl']}}
    if lora and base['arch'] == 'minimax-h3':
        ref2va = bool(re.search(r'ref2va|r2v', version['name'] + ' ' + file['name'], re.I))
        entry.update(kind='video-lora', format='LoRA', families=['minimax-h3', 'fasth3', 'minimax-h3-ref2va'], made_for='Ref2VA' if ref2va else 'MiniMax H3',
                     trained_words=[w.strip() for w in version.get('trainedWords') or [] if w.strip()],
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'minimax-h3':
        ref2va = bool(re.search(r'ref2va|r2v', version['name'] + ' ' + file['name'] + ' ' + model['name'], re.I))
        template = models.entry('h3-ref-base-int8' if ref2va else 'h3-int8')
        gguf = target.suffix.lower() == '.gguf'
        entry.update(kind=template['kind'], family=template['family'], format='GGUF' if gguf else ((file.get('metadata') or {}).get('fp') or 'BF16').upper(),
                     dependencies=list(template['dependencies']), generation_defaults=dict(template['generation_defaults'], source=entry['civitai']['url']))
        if template.get('modes'):
            entry['modes'] = list(template['modes'])
    elif lora and base['arch'] == 'ltx25':
        entry.update(kind='video-lora', format='LoRA', families=['ltx25', 'ltx25-msr'], made_for='LTX 2.5',
                     trained_words=[w.strip() for w in version.get('trainedWords') or [] if w.strip()],
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'ltx25':
        # Only distilled LTX 2.5 checkpoints suit the fixed 8 + 3 step schedule that Create video runs.
        gguf = target.suffix.lower() == '.gguf'
        page = _text(version.get('description')) + '\n' + _text(model.get('description'))
        dev = ltx_variant(model['name'], version['name'], author_settings(_text(version.get('description')), _text(model.get('description'))), page) == 'dev'
        entry.update(kind='video', family='ltx25', modes=['text', 'image'],
                     format='GGUF' if gguf else ((file.get('metadata') or {}).get('fp') or 'BF16').upper(),
                     dependencies=['ltx25-gemma-int8', 'ltx25-video-vae', 'ltx25-audio-vae', *(['ltx25-distill-lora'] if dev else [])],
                     generation_defaults={'steps': 8, 'source': entry['civitai']['url'],
                                          'note': ('Dev checkpoint: the official distilled LoRA at 0.5, then the distilled schedule (as the Lightricks '
                                                   'and model pages describe).' if dev else 'LTX 2.5 distilled schedule.')})
        if dev:
            entry['distill_lora_strength'] = 0.5
    elif lora and base['arch'] == 'krea-2':
        entry.update(kind='image-lora', format='LoRA', families=list(KREA2_FAMILIES),
                     trained_words=[w.strip() for w in version.get('trainedWords') or [] if w.strip()],
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'krea-2':
        # Krea 2 uses its own Qwen3-VL 4B encoder and Qwen-Image VAE (an all-in-one file loads only its DiT).
        gguf = target.suffix.lower() == '.gguf'
        entry.update(kind='image', family='krea-2-turbo' if turbo else 'krea-2',
                     format='GGUF' if gguf else ((file.get('metadata') or {}).get('fp') or 'BF16').upper(),
                     dependencies=['krea2-encoder-bf16', 'krea2-vae'], text_encoder='krea2-encoder-bf16', recommended_vae='krea2-vae',
                     generation_defaults=image_defaults(turbo, settings, entry['civitai']['url'], 'krea-2'), settings_version=SETTINGS_VERSION)
        if not gguf:
            entry['layout'] = 'comfy'
    elif lora:
        entry.update(kind='image-lora', format='LoRA', families=['qwen-image-2.1', 'qwen-image-2.1-turbo'],
                     trained_words=[w.strip() for w in version.get('trainedWords') or [] if w.strip()],
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    else:
        gguf = target.suffix.lower() == '.gguf'
        family = 'qwen-image-2.1-turbo' if turbo else 'qwen-image-2.1'
        entry.update(kind='image', family=family, format='GGUF' if gguf else ((file.get('metadata') or {}).get('fp') or 'BF16').upper(),
                     dependencies=['encoder-int8' if gguf else 'encoder-bf16', 'vae'],
                     generation_defaults=image_defaults(turbo, settings, entry['civitai']['url']), settings_version=SETTINGS_VERSION)
        vae_id = _install_vae(version, model, progress) or recommended_vae(description)
        if vae_id:
            entry['recommended_vae'] = vae_id
        if not gguf:
            entry['layout'] = 'comfy'
    _save_preview(entry_id, version)
    entry['preview'] = f'/api/civitai/previews/{entry_id}' if (PREVIEWS / f'{entry_id}.jpg').exists() else None
    _register(entry)
    with models.lock:
        saved = models.installed_manifest()
        saved[entry_id] = {'revision': entry['revision'], 'verified': True}
        write_json(models.INSTALL_PATH, saved)
    return entry


SETTINGS_VERSION = 3  # 3: settings lists, Euler A, en dash ranges and "Best:" lines on the page


def _known_vae(name: str) -> str | None:
    if re.search(r'texture[_-]?fix', name, re.I):
        return 'vae-texture-fix'
    if re.fullmatch(r'qwen_image_2\.1_vae(_bf16|_fp16|_fp32)?\.safetensors', name.lower()):
        return 'vae'
    return None


def _install_vae(version: dict, model: dict, progress) -> str | None:
    """Download a VAE that ships with the version, if it is a Qwen-Image-2.1 VAE. Returns its catalog id."""
    from . import models
    file = next((f for f in version.get('files', []) if f.get('type') == 'VAE' and f['name'].lower().endswith('.safetensors')), None)
    if not file:
        return None
    # The two common Qwen-Image-2.1 VAEs are already in the app.
    known = _known_vae(file['name'])
    if known:
        return known
    entry_id = f"civitai-vae-{file['id']}"
    if models.is_installed(entry_id) if any(i['id'] == entry_id for i in user_catalog()) else False:
        return entry_id
    relative = f"{file['id']}_{re.sub(r'[^A-Za-z0-9._-]+', '_', file['name'])}"
    target = MODELS_DIR / 'civitai/vae' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _download(file['downloadUrl'], target, int(file.get('sizeKB', 0) * 1024), ((file.get('hashes') or {}).get('SHA256') or '').lower(), progress)
    reference = models.entry('vae')
    reference_path = models.local_path(reference, reference['files'][0])
    if reference_path.is_file() and _safetensors_shapes(target) != _safetensors_shapes(reference_path):
        target.unlink(missing_ok=True)  # not the Qwen-Image-2.1 latent space
        return None
    size = target.stat().st_size
    _register({'id': entry_id, 'title': f"{model['name']} VAE", 'kind': 'vae', 'format': ((file.get('metadata') or {}).get('fp') or 'BF16').upper(),
               'repo': 'civitai', 'source': 'civitai', 'revision': str(file['id']), 'directory': 'civitai/vae',
               'files': [{'path': relative, 'bytes': size, 'sha256': None}], 'bytes': size, 'dependencies': [],
               'civitai': {'model_id': model['id'], 'version_id': version['id'], 'url': f"{SITE}/models/{model['id']}?modelVersionId={version['id']}",
                           'creator': (model.get('creator') or {}).get('username', ''), 'nsfw': bool(model.get('nsfw')), 'download_url': file['downloadUrl']}})
    with models.lock:
        saved = models.installed_manifest()
        saved[entry_id] = {'revision': str(file['id']), 'verified': True}
        write_json(models.INSTALL_PATH, saved)
    return entry_id


def repair_entries() -> list[str]:
    """Recheck Civitai image models installed by older versions of the app: Base models that were saved
    as Turbo got the wrong steps and CFG, and older page parsing missed settings lists and sampler names.
    Uses the page again; changes only the settings."""
    fixed = []
    for item in user_catalog():
        if item.get('kind') != 'image' or 'civitai' not in item or item.get('settings_version', 0) >= SETTINGS_VERSION:
            continue
        try:
            version = _get(f"/api/v1/model-versions/{item['civitai']['version_id']}")
            model = _get(f"/api/v1/models/{item['civitai']['model_id']}")
        except Exception:
            continue  # offline: try again on the next start
        version_text, model_text = _text(version.get('description')), _text(model.get('description'))
        settings = author_settings(version_text, model_text)
        if item.get('family', '').startswith('krea-2'):
            # A Krea 2 entry keeps the type it was installed as; only the page settings are read again.
            turbo = item['family'] == 'krea-2-turbo'
            item.update(generation_defaults=image_defaults(turbo, settings, item['civitai']['url'], 'krea-2'), settings_version=SETTINGS_VERSION)
        else:
            detected = detect_turbo(version['name'], settings)
            turbo = item.get('family') == 'qwen-image-2.1-turbo' if detected is None else detected
            item.update(family='qwen-image-2.1-turbo' if turbo else 'qwen-image-2.1',
                        generation_defaults=image_defaults(turbo, settings, item['civitai']['url']), settings_version=SETTINGS_VERSION)
        vae_file = next((f['name'] for f in version.get('files', []) if f.get('type') == 'VAE'), '')
        vae_id = _known_vae(vae_file) or recommended_vae(version_text + '\n' + model_text)
        if vae_id and not item['family'].startswith('krea-2'):
            item['recommended_vae'] = vae_id
        _register(item)
        fixed.append(item['id'])
    return fixed


def _download(url: str, target: Path, total: int, expected: str, progress, parts: int = 8) -> None:
    """Download with parallel ranged connections (one connection is slow on Civitai's storage), resumable."""
    token = access_token()
    headers = {**HEADERS, **({'Authorization': f'Bearer {token}'} if token else {})}
    partial = target.with_name(target.name + '.part')
    # Resolve the signed storage address once and learn whether it accepts ranges.
    with httpx.stream('GET', url, headers={**headers, 'Range': 'bytes=0-0'}, timeout=60, follow_redirects=True) as probe:
        if probe.status_code in (401, 403):
            if not token or token_rejected(token):
                if token:
                    _forget()
                raise PermissionError('Civitai needs you to sign in for this download. Click Connect Civitai account, then try again.')
            try:
                reason = json.loads(probe.read() or b'{}').get('message') or ''
            except ValueError:
                reason = ''
            raise PermissionError(f"Civitai refused this file for your account: {reason or 'for example an early access or paid version.'} Open it on Civitai to see the terms.")
        if probe.status_code not in (200, 206):
            raise ConnectionError(f'Civitai download failed ({probe.status_code}).')
        final = str(probe.url)
        ranged = probe.status_code == 206
        if ranged:
            total = int(probe.headers.get('content-range', '/0').rsplit('/', 1)[-1] or total)
    if not ranged or total < 64 * 1024 * 1024:
        _single(final, partial, total, progress)
    else:
        _parallel(final, partial, total, progress, parts)
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


def _single(url: str, partial: Path, total: int, progress) -> None:
    done = partial.stat().st_size if partial.exists() else 0
    headers = {**HEADERS, **({'Range': f'bytes={done}-'} if done else {})}
    with httpx.stream('GET', url, headers=headers, timeout=60, follow_redirects=True) as response:
        if response.status_code == 200:
            done = 0
        elif response.status_code != 206:
            raise ConnectionError(f'Civitai download failed ({response.status_code}).')
        with partial.open('ab' if done else 'wb') as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)
                done += len(chunk)
                progress(done, total, 'downloading')


def _parallel(url: str, partial: Path, total: int, progress, parts: int) -> None:
    size = -(-total // parts)
    pieces = [(i, i * size, min(total, (i + 1) * size) - 1) for i in range(parts) if i * size < total]
    files = [partial.with_name(f'{partial.name}{i}') for i, _, _ in pieces]
    done = [f.stat().st_size if f.exists() else 0 for f in files]
    errors: list[BaseException] = []
    lock = threading.Lock()
    last = [0.0]

    def fetch(index: int, first: int, final: int) -> None:
        try:
            for attempt in range(5):
                start = first + done[index]
                if start > final:
                    return
                try:
                    with httpx.stream('GET', url, headers={**HEADERS, 'Range': f'bytes={start}-{final}'}, timeout=60) as response:
                        if response.status_code != 206:
                            raise ConnectionError(f'Civitai download failed ({response.status_code}).')
                        with files[index].open('ab') as handle:
                            for chunk in response.iter_bytes(1024 * 1024):
                                if errors:
                                    return
                                handle.write(chunk)
                                with lock:
                                    done[index] += len(chunk)
                                    now = time.monotonic()
                                    if now - last[0] > .25:
                                        last[0] = now
                                        progress(sum(done), total, 'downloading')
                    return
                except httpx.HTTPError:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        except BaseException as exc:  # includes DownloadCancelled from progress()
            errors.append(exc)

    threads = [threading.Thread(target=fetch, args=piece, daemon=True) for piece in pieces]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    with partial.open('wb') as handle:
        for file in files:
            with file.open('rb') as source:
                shutil.copyfileobj(source, handle, 16 * 1024 * 1024)
    for file in files:
        file.unlink(missing_ok=True)


def _save_preview(entry_id: str, version: dict) -> None:
    images = [i for i in version.get('images', []) if i.get('type') == 'image']
    if not images:
        return
    try:
        response = httpx.get(_image(images[0], 320)['url'], headers=HEADERS, timeout=30, follow_redirects=True)
        if response.status_code == 200:
            PREVIEWS.mkdir(parents=True, exist_ok=True)
            (PREVIEWS / f'{entry_id}.jpg').write_bytes(response.content)
    except httpx.HTTPError:
        pass

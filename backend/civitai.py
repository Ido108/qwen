"""Civitai (civitai.red) browsing, account connection and model installation.

Images run on Qwen-Image-2.1, Krea 2 and Z-Image files. Civitai calls those base models
"Qwen 2.1", "Krea 2", "Krea 2 Turbo", "ZImageBase" and "ZImageTurbo"; "Qwen" is Qwen-Image 1,
a different architecture. Every file is also checked after download: its tensor names and
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
    # Civitai keeps the two Z-Image types apart, so the tag alone tells Turbo from Base.
    'ZImageTurbo': {'arch': 'z-image', 'area': 'image', 'lora_kind': 'image-lora', 'model_kind': 'image', 'turbo': True},
    'ZImageBase': {'arch': 'z-image', 'area': 'image', 'lora_kind': 'image-lora', 'model_kind': 'image', 'turbo': False},
    'Flux.3 Klein': {'arch':'flux3-klein','area':'image','lora_kind':'image-lora','model_kind':'image','turbo':True},
    'Flux.3 Klein-base': {'arch':'flux3-klein','area':'image','lora_kind':'image-lora','model_kind':'image','turbo':False},
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
ZIMAGE_HIDDEN = 3840   # Z-Image single-stream DiT (layers.N.attention, cap_embedder.1 3840 x 2560)
ZIMAGE_FAMILIES = ['z-image', 'z-image-turbo']
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
    return {item['civitai']['version_id'] for item in user_catalog() if (item.get('civitai') or {}).get('version_id') is not None}


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
    # v6 image indexes include FLUX.3 and its author-described Base variants.
    version='v6' if key[2]=='image' else 'v5'
    return INDEX_DIR / ('-'.join(str(part) for part in key).replace(' ', '_') + '-'+version+'.json')


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
    base = BASE_MODELS.get(version.get('baseModel')) or {}
    if base.get('arch') == 'minimax-h3':
        from .civitai_settings import video_variant
        return video_variant(model, version)
    if base.get('arch')=='flux3-klein':
        name=version.get('name','')
        if re.search(r'\bdistilled\b|\bsd\b',name,re.I):return True
        if re.search(r'\bbase\b',name,re.I):return False
        description=_text(model.get('description'))+'\n'+_text(version.get('description'))
        # Some authors document the older *_sd_* naming scheme and tag their
        # Base quantizations as generic Klein. Follow that explicit mapping;
        # absence of SD alone is not evidence for other pages/new filenames.
        if (re.search(r'Klein Base\s*\(flux3_image_klein_\*\)',description,re.I)
                and re.search(r'Klein Distilled\s*\(flux3_image_klein_sd_\*\)',description,re.I)
                and re.search(r'\bklein\s+(?:int\d|bf16|nvfp4)',name,re.I)):
            return False
    if 'turbo' in base:
        return base['turbo']  # the base model tag names the type (Krea 2 Turbo, ZImageTurbo, ZImageBase)
    if settings is None:
        settings = author_settings(_text(version.get('description')), _text(model.get('description')), arch=base.get('arch')) or example_settings(version)
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
    from .lora_author import strength
    return strength(description)


def trained_words(version, model):
    from .lora_author import profile
    return profile(version, model)['trained_words']


SAMPLERS = ('res_2m_ode', 'res_2s_ode', 'res_2m', 'res_2s', 'euler_ancestral_cfg_pp', 'euler_ancestral', 'euler_cfg_pp', 'euler', 'res_multistep', 'dpmpp_2m_sde', 'dpmpp_2m',
            'dpmpp_sde', 'dpmpp_3m_sde', 'uni_pc', 'lcm', 'heun', 'ddim', 'deis', 'er_sde', 'sa_solver')
SCHEDULERS = ('simple', 'normal', 'beta', 'karras', 'sgm_uniform', 'exponential', 'ddim_uniform', 'linear_quadratic', 'kl_optimal')


# Names authors write for ComfyUI samplers, longest first ("Euler A" is euler_ancestral, not euler).
SAMPLER_ALIASES = (('res_mulistep', 'res_multistep'), ('euler ancestral cfg++', 'euler_ancestral_cfg_pp'), ('euler a cfg++', 'euler_ancestral_cfg_pp'), ('euler cfg++', 'euler_cfg_pp'),
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


# Words a page uses to name an architecture in a settings heading ("ZIT Usage", "For Krea 2:").
ARCH_WORDS = {'flux3-klein': r'flux[ ._-]*3|klein', 'z-image': r'\bzit\b|\bzib\b|z[ _-]?image', 'krea-2': r'krea', 'qwen-image-2.1': r'qwen',
              'ltx25': r'\bltx', 'minimax-h3': r'\bh3\b|minimax'}
# Architectures the app does not run; a page paragraph about one of them is not about the model's own versions.
OTHER_ARCH_WORDS = r'\bsdxl\b|\bpony\b|\bflux\b|\bklein\b|\bernie\b|\blongcat\b|illustrious|\bsd ?1\.5\b|hidream|chroma|\bwan\b|hunyuan'
# Settings for a second pass (upscaling, img2img, hires fix), not for making the image.
SECOND_PASS = re.compile(r'upscal|hires|hi-res|img2img|image to image|refine|detailer|denoise', re.I)
# A sentence that says what a version is built on ("Zia is a fine-tune of Z-Image"); a file name such as
# qwen3_4b.safetensors in a Z-Image paragraph does not change the paragraph's architecture.
BUILT_ON = re.compile(r'fine-?tunes?\s+of|based\s+on|merges?\s+of|built\s+on|trained\s+on|distill\w*\s+(?:of|from)', re.I)
# A faster or optional choice next to the recommended settings ("If you want it 30% faster, you can use 4 steps").
OPTIONAL = re.compile(r'\bif you\b|\byou can\b|\bor even\b|faster|speed', re.I)


def _setting_key(key: str) -> str:
    key = key.lower()
    return ('sampler' if key.startswith(('sampler', 'sampling method')) else 'scheduler' if key.startswith('schedule')
            else 'cfg' if key.startswith(('cfg', 'guidance')) else 'steps')


STEP_SPEC = re.compile(r'(?<![\d+/])\b(\d{1,3})(?:\s*([-\u2013~/]|to|till|or)\s*(\d{1,3}))?[\s-]*steps?\b', re.I)


def step_settings(text):
    text=text.translate(str.maketrans({char:'-' for char in '\u2010\u2011\u2012\u2013\u2014\u2015\u2212'}))
    # Quantization error ("within +/-1 step"), training iterations and numbered
    # installation instructions are not denoising recommendations.
    text=re.sub(r'(?:\u00b1|\+/-)\s*\d+\s*steps?\b','',text,flags=re.I)
    text=re.sub(r'\b(?:training|trained for|calibration)\s*(?:steps?\s*[:=]?\s*\d+|\d+\s*steps?)\b','',text,flags=re.I)
    match = STEP_SPEC.search(text) or re.search(r'\bsteps\s*[:=]?\s*(\d{1,3})\b|\bstep\s*[:=]\s*(\d{1,3})\b', text, re.I)
    if match and match.re is not STEP_SPEC and match.group(1) is None:
        return {'steps':int(match.group(2))} if 1 <= int(match.group(2)) <= 150 else {}
    if not match or not 1 <= int(match.group(1)) <= 150:
        return {}
    result = {'steps': int(match.group(1))}
    if match.re is STEP_SPEC and match.group(3) and 1 <= int(match.group(3)) <= 150:
        values = sorted({int(match.group(1)), int(match.group(3))})
        result['step_options' if match.group(2).lower() in ('or','/') else 'steps_range'] = values
    return result


def _labeled_settings(text: str, arch: str | None = None) -> dict:
    """A settings list such as "Sampler: Euler A + BETA" and "Steps: 8" on their own lines. A setting named twice
    starts a new list. Among several lists, the one whose heading names the model's architecture wins (a page can
    list "DMD2 Usage", "SDXL Usage" and "ZIT Usage"), then the first one after a "recommended settings" line,
    else the first list."""
    lines = [line.strip(' \t•*-·>') for line in (text or '').split('\n')]
    label = re.compile(r'(sampler|sampling method|scheduler|schedule type|sampling steps|steps?|cfg(?:\s*scale)?|guidance(?:\s*scale)?)\s*[:：=]\s*(.+)', re.I)
    # A heading glued to the end of a value when the page lost a line break: "CFG: 1SDXL Usage".
    glued = re.compile(r'(?<=\d)\s*([A-Za-z][^:：]{0,40}?\b(?:usage|settings?|workflow))\s*$', re.I)
    groups, current, last, recommended, heading = [], None, -10, False, ''
    for index, line in enumerate(lines):
        # "Steps: 10, Sampler: Euler a, CFG scale: 2" holds three settings on one line.
        matches = [label.match(part) for part in re.split(r'\s*[,;|]\s*(?=[A-Za-z][A-Za-z ]{1,20}[:：=])', line)]
        matches = [match for match in matches if match]
        if not matches:
            if re.search(r'recommend|setting', line, re.I):
                recommended = True
            if line and len(line) <= 80:
                heading = line
            continue
        keys = {_setting_key(match.group(1)) for match in matches}
        if current is None or index - last > 3 or keys & current['keys']:
            current = {'recommended': recommended, 'lines': [], 'keys': set(), 'heading': heading}
            groups.append(current)
            recommended = False
        for match in matches:
            key, value = match.groups()
            tail = glued.search(value)
            if tail:
                heading = tail.group(1)  # names the next list
                value = value[:tail.start()]
            current['lines'].append((key, value))
            current['keys'].add(_setting_key(key))
        if not any(glued.search(match.group(2)) for match in matches):
            heading = ''
        last = index
    named = lambda g: bool(arch and arch in ARCH_WORDS and re.search(ARCH_WORDS[arch], g['heading'], re.I))
    for group in sorted(groups, key=lambda g: (not named(g), not g['recommended'])):
        found = {}
        for key, value in group['lines']:
            kind = _setting_key(key)
            if kind == 'sampler':
                if find_sampler(value):
                    found.setdefault('sampler', find_sampler(value))
                if find_scheduler(value):
                    found.setdefault('scheduler', find_scheduler(value))
            elif kind == 'scheduler':
                if find_scheduler(value):
                    found.setdefault('scheduler', find_scheduler(value))
            elif kind == 'cfg':
                number = re.match(r'\s*(\d+(?:\.\d+)?)', value)
                if number and 0.5 <= float(number.group(1)) <= 15:
                    found.setdefault('cfg', float(number.group(1)))
            else:
                for name, number in step_settings(value+' steps').items():
                    found.setdefault(name, number)
        if 'steps' in found:
            found['quote'] = ((group['heading'] + ': ') if group['heading'] else '') + '; '.join(f'{key}: {value.strip()}' for key, value in group['lines'])
            found['quote'] = found['quote'][:300]
            return found
    return {}


def author_settings(*texts: str, arch: str | None = None) -> dict:
    """Steps, CFG, sampler and scheduler that the page recommends. The version text wins over the model text,
    a labeled settings list wins over prose, and a sentence that names the author or a recommendation wins
    over example-image captions. An optional faster choice ("you can use 4 steps") counts only when no text
    gives the main settings."""
    optional = {}
    for text in texts:
        labeled = _labeled_settings(text, arch)
        if labeled:
            return labeled
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text or '') if s.strip()]
        # A page about several architectures ("KreaGlam is a finetune of Krea 2. Settings are 8 steps...
        # Zia is a fine-tune of Z-Image. Settings are..."): each sentence belongs to the last architecture named before it.
        about, context = [], None
        for sentence in sentences:
            if BUILT_ON.search(sentence):
                own = [name for name, words in ARCH_WORDS.items() if re.search(words, sentence, re.I)]
                other = re.search(OTHER_ARCH_WORDS, sentence, re.I)
                if len(own) == 1 and not other:
                    context = own[0]
                elif other and not own:
                    context = 'other'
            about.append(context)
        mine = lambda i: about[i] is None or about[i] == arch or (arch is None and about[i] != 'other')
        rank = lambda i: (bool(OPTIONAL.search(sentences[i])), 0 if re.search(r'author|recommend|\bbest\b|suggested|default', sentences[i], re.I) else 1 if re.search(r'setting|use', sentences[i], re.I) else 2)
        for sentence in [sentences[i] for i in sorted((i for i in range(len(sentences)) if mine(i)), key=rank)]:
            # "Use 4-5 steps when upscaling" is a second pass; "8 steps (3-4 for upscaling)" keeps its first number.
            lead = re.split(r'\(|\bfor upscal', sentence, maxsplit=1, flags=re.I)[0]
            if SECOND_PASS.search(lead):
                continue
            parsed = step_settings(sentence)
            if not parsed:
                continue
            found = {**parsed, 'quote': sentence[:300]}
            cfg = re.search(r'\bcfg\s*(?:scale)?\s*[:=]?\s*(\d+(?:\.\d+)?)', sentence, re.I)
            if cfg and 0.5 <= float(cfg.group(1)) <= 15:
                found['cfg'] = float(cfg.group(1))
            sampler, scheduler = find_sampler(sentence), find_scheduler(sentence)
            if sampler:
                found['sampler'] = sampler
            if scheduler:
                found['scheduler'] = scheduler
            if not OPTIONAL.search(sentence):
                return found
            optional = optional or found
    return optional


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


def ltx_variant(model_name: str, version_name: str, settings: dict, text: str, file: dict | None = None) -> str | None:
    """'distilled' or 'dev' for an LTX 2.5 checkpoint, None when the page does not say. Create video runs the
    official distilled 8 + 3 step schedule, which breaks a dev checkpoint."""
    for name in ((file or {}).get('name','').replace('_',' ').replace('-',' '), version_name, model_name):
        # "dev ... bring your own distilled LoRA" is a dev model, so dev wins within one name.
        if re.search(r'\bdev\b|\bnon[ -]?distill', name, re.I):
            return 'dev'
        if re.search(r'distill', name, re.I):
            return 'distilled'
    distilled, dev = re.search(r'distill', text, re.I), re.search(r'\bdev\b', text, re.I)
    return 'distilled' if distilled and not dev else 'dev' if dev and not distilled else None


def ltx_problem(model: dict, version: dict, settings: dict, text: str) -> str | None:
    if model.get('type') != 'Checkpoint' or version.get('baseModel') != 'LTXV 2.5':
        return None
    variant = ltx_variant(model['name'], version['name'], settings, text)
    if variant is None:
        if any(ltx_variant('', '', {}, '', file) for file in version.get('files', [])):
            return None
        return 'This version does not identify a distilled or dev LTX 2.5 checkpoint. Choose a version that identifies its model type.'
    if variant == 'distilled' and settings.get('steps') not in (None, 8):
        return 'This version recommends a different sampling schedule. The supported LTX 2.5 workflow uses the official 8-step distilled schedule.'
    return None


def recommended_vae(text: str) -> str | None:
    if re.search(r'texture[ _-]?fix|checkerboard|madebyollin', text, re.I):
        return 'vae-texture-fix'
    return None


def image_defaults(turbo: bool, settings: dict, url: str, arch: str = 'qwen-image-2.1') -> dict:
    from . import models
    if arch in ('krea-2', 'z-image','flux3-klein'):
        # The official settings (Krea 2 Turbo: 8 steps, CFG 1, mu 1.15; Raw: 52 steps, CFG 4.5, size-based mu.
        # Z-Image Turbo: 8 steps, CFG 1; Base: 30 steps, CFG 4; res_multistep, shift 3). Neither has a fixed
        # sigma list, so the page's steps, CFG, sampler and scheduler all apply.
        template = {'krea-2': ('krea2-turbo-fp8', 'krea2-raw-fp8'), 'z-image': ('zimage-turbo-bf16', 'zimage-base-bf16'),'flux3-klein':('flux3-klein-int8','flux3-base-int8')}[arch]
        defaults = dict(models.entry(template[0] if turbo else template[1])['generation_defaults'], source=url)
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
    from .civitai_settings import version_settings, h3_defaults, h3_capabilities, ltx_defaults
    model = _get(f'/api/v1/models/{int(model_id)}')
    from .civitai_workflows import attach_workflows
    attach_workflows(model.get('modelVersions', []))
    summary = _summary(model)
    description = _text(model.get('description'))
    versions = []
    for version in model.get('modelVersions', []):
        version_text = _text(version.get('description'))
        from .lora_author import profile as lora_profile
        lora_advice = lora_profile(version, model)
        info = BASE_MODELS.get(version.get('baseModel'))
        settings = version_settings(model, version)
        turbo = version_turbo(model, version, settings)
        # Where the type comes from, so the app can say it: the Civitai tag, the version name, the page settings or the model name.
        source = ('page' if (info or {}).get('arch')=='flux3-klein' and turbo!=(info or {}).get('turbo') else 'tag' if 'turbo' in (info or {}) else 'name' if detect_turbo(version['name'], {}) is not None
                  else 'page' if detect_turbo(version['name'], settings) is not None else 'model name' if turbo is not None else None)
        files = [{'id': f['id'], 'name': f['name'], 'type': f.get('type', 'Model'), 'bytes': int(f.get('sizeKB', 0) * 1024),
                  'format': _file_format(f), 'precision': _file_precision(f),
                  'primary': bool(f.get('primary')), 'sha256': ((f.get('hashes') or {}).get('SHA256') or '').lower()}
                 for f in version.get('files', [])]
        usable = [f for f in files if _usable_file(model['type'], f)]
        rank = lambda f: _file_rank(f, _best_quality(model['type'], info))
        problem = ltx_problem(model, version, settings, version_text + '\n' + description)
        if not info:
            problem = f"This app does not support {version.get('baseModel') or 'this model architecture'}."
        elif not usable:
            problem = 'This version has no supported diffusion model or LoRA file.'
        sampling = None
        if (info or {}).get('arch') == 'minimax-h3' and model['type'] == 'Checkpoint':
            from . import models
            ref2va, modes = h3_capabilities(model,version)
            sampling = h3_defaults(model, version, models.entry('h3-ref-base-int8' if ref2va else 'h3-int8'), f"{SITE}/models/{model['id']}?modelVersionId={version['id']}")
        if (info or {}).get('arch')=='flux3-klein' and model['type']=='Checkpoint':
            sampling=image_defaults(bool(turbo),settings,f"{SITE}/models/{model['id']}?modelVersionId={version['id']}",'flux3-klein')
        if (info or {}).get('arch')=='ltx25' and model['type']=='Checkpoint':
            sampling=ltx_defaults(model,version,f"{SITE}/models/{model['id']}?modelVersionId={version['id']}")
        if sampling and (info or {}).get('arch')=='ltx25':
            for f in usable:
                f['generation_defaults']=ltx_defaults(model,version,sampling['source'],f)
        if sampling and (info or {}).get('arch')=='minimax-h3':
            for f in usable:
                file_ref, file_modes = h3_capabilities(model,version,f)
                f.update(family='minimax-h3-ref2va' if file_ref else 'minimax-h3', modes=file_modes)
                f['generation_defaults']=h3_defaults(model,version,models.entry('h3-ref-base-int8' if file_ref else 'h3-int8'),sampling['source'],f)
        if sampling and usable and (info or {}).get('arch')=='minimax-h3':
            sampling=sorted(usable,key=rank)[0]['generation_defaults']
        versions.append({'id': version['id'], 'name': version['name'], 'base_model': version.get('baseModel', ''),
                         'area': info['area'] if info else None, 'problem': problem,
                         'ltx_variant': ltx_variant(model['name'], version['name'], settings, version_text + '\n' + description) if version.get('baseModel') == 'LTXV 2.5' and model['type'] == 'Checkpoint' else None,
                         'supported': bool(info) and bool(usable) and not problem,
                         'trained_words': trained_words(version, model),
                         'description': version_text, 'files': files, 'default_file': (sorted(usable, key=rank)[0]['id'] if usable else None),
                         'suggested_strength': lora_advice['strength'],
                         'turbo_hint': bool(turbo), 'turbo_known': turbo is not None, 'turbo_source': source,
                         'author_settings': settings or None,
                         'generation_defaults': sampling, 'modes': h3_capabilities(model,version)[1] if (info or {}).get('arch')=='minimax-h3' else [],
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


def _file_precision(file: dict) -> str:
    value=(file.get('metadata') or {}).get('fp')
    # Civitai has no six-bit metadata choice; W6A8 files often say INT8 there.
    # An explicit precision suffix in the author's filename is more specific.
    named=re.findall(r'(int[2-8]|w[2-8]a[24816]+)(?=[_. -]|$)',file.get('name',''),re.I)
    if named and str(value).lower()=='int8':return named[-1].upper()
    if value:
        return value
    match=re.search(r'(?:^|[_. -])(nvfp4|mxfp8|fp8_scaled|bf16|fp16|fp32|fp8|int8|int4|q[2-8](?:_[a-z0-9]+)*)(?:[_. -]|$)',file.get('name',''),re.I)
    return match[1].upper() if match else ''


def _usable_file(model_type: str, file: dict) -> bool:
    if model_type == 'LORA':
        return file['format'] == 'SafeTensor' and file['type'] == 'Model'
    return file['type'] in ('Model', 'Diffusion Model', 'UNet') and file['format'] in ('SafeTensor', 'GGUF')


CHECKPOINT_ORDER = {'fp8': 0, 'fp8_scaled': 0, 'mxfp8': 1, 'nvfp4': 2, 'bf16': 3, 'fp16': 4, 'fp32': 5, 'int8': 6, 'int4': 7}


def _best_quality(model_type: str, info: dict | None) -> bool:
    """Qwen-Image-2.1, Krea 2 and Z-Image checkpoints: choose by quality, not by the page's primary file."""
    return model_type != 'LORA' and bool(info) and info['arch'] in ('qwen-image-2.1', 'krea-2', 'z-image')


def _file_rank(file: dict, best_quality: bool = False):
    if best_quality:
        # The best file for a 32 GB RTX 5090: FP8 (near BF16 quality, native and fast, fits next to the
        # text encoder), then NVFP4, BF16, INT8, INT4; GGUF last. The primary file is often INT4.
        gguf = file.get('format') == 'GGUF' or file.get('name', '').lower().endswith('.gguf')
        # Within one precision the larger file keeps more weights at full precision (a mislabeled small file is 4-bit).
        return (gguf, CHECKPOINT_ORDER.get((file['precision'] or '').lower(), 8), not file['primary'], -file['bytes'])
    # Prefer the primary file, then full precision, then smaller quantized files.
    order = {'bf16': 0, 'fp16': 1, 'fp8': 2, 'int8': 3}
    return (not file['primary'], order.get((file['precision'] or '').lower(), 4), file['bytes'])


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
    flux3=any(re.search(r'content_mode_blocks[._]image[._]\d+[._]attn_out',name) for name in names)
    if flux3:
        return {'kind':'lora' if lora else 'model','arch':'flux3-klein','tensors':len(names),
                'variant':None if lora else ('distilled' if any('ref_time_in' in name for name in names) else 'base')}
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
        # Z-Image: cap_embedder.1 (3840 x 2560) in a model; 3840-wide layers.N.attention or refiner blocks in a LoRA.
        if not lora and re.search(r'(^|\.)cap_embedder\.1\.weight$', name) and sorted(shape) == [2560, ZIMAGE_HIDDEN]:
            arch = 'z-image'; break
        if lora and re.search(r'(^|[._])(layers|context_refiner|noise_refiner)[._]\d+[._]attention[._](to_q|to_k|to_v|qkv|to_out|out)', name) \
                and re.search(r'lora_(?:A|B|up|down)', name) and max(shape) == ZIMAGE_HIDDEN:
            arch = 'z-image'; break
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
    from .civitai_workflows import attach_workflows
    attach_workflows([version])
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
    file = sorted(candidates, key=lambda f: _file_rank({'primary': bool(f.get('primary')), 'precision': _file_precision(f), 'bytes': f.get('sizeKB', 0),
                                                        'format': _file_format(f), 'name': f['name']},
                                                       _best_quality(model['type'], base)))[0]
    if base['arch']=='ltx25' and model['type']=='Checkpoint':
        from .civitai_settings import version_settings
        selected_settings=version_settings(model,version,file)
        variant=ltx_variant(model['name'],version['name'],selected_settings,page_text,file)
        if variant is None:
            raise ValueError('The selected file does not identify a distilled or dev LTX 2.5 model.')
        if variant=='distilled' and selected_settings.get('steps') not in (None,8):
            raise ValueError('This file requires a different sampling schedule. Choose a model that supports the official 8-step distilled workflow.')
    lora = model['type'] == 'LORA'
    entry_id = f"civitai-{'lora' if lora else 'model'}-{version['id']}"
    existing=next((item for item in user_catalog() if item['id']==entry_id),None)
    if existing and existing.get('civitai',{}).get('file_id') not in (None,file['id']):
        entry_id += f"-file-{file['id']}"
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
    from .civitai_settings import version_settings, h3_defaults, h3_capabilities, ltx_defaults
    settings = version_settings(model, version)
    detected = version_turbo(model, version, settings)
    turbo = turbo if detected is None else detected  # the type the user chose when the page does not say
    size = target.stat().st_size
    entry = {'id': entry_id, 'title': model['name'] if len(model.get('modelVersions', [])) < 2 else f"{model['name']} · {version['name']}",
             'repo': 'civitai', 'source': 'civitai', 'revision': str(version['id']), 'directory': directory,
             'files': [{'path': relative, 'bytes': size, 'sha256': expected or None}], 'bytes': size, 'dependencies': [],
             'civitai': {'model_id': model['id'], 'version_id': version['id'], 'url': f"{SITE}/models/{model['id']}?modelVersionId={version['id']}",
                         'creator': (model.get('creator') or {}).get('username', ''), 'nsfw': bool(model.get('nsfw')),
                         'download_url': file['downloadUrl'], 'file_id': file['id'], 'file_name': file['name']}}
    if lora and base['arch'] == 'minimax-h3':
        ref2va, _ = h3_capabilities(model,version,file)
        entry.update(kind='video-lora', format='LoRA', families=['minimax-h3-ref2va'] if ref2va else ['minimax-h3'], made_for='Ref2VA' if ref2va else 'FL2VA',
                     trained_words=trained_words(version, model),
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'minimax-h3':
        ref2va, modes = h3_capabilities(model,version,file)
        template = models.entry('h3-ref-base-int8' if ref2va else 'h3-int8')
        gguf = target.suffix.lower() == '.gguf'
        entry.update(kind=template['kind'], family=template['family'], format='GGUF' if gguf else (_file_precision(file) or 'SafeTensor').upper(),
                     dependencies=list(template['dependencies']), generation_defaults=h3_defaults(model, version, template, entry['civitai']['url'],file), settings_version=SETTINGS_VERSION)
        if modes:
            entry['modes'] = modes
    elif lora and base['arch'] == 'ltx25':
        entry.update(kind='video-lora', format='LoRA', families=['ltx25', 'ltx25-msr'], made_for='LTX 2.5',
                     trained_words=trained_words(version, model),
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'ltx25':
        # Select the exact file's variant. Dev files need the distillation adapter;
        # already distilled files must not receive it a second time.
        gguf = target.suffix.lower() == '.gguf'
        page = _text(version.get('description')) + '\n' + _text(model.get('description'))
        variant = ltx_variant(model['name'], version['name'], author_settings(_text(version.get('description')), _text(model.get('description'))), page, file)
        if variant is None:
            raise ValueError('The selected file does not identify a distilled or dev LTX 2.5 model.')
        dev = variant == 'dev'
        entry.update(kind='video', family='ltx25', modes=['text', 'image'],
                     format='GGUF' if gguf else (_file_precision(file) or 'SafeTensor').upper(),
                     dependencies=['ltx25-gemma-int8', 'ltx25-video-vae', 'ltx25-audio-vae', *(['ltx25-distill-lora'] if dev else [])],
                     generation_defaults=ltx_defaults(model,version,entry['civitai']['url'],file),settings_version=SETTINGS_VERSION)
        if dev:
            entry['distill_lora_strength'] = entry['generation_defaults']['distill_lora_strength']
    elif base['arch']=='flux3-klein':
        families=['flux3-klein','flux3-klein-base']
        if lora:
            entry.update(kind='image-lora',format='LoRA',families=families,
                trained_words=trained_words(version, model),
                default_strength=suggested_strength(description) or 1.0,summary=description[:400])
        else:
            distilled=info['variant']=='distilled'
            entry.update(kind='image',family='flux3-klein' if distilled else 'flux3-klein-base',
                format='GGUF' if target.suffix.lower()=='.gguf' else (_file_precision(file) or 'SafeTensor').upper(),
                dependencies=['flux3-encoder-int8','flux3-vae','flux3-prompt'],text_encoder='flux3-encoder-int8',
                recommended_vae='flux3-vae',layout='comfy',
                generation_defaults=image_defaults(distilled,settings,entry['civitai']['url'],'flux3-klein'),settings_version=SETTINGS_VERSION)
    elif lora and base['arch'] == 'krea-2':
        entry.update(kind='image-lora', format='LoRA', families=list(KREA2_FAMILIES),
                     trained_words=trained_words(version, model),
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'krea-2':
        # Krea 2 uses its own Qwen3-VL 4B encoder and Qwen-Image VAE (an all-in-one file loads only its DiT).
        gguf = target.suffix.lower() == '.gguf'
        entry.update(kind='image', family='krea-2-turbo' if turbo else 'krea-2',
                     format='GGUF' if gguf else (_file_precision(file) or 'SafeTensor').upper(),
                     dependencies=['krea2-encoder-bf16', 'krea2-vae'], text_encoder='krea2-encoder-bf16', recommended_vae='krea2-vae',
                     generation_defaults=image_defaults(turbo, settings, entry['civitai']['url'], 'krea-2'), settings_version=SETTINGS_VERSION)
        if not gguf:
            entry['layout'] = 'comfy'
    elif lora and base['arch'] == 'z-image':
        # Z-Image Base and Turbo share one architecture; Civitai LoRAs are used with both.
        entry.update(kind='image-lora', format='LoRA', families=list(ZIMAGE_FAMILIES),
                     trained_words=trained_words(version, model),
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    elif base['arch'] == 'z-image':
        # Z-Image uses its own Qwen3 4B encoder and Flux VAE (an all-in-one file loads only its DiT).
        gguf = target.suffix.lower() == '.gguf'
        entry.update(kind='image', family='z-image-turbo' if turbo else 'z-image',
                     format='GGUF' if gguf else (_file_precision(file) or 'SafeTensor').upper(),
                     dependencies=['zimage-encoder-bf16', 'zimage-vae'], text_encoder='zimage-encoder-bf16', recommended_vae='zimage-vae',
                     generation_defaults=image_defaults(turbo, settings, entry['civitai']['url'], 'z-image'), settings_version=SETTINGS_VERSION)
        if not gguf:
            entry['layout'] = 'comfy'
    elif lora:
        entry.update(kind='image-lora', format='LoRA', families=['qwen-image-2.1', 'qwen-image-2.1-turbo'],
                     trained_words=trained_words(version, model),
                     default_strength=suggested_strength(description) or 1.0, summary=description[:400])
    else:
        gguf = target.suffix.lower() == '.gguf'
        family = 'qwen-image-2.1-turbo' if turbo else 'qwen-image-2.1'
        entry.update(kind='image', family=family, format='GGUF' if gguf else (_file_precision(file) or 'SafeTensor').upper(),
                     dependencies=['encoder-int8' if gguf else 'encoder-bf16', 'vae'],
                     generation_defaults=image_defaults(turbo, settings, entry['civitai']['url']), settings_version=SETTINGS_VERSION)
        vae_id = _install_vae(version, model, progress) or recommended_vae(description)
        if vae_id:
            entry['recommended_vae'] = vae_id
        if not gguf:
            entry['layout'] = 'comfy'
    if lora:
        from .lora_author import profile
        advice = profile(version, model)
        entry.update(lora_author=advice, trained_words=advice['trained_words'],
                     default_strength=advice['strength'] if advice['strength'] is not None else 1.0,
                     summary=advice['description'][:400])
    if not lora:
        precision=_file_precision(file) or entry['format']
        entry['quantization']=precision
        entry['title'] += ' \u00b7 '+precision
    _save_preview(entry_id, version)
    entry['preview'] = f'/api/civitai/previews/{entry_id}' if (PREVIEWS / f'{entry_id}.jpg').exists() else None
    _register(entry)
    with models.lock:
        saved = models.installed_manifest()
        saved[entry_id] = {'revision': entry['revision'], 'verified': True}
        write_json(models.INSTALL_PATH, saved)
    return entry


SETTINGS_VERSION = 8  # Partition-scoped H3 settings; reject non-inference step counts.


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
    _register({'id': entry_id, 'title': f"{model['name']} VAE", 'kind': 'vae', 'format': (_file_precision(file) or 'SafeTensor').upper(),
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
        required_version = SETTINGS_VERSION if item.get('family') in ('minimax-h3','minimax-h3-ref2va') else 7
        if item.get('kind') not in ('image','video','reference-video') or 'civitai' not in item or item.get('settings_version', 0) >= required_version:
            continue
        try:
            version = _get(f"/api/v1/model-versions/{item['civitai']['version_id']}")
            model = _get(f"/api/v1/models/{item['civitai']['model_id']}")
        except Exception:
            continue  # offline: try again on the next start
        version_text, model_text = _text(version.get('description')), _text(model.get('description'))
        from .civitai_workflows import attach_workflows
        attach_workflows([version])
        arch = (BASE_MODELS.get(version.get('baseModel')) or {}).get('arch')
        from .civitai_settings import version_settings, h3_defaults, h3_capabilities, ltx_defaults
        settings = version_settings(model, version)
        selected_file = next((file for file in version.get('files',[]) if any(
            file.get('hashes',{}).get('SHA256','').lower() == saved.get('sha256') and saved.get('sha256')
            or saved['path'].endswith(re.sub(r'[^A-Za-z0-9._-]+','_',file['name'])) for saved in item['files'])), None)
        if selected_file:
            item['civitai'].update(file_id=selected_file['id'],file_name=selected_file['name'])
            item['quantization']=_file_precision(selected_file) or item['format']
            if not item['title'].endswith(' \u00b7 '+item['quantization']):
                item['title']+=' \u00b7 '+item['quantization']
        if arch == 'minimax-h3':
            from . import models
            ref2va,modes=h3_capabilities(model,version,selected_file)
            template = models.entry('h3-ref-base-int8' if ref2va else 'h3-int8')
            item.update(kind=template['kind'],family=template['family'],modes=modes)
            item.update(generation_defaults=h3_defaults(model,version,template,item['civitai']['url'],selected_file),settings_version=SETTINGS_VERSION)
            _register(item)
            fixed.append(item['id'])
            continue
        if arch == 'ltx25':
            profile=ltx_defaults(model,version,item['civitai']['url'],selected_file)
            item.update(generation_defaults=profile,settings_version=SETTINGS_VERSION)
            if profile['variant'] in ('dev','distilled'):
                item['dependencies']=['ltx25-gemma-int8','ltx25-video-vae','ltx25-audio-vae']+(['ltx25-distill-lora'] if profile['variant']=='dev' else [])
                if profile['variant']=='dev':item['distill_lora_strength']=profile['distill_lora_strength']
                else:item.pop('distill_lora_strength',None)
            _register(item);fixed.append(item['id'])
            continue
        own = next((family for family in ('flux3-klein','krea-2','z-image') if item.get('family','').startswith(family)),None)
        if own:
            # A Krea 2 or Z-Image entry keeps the type it was installed as; only the page settings are read again.
            turbo = item['family']=='flux3-klein' if own=='flux3-klein' else item['family'].endswith('-turbo')
            item.update(generation_defaults=image_defaults(turbo, settings, item['civitai']['url'], own), settings_version=SETTINGS_VERSION)
        else:
            detected = detect_turbo(version['name'], settings)
            turbo = item.get('family') == 'qwen-image-2.1-turbo' if detected is None else detected
            item.update(family='qwen-image-2.1-turbo' if turbo else 'qwen-image-2.1',
                        generation_defaults=image_defaults(turbo, settings, item['civitai']['url']), settings_version=SETTINGS_VERSION)
        vae_file = next((f['name'] for f in version.get('files', []) if f.get('type') == 'VAE'), '')
        vae_id = _known_vae(vae_file) or recommended_vae(version_text + '\n' + model_text)
        if vae_id and not own:
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

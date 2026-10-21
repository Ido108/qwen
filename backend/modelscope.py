"""ModelScope (modelscope.cn) as a model source, with the same interface as the Hugging Face source in hf_hub.

Search uses the site's own model list (PUT /api/v1/dolphin/models) with its foundation tags. Each repo also names
its Hugging Face base repo (BaseModel), which maps to the app's architectures exactly like on Hugging Face.
ModelScope has no FLUX.3 Klein or LTX 2.5 tag: those come from a name search kept only when BaseModel matches.
"""
from __future__ import annotations
import json
import re
from types import SimpleNamespace
from urllib.parse import quote

import httpx

SITE = 'https://www.modelscope.cn'
HEADERS = {'User-Agent': 'QwenStudio/1.0', 'Content-Type': 'application/json'}
# Foundation tag -> base repo. Krea 2 Raw LoRAs are tagged KREA_2_TURBO too; LoRAs run on both types.
TAGS = {'QWEN_IMAGE_2_1': 'Qwen/Qwen-Image-2.1', 'QWEN_IMAGE_2_1_TURBO': 'Qwen/Qwen-Image-2.1-Turbo', 'KREA_2_TURBO': 'krea/Krea-2-Turbo',
        'Z_IMAGE': 'Tongyi-MAI/Z-Image', 'Z_IMAGE_TURBO': 'Tongyi-MAI/Z-Image-Turbo', 'MINIMAX_H3': 'MiniMaxAI/MiniMax-H3',
        'MOSS_IMAGE_SFT': 'OpenMOSS-Team/MOSS-Image-SFT'}
UNTAGGED = {'black-forest-labs/flux-3-image-klein': 'klein', 'black-forest-labs/flux-3-image-klein-base': 'klein', 'Lightricks/LTX-2.5': 'ltx'}
SORT = {'Most Downloaded': 'DownloadsCount', 'Highest Rated': 'StarsCount', 'Newest': 'GmtModified'}
PAGE = 24


def _call(method: str, path: str, body=None, params=None) -> dict:
    response = httpx.request(method, SITE + path, content=json.dumps(body) if body is not None else None, params=params, headers=HEADERS, timeout=40, follow_redirects=True)
    if response.status_code != 200:
        raise ConnectionError(f'ModelScope answered {response.status_code}. Try again in a moment.')
    data = response.json()
    if data.get('Code') not in (None, 200) or data.get('Success') is False:
        raise ConnectionError(f"ModelScope could not answer: {data.get('Message') or data.get('Code')}.")
    return data.get('Data') or {}


def _images(raw: dict) -> list[str]:
    return list(dict.fromkeys(re.findall(r'https://resources\.modelscope\.cn/cover-images/[^"\'\\\s]+?\.(?:png|jpe?g|webp)', json.dumps(raw))))[:8]


def _bases(raw: dict) -> list[str]:
    declared = raw.get('BaseModel') or []
    declared = [declared] if isinstance(declared, str) else list(declared)
    tag = TAGS.get(raw.get('SubVisionFoundation') or '')
    return list(dict.fromkeys([*declared, *([tag] if tag else [])]))


def _item(raw: dict) -> dict:
    images = _images(raw)
    repo = f"{raw['Path']}/{raw['Name']}"
    kind = {'LoRA': 'adapter'}.get(raw.get('AigcType'))
    return {'repo': repo, 'name': raw['Name'], 'creator': raw.get('NickName') or raw['Path'], 'downloads': raw.get('Downloads') or 0,
            'likes': raw.get('Stars') or 0, 'bases': _bases(raw), 'preview': images[0] if images else None,
            'relation': raw.get('BaseModelRelation') or kind}


def listing(bases: list[str], kind: str, sort: str, query: str, cursor: str | None) -> tuple[list[dict], str | None, int | None]:
    page = int(cursor or 1)
    tags = [tag for tag, repo in TAGS.items() if repo in bases]
    aigc = {'category': 'aigc_type', 'predicate': 'contains', 'values': ['LoRA' if kind == 'LORA' else 'Checkpoint']}
    items, total, more = [], 0, False
    if tags:
        data = _call('PUT', '/api/v1/dolphin/models', {'PageSize': PAGE, 'PageNumber': page, 'SortBy': SORT.get(sort, 'DownloadsCount'), 'Target': '', 'Name': query,
                                                     'Criterion': [{'category': 'sub_vision_foundation', 'predicate': 'contains', 'values': tags}, aigc]}).get('Model') or {}
        items += [_item(raw) for raw in data.get('Models') or []]
        total += data.get('TotalCount') or 0
        more = page * PAGE < (data.get('TotalCount') or 0)
    for word in dict.fromkeys(word for repo, word in UNTAGGED.items() if repo in bases):
        data = _call('PUT', '/api/v1/dolphin/models', {'PageSize': PAGE, 'PageNumber': page, 'SortBy': SORT.get(sort, 'DownloadsCount'), 'Target': '',
                                                     'Name': f'{query} {word}'.strip(), 'Criterion': [aigc]}).get('Model') or {}
        found = [_item(raw) for raw in data.get('Models') or []]
        items += [item for item in found if set(item['bases']) & set(bases)]
        more = more or page * PAGE < (data.get('TotalCount') or 0)
    return items, str(page + 1) if more else None, total or None


def card(repo: str) -> dict:
    raw = _call('GET', f'/api/v1/models/{repo}')
    revision = raw.get('Revision') or 'master'
    listed = _call('GET', f'/api/v1/models/{repo}/repo/files', params={'Revision': revision, 'Recursive': 'true'}).get('Files') or []
    files = [{'path': f['Path'], 'bytes': int(f.get('Size') or 0), 'sha256': (f.get('Sha256') or '').lower()} for f in listed if f.get('Type') == 'blob']
    words = raw.get('TriggerWords') or []
    words = re.split(r'[,\n]', words) if isinstance(words, str) else [w for w in words if isinstance(w, str)]
    item = _item(raw)
    return {**item, 'source': 'modelscope', 'url': f'{SITE}/models/{repo}', 'revision': revision, 'gated': False,
            'type': {'LoRA': 'LORA', 'Checkpoint': 'Checkpoint'}.get(raw.get('AigcType')),
            'text': '\n\n'.join(part for part in (raw.get('Description') or '', raw.get('ReadMeContent') or '') if part)[:20000],
            'trained_words': [w.strip() for w in words if w.strip()], 'images': _images(raw), 'files': files}


def file_url(repo: str, revision: str, path: str) -> str:
    return f'{SITE}/api/v1/models/{repo}/repo?Revision={quote(revision)}&FilePath={quote(path)}'


def headers() -> dict:
    return {'User-Agent': HEADERS['User-Agent']}


SOURCE = SimpleNamespace(name='ModelScope', listing=listing, card=card, file_url=file_url, headers=headers)

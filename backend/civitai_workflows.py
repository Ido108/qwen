"""Read small publisher workflows as data. Never execute nodes or fetch weights."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

_cache = {}
MAX_BYTES = 1024 * 1024


def sampling_from_workflow(data):
    """Follow the connected sampler, not disconnected example/settings nodes."""
    from .civitai_settings import partition_hint
    if not isinstance(data, dict):
        return None
    nodes = {n['id']: n for n in data.get('nodes', []) if n.get('mode', 0) == 0}
    links = {link[0]: link for link in data.get('links', []) if isinstance(link, list) and len(link) >= 6}

    def input_node(node, name):
        link_id = next((i.get('link') for i in node.get('inputs', []) if i.get('name') == name), None)
        link = links.get(link_id)
        return nodes.get(link[1], {}) if link else {}

    def model_path(node):
        seen, shift, checkpoint = set(), None, None
        while node and node['id'] not in seen:
            seen.add(node['id'])
            kind, values = node.get('type', ''), node.get('widgets_values', [])
            # A separate acceleration/style LoRA changes the checkpoint contract.
            if 'lora' in kind.lower():
                return None
            if kind == 'MiniMaxH3SigmaShift' and shift is None:
                if len(values) < 2 or any(not isinstance(v,(int,float)) or not 0 < v <= 30 for v in values[:2]):
                    return None
                shift = values[:2]
            if 'loader' in kind.lower() and values and isinstance(values[0], str):
                checkpoint = values[0]
                break
            node = input_node(node, 'model')
        if not checkpoint:
            return None
        return checkpoint, shift or [12.0, 3.0]

    samplers = [n for n in nodes.values() if n.get('type') == 'SamplerCustomAdvanced']
    if len(samplers) != 1:
        return None  # Multi-pass workflows need an explicit contract.
    sampler = samplers[0]
    guider, scheduler, selection = (input_node(sampler, key) for key in ('guider', 'sigmas', 'sampler'))
    if guider.get('type') != 'BasicGuider' or scheduler.get('type') != 'BasicScheduler':
        return None
    model = model_path(input_node(guider, 'model'))
    schedule_model = model_path(input_node(scheduler, 'model'))
    values = scheduler.get('widgets_values', [])
    if not model or not schedule_model or model[0] != schedule_model[0] or len(values) < 3:
        return None
    schedule, steps, denoise = values[:3]
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 60 or denoise != 1:
        return None
    from .civitai import SAMPLERS, SCHEDULERS
    kind = selection.get('type')
    name = (selection.get('widgets_values') or [None])[0] if kind == 'KSamplerSelect' else None
    if kind == 'MiniMaxH3TurboSampler':
        # The publisher node is stock Euler on ModelSamplingAV (our pinned H3).
        name = 'euler'
    if name not in SAMPLERS or schedule not in SCHEDULERS:
        return None
    reference = partition_hint(model[0])
    if reference is None:
        return None
    return dict(reference=reference, checkpoint=model[0], steps=steps, cfg=1.0,
                sampler=name, scheduler=schedule, video_shift=model[1][0], audio_shift=model[1][1],
                scheduler_video_shift=schedule_model[1][0], scheduler_audio_shift=schedule_model[1][1])


def _read_config(version_id, file):
    from .civitai import SITE, HEADERS, access_token
    key = (version_id, file['id'], (file.get('hashes') or {}).get('SHA256'))
    saved = _cache.get(key)
    if saved and time.monotonic() - saved[0] < (3600 if saved[1] else 60):
        return saved[1]
    url = f"{SITE}/api/download/models/{int(version_id)}?fileId={int(file['id'])}"
    result = None
    try:
        token = access_token()
        headers = {**HEADERS, **({'Authorization': f'Bearer {token}'} if token else {})}
        with httpx.stream('GET', url, headers=headers, follow_redirects=True, timeout=15) as response:
            response.raise_for_status()
            if int(response.headers.get('content-length', 0)) > MAX_BYTES:
                raise ValueError('Workflow too large')
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > MAX_BYTES:
                    raise ValueError('Workflow too large')
        result = sampling_from_workflow(json.loads(raw))
        if result:
            result['source'] = url
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        pass  # A missing/unsupported workflow must not block browsing.
    _cache[key] = (time.monotonic(), result)
    return result


def attach_workflows(versions):
    jobs = [(version, file) for version in versions if version.get('baseModel') == 'MiniMax H3'
            for file in version.get('files', [])[:50]
            if file.get('type') == 'Config' and file.get('name', '').lower().endswith('.json')
            and 0 < file.get('sizeKB', 0) <= MAX_BYTES / 1024]
    if not jobs:
        return
    # These are JSON settings only. Bound concurrent requests and response sizes.
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda job: _read_config(job[0]['id'], job[1]), jobs[:12]))
    for (version, _), result in zip(jobs, results):
        if result:
            version.setdefault('_workflow_settings', []).append(result)

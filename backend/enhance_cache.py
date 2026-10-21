"""Saved Enhance results: the same request, input pictures and enhancer give the same rewrite, so a repeat (any seed)
reuses it instead of running the enhancer again."""
import hashlib
import json
import threading
from .common import DATA_DIR, write_json

PATH = DATA_DIR / 'enhance-cache.json'
SIZE = 300
_lock = threading.Lock()


def key(*values, images=()):
    digest = hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, default=str).encode())
    for image in images:
        digest.update(f'{image.mode}{image.size}'.encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


def get(name):
    try:
        return json.loads(PATH.read_text(encoding='utf-8')).get(name)
    except (OSError, ValueError):
        return None


def put(name, value):
    if not value:
        return
    with _lock:
        try:
            data = json.loads(PATH.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = {}
        data.pop(name, None)
        data[name] = value
        write_json(PATH, dict(list(data.items())[-SIZE:]))

"""People: named avatars with reference photos, for quick selection as input images.

Moise (the WhatsApp assistant on Railway) keeps contact names and profile
photos. Import reads both with the dashboard address and password: from the
saved, encrypted copy, or once through the logged-in Railway CLI. Set
QWEN_MOISE_URL to use another address.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from .secure_store import AUTH_DIR, _crypt
from .common import APP_DIR, DATA_DIR, UPLOADS_DIR, write_json

PEOPLE_FILE = DATA_DIR / 'people.json'
AVATARS = DATA_DIR / 'avatars'
MOISE_FILE = AUTH_DIR / 'moise.dpapi'
MOISE_REPO = APP_DIR.parent / 'moise'
DETECTOR = APP_DIR / 'reference' / 'face_detection_yunet_2023mar.onnx'
_lock = threading.RLock()
status = {'state': 'idle'}


def load() -> list[dict]:
    try:
        return json.loads(PEOPLE_FILE.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return []


def _save(people: list[dict]) -> None:
    write_json(PEOPLE_FILE, people)


def public(person: dict) -> dict:
    cover = person['image_ids'][0] if person['image_ids'] else ''
    return {'id': person['id'], 'name': person['name'], 'image_ids': person['image_ids'], 'source': person.get('source', 'manual'),
            'image_id': cover, 'url': f'/api/uploads/{cover}', 'thumb_url': f"/api/people/{person['id']}/avatar?v={person.get('updated_at', 0)}",
            'label': person.get('label')}


def listing() -> list[dict]:
    return [public(p) for p in sorted(load(), key=lambda p: p['name'].casefold())]


def save_image(picture: Image.Image) -> str:
    picture = ImageOps.exif_transpose(picture).convert('RGBA')
    image_id = hashlib.sha256(picture.tobytes() + f'{picture.size}'.encode()).hexdigest()[:32] + '.png'
    path = UPLOADS_DIR / image_id
    if not path.is_file():
        temporary = UPLOADS_DIR / f'.{uuid.uuid4().hex}.tmp'
        picture.save(temporary, format='PNG')
        temporary.replace(path)
    return image_id


def upsert(name: str, image_ids: list[str], source: str = 'manual', source_id: str | None = None) -> dict:
    name = re.sub(r'\s+', ' ', name).strip()[:80]
    if not name:
        raise ValueError('Enter a name.')
    if not image_ids:
        raise ValueError('Choose at least one photo.')
    with _lock:
        people = load()
        match = next((p for p in people if (source_id and p.get('source_id') == source_id) or (not source_id and p['name'].casefold() == name.casefold())), None)
        if match:
            match['name'] = name
            match['image_ids'] = list(dict.fromkeys(match['image_ids'] + image_ids))
            match['updated_at'] = time.time()
            person = match
        else:
            person = {'id': uuid.uuid4().hex[:12], 'name': name, 'image_ids': list(dict.fromkeys(image_ids)), 'source': source,
                      'source_id': source_id, 'updated_at': time.time()}
            people.append(person)
        _save(people)
    (AVATARS / f"{person['id']}.webp").unlink(missing_ok=True)
    return public(person)


def update(person_id: str, name: str | None = None, image_ids: list[str] | None = None, label: str | None = None) -> dict:
    with _lock:
        people = load()
        person = next((p for p in people if p['id'] == person_id), None)
        if not person:
            raise ValueError('This person no longer exists.')
        if name is not None:
            if not name.strip():
                raise ValueError('Enter a name.')
            person['name'] = name.strip()[:80]
        if image_ids is not None:
            if not image_ids:
                raise ValueError('A person needs at least one photo.')
            person['image_ids'] = list(dict.fromkeys(image_ids))
        if label is not None:
            # What a mention of this person's photos reads as ("woman", "man"), kept for every photo of them.
            person['label'] = label.strip().lower()[:48] or None
        person['updated_at'] = time.time()
        _save(people)
    (AVATARS / f'{person_id}.webp').unlink(missing_ok=True)
    return public(person)


def delete(person_id: str) -> None:
    with _lock:
        _save([p for p in load() if p['id'] != person_id])
    (AVATARS / f'{person_id}.webp').unlink(missing_ok=True)


def avatar(person_id: str) -> Path:
    """A square, face-centred 256 px avatar of the person's first photo."""
    cached = AVATARS / f'{person_id}.webp'
    if cached.is_file():
        return cached
    person = next((p for p in load() if p['id'] == person_id), None)
    if not person or not person['image_ids']:
        raise ValueError('No photo for this person.')
    with Image.open(UPLOADS_DIR / person['image_ids'][0]) as source:
        picture = source.convert('RGB')
    box = _face_square(picture)
    AVATARS.mkdir(parents=True, exist_ok=True)
    temporary = AVATARS / f'{uuid.uuid4().hex}.tmp'
    picture.crop(box).resize((256, 256), Image.Resampling.LANCZOS).save(temporary, format='WEBP', quality=88)
    temporary.replace(cached)
    return cached


def _face_square(picture: Image.Image) -> tuple[int, int, int, int]:
    width, height = picture.size
    side = min(width, height)
    fallback = ((width - side) // 2, max(0, (height - side) // 4), (width - side) // 2 + side, max(0, (height - side) // 4) + side)
    try:
        import cv2
        import numpy as np
        scale = min(1.0, 960 / max(width, height))
        sample = np.array(picture.resize((max(1, int(width * scale)), max(1, int(height * scale)))))[:, :, ::-1].copy()
        detector = cv2.FaceDetectorYN.create(str(DETECTOR), '', (sample.shape[1], sample.shape[0]), score_threshold=0.6)
        _, faces = detector.detect(sample)
    except Exception:
        return fallback
    if faces is None or not len(faces):
        return fallback
    x, y, w, h = (max(faces, key=lambda f: f[2] * f[3])[:4] / scale)
    # Head and shoulders: the face fills about 45% of the square.
    side = int(min(max(w, h) / 0.45, width, height))
    cx, cy = x + w / 2, y + h / 2 + h * 0.15
    left = int(min(max(0, cx - side / 2), width - side))
    top = int(min(max(0, cy - side / 2), height - side))
    return left, top, left + side, top + side


# ---- Moise import -----------------------------------------------------------

def _moise_login() -> tuple[str, str]:
    """The Moise address and dashboard password: saved (encrypted), or read once from the Railway project."""
    saved = {}
    if MOISE_FILE.exists():
        try:
            saved = json.loads(_crypt(MOISE_FILE.read_bytes(), decrypt=True))
        except Exception:
            saved = {}
    url = os.environ.get('QWEN_MOISE_URL') or saved.get('url')
    if saved.get('password') and url:
        return url.rstrip('/'), saved['password']
    if not (MOISE_REPO / 'railway.json').exists() and not (MOISE_REPO / 'railway.toml').exists():
        raise PermissionError('Moise was not found next to this app. Set QWEN_MOISE_URL and connect it once.')
    railway = shutil.which('railway')
    if not railway:
        raise PermissionError('The Railway CLI is not installed, so the Moise address and password cannot be read.')
    result = subprocess.run([railway, 'variables', '--json'], cwd=MOISE_REPO, capture_output=True, text=True, timeout=60,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        variables = json.loads(result.stdout)
        password = saved.get('password') or variables['DASHBOARD_PASSWORD']
        url = url or f"https://{variables['RAILWAY_PUBLIC_DOMAIN']}"
    except (ValueError, KeyError):
        raise PermissionError('Railway did not return the Moise address and password. Run "railway login" first.')
    return url.rstrip('/'), password


def _remember(url: str, password: str) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MOISE_FILE.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temporary.write_bytes(_crypt(json.dumps({'url': url, 'password': password}).encode()))
    temporary.replace(MOISE_FILE)


def import_moise(password: str | None = None) -> dict:
    with _lock:
        if status.get('state') == 'running':
            raise ValueError('An import is already running.')
        status.clear(); status.update(state='running', message='Connecting to Moise…', done=0, total=0)
    try:
        url, saved_password = _moise_login()
        password = password or saved_password
        client = httpx.Client(base_url=url, headers={'Authorization': f'Bearer {password}', 'User-Agent': 'QwenStudio/1.0'}, timeout=90)
        summary = client.get('/api/get-profile-images', params={'summary': 'true'})
        if summary.status_code in (401, 403):
            MOISE_FILE.unlink(missing_ok=True)
            raise PermissionError('Moise did not accept the password.')
        summary.raise_for_status()
        _remember(url, password)
        names = json.loads(client.get('/api/get-contacts-json').json().get('jsonContent') or '{}')
        contacts = [wa for wa, info in (summary.json().get('profileImages') or {}).items() if info.get('count') and not wa.endswith('@g.us')]
        status.update(total=len(contacts), message='Importing photos…')
        imported = 0
        for wa in contacts:
            response = client.get('/api/get-profile-images', params={'userId': wa})
            if response.status_code != 200:
                continue
            images = _images(response.json(), wa)
            ids = []
            for item in images:
                raw = _bytes(client, item)
                if raw:
                    try:
                        with Image.open(io.BytesIO(raw)) as picture:
                            ids.append(save_image(picture))
                    except Exception:
                        continue
            name = names.get(wa) or next((i.get('contactName') for i in images if i.get('contactName')), '') or wa.split('@')[0]
            if ids:
                upsert(name, ids, 'moise', wa)
                imported += 1
            status.update(done=status['done'] + 1, message=f'Imported {name}')
        status.update(state='done', message=f'Imported {imported} people from Moise.', imported=imported)
    except Exception as exc:
        status.update(state='failed', message=str(exc)[:300])
    return dict(status)


def _images(payload, wa: str) -> list[dict]:
    data = payload.get('profileImages', payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        data = data.get(wa) or data.get('images') or next(iter(data.values()), [])
    return [item for item in (data or []) if isinstance(item, dict)]


def _bytes(client: httpx.Client, item: dict) -> bytes | None:
    if item.get('data'):
        value = item['data'].split(',', 1)[-1]
        try:
            return base64.b64decode(value)
        except ValueError:
            return None
    path = item.get('path') or ''
    if path:
        relative = path.split('profile_images_files/')[-1].lstrip('/')
        response = client.get(f'/profile-files/{relative}')
        if response.status_code == 200:
            return response.content
    return None

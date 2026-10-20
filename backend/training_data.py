"""Bounded dataset uploads. Archive paths are names only; nothing is extracted."""
from __future__ import annotations
import hashlib
import io
import json
import re
import shutil
import stat
import threading
import time
import uuid
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from PIL import Image, ImageOps
from .common import DATA_DIR, read_json, write_json

ROOT = DATA_DIR / 'training-imports'
MAX_BYTES = 1024 ** 3
MAX_FILE = 32 * 1024 ** 2
MAX_CHUNK = 4 * 1024 ** 2
MAX_IMAGES = 300
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.heic', '.heif'}
CAPTION_EXT = {'.txt', '.caption'}
lock = threading.RLock()


def safe_name(name):
    name = name.replace('\\', '/')
    parts = name.split('/')
    if not name or len(name) > 600 or name.startswith('/') or any(p in ('..', '.', '') for p in parts) or re.search(r'[\x00-\x1f:]', name):
        raise ValueError('A file has an invalid folder path.')
    return str(PurePosixPath(name))


def directory(import_id):
    if not re.fullmatch(r'[a-f0-9]{32}', import_id):
        raise ValueError('This upload does not exist. Select the files again.')
    path = ROOT / import_id
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Invalid upload folder.')
    return path


def create(files):
    if not 1 <= len(files) <= 1500:
        raise ValueError('Select up to 300 images with their captions, or one ZIP file.')
    names = [safe_name(f['name']) for f in files]
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValueError('Two files use the same folder path. Rename one before import.')
    if any(f['size'] < 0 or f['size'] > MAX_BYTES for f in files) or sum(f['size'] for f in files) > MAX_BYTES:
        raise ValueError('The dataset must be at most 1 GB.')
    ident = uuid.uuid4().hex
    path = directory(ident)
    path.mkdir(parents=True)
    write_json(path / 'manifest.json', {'files': [dict(name=n, size=f['size']) for n, f in zip(names, files)], 'created_at': time.time()})
    return {'id': ident, 'chunk_bytes': MAX_CHUNK}


def append(import_id, index, offset, raw):
    with lock:
        root = directory(import_id)
        manifest = read_json(root / 'manifest.json')
        if offset < 0 or not 0 <= index < len(manifest['files']) or not 0 <= len(raw) <= MAX_CHUNK:
            raise ValueError('Invalid upload part.')
        target = root / f'{index}.part'
        size = target.stat().st_size if target.exists() else 0
        expected = manifest['files'][index]['size']
        # An acknowledged part may be retried after a remote connection drops.
        if offset < size:
            with target.open('rb') as stream:
                stream.seek(offset)
                if stream.read(len(raw)) == raw:
                    return {'offset': size}
        if offset != size or size + len(raw) > expected:
            raise ValueError('The upload part is out of order. Select the files again.')
        with target.open('ab') as stream:
            stream.write(raw)
        return {'offset': size + len(raw)}


def discard(import_id):
    with lock:
        target = directory(import_id)
        if target.exists():
            shutil.rmtree(target)  # only the validated UUID staging folder


def _caption(raw, name):
    if len(raw) > 32_000:
        raise ValueError(f'Caption is too long: {name}. Use at most 8,000 characters.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError:
        if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
            text = raw.decode('utf-16')
        else:
            raise ValueError(f'Cannot read caption {name}. Save it as UTF-8 text.')
    if len(text) > 8000 or '\x00' in text:
        raise ValueError(f'Invalid caption: {name}. Use at most 8,000 characters.')
    return text


def import_files(files, save_image):
    """files are (relative name, open stream callback, byte size). No full archive in RAM."""
    entries = {}; opened = ExitStack(); ignored = 0; total = 0
    def add(name, opener, size):
        nonlocal total, ignored
        name = safe_name(name)
        if any(p.startswith('.') or p == '__MACOSX' for p in PurePosixPath(name).parts):
            ignored += 1; return
        ext = PurePosixPath(name).suffix.lower()
        if ext not in IMAGE_EXT | CAPTION_EXT:
            ignored += 1; return
        limit = MAX_FILE if ext in IMAGE_EXT else 32_000
        total += size
        if size > limit or total > MAX_BYTES:
            raise ValueError('Dataset too large. Use images under 32 MB and a dataset under 1 GB.')
        key = name.casefold()
        if key in entries:
            raise ValueError(f'Two files use the same folder path: {name}.')
        entries[key] = (name, opener, size)
    try:
        for name, opener, size in files:
            if PurePosixPath(name).suffix.lower() == '.zip':
                stream = opened.enter_context(opener())
                archive = opened.enter_context(zipfile.ZipFile(stream))
                if len(archive.infolist()) > 1500:
                    raise ValueError('The ZIP contains too many files. Use up to 300 images with captions.')
                prefix = str(PurePosixPath(name).with_suffix('')) + '/'
                for member in archive.infolist():
                    if member.is_dir(): continue
                    safe_name(member.filename)
                    if member.flag_bits & 1 or stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError('Encrypted files and links are not supported in datasets.')
                    if member.file_size > MAX_FILE or member.file_size / max(member.compress_size, 1) > 2000:
                        raise ValueError('A ZIP entry exceeds the image size limit.')
                    add(prefix + member.filename, lambda a=archive, m=member: a.open(m), member.file_size)
            else:
                add(name, opener, size)
        pictures = [e for e in entries.values() if PurePosixPath(e[0]).suffix.lower() in IMAGE_EXT]
        if not 1 <= len(pictures) <= MAX_IMAGES:
            raise ValueError('Select between 1 and 300 images. Captions must be TXT or CAPTION files beside each image.')
        items = []; seen = set(); duplicates = 0; paired = set()
        # Decode one image at a time. Imported originals share the existing pixel-hash upload store.
        for name, opener, _ in pictures:
            stem = str(PurePosixPath(name).with_suffix('')).casefold()
            candidates = [stem + ext for ext in ('.txt', '.caption')] + [name.casefold() + '.txt']
            matches = [entries[k] for k in candidates if k in entries]
            captions = []
            for capname, capopen, _ in matches:
                with capopen() as stream: captions.append(_caption(stream.read(32_001), capname))
                paired.add(capname.casefold())
            if len(set(captions)) > 1:
                raise ValueError(f'Different captions match {name}. Keep one caption file for this image.')
            caption = captions[0] if captions else ''
            with opener() as stream, Image.open(stream) as source:
                if source.width * source.height > 64_000_000:
                    raise ValueError(f'Image exceeds 64 megapixels: {name}.')
                if getattr(source, 'n_frames', 1) > 1:
                    raise ValueError(f'Use a still image for {name}.')
                picture = ImageOps.exif_transpose(source).convert('RGBA')
                item = save_image(picture, PurePosixPath(name).name)
            signature = (item['id'], caption)
            if signature in seen:
                duplicates += 1; continue
            seen.add(signature)
            key = hashlib.sha256((item['id'] + '\0' + caption).encode()).hexdigest()[:24]
            items.append({**item, 'dataset_key': key, 'relative_path': name, 'caption': caption,
                          'caption_source': 'file' if caption.strip() else 'missing'})
        unpaired = sum(1 for key in entries if PurePosixPath(key).suffix in CAPTION_EXT and key not in paired)
        return {'items': items, 'captions': sum(bool(i['caption'].strip()) for i in items),
                'duplicates': duplicates, 'ignored': ignored, 'unpaired_captions': unpaired}
    finally:
        opened.close()


def finish(import_id, save_image):
    with lock:
        root = directory(import_id)
        manifest = read_json(root / 'manifest.json')
        files = []
        for index, item in enumerate(manifest['files']):
            path = root / f'{index}.part'
            if not path.exists() or path.stat().st_size != item['size']:
                raise ValueError('The upload is incomplete. Select the files again.')
            files.append((item['name'], lambda p=path: p.open('rb'), item['size']))
        try:
            return import_files(files, save_image)
        finally:
            discard(import_id)

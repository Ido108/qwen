"""Turn a saved output into a normal, persistent video input without transcoding."""
from __future__ import annotations

import os
import json
from pathlib import Path
import re
import shutil
import threading
import uuid

from .common import read_json, write_json

_lock = threading.Lock()


def _read_optional(path: Path) -> dict:
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def inspect_video(path: Path, poster_path: Path) -> dict:
    import av

    with av.open(str(path)) as media:
        stream = next(iter(media.streams.video), None)
        if stream is None:
            raise ValueError('This file has no video track.')
        frame = next(media.decode(stream))
        duration = (float(media.duration) / av.time_base if media.duration else
                    float(stream.duration * stream.time_base) if stream.duration else
                    float(stream.frames / stream.average_rate) if stream.frames and stream.average_rate else None)
        poster = frame.to_image()
        poster.thumbnail((960, 540))
        poster.save(poster_path, format='JPEG', quality=88)
        return {'width': frame.width, 'height': frame.height,
                'fps': float(stream.average_rate) if stream.average_rate else None,
                'duration_seconds': duration, 'has_audio': bool(media.streams.audio)}


def reuse_video_file(source: Path, uploads: Path, job_folder: Path, part: str = 'final') -> dict:
    # A separate lock never blocks the generation queue. Repeated sends share one input.
    with _lock:
        stat = source.stat()
        stamp = {'path': str(source.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        cache_file = job_folder / f'reused-video-{part}.json'
        cached = _read_optional(cache_file)
        video_id = cached.get('id', '')
        if cached.get('source') == stamp and re.fullmatch(r'[a-f0-9]{32}\.mp4', video_id):
            saved = _read_optional(uploads / (video_id + '.json'))
            if saved.get('id') == video_id and (uploads / video_id).is_file() and (uploads / (video_id + '.jpg')).is_file():
                return saved
        video_id = uuid.uuid4().hex + '.mp4'
        destination = uploads / video_id
        temporary = uploads / (video_id + '.part')
        poster_path = uploads / (video_id + '.jpg')
        metadata_path = uploads / (video_id + '.json')
        uploads.mkdir(parents=True, exist_ok=True)
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            info = inspect_video(temporary, poster_path)
            result = {'id': video_id, 'name': source.name, 'bytes': stat.st_size,
                      'url': f'/api/video-uploads/{video_id}',
                      'poster_url': f'/api/video-uploads/{video_id}/poster', **info}
            temporary.replace(destination)
            write_json(metadata_path, result)
            write_json(cache_file, {'source': stamp, 'id': video_id})
            return result
        except Exception:
            for path in (temporary, destination, poster_path, metadata_path):
                path.unlink(missing_ok=True)
            raise

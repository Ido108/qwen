"""Bounded RAM reuse of source images and deterministic image-tool preparation.

Uploads remain the durable originals. Job folders do not need input PNG copies.
Returned images belong to the caller; mutations cannot affect a later job.
"""
from collections import OrderedDict
import json
from pathlib import Path

from PIL import Image

from .image_cache import ConditioningCache


def _image_bytes(value):
    if isinstance(value, Image.Image):
        return value.width * value.height * len(value.getbands())
    if isinstance(value, (list, tuple)):
        return sum(_image_bytes(item) for item in value)
    return 0


def _copy_images(value):
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, list):
        return [_copy_images(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_images(item) for item in value)
    return value


class ImageInputs:
    def __init__(self, max_bytes=256 * 1024 * 1024, max_entries=24):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0

    def _get(self, key):
        item = self.entries.get(key)
        if item is None:
            return None
        self.entries.move_to_end(key)
        return _copy_images(item[0])

    def _put(self, key, value):
        size = _image_bytes(value)
        if size > self.max_bytes or self.max_entries < 1:
            return
        previous = self.entries.pop(key, None)
        if previous:
            self.bytes -= previous[1]
        while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            _, (_, old_size) = self.entries.popitem(last=False)
            self.bytes -= old_size
        self.entries[key] = (_copy_images(value), size)
        self.bytes += size

    def load(self, path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        key = ('source', str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._get(key)
        if cached is not None:
            return cached, True
        with Image.open(path) as source:
            image = source.convert('RGBA')
        # A replacement at this path invalidates its old decoded image.
        for old in list(self.entries):
            if old[0:2] == key[0:2]:
                self.bytes -= self.entries.pop(old)[1]
        self._put(key, image)
        return image, False

    def prepare(self, tool, request, images):
        from .image_tools import prepare
        # These are all request fields read by image_tools.prepare/budget.
        settings = {name: request.get(name) for name in
                    ('custom_size', 'width', 'height', 'auto_resolution', 'tool_options')}
        key = ('prepared', ConditioningCache.fingerprint(
            json.dumps([tool, settings], sort_keys=True), '', images))
        cached = self._get(key)
        if cached is not None:
            return cached, True
        prepared = prepare(tool, request, images)
        self._put(key, prepared)
        return prepared, False

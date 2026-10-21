"""Large sequential reads of model files into the Windows file cache, ahead of the mapped load.

ComfyUI's read-only mapping fills tensors through page faults. On the external USB drive those
small reads measured 0.51 GB/s; four 64 MB readers running ahead of the same mapped copy
measured 1.39 GB/s (CPU only, 3 GB regions of unused files). The cached pages are standby
memory: they need no commit charge and Windows drops them under pressure, so a prefetch can
make a load faster but never makes it fail.
"""
from __future__ import annotations
import threading
from pathlib import Path

CHUNK = 64 * 1024 * 1024
READERS = 4
MINIMUM = 256 * 1024 * 1024
_queue: list[Path] = []
_done: set[Path] = set()
_condition = threading.Condition()
_worker: threading.Thread | None = None


def _read(path: Path) -> None:
    size = path.stat().st_size
    def reader(index: int) -> None:
        buffer = bytearray(CHUNK)
        view = memoryview(buffer)
        with open(path, 'rb', buffering=0) as handle:
            offset = index * CHUNK
            while offset < size:
                handle.seek(offset)
                if not handle.readinto(view):
                    break
                offset += READERS * CHUNK
    threads = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(READERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _run() -> None:
    while True:
        with _condition:
            while not _queue:
                _condition.wait()
            path = _queue[0]
        try:
            _read(path)
        except OSError:
            pass  # The load itself reports a missing or unreadable file.
        with _condition:
            _queue.pop(0)
            _done.add(path)
            _condition.notify_all()


def prefetch(*paths) -> None:
    """Queue files to read in order in the background. Small and already queued files are skipped."""
    global _worker
    with _condition:
        for value in paths:
            path = Path(value)
            try:
                if path.stat().st_size < MINIMUM:
                    continue
            except OSError:
                continue
            if path not in _done and path not in _queue:
                _queue.append(path)
        if _worker is None:
            _worker = threading.Thread(target=_run, name='model-file-prefetch', daemon=True)
            _worker.start()
        _condition.notify_all()

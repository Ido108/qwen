"""A long video that is not finished: its finished parts, joined in order into one playable file.

The parts are joined by stream copy (no new encoding, about a second), so each join is a plain cut. The file is in
the parts folder, next to the parts, and is built again when another part is saved. A run that stops in the middle
(stopped by the owner, a failure, the app closed) keeps the file of the parts it finished.
"""
from __future__ import annotations

import os
import subprocess
import threading
import uuid
from pathlib import Path

_lock = threading.Lock()


def unfinished_with_parts(job: dict) -> bool:
    """A job that did not finish and has at least one saved part."""
    return job.get('status') != 'succeeded' and int(job.get('parts_saved') or 0) >= 1 and bool(job.get('parts_folder'))


def joined_parts(folder: Path, count: int) -> Path | None:
    """The file of parts 1 to count of folder joined in order; None when part 1 is not there.
    Parts after a missing part are left out."""
    import imageio_ffmpeg
    folder = Path(folder)
    parts = []
    for index in range(1, count + 1):
        part = folder / f'part-{index:03d}.mp4'
        if not part.is_file():
            break
        parts.append(part)
    if not parts:
        return None
    target = folder / f'parts-001-to-{len(parts):03d}.mp4'
    with _lock:
        if not target.is_file():
            listing, temporary = folder / f'join-{uuid.uuid4().hex}.txt', folder / f'join-{uuid.uuid4().hex}.mp4'
            try:
                listing.write_text(''.join(f"file '{part.name}'\n" for part in parts), encoding='utf-8')
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                                '-safe', '0', '-i', listing.name, '-c', 'copy', '-movflags', '+faststart', temporary.name],
                               check=True, capture_output=True, cwd=str(folder),
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
                temporary.replace(target)
            finally:
                listing.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
        for old in folder.glob('parts-001-to-*.mp4'):
            if old != target:
                try:
                    old.unlink()
                except OSError:
                    pass  # it is being played; the next join removes it
    return target

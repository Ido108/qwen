"""MiniMax H3 clip boundaries and its 17n+5 output-frame grid."""
from __future__ import annotations

import math
from pathlib import Path

FPS = 24
MIN_FRAMES = 22
MAX_FRAMES = 362


def nearest_frames(seconds: float, max_frames: int = MAX_FRAMES) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Select a video segment with a valid duration.')
    maximum = min(MAX_FRAMES, max_frames)
    if maximum < MIN_FRAMES or maximum % 17 != 5:
        raise ValueError('The selected model has an invalid video duration limit.')
    count = round((seconds * FPS - 5) / 17)
    return min(maximum, max(MIN_FRAMES, 5 + 17 * count))


def probe_video_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as media:
        stream = next((item for item in media.streams if item.type == 'video'), None)
        if stream is None:
            raise ValueError('The source file has no video stream.')
        if media.duration and media.duration > 0:
            return media.duration / 1_000_000
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if stream.frames and stream.average_rate:
            return stream.frames / float(stream.average_rate)
        last_time = None
        rate = float(stream.average_rate) if stream.average_rate else FPS
        for index, frame in enumerate(media.decode(stream)):
            last_time = frame.time if frame.time is not None else index / rate
        if last_time is None:
            raise ValueError('The source video has no readable frames.')
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None and stream.time_base else 0
        return last_time - origin + 1 / rate


def resolve_reference_clip(request, source_duration: float, max_frames: int = MAX_FRAMES):
    """Resolve a saved selection; legacy requests without an end use their frame length."""
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError('Could not determine the source video duration.')
    start = request.source_start_seconds
    if not math.isfinite(start) or start < 0 or start >= source_duration:
        raise ValueError('The selected start is outside the source video.')
    end = request.source_end_seconds
    if end is None:
        end = min(source_duration, start + request.frames / FPS)
    if not math.isfinite(end) or end <= start or end > source_duration + 1 / FPS:
        raise ValueError('The selected end is outside the source video.')
    end = min(end, source_duration)
    selected_seconds = end - start
    if selected_seconds < MIN_FRAMES / FPS - 1 / FPS:
        raise ValueError('Select at least 0.9 seconds of source video.')
    if selected_seconds > max_frames / FPS + 1 / FPS:
        raise ValueError(f'Select at most {max_frames / FPS:.1f} seconds for this model.')
    frames = nearest_frames(selected_seconds, max_frames)
    request.source_start_seconds = round(start, 4)
    request.source_end_seconds = round(end, 4)
    request.frames = frames
    return request

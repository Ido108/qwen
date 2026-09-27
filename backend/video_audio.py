"""Bounded audio muxing; the completed video stream is copied unchanged."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import time


def add_source_audio(video: Path, source: Path, destination: Path, update, *, preserve_audio=True) -> None:
    import av
    import imageio_ffmpeg
    from .reference_timing import probe_video_duration

    duration = probe_video_duration(video)
    if duration <= 0:
        raise ValueError('The upscaled video has no valid duration.')
    with av.open(str(source)) as media:
        has_audio = preserve_audio and bool(media.streams.audio)
    with av.open(str(video)) as media:
        browser_ready=media.streams.video[0].codec_context.name=='h264'
    if not has_audio and browser_ready:
        update(stage='saving', message='Saving video without audio…', step=0, total_steps=0)
        shutil.copyfile(video, destination)
        return
    seconds = f'{duration:.9f}'
    codec=['-c:v','copy'] if browser_ready else ['-c:v','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','12']
    # An unbounded apad combined with video stream copy and -shortest can hang.
    # Bound both the filter and output to the actual video duration.
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video)]
    if has_audio:command.extend(['-i',str(source)])
    command.extend(['-map','0:v:0',*codec])
    if has_audio:command.extend(['-map','1:a:0','-c:a','aac','-b:a','192k','-af',f'apad=whole_dur={seconds},atrim=duration={seconds}'])
    else:command.append('-an')
    command.extend(['-t',seconds,'-movflags','+faststart',str(destination)])
    action='Adding source audio' if has_audio else 'Saving playable video'
    update(stage='muxing_audio' if has_audio else 'saving', message=action+'…', step=0, total_steps=0)
    started = time.monotonic()
    timeout = max(120, min(600, duration * .5))
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        last_notice = 0
        while True:
            try:
                _, error = process.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise RuntimeError('Saving the audio took too long. The upscaled frames were kept for recovery.')
                if elapsed - last_notice >= 5:
                    last_notice = elapsed
                    update(message=f'{action} · {elapsed:.0f}s elapsed')
        if process.returncode:
            raise RuntimeError('Could not preserve source audio: ' + error[-900:])
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

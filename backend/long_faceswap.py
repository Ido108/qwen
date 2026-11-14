"""Face swap of any length, one part at a time.

One MiniMax H3 pass costs more per frame the longer it is: attention grows with the square of the
length, and a long pass leaves no room for the weights on the GPU. Measured on 2026-11-14 on the
RTX 5090 (Turbo INT8, 1024x576), seconds per frame and step: 124 frames 0.079, 192 frames 0.102,
260 frames 0.127, 362 frames 0.173. With the fixed cost of a part, 124 frames is the cheapest per
second of video. Each part is read from the source, swapped (by shot, as before) and written to the
MP4 at once, so the length of a selection is limited by time and disk space, not by memory.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

FPS = 24
PART_FRAMES = 124
# Each part after the first starts this many frames early; the last frames made by the previous part are
# anchored there as a guide clip (H3 anchors clips of 5, 22, 39... frames), and the repeated frames are dropped.
OVERLAP_FRAMES = 5


def part_plan(total_frames: int) -> list[tuple[int, int]]:
    """[start, end) frame ranges of the parts, in order. Every part is one 124-frame pass: the first
    part 124 frames, each later part 119 frames plus the 5 overlap frames. Only the last part is
    shorter; a last part under 17 frames joins the part before it instead of paying a whole pass."""
    if total_frames < 1:
        raise ValueError('Select a longer video segment.')
    edges = [0, min(total_frames, PART_FRAMES)]
    while edges[-1] < total_frames:
        edges.append(min(total_frames, edges[-1] + PART_FRAMES - OVERLAP_FRAMES))
    if len(edges) > 2 and edges[-1] - edges[-2] < 17:
        edges.pop(-2)
    return list(zip(edges, edges[1:]))


def is_long_faceswap(request) -> bool:
    return request.get('kind') == 'reference-video' and request.get('workflow') == 'faceswap' and int(request['frames']) > PART_FRAMES


class Mp4PartWriter:
    """An H.264 video written frame by frame; the audio is added in one pass at the end."""

    def __init__(self, path: Path, fps: int = FPS):
        import av
        self.path = Path(path)
        self.video_path = self.path.with_name(self.path.stem + '.video.mp4')
        self.fps = fps
        self.output = av.open(str(self.video_path), 'w')
        self.stream = None
        self.frames = 0
        self.size = None

    def add(self, frames):
        import av
        import numpy as np
        for pixels in frames:
            array = pixels.numpy()
            if not np.isfinite(array).all():
                raise RuntimeError('The model returned invalid video pixels. Retry the video.')
            if self.stream is None:
                height, width = array.shape[:2]
                self.size = (width, height)
                self.stream = self.output.add_stream('libx264', rate=self.fps)
                self.stream.width, self.stream.height = width, height
                self.stream.pix_fmt = 'yuv420p'
                self.stream.options = {'crf': '18', 'preset': 'medium'}
            frame = av.VideoFrame.from_ndarray((array.clip(0, 1) * 255).round().astype(np.uint8), format='rgb24')
            frame.pts, frame.time_base = self.frames, Fraction(1, self.fps)
            for packet in self.stream.encode(frame):
                self.output.mux(packet)
            self.frames += 1

    def finish(self, audio=None):
        """Close the video and write the final MP4 with the audio, if any."""
        import av
        import numpy as np
        if self.stream is None:
            self.output.close()
            raise RuntimeError('The model returned no video frames.')
        for packet in self.stream.encode():
            self.output.mux(packet)
        self.output.close()
        with av.open(str(self.video_path)) as source, av.open(str(self.path), 'w', options={'movflags': '+faststart'}) as output:
            video_in = source.streams.video[0]
            video_out = output.add_stream_from_template(video_in)
            audio_stream = None
            if audio:
                waveform = audio['waveform'][0].numpy()
                if not np.isfinite(waveform).all():
                    raise RuntimeError('The model returned invalid audio samples. Retry the video.')
                rate = int(audio['sample_rate'])
                target = round(self.frames / self.fps * rate)
                waveform = np.pad(waveform[:, :target], ((0, 0), (0, max(0, target - waveform.shape[1]))))
                layout = 'stereo' if waveform.shape[0] == 2 else 'mono'
                audio_stream = output.add_stream('aac', rate=rate)
                audio_stream.layout = layout
                audio_stream.bit_rate = 192000
            for packet in source.demux(video_in):
                if packet.dts is None:
                    continue
                packet.stream = video_out
                output.mux(packet)
            if audio_stream:
                for offset in range(0, waveform.shape[1], 1024):
                    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform[:, offset:offset + 1024]), format='fltp', layout=layout)
                    frame.sample_rate = rate
                    frame.pts, frame.time_base = offset, Fraction(1, rate)
                    for packet in audio_stream.encode(frame):
                        output.mux(packet)
                for packet in audio_stream.encode():
                    output.mux(packet)
        self.video_path.unlink(missing_ok=True)


def swap_long(request, prompt, references, source_path, update, generate, destination, preview_directory=None):
    """Swap the whole selection part by part and write it to destination.

    references: the reference images (ref_image_n). Returns (frames written, first frame, has audio).
    """
    import torch
    from .reference_media import read_frames, read_audio
    from .reference_shots import grid_length, swap_by_shot, _fit_audio

    clip_start = float(request.get('source_start_seconds') or 0)
    total = int(request['frames'])
    plan = part_plan(total)
    writer = Mp4PartWriter(destination)
    sounds, first_frame, rate, tail = [], None, None, None
    want_audio = request['source_audio'] or request['output_audio'] == 'source'
    for index, (start, end) in enumerate(plan):
        keep = end - start
        lead = OVERLAP_FRAMES if tail is not None else 0
        length = grid_length(keep + lead)
        part_start = clip_start + (start - lead) / FPS
        part_end = part_start + length / FPS
        update(part=index + 1, total_parts=len(plan), stage='decoding_source',
               message=f'Decoding source video, part {index + 1} of {len(plan)}…')
        frames = read_frames(source_path, length, start_seconds=part_start, end_seconds=part_end,
                             progress=lambda done, count: update(input_frames_done=done, input_frames_total=count))
        source_audio = read_audio(source_path, length / FPS, start_seconds=part_start, end_seconds=part_end) if want_audio else None
        images = {**references, 'source_video': frames}
        if tail is not None:
            images['continuity_guide'] = tail
        if request['source_audio'] and source_audio is not None:
            images['source_audio'] = source_audio
        part_request = {**request, 'frames': length, 'source_start_seconds': part_start, 'source_end_seconds': part_end}

        def relay(**values):
            values.setdefault('part', index + 1)
            values.setdefault('total_parts', len(plan))
            update(**values)

        pixels, audio = swap_by_shot(part_request, prompt, images, relay, generate, preview_directory=preview_directory)
        pixels = pixels[lead:lead + keep]
        if first_frame is None:
            first_frame = pixels[0].clone()
        writer.add(pixels)
        tail = pixels[-OVERLAP_FRAMES:].clone() if keep >= OVERLAP_FRAMES else None
        if request['output_audio'] == 'source':
            audio = source_audio
        elif request['output_audio'] == 'mute':
            audio = None
        if audio is not None and lead:
            # The overlap frames were written by the previous part; drop their sound too.
            skip = round(lead / FPS * int(audio['sample_rate']))
            audio = {'waveform': audio['waveform'][..., skip:], 'sample_rate': audio['sample_rate']}
        if audio is not None:
            rate = rate or int(audio['sample_rate'])
            sounds.append((keep, _fit_audio(audio, rate, round(keep / FPS * rate))))
        else:
            sounds.append((keep, None))
        del frames, pixels, images, source_audio, audio
        torch.cuda.empty_cache()
    combined = None
    if rate:
        pieces = [piece if piece is not None else torch.zeros(1, 2, round(keep / FPS * rate)) for keep, piece in sounds]
        combined = {'waveform': torch.cat(pieces, dim=-1), 'sample_rate': rate}
    update(stage='saving', message='Saving MP4…')
    writer.finish(combined)
    return writer.frames, first_frame, combined is not None, writer.size, rate

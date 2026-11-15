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
# Each part after the first starts this many frames early, so it also makes the last frames of the part before it;
# the two versions of those frames are blended into one seam. Parts are never anchored to frames an earlier part
# made: on 2026-11-15 a 15-part run anchored that way drifted part after part (brightness 64 to 70, sharpness
# 675 to 1319), and the last parts came out orange and over-sharpened.
OVERLAP_FRAMES = 5
# A shorter last part joins the part before it: one longer pass (at most 192 frames, still mostly on the GPU)
# costs less than a part of its own, which pays the whole fixed cost of a part (about 65 s on the RTX 5090).
MIN_LAST_PART = 60


def part_plan(total_frames: int) -> list[tuple[int, int]]:
    """[start, end) frame ranges of the parts, in order. Every part is one 124-frame pass: the first
    part 124 frames, each later part 119 frames plus the 5 overlap frames. Only the last part is
    shorter; a last part under MIN_LAST_PART frames joins the part before it instead of paying a whole pass."""
    if total_frames < 1:
        raise ValueError('Select a longer video segment.')
    edges = [0, min(total_frames, PART_FRAMES)]
    while edges[-1] < total_frames:
        edges.append(min(total_frames, edges[-1] + PART_FRAMES - OVERLAP_FRAMES))
    if len(edges) > 2 and edges[-1] - edges[-2] < MIN_LAST_PART:
        edges.pop(-2)
    return list(zip(edges, edges[1:]))


# Video to video workflows that run in parts. Camera view builds the camera guide of each part from that part of
# the source; with its own guide video or with Replace subjects first (two passes) it keeps one pass.
PART_WORKFLOWS = ('faceswap', 'reference', 'crossview')


def runs_in_parts(request) -> bool:
    """The workflow, without its length: can this request run in parts?"""
    if request.get('kind') != 'reference-video' or request.get('workflow') not in PART_WORKFLOWS:
        return False
    return request['workflow'] != 'crossview' or not (request.get('warp_video') or request.get('replace_subjects_first'))


def is_long_faceswap(request) -> bool:
    return runs_in_parts(request) and int(request['frames']) > PART_FRAMES


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


def swap_long(request, prompt, references, source_path, update, generate, destination, preview_directory=None, encoder=None):
    """Swap the whole selection part by part and write it to destination.

    references: the reference images (ref_image_n). encoder: a video_engine.ReferenceEncoder; with it every part is
    encoded first (one load of each encoder for the whole video) and then generated from its ready conditioning.
    Returns (frames written, first frame, has audio, size, sample rate).
    """
    import time
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from .reference_media import read_frames, read_audio
    from .reference_shots import grid_length, swap_by_shot, _fit_audio

    clip_start = float(request.get('source_start_seconds') or 0)
    total = int(request['frames'])
    plan = part_plan(total)
    writer = Mp4PartWriter(destination)
    # Every finished part is also saved as a playable MP4 with its sound and a poster, next to the whole video:
    # a stopped run keeps its work, and each part can be watched on its own.
    parts_folder = Path(destination).with_name(Path(destination).stem + ' parts')
    parts_folder.mkdir(parents=True, exist_ok=True)
    sounds, first_frame, rate, held, durations, saved_parts = [], None, None, None, [], []
    want_audio = request['source_audio'] or request['output_audio'] == 'source'

    def bounds(index):
        start, end = plan[index]
        # Every part after the first starts OVERLAP_FRAMES early, where the previous part's last frames are anchored.
        lead = OVERLAP_FRAMES if index else 0
        length = grid_length(end - start + lead)
        part_start = clip_start + (start - lead) / FPS
        return lead, length, part_start, part_start + length / FPS

    def read_part(index):
        _, length, part_start, part_end = bounds(index)
        frames = read_frames(source_path, length, start_seconds=part_start, end_seconds=part_end)
        audio = read_audio(source_path, length / FPS, start_seconds=part_start, end_seconds=part_end) if want_audio else None
        return frames, audio

    def source_parts():
        """Each part's source frames and sound, in order; the next part is read on the CPU while the GPU works."""
        reader = ThreadPoolExecutor(max_workers=1)
        pending = reader.submit(read_part, 0)
        try:
            for index in range(len(plan)):
                result = pending.result()
                if index + 1 < len(plan):
                    pending = reader.submit(read_part, index + 1)
                yield index, result
        finally:
            reader.shutdown(cancel_futures=True)

    def part_images(frames, source_audio):
        images = {**references, 'source_video': frames}
        if request['source_audio'] and source_audio is not None:
            images['source_audio'] = source_audio
        return images

    measured = {}

    def part_request(index):
        _, length, part_start, part_end = bounds(index)
        part = {**request, 'frames': length, 'source_start_seconds': part_start, 'source_end_seconds': part_end}
        if measured:
            # The working memory the part before used, scaled to this part's length, with 2 GB of margin.
            part['working_memory_gb'] = round(measured['gb'] * max(1.0, length / measured['frames']) + 2.0, 2)
        return part

    # A replacement is swapped shot by shot (reference_shots); Reference-guided follows the whole part at once.
    by_shot = swap_by_shot if request['workflow'] == 'faceswap' else (
        lambda shot_request, shot_prompt, shot_images, progress, make, preview_directory=None:
        make(shot_request, shot_prompt, shot_images, progress, preview_directory=preview_directory))
    ready = None
    if encoder is not None:
        # Passes 1 and 2: every part's references through the VAE, then every part's prompt through the text
        # encoder (video_engine.ReferenceEncoder). swap_by_shot runs as in pass 3, so each shot gets its own entry.
        units = []
        for index, (frames, source_audio) in source_parts():
            update(parts_phase='prepare', preparing_part=index + 1, total_parts=len(plan), stage='preparing_parts',
                   message=f'Preparing part {index + 1} of {len(plan)}: reading and encoding the source video…')

            def prepare(shot_request, shot_prompt, shot_images, progress, preview_directory=None):
                units.append(encoder.prepare(shot_request, shot_prompt, shot_images))
                return torch.zeros(shot_request['frames'], shot_request['height'], shot_request['width'], 3), None

            def relay_prepare(**values):
                values.update(parts_phase='prepare', preparing_part=index + 1, total_parts=len(plan))
                values.pop('step', None)
                update(**values)

            by_shot(part_request(index), prompt, part_images(frames, source_audio), relay_prepare, prepare)
            del frames, source_audio
        update(parts_phase='prepare', preparing_part=len(plan), total_parts=len(plan))
        ready = iter(zip(units, encoder.encode_all(units)))

    def run(shot_request, shot_prompt, shot_images, progress, preview_directory=None):
        if ready is not None:
            unit, positive = next(ready)
            if unit['frames'] != shot_request['frames']:
                raise RuntimeError('The prepared parts do not match the video. Retry the video.')
            shot_images = {**shot_images, 'precomputed_positive': positive}
        return generate(shot_request, shot_prompt, shot_images, progress, preview_directory=preview_directory)

    for index, (frames, source_audio) in source_parts():
        start, end = plan[index]
        began = time.monotonic()
        keep = end - start
        lead, length, part_start, part_end = bounds(index)
        update(part=index + 1, total_parts=len(plan), stage='decoding_source', parts_phase='generate',
               part_start_seconds=round(clip_start + start / FPS, 3), part_end_seconds=round(clip_start + end / FPS, 3),
               message=f'Decoding source video, part {index + 1} of {len(plan)}…')
        images = part_images(frames, source_audio)

        def relay(**values):
            values.setdefault('part', index + 1)
            values.setdefault('total_parts', len(plan))
            if 'preview_step' in values:
                values['preview_part'] = index + 1
            if values.get('sampling_working_gb'):
                measured.update(gb=max(values['sampling_working_gb'], measured.get('gb', 0)), frames=length)
            update(**values)

        made, audio = by_shot(part_request(index), prompt, images, relay, run, preview_directory=preview_directory)
        pixels = made[lead:lead + keep]
        if first_frame is None:
            first_frame = pixels[0].clone()
        if held is not None and lead:
            # The seam: the previous part's last frames fade into this part's version of the same frames.
            weight = (torch.arange(1, lead + 1, dtype=held.dtype) / (lead + 1)).view(-1, 1, 1, 1)
            writer.add(held * (1 - weight) + made[:lead].to(held.dtype) * weight)
        last = index == len(plan) - 1
        hold = 0 if last or keep <= OVERLAP_FRAMES else OVERLAP_FRAMES
        writer.add(pixels[:keep - hold])
        held = pixels[keep - hold:].clone() if hold else None
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
        from .video_worker import save_mp4
        part_sound = {'waveform': sounds[-1][1], 'sample_rate': rate} if sounds[-1][1] is not None else None
        save_mp4(parts_folder / f'part-{index + 1:03d}.mp4', pixels, part_sound)
        from PIL import Image
        Image.fromarray((pixels[0].numpy().clip(0, 1) * 255).round().astype('uint8')).save(
            parts_folder / f'part-{index + 1:03d}.jpg', quality=88)
        durations.append(round(time.monotonic() - began, 1))
        saved_parts.append({'index': index + 1, 'start_seconds': round(start / FPS, 3), 'end_seconds': round(end / FPS, 3)})
        update(parts_saved=index + 1, parts_folder=str(parts_folder), part_seconds=durations, parts=saved_parts)
        del frames, made, pixels, images, source_audio, audio
        torch.cuda.empty_cache()
    combined = None
    if rate:
        pieces = [piece if piece is not None else torch.zeros(1, 2, round(keep / FPS * rate)) for keep, piece in sounds]
        combined = {'waveform': torch.cat(pieces, dim=-1), 'sample_rate': rate}
    update(stage='saving', message='Saving MP4…')
    writer.finish(combined)
    return writer.frames, first_frame, combined is not None, writer.size, rate

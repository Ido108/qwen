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
# Each part after the first starts this many frames early, anchored there to the last frames of the part before it
# (H3 anchors clips of 5, 22, 39... frames), so the background, clothing and light carry over; the two versions of
# those frames are blended into one seam. Parts made without that anchor did not match: with a prompt that asks for
# a new background ("make the background nvidia office"), part 1 came out in an open office, part 2 in the source
# room and part 14 with a large logo (2026-11-15). The anchor frames are first matched to the look of part 1 and
# softened (LookMatcher.guide), because the model adds fine detail to the frames it starts from: anchored on the
# plain frames, a chain drifted part after part (brightness 64 to 70 over 15 parts; with the color matched, the
# fine texture on the face still rose 4.20, 4.99, 5.49 in parts 1 to 3).
OVERLAP_FRAMES = 5


class LookMatcher:
    """Keeps every part of a long video in the look of its first part.

    Part 1 sets the reference: its color statistics (Lab mean and spread) and sharpness, and those of its source.
    Each later part is mapped to part 1's statistics shifted by the change in its own source, so a real change of
    light in the source carries through while the drift a chain of anchored parts adds does not. A part sharper than
    the reference (relative to its source) is softened to it; detail is never added.
    """
    SIGMAS = (0.4, 0.6, 0.8, 1.0, 1.3, 1.6)
    # The model added about 10 to 19% fine detail to the frames a part starts from (face texture 4.20, 4.99, 5.49
    # in an anchored chain); the anchor frames are softened by that much, so each part makes its own detail.
    GUIDE_SHARPNESS = 0.85

    def __init__(self):
        self.reference = None

    @staticmethod
    def _uint8(frames):
        return (frames.clamp(0, 1).numpy() * 255).round().astype('uint8')

    @classmethod
    def _sharpness(cls, frames, step=6):
        import cv2
        import numpy as np
        return float(np.mean([cv2.Laplacian(cv2.cvtColor(cv2.resize(f, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY),
                                            cv2.CV_32F).var() for f in frames[::step]]))

    @classmethod
    def look(cls, frames):
        """(Lab mean, Lab spread, sharpness) of [N, H, W, 3] frames in 0..1."""
        import cv2
        import numpy as np
        pixels = cls._uint8(frames)
        lab = np.stack([cv2.cvtColor(cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
                        for f in pixels[::4]]).astype(np.float32).reshape(-1, 3)
        return lab.mean(0), lab.std(0), cls._sharpness(pixels)

    def set_reference(self, frames, source):
        self.reference = (self.look(frames), self.look(source))

    def guide(self, frames, source=None):
        """The last frames of a part (already matched) softened for the next part to start from."""
        import cv2
        import numpy as np
        import torch
        if self.reference is None:
            return frames
        (_, _, ref_sharp), (_, _, ref_src_sharp) = self.reference
        pixels = self._uint8(frames)
        src_sharp = self._sharpness(self._uint8(source), step=1) if source is not None else ref_src_sharp
        ratio = src_sharp / ref_src_sharp if ref_src_sharp > 1.0 and src_sharp > 1.0 else 1.0
        target = self.GUIDE_SHARPNESS * ref_sharp * ratio
        if target <= 0 or self._sharpness(pixels, step=1) <= target:
            return frames
        sigma = min(self.SIGMAS, key=lambda s: abs(self._sharpness(np.stack([cv2.GaussianBlur(f, (0, 0), s) for f in pixels]), step=1) - target))
        return torch.from_numpy(np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in pixels]).astype(np.float32) / 255)

    def apply(self, frames, source):
        """frames matched to the reference look; the first call sets the reference and returns frames unchanged."""
        import cv2
        import numpy as np
        import torch
        if self.reference is None:
            self.set_reference(frames, source)
            return frames
        (ref_mean, ref_spread, ref_sharp), (ref_src_mean, ref_src_spread, ref_src_sharp) = self.reference
        mean, spread, _ = self.look(frames)
        src_mean, src_spread, src_sharp = self.look(source)
        ratio = np.where(ref_src_spread > 1.0, src_spread / np.maximum(ref_src_spread, 1e-3), 1.0)
        target_mean, target_spread = ref_mean + (src_mean - ref_src_mean), ref_spread * ratio
        scale = target_spread / np.maximum(spread, 1e-3)
        pixels = self._uint8(frames)
        matched = np.empty_like(pixels)
        for i, frame in enumerate(pixels):
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
            matched[i] = cv2.cvtColor(np.clip((lab - mean) * scale + target_mean, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        target_sharp = ref_sharp * (src_sharp / ref_src_sharp if ref_src_sharp > 1.0 and src_sharp > 1.0 else 1.0)
        sharp = self._sharpness(matched)
        if target_sharp > 0 and sharp > 1.08 * target_sharp:
            sample = matched[::6]
            sigma = min(self.SIGMAS, key=lambda s: abs(self._sharpness(np.stack([cv2.GaussianBlur(f, (0, 0), s) for f in sample]), step=1) - target_sharp))
            matched = np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in matched])
        return torch.from_numpy(matched.astype(np.float32) / 255)
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


def plan_with_parts(total_frames: int, parts: int) -> list[tuple[int, int]]:
    """The plan a finished job used: the current rule, or the earlier one (a last part under 17 frames joined)."""
    plan = part_plan(total_frames)
    if len(plan) == parts:
        return plan
    edges = [0, min(total_frames, PART_FRAMES)]
    while edges[-1] < total_frames:
        edges.append(min(total_frames, edges[-1] + PART_FRAMES - OVERLAP_FRAMES))
    if len(edges) > 2 and edges[-1] - edges[-2] < 17:
        edges.pop(-2)
    plan = list(zip(edges, edges[1:]))
    if len(plan) != parts:
        raise ValueError('The parts of the original video could not be matched. Run the whole video again.')
    return plan


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
    first_new, kept_frames, kept_audio = 0, None, None
    if request.get('repair_from_part'):
        # A repair: the parts before repair_from_part are taken from the original video as they are, the others are
        # made again with the original settings, on the original part boundaries.
        plan, first_new, kept_frames, kept_audio = _original_parts(request, total)
    writer = Mp4PartWriter(destination)
    # Every finished part is also saved as a playable MP4 with its sound and a poster, next to the whole video:
    # a stopped run keeps its work, and each part can be watched on its own.
    parts_folder = Path(destination).with_name(Path(destination).stem + ' parts')
    parts_folder.mkdir(parents=True, exist_ok=True)
    sounds, first_frame, rate, held, tail, durations, saved_parts = [], None, None, None, None, [], []
    looks = LookMatcher()
    want_audio = request['source_audio'] or request['output_audio'] == 'source'

    def bounds(index):
        start, end = plan[index]
        # Every part after the first starts OVERLAP_FRAMES early: those frames become the seam with the previous part.
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
        """Each part's source frames and sound, in order; the next part is read on the CPU while the GPU works.
        A part kept from the original video (a repair) needs no source and gives (None, None)."""
        reader = ThreadPoolExecutor(max_workers=1)
        indices = list(range(first_new, len(plan)))
        pending = reader.submit(read_part, indices[0]) if indices else None
        try:
            for index in range(len(plan)):
                if index < first_new:
                    yield index, (None, None)
                    continue
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
            if index < first_new:
                continue
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
        if index < first_new:
            # Kept from the original video, frames and sound as they are.
            pixels = kept_frames[start:end].float() / 255
            if index == 0:
                # The kept part 1 sets the look the new parts are matched to.
                looks.set_reference(pixels, read_part(0)[0])
            if first_frame is None:
                first_frame = pixels[0].clone()
            if held is not None:
                writer.add(held)
            hold = 0 if keep <= OVERLAP_FRAMES else OVERLAP_FRAMES
            writer.add(pixels[:keep - hold])
            held = pixels[keep - hold:].clone() if hold else None
            tail = looks.guide(pixels[-OVERLAP_FRAMES:]) if keep >= OVERLAP_FRAMES else None
            audio = None
            if request['output_audio'] == 'source':
                audio = read_audio(source_path, keep / FPS, start_seconds=clip_start + start / FPS, end_seconds=clip_start + end / FPS)
            elif request['output_audio'] != 'mute' and kept_audio is not None:
                kept_rate = int(kept_audio['sample_rate'])
                audio = {'waveform': kept_audio['waveform'][..., round(start / FPS * kept_rate):round(end / FPS * kept_rate)],
                         'sample_rate': kept_rate}
            if audio is not None:
                rate = rate or int(audio['sample_rate'])
                sounds.append((keep, _fit_audio(audio, rate, round(keep / FPS * rate))))
            else:
                sounds.append((keep, None))
            _save_part(parts_folder, index, pixels, sounds[-1][1], rate)
            durations.append(round(time.monotonic() - began, 1))
            saved_parts.append({'index': index + 1, 'start_seconds': round(start / FPS, 3), 'end_seconds': round(end / FPS, 3)})
            update(parts_saved=index + 1, parts_folder=str(parts_folder), part_seconds=durations, parts=saved_parts)
            continue
        images = part_images(frames, source_audio)
        # Camera view anchors its camera guide on every frame; the other workflows start from the previous part's end.
        if tail is not None and request['workflow'] != 'crossview':
            images['continuity_guide'] = tail

        def relay(**values):
            values.setdefault('part', index + 1)
            values.setdefault('total_parts', len(plan))
            if 'preview_step' in values:
                values['preview_part'] = index + 1
            if values.get('sampling_working_gb'):
                measured.update(gb=max(values['sampling_working_gb'], measured.get('gb', 0)), frames=length)
            update(**values)

        made, audio = by_shot(part_request(index), prompt, images, relay, run, preview_directory=preview_directory)
        made = looks.apply(made, frames[:made.shape[0]])
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
        tail = (looks.guide(pixels[-OVERLAP_FRAMES:], frames[lead + keep - OVERLAP_FRAMES:lead + keep])
                if keep >= OVERLAP_FRAMES else None)
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
        _save_part(parts_folder, index, pixels, sounds[-1][1], rate)
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


def _save_part(folder, index, pixels, sound, rate):
    """A finished part as its own MP4 with its sound, and a poster."""
    from PIL import Image
    from .video_worker import save_mp4
    save_mp4(folder / f'part-{index + 1:03d}.mp4', pixels, {'waveform': sound, 'sample_rate': rate} if sound is not None else None)
    Image.fromarray((pixels[0].numpy().clip(0, 1) * 255).round().astype('uint8')).save(folder / f'part-{index + 1:03d}.jpg', quality=88)


def _original_parts(request, total):
    """For a repair: the original job's plan, the index of the first part made again, and the kept frames and sound
    (frames 0 up to that part, read from the original video)."""
    import av
    import numpy as np
    import torch
    from .common import JOBS_DIR, read_json
    from .reference_media import read_audio
    original = read_json(JOBS_DIR / request['repair_of'] / 'status.json')
    if int(original['request']['frames']) != total or original.get('status') != 'succeeded':
        raise ValueError('The original video does not match this repair. Run the whole video again.')
    parts = original.get('parts') or []
    if len(parts) == original.get('total_parts') and parts:
        plan = [(round(part['start_seconds'] * FPS), round(part['end_seconds'] * FPS)) for part in parts]
    else:
        plan = plan_with_parts(total, int(original.get('total_parts') or len(part_plan(total))))
    first_new = int(request['repair_from_part']) - 1
    if not 1 <= first_new < len(plan):
        raise ValueError(f'Choose a part from 2 to {len(plan)} to make again.')
    count = plan[first_new][0]
    video = Path(original['output_path'])
    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError('The original video is shorter than its parts. Run the whole video again.')
    kept = torch.from_numpy(np.stack(frames))  # 8-bit; each kept part is converted when it is written
    audio = read_audio(video, count / FPS, start_seconds=0, end_seconds=count / FPS) if original.get('has_audio') else None
    return plan, first_new, kept, audio

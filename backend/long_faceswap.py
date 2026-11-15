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
# (H3 anchors clips of 5, 22, 39... frames), so the background, clothing and light carry over. The two versions
# of those frames become the seam (seam_morph): each is moved toward the other along the motion between them and
# then mixed. A plain mix of two versions that differ showed two faces at once, and a cut showed a jump
# (2026-11-15, parts made without the anchor).
# Parts made without that anchor did not match: with a prompt that asks for
# a new background ("make the background nvidia office"), part 1 came out in an open office, part 2 in the source
# room and part 14 with a large logo (2026-11-15). The model makes its output sharper than its input (the face
# about 1.5 times its source), so a part that starts from the frames of the part before sharpens them again: a plain
# chain drifted part after part (brightness 64 to 70 over 15 parts; face sharpness against the source 1.47, 1.56,
# 2.00, 2.36 in parts 1 to 4). Each part is therefore matched to part 1 before the next part starts from it, the
# whole frame and the face area on its own (LookMatcher): every part then starts from frames at part 1's level.
OVERLAP_FRAMES = 5


# Parts 2 and later of a video with one input image get, next to that image, the frame of part 1 picked by
# best_reference_frame as a second picture, with this added to the prompt: the clothing and background part 1 made
# stay the same in every part, also when the prompt asks for a new background (2026-11-15: with "make the background
# nvidia office" the parts, made on their own, showed an open office, the source room and a large logo). The input
# image stays the first picture: with the frame of part 1 as the only picture and a sentence about the person and
# face, the swap went back to the source face in the middle of a part (the FaceSwap LoRA needs its face picture, and
# longer identity text in the prompt makes the swap fade, see reference_prompt). Those parts do not also start from
# the last frames of the part before.
FOLLOW_PROMPT = '\n\nThe clothing, background and lighting are exactly as in <Picture 2>.'
# A part is made again (another seed; the last of PART_RETRIES tries starts from the last frames of the part before)
# when it does not hold what part 1 set:
# - the face: in any of its checked frames less like the input image than FACE_KEPT of part 1's likeness (the swap
#   can fade back to the source face in the middle of a part);
# - the scene: its still picture (scene_plate) differs from part 1's by more than SCENE_LIMIT beyond what the source
#   itself changed between the two parts (2026-11-15: the short last part of a video came out in a dark jacket where
#   parts 1 to 15 had a white T-shirt; made on their own, parts came out with other backgrounds).
FACE_KEPT = 0.6
SCENE_LIMIT = 3.0
PART_RETRIES = 3


def seam_morph(held, ahead):
    """The seam frames between two parts: held is the previous part's version of the overlap frames, ahead the new
    part's version of the same frames ([N, H, W, 3] in 0..1). Each output frame moves both versions toward each other
    along the motion measured between them (dense optical flow) and then mixes them, from held to ahead. A plain mix
    of two versions that differ showed two faces at once, and a cut showed a jump (2026-11-15)."""
    import cv2
    import numpy as np
    import torch
    count, height, width = held.shape[0], held.shape[1], held.shape[2]
    first = (held.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    second = (ahead.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    flow_finder = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    out = np.empty_like(first)
    for index in range(count):
        weight = (index + 1) / (count + 1)
        gray_first, gray_second = cv2.cvtColor(first[index], cv2.COLOR_RGB2GRAY), cv2.cvtColor(second[index], cv2.COLOR_RGB2GRAY)
        # Where each point of the first version is in the second one: the mean of both directions, smoothed.
        flow = 0.5 * (flow_finder.calc(gray_first, gray_second, None) - flow_finder.calc(gray_second, gray_first, None))
        flow = cv2.GaussianBlur(flow, (0, 0), 3)
        moved_first = cv2.remap(first[index], xs - weight * flow[..., 0], ys - weight * flow[..., 1], cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        moved_second = cv2.remap(second[index], xs + (1 - weight) * flow[..., 0], ys + (1 - weight) * flow[..., 1],
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out[index] = cv2.addWeighted(moved_first, 1 - weight, moved_second, weight, 0)
    return torch.from_numpy(out.astype(np.float32) / 255).to(held.dtype)


def _face_embedder():
    """embed(bgr uint8 image) -> unit face embedding (SFace) of the largest face, or None without a face."""
    import cv2
    import numpy as np
    from .reference_people import ensure_recognizer, face_detector
    recognizer = cv2.FaceRecognizerSF.create(str(ensure_recognizer()), '')
    detector = face_detector()

    def embed(bgr):
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = detector.detect(bgr)
        if faces is None:
            return None
        face = max(faces, key=lambda item: item[2] * item[3])
        embedding = recognizer.feature(recognizer.alignCrop(bgr, face)).flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm else None

    return embed


def scene_plate(frames, count=12):
    """The still picture of a part: the median of count evenly spaced frames ([N, H, W, 3] in 0..1), small, in Lab.
    The median keeps what stays in place (background, clothing) and drops what moves."""
    import cv2
    import numpy as np
    picks = sorted({round(i * (frames.shape[0] - 1) / max(1, count - 1)) for i in range(count)})
    small = [cv2.cvtColor(cv2.resize((frames[i].clamp(0, 1).numpy() * 255).round().astype(np.uint8), (64, 36),
                                     interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32) for i in picks]
    return np.median(np.stack(small), axis=0)


def face_likeness(frames, reference, count=12):
    """Likeness of the face in count evenly spaced frames ([N, H, W, 3] in 0..1) to the face in reference (a PIL
    image), one value per frame with a face. None when reference has no face."""
    import numpy as np
    embed = _face_embedder()
    target = embed(np.ascontiguousarray(np.asarray(reference.convert('RGB'))[..., ::-1]))
    if target is None:
        return None
    values = []
    for index in sorted({round(i * (frames.shape[0] - 1) / max(1, count - 1)) for i in range(count)}):
        pixels = (frames[index].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        embedding = embed(np.ascontiguousarray(pixels[..., ::-1]))
        if embedding is not None:
            values.append(float(np.dot(embedding, target)))
    return values


def best_reference_frame(frames, reference):
    """The frame of frames ([N, H, W, 3] in 0..1) that moves least and whose face is most like the face in reference
    (a PIL image): among the stillest third of the frames, the one with the highest face likeness (SFace).
    Returns (PIL image, frame index, likeness), or None when reference or no frame has a face."""
    import cv2
    import numpy as np
    from PIL import Image
    embed = _face_embedder()
    target = embed(np.ascontiguousarray(np.asarray(reference.convert('RGB'))[..., ::-1]))
    if target is None:
        return None
    pixels = (frames.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    gray = np.stack([cv2.cvtColor(cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
                     for f in pixels]).astype(np.float32)
    motion = np.zeros(len(pixels), np.float32)
    if len(pixels) > 1:
        steps = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
        motion[:-1] += steps
        motion[1:] += steps
        motion[1:-1] /= 2
    best = None
    for index in sorted(np.argsort(motion, kind='stable')[:max(1, len(pixels) // 3)]):
        embedding = embed(np.ascontiguousarray(pixels[index][..., ::-1]))
        if embedding is not None:
            likeness = float(np.dot(embedding, target))
            if best is None or likeness > best[1]:
                best = (int(index), likeness)
    if best is None:
        return None
    return Image.fromarray(pixels[best[0]]), best[0], best[1]


class LookMatcher:
    """Keeps every part of a long video in the look of its first part.

    Part 1 sets the reference: its color statistics (Lab mean and spread) and the strength of its fine, medium and
    coarse detail (the spread of each band of the brightness), and those of its source. Each later part is mapped to
    part 1's statistics shifted by the change in its own source, so a real change of light in the source carries
    through while the drift a chain of parts adds does not. A detail band stronger than part 1's is turned down to
    it; detail is never added. The face area gets the same match on its own, frame by frame, under a soft oval mask.

    The model makes its output sharper and more contrasty than its input, so in a chain each part did that again to
    the part before (2026-11-15: the wall broke into blocks, the face turned orange and crunchy by part 15). Turning
    down only the finest detail (a blur) brought the measured sharpness back but left the stronger wrinkles and
    local contrast, which sit in the medium band; each band is therefore matched.
    """
    BAND_SIGMAS = (1.0, 2.5, 6.0)
    # A detail band is never turned down below half, and only when stronger than part 1's by more than 3%.
    MIN_GAIN = 0.5
    # Limits of the change a source can carry through to the targets, and of one face color correction.
    SOURCE_RATIO = (0.8, 1.25)
    FACE_SCALE = (0.7, 1.3)
    FACE_SHIFT = 15.0

    def __init__(self):
        self.reference = None
        self.face = None

    @staticmethod
    def _uint8(frames):
        return (frames.clamp(0, 1).numpy() * 255).round().astype('uint8')

    @classmethod
    def _split(cls, lum):
        """Detail bands (fine to coarse) and the rest of a float brightness image."""
        import cv2
        blurred = [lum] + [cv2.GaussianBlur(lum, (0, 0), s) for s in cls.BAND_SIGMAS]
        return [blurred[i] - blurred[i + 1] for i in range(len(cls.BAND_SIGMAS))], blurred[-1]

    @classmethod
    def _bands(cls, images):
        """Mean spread of each detail band over uint8 RGB images (frames or face crops)."""
        import cv2
        import numpy as np
        values = []
        for image in images:
            bands, _ = cls._split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32))
            values.append([float(b.std()) for b in bands])
        return np.mean(values, 0) if values else None

    @classmethod
    def _gains(cls, current, reference, source, reference_source):
        """Gain of each detail band (MIN_GAIN to 1) that brings current to reference, shifted by the source change."""
        import numpy as np
        low, high = cls.SOURCE_RATIO
        change = np.clip(np.where(reference_source > 1e-3, source / np.maximum(reference_source, 1e-3), 1.0), low, high)
        gains = np.clip(reference * change / np.maximum(current, 1e-3), cls.MIN_GAIN, 1.0)
        return np.where(gains > 0.97, 1.0, gains)

    @classmethod
    def _apply_gains(cls, lab, gains):
        """lab (float Lab image) with its detail bands scaled by gains."""
        import numpy as np
        bands, rest = cls._split(lab[..., 0])
        lab = lab.copy()
        lab[..., 0] = np.clip(rest + sum(g * b for g, b in zip(gains, bands)), 0, 255)
        return lab

    @classmethod
    def look(cls, frames):
        """(Lab mean, Lab spread, detail bands) of [N, H, W, 3] frames in 0..1."""
        import cv2
        import numpy as np
        pixels = cls._uint8(frames)
        lab = np.stack([cv2.cvtColor(cv2.resize(f, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
                        for f in pixels[::4]]).astype(np.float32).reshape(-1, 3)
        return lab.mean(0), lab.std(0), cls._bands(pixels[::6])

    @staticmethod
    def _face_boxes(pixels, step=2):
        """(x, y, w, h) of the largest face in each frame of uint8 RGB pixels, or None; found every step frames."""
        import numpy as np
        from .reference_people import face_detector
        detector = face_detector()
        boxes, box = [], None
        for index, frame in enumerate(pixels):
            if index % step == 0:
                bgr = np.ascontiguousarray(frame[..., ::-1])
                detector.setInputSize((bgr.shape[1], bgr.shape[0]))
                _, faces = detector.detect(bgr)
                box = None
                if faces is not None:
                    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])[:4]
                    if w >= 24 and h >= 24:
                        box = (int(x), int(y), int(w), int(h))
            boxes.append(box)
        return boxes

    @staticmethod
    def _crop(frame, box):
        x, y, w, h = box
        return frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]

    @classmethod
    def face_look(cls, pixels, boxes):
        """(Lab mean, Lab spread, detail bands) of the face area over the frames, or None without a face."""
        import cv2
        import numpy as np
        crops = [cls._crop(f, b) for f, b in zip(pixels, boxes) if b is not None]
        crops = [c for c in crops if c.size][::3]
        if len(crops) < 2:
            return None
        lab = np.concatenate([cv2.cvtColor(cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).reshape(-1, 3)
                              for c in crops]).astype(np.float32)
        # Detail measured on crops of one size, so a face nearer the camera does not count as more detailed.
        return lab.mean(0), lab.std(0), cls._bands([cv2.resize(c, (160, 160), interpolation=cv2.INTER_AREA) for c in crops])

    @staticmethod
    def _mask(shape, box):
        """Soft oval over the face, hair and jaw: 1 inside, fading to 0."""
        import cv2
        import numpy as np
        x, y, w, h = box
        mask = np.zeros(shape[:2], np.float32)
        cv2.ellipse(mask, (int(x + w / 2), int(y + h * 0.42)), (int(w * 0.72), int(h * 0.88)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (0, 0), max(2.0, w * 0.12))[..., None]

    def set_reference(self, frames, source):
        self.reference = (self.look(frames), self.look(source))
        pixels, source_pixels = self._uint8(frames), self._uint8(source)
        face, source_face = self.face_look(pixels, self._face_boxes(pixels)), self.face_look(source_pixels, self._face_boxes(source_pixels))
        self.face = (face, source_face) if face is not None and source_face is not None else None

    def _match_face(self, pixels, source):
        """pixels (uint8, already matched as a whole) with the face area matched to part 1's face, frame by frame."""
        import cv2
        import numpy as np
        (ref_mean, ref_spread, ref_bands), (src0_mean, src0_spread, src0_bands) = self.face
        boxes = self._face_boxes(pixels)
        look, source_look = self.face_look(pixels, boxes), self.face_look(source, self._face_boxes(source))
        if look is None or source_look is None:
            return pixels
        mean, spread, bands = look
        src_mean, src_spread, src_bands = source_look
        low, high = self.SOURCE_RATIO
        target_mean = ref_mean + np.clip(src_mean - src0_mean, -self.FACE_SHIFT, self.FACE_SHIFT)
        target_spread = ref_spread * np.clip(np.where(src0_spread > 1.0, src_spread / np.maximum(src0_spread, 1e-3), 1.0), low, high)
        scale = np.clip(target_spread / np.maximum(spread, 1e-3), *self.FACE_SCALE)
        shift = np.clip(target_mean - mean, -self.FACE_SHIFT, self.FACE_SHIFT)
        gains = self._gains(bands, ref_bands, src_bands, src0_bands)
        out = pixels.copy()
        for index, (frame, box) in enumerate(zip(pixels, boxes)):
            if box is None:
                continue
            lab = (cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32) - mean) * scale + mean + shift
            if (gains < 1.0).any():
                lab = self._apply_gains(lab, gains)
            fixed = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            mask = self._mask(frame.shape, box)
            out[index] = np.clip(frame * (1 - mask) + fixed * mask, 0, 255).round().astype(np.uint8)
        return out

    def apply(self, frames, source):
        """frames matched to the reference look; the first call sets the reference and returns frames unchanged."""
        import cv2
        import numpy as np
        import torch
        if self.reference is None:
            self.set_reference(frames, source)
            return frames
        (ref_mean, ref_spread, ref_bands), (ref_src_mean, ref_src_spread, ref_src_bands) = self.reference
        mean, spread, _ = self.look(frames)
        src_mean, src_spread, src_bands = self.look(source)
        ratio = np.where(ref_src_spread > 1.0, src_spread / np.maximum(ref_src_spread, 1e-3), 1.0)
        target_mean, target_spread = ref_mean + (src_mean - ref_src_mean), ref_spread * ratio
        scale = target_spread / np.maximum(spread, 1e-3)
        pixels = self._uint8(frames)
        labs = [(cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32) - mean) * scale + target_mean for frame in pixels]
        # Detail bands measured after the color match, on the frames as they will be.
        colored = np.stack([cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB) for lab in labs])
        gains = self._gains(self._bands(colored[::6]), ref_bands, src_bands, ref_src_bands)
        if (gains < 1.0).any():
            matched = np.stack([cv2.cvtColor(np.clip(self._apply_gains(lab, gains), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                                for lab in labs])
        else:
            matched = colored
        if self.face is not None:
            matched = self._match_face(matched, self._uint8(source))
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
    # long_parts. By default, with one input image, parts 2 and later take their input image from part 1
    # (best_reference_frame, 'follow'): the background and clothing part 1 made stay in every part. Parts made only
    # from the input images and the source ('independent', also the default with more input images) kept the quality
    # but not always the background: one video came out the same in all 16 parts, another with a different
    # background in each of its first 3 parts, from the same prompt and settings. 'chain': each part starts from the
    # last frames of the part before; every chain lost quality (a copy of a copy).
    mode = request.get('long_parts')
    follow = (mode in (None, 'follow') and request['workflow'] != 'crossview'
              and len(request.get('reference_images') or []) == 1 and len(plan) > 1)
    independent = mode != 'chain'
    current = {'references': references, 'prompt': prompt}
    # The face of every part is checked against the input image when the video has one input image with a face.
    guard = (request['workflow'] != 'crossview' and len(request.get('reference_images') or []) == 1
             and references.get('ref_image_1') is not None)
    face_level = {}
    # The still pictures of part 1 and of its source (scene_plate).
    scene = {}

    def scene_change(made, source_frames):
        """How far this part's still picture is from part 1's, beyond the change in the source itself."""
        import numpy as np
        return float(np.abs(scene_plate(made) - scene['made']).mean() - np.abs(scene_plate(source_frames) - scene['source']).mean())

    def follow_part_1(frames):
        """Parts 2 and later: part 1's stillest frame most like the input image becomes their second picture."""
        if not follow or references.get('ref_image_1') is None:
            return
        chosen = best_reference_frame(frames, references['ref_image_1'])
        if chosen is None:
            return
        image, frame, likeness = chosen
        if preview_directory is not None:
            image.save(Path(preview_directory) / 'reference-from-part-1.png')
        current.update(references={**references, 'ref_image_2': image}, prompt=prompt + FOLLOW_PROMPT, frame=frame)
        update(reference_frame=frame, reference_likeness=round(likeness, 3))

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

    def source_parts(wanted=None):
        """Each part's source frames and sound, in order (all parts, or the wanted ones); the next part is read on
        the CPU while the GPU works. A part kept from the original video (a repair) needs no source and gives
        (None, None)."""
        reader = ThreadPoolExecutor(max_workers=1)
        wanted = list(range(len(plan))) if wanted is None else list(wanted)
        indices = [index for index in wanted if index >= first_new]
        pending = reader.submit(read_part, indices[0]) if indices else None
        try:
            for index in wanted:
                if index < first_new:
                    yield index, (None, None)
                    continue
                result = pending.result()
                position = indices.index(index) + 1
                if position < len(indices):
                    pending = reader.submit(read_part, indices[position])
                yield index, result
        finally:
            reader.shutdown(cancel_futures=True)

    def part_images(frames, source_audio):
        images = {**current['references'], 'source_video': frames}
        if request['source_audio'] and source_audio is not None:
            images['source_audio'] = source_audio
        return images

    measured = {}

    def part_request(index):
        _, length, part_start, part_end = bounds(index)
        part = {**request, 'frames': length, 'source_start_seconds': part_start, 'source_end_seconds': part_end}
        if index and 'frame' in current:
            # The frame of part 1 is the second picture of the later parts.
            part['reference_images'] = [*request['reference_images'], 'frame-of-part-1']
        if measured:
            # The working memory the part before used, scaled to this part's length, with 2 GB of margin.
            part['working_memory_gb'] = round(measured['gb'] * max(1.0, length / measured['frames']) + 2.0, 2)
        return part

    # A replacement is swapped shot by shot (reference_shots); Reference-guided follows the whole part at once.
    by_shot = swap_by_shot if request['workflow'] == 'faceswap' else (
        lambda shot_request, shot_prompt, shot_images, progress, make, preview_directory=None:
        make(shot_request, shot_prompt, shot_images, progress, preview_directory=preview_directory))
    def prepare_parts(wanted):
        """Passes 1 and 2 for the wanted parts: their references through the VAE, then their prompts through the
        text encoder (video_engine.ReferenceEncoder). swap_by_shot runs as in pass 3, so each shot gets its own entry."""
        units = []
        for index, (frames, source_audio) in source_parts(wanted):
            if index < first_new:
                continue
            update(parts_phase='prepare', preparing_part=index + 1, total_parts=len(plan), stage='preparing_parts',
                   message=f'Preparing part {index + 1} of {len(plan)}: reading and encoding the source video…')

            def prepare(shot_request, shot_prompt, shot_images, progress, preview_directory=None):
                units.append({**encoder.prepare(shot_request, shot_prompt, shot_images), 'part': index + 1, 'parts': len(plan)})
                return torch.zeros(shot_request['frames'], shot_request['height'], shot_request['width'], 3), None

            def relay_prepare(**values):
                values.update(parts_phase='prepare', preparing_part=index + 1, total_parts=len(plan))
                values.pop('step', None)
                update(**values)

            by_shot(part_request(index), current['prompt'] if index else prompt, part_images(frames, source_audio),
                    relay_prepare, prepare)
            del frames, source_audio
        update(parts_phase='prepare', preparing_part=wanted[-1] + 1, total_parts=len(plan))
        return iter(zip(units, encoder.encode_all(units)))

    new_parts = list(range(first_new, len(plan)))
    if first_new:
        # A repair keeps part 1, so the new parts follow the kept part 1.
        follow_part_1(kept_frames[plan[0][0]:plan[0][1]].float() / 255)
    # Without a kept part 1, the later parts are prepared again once part 1 is made (their input image comes from
    # it). Part 1 is still prepared together with all the parts: prepared alone, the same seed gave a different
    # part 1 (one video: a plant in the background that the run with all parts prepared did not have).
    split = follow and not first_new
    ready = None
    if encoder is not None and new_parts:
        ready = prepare_parts(new_parts)

    # The prepared conditioning a part used, kept so the part can be made again from it with another seed.
    used, replay = [], []

    def run(shot_request, shot_prompt, shot_images, progress, preview_directory=None):
        if ready is not None:
            unit, positive = replay.pop(0) if replay else next(ready)
            used.append((unit, positive))
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
                # The kept part 1 sets the look the new parts are matched to, and what the new parts must hold.
                first_source = read_part(0)[0]
                looks.set_reference(pixels, first_source)
                if guard:
                    scene.update(made=scene_plate(pixels), source=scene_plate(first_source[:pixels.shape[0]]))
                    values = face_likeness(pixels, references['ref_image_1'])
                    if values:
                        face_level['value'] = sorted(values)[len(values) // 2]
                del first_source
            if first_frame is None:
                first_frame = pixels[0].clone()
            if held is not None:
                writer.add(held)
            hold = 0 if keep <= OVERLAP_FRAMES else OVERLAP_FRAMES
            writer.add(pixels[:keep - hold])
            held = pixels[keep - hold:].clone() if hold else None
            tail = pixels[-OVERLAP_FRAMES:].clone() if keep >= OVERLAP_FRAMES else None
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
        # Camera view anchors its camera guide on every frame; the other workflows start from the previous part's end,
        # unless they follow a frame of part 1.
        if tail is not None and request['workflow'] != 'crossview' and 'frame' not in current and not independent:
            images['continuity_guide'] = tail

        def relay(**values):
            values.setdefault('part', index + 1)
            values.setdefault('total_parts', len(plan))
            if 'preview_step' in values:
                values['preview_part'] = index + 1
            if values.get('sampling_working_gb'):
                measured.update(gb=max(values['sampling_working_gb'], measured.get('gb', 0)), frames=length)
            update(**values)

        attempt, tries, best = part_request(index), 0, None
        while True:
            used.clear()
            made, audio = by_shot(attempt, current['prompt'] if index else prompt, images, relay, run,
                                  preview_directory=preview_directory)
            values = face_likeness(made, references['ref_image_1']) if guard else None
            if index == 0 and values and 'value' not in face_level:
                # Part 1 sets how alike the swapped face is to the input image (its middle value).
                face_level['value'] = sorted(values)[len(values) // 2]
            lowest = min(values) if values else None
            face_kept = not face_level.get('value') or lowest is None or lowest >= FACE_KEPT * face_level['value']
            change = scene_change(made, frames) if guard and scene else 0.0
            scene_kept = change <= SCENE_LIMIT
            rank = (face_kept and scene_kept, face_kept, -change)
            if best is None or rank > best[0]:
                best = (rank, made, audio)
            if (face_kept and scene_kept) or tries == PART_RETRIES:
                made, audio = best[1], best[2]
                break
            # The face or the scene of part 1 is lost in this part: the same part again, with another seed. The last
            # try starts from the last frames of the part before, which already has the face and the scene (one such
            # start is a small step; only a chain of them lost quality).
            tries += 1
            replay[:] = list(used)
            if tries == PART_RETRIES and tail is not None and request['workflow'] != 'crossview':
                images = {**images, 'continuity_guide': tail}
                attempt = {**attempt, 'seed': int(request['seed'])}
            else:
                attempt = {**attempt, 'seed': int(request['seed']) + tries}
            face_level['retries'] = face_level.get('retries', 0) + 1
            update(part_retries=face_level['retries'])
        best = None
        if index == 0 and guard and not scene:
            scene.update(made=scene_plate(made), source=scene_plate(frames))
        made = looks.apply(made, frames[:made.shape[0]])
        pixels = made[lead:lead + keep]
        if index == 0 and split:
            follow_part_1(pixels)
            if encoder is not None:
                ready = prepare_parts(new_parts[1:])
        if first_frame is None:
            first_frame = pixels[0].clone()
        if held is not None and lead:
            # The seam: the previous part's version of the overlap frames morphs into this part's version.
            writer.add(seam_morph(held, made[:lead].to(held.dtype)))
        last = index == len(plan) - 1
        hold = 0 if last or keep <= OVERLAP_FRAMES else OVERLAP_FRAMES
        writer.add(pixels[:keep - hold])
        held = pixels[keep - hold:].clone() if hold else None
        # Matched to part 1 above, so the next part starts from frames at part 1's level.
        tail = pixels[-OVERLAP_FRAMES:].clone() if keep >= OVERLAP_FRAMES else None
        if request['output_audio'] == 'source':
            audio = source_audio
        elif request['output_audio'] == 'mute':
            audio = None
        if audio is not None:
            rate = rate or int(audio['sample_rate'])
            overlap = round(lead / FPS * rate)
            piece = _fit_audio(audio, rate, overlap + round(keep / FPS * rate))
            before = sounds[-1][1] if sounds and overlap else None
            if before is not None and before.shape[-1] >= overlap:
                # The seam in the sound: the previous part's sound of the overlap frames fades into this part's sound
                # of the same frames. Joined with a hard cut, the sound jumped at every seam.
                fade = torch.linspace(0, 1, overlap + 2)[1:-1].to(before.dtype)
                mixed = before[..., -overlap:] * (1 - fade) + piece[..., :overlap] * fade
                sounds[-1] = (sounds[-1][0], torch.cat([before[..., :-overlap], mixed], dim=-1))
            sounds.append((keep, piece[..., overlap:]))
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
    # The last frames of the last kept part are a seam in the original video, already morphed toward the part that is
    # made again; its own version of them is in its saved part file.
    part_file = Path(original.get('parts_folder') or '') / f'part-{first_new:03d}.mp4'
    if original.get('parts_folder') and part_file.is_file():
        with av.open(str(part_file)) as container:
            own = [frame.to_ndarray(format='rgb24') for frame in container.decode(container.streams.video[0])]
        if len(own) == plan[first_new - 1][1] - plan[first_new - 1][0] and own[0].shape == frames[0].shape:
            frames[-OVERLAP_FRAMES:] = own[-OVERLAP_FRAMES:]
    kept = torch.from_numpy(np.stack(frames))  # 8-bit; each kept part is converted when it is written
    audio = read_audio(video, count / FPS, start_seconds=0, end_seconds=count / FPS) if original.get('has_audio') else None
    return plan, first_new, kept, audio

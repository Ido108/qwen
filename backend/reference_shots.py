"""Face swap one shot at a time.

MiniMax H3 Ref2VA with the FaceSwap LoRA keeps a single continuous shot. Given a
multi-shot source it builds a new edit instead: shots in another order, cuts at
other times (measured 2026-10-30 on a six-shot clip, with both prompt forms).
Each shot is swapped on its own and the shots are joined in source order.
"""
from __future__ import annotations

FPS = 24


def shot_bounds(frames) -> list[tuple[int, int]]:
    """[start, end) frame ranges between hard cuts in [N, H, W, 3] frames.

    A cut is a frame whose change from the previous one is large and at least three
    times every change in the three frames before it, so fast motion is not a cut.
    """
    import torch
    count = frames.shape[0]
    if count < 2:
        return [(0, count)]
    small = torch.nn.functional.interpolate(frames.permute(0, 3, 1, 2), size=(64, 36), mode='area')
    change = (small[1:] - small[:-1]).abs().mean(dim=(1, 2, 3)).tolist()
    typical = sorted(change)[len(change) // 2]
    cuts = [index + 1 for index, value in enumerate(change)
            if value >= 0.1 and value >= 3 * max(change[max(0, index - 3):index] + [typical])]
    edges = [0] + cuts + [count]
    return [(start, end) for start, end in zip(edges, edges[1:])]


def has_face(frames) -> bool:
    """True when the app's face detector finds a face in one of three frames of the shot."""
    import numpy as np
    from .reference_people import face_detector
    detector = face_detector()
    count = frames.shape[0]
    for index in sorted({0, count // 2, count - 1}):
        rgb = (frames[index].numpy() * 255).astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = detector.detect(bgr)
        if faces is not None and any(face[2] >= 16 and face[3] >= 16 for face in faces):
            return True
    return False


def grid_length(frames: int) -> int:
    """H3 renders 17k+5 frames; the smallest such length that holds the shot."""
    length = max(5, frames)
    while length % 17 != 5:
        length += 1
    return length


def _audio_slice(audio, start: int, end: int, length: int):
    import torch
    rate = audio['sample_rate']
    first, last, total = round(start / FPS * rate), round(end / FPS * rate), round(length / FPS * rate)
    waveform = audio['waveform'][..., first:last]
    waveform = torch.nn.functional.pad(waveform, (0, max(0, total - waveform.shape[-1])))
    return {'waveform': waveform, 'sample_rate': rate}


def _fit_audio(audio, rate: int, samples: int):
    import torch
    import torchaudio
    waveform = audio['waveform']
    if audio['sample_rate'] != rate:
        waveform = torchaudio.functional.resample(waveform, audio['sample_rate'], rate)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return torch.nn.functional.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1])))


def swap_by_shot(request, prompt, images, progress, generate, preview_directory=None):
    """Run generate() on each shot with a face; keep the other shots' source frames.

    Returns (frames, audio) for the whole selection, like one generate() call.
    """
    import torch
    source = images['source_video']
    shots = shot_bounds(source)
    plan = [(start, end, end - start >= 5 and has_face(source[start:end])) for start, end in shots]
    swapped = [shot for shot in plan if shot[2]]
    if len(shots) == 1 or not swapped:
        return generate(request, prompt, images, progress, preview_directory=preview_directory)
    steps = request['steps']
    progress(shots=[{'start': start, 'end': end, 'swap': swap} for start, end, swap in plan])
    clip_start = float(request.get('source_start_seconds') or 0)
    parts, sounds = [], []
    for start, end, swap in plan:
        if not swap:
            # Source frames keep their own size; the swapped shots come out at the output size.
            kept = torch.nn.functional.interpolate(source[start:end].permute(0, 3, 1, 2), size=(request['height'], request['width']),
                                                   mode='bicubic', antialias=True).clamp(0, 1).permute(0, 2, 3, 1)
            parts.append(kept)
            sounds.append(None)
            continue
        offset = swapped.index((start, end, swap)) * steps
        def relay(**values):
            if 'total_steps' in values:
                values['total_steps'] = steps * len(swapped)
            if values.get('stage') == 'generating':
                values['step'] = values.get('step', 0) + offset
            progress(**values)
        length = grid_length(end - start)
        frames = source[start:end]
        if length > frames.shape[0]:
            frames = torch.cat([frames, frames[-1:].repeat(length - frames.shape[0], 1, 1, 1)])
        shot_images = {**images, 'source_video': frames}
        if images.get('source_audio') is not None:
            shot_images['source_audio'] = _audio_slice(images['source_audio'], start, end, length)
        shot_request = {**request, 'frames': length, 'source_start_seconds': clip_start + start / FPS,
                        'source_end_seconds': clip_start + end / FPS}
        progress(shot=swapped.index((start, end, swap)) + 1, total_shots=len(swapped))
        pixels, audio = generate(shot_request, prompt, shot_images, relay, preview_directory=preview_directory)
        parts.append(pixels[:end - start])
        sounds.append(audio)
    frames = torch.cat(parts)
    made = [audio for audio in sounds if audio is not None]
    if not made:
        return frames, None
    rate = made[0]['sample_rate']
    pieces = []
    for (start, end, _), audio in zip(plan, sounds):
        samples = round((end - start) / FPS * rate)
        if audio is None:
            kept = images.get('source_audio')
            pieces.append(_fit_audio(_audio_slice(kept, start, end, end - start), rate, samples) if kept is not None
                          else torch.zeros(1, 2, samples))
        else:
            pieces.append(_fit_audio(audio, rate, samples))
    return frames, {'waveform': torch.cat(pieces, dim=-1), 'sample_rate': rate}

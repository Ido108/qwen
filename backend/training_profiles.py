"""Transparent starting points, not a claim to infer optimal learning from image count."""
import math

VERSION = '2026-10-21-v3'


def image_facts(photo):
    """Size, sharpness (Laplacian variance of the sharpest 64 px tile of a 512 px gray copy) and a 64-bit difference hash."""
    import numpy as np
    from PIL import Image
    gray = photo.convert('L'); gray.thumbnail((512, 512))
    a = np.asarray(gray, dtype=np.float32)
    laplacian = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    # The sharpest 64 px tile: a portrait with a blurred background is still sharp where the face is.
    tiles = [laplacian[y:y + 64, x:x + 64].var() for y in range(0, laplacian.shape[0] - 31, 64) for x in range(0, laplacian.shape[1] - 31, 64)]
    small = np.asarray(photo.convert('L').resize((9, 8), Image.Resampling.BILINEAR), dtype=np.float32)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return {'width': photo.width, 'height': photo.height, 'sharpness': float(max(tiles) if tiles else laplacian.var()),
            'hash': int(''.join('1' if b else '0' for b in bits), 2)}


def dataset_notes(subject, images):
    """Warnings from the dataset research (docs/lora-dataset-captioning.md): count, framing, size, blur, near-duplicates."""
    notes = []
    count = len(images)
    if subject != 'style' and count < 15:
        notes.append(f'Only {count} photos: 15 to 40 varied photos of the person give the most reliable likeness.')
    if subject != 'style':
        notes.append('Mix close-ups, half-body and full-body photos (about 40, 30 and 30 percent) so full-body prompts keep the likeness.')
    small = sum(1 for i in images if 0 < min(i.get('width') or 0, i.get('height') or 0) < 768)
    if small:
        notes.append(f'{small} photo{"s are" if small > 1 else " is"} smaller than 768 px on the short side. Replace {"them with larger copies" if small > 1 else "it with a larger copy"} if you can.')
    sharp = sorted(i['sharpness'] for i in images if i.get('sharpness'))
    if len(sharp) >= 4:
        median = sharp[len(sharp) // 2]
        blurry = sum(1 for value in sharp if value < median * 0.35)
        if blurry:
            notes.append(f'{blurry} photo{"s look" if blurry > 1 else " looks"} much blurrier than the rest. Blurry photos teach blur; remove {"them" if blurry > 1 else "it"} if you can.')
    hashes = [i['hash'] for i in images if i.get('hash') is not None]
    near = sum(1 for index, value in enumerate(hashes) if any(bin(value ^ other).count('1') <= 4 for other in hashes[:index]))
    if near:
        notes.append(f'{near} photo{"s are" if near > 1 else " is"} nearly the same as another one. Near copies repeat one pose and background; keep one of each.')
    return notes


def recommend(arch, subject, images):
    from .training import ARCHS
    info = ARCHS[arch]
    count = max(1, len(images))
    turbo = bool(info.get('assistant_lora_path'))
    # The pinned toolkit example recommends 500 to 4000 total steps. These bounded
    # exposure targets are app policy, not undocumented model-author settings.
    # A synthesis of documented first runs of 2000 to 3000 steps (ai-toolkit examples 2000, Krea's workflow 2500,
    # MOSS guides 3000, Qwen 2000 for 31 images), not a published formula. Turbo with its adapter keeps a 2000 cap.
    exposures = 60 if subject == 'style' else 100
    steps = int(min(3000, max(1500, math.ceil(count * exposures / 50) * 50)))
    if turbo: steps = min(steps, 2000)
    sides = [min(i.get('width') or 0, i.get('height') or 0) for i in images]
    known = sorted(s for s in sides if s > 0)
    median = known[len(known) // 2] if known else 1024
    sizes = [s for s in (512, 768, 1024) if s <= max(512, median)]
    notes = [f'{count} images; about {steps / count:.0f} training steps per image.']
    if count < 10: notes.append('A small dataset can repeat its poses and backgrounds. Check the saved samples before continuing.')
    if turbo: notes.append('Short training helps preserve Turbo speed. The training adapter is removed for samples and the final LoRA.')
    if median < 1024: notes.append('Image sizes follow the dataset. Larger settings cannot add missing detail.')
    # Rank: ostris' Krea 2 docs and the MOSS guides (DiffSynth, SimpleTuner, RunComfy) use 32 for a character; Qwen keeps 16.
    # Learning rate: every ai-toolkit source uses 1e-4; half of it with the fewest steps under-trained small datasets.
    rank = 32 if subject == 'style' or count > 40 or info['arch'] in ('krea2', 'moss_image') else 16
    return {'version': VERSION, 'steps': steps, 'rank': rank,
            'learning_rate': 1e-4, 'resolutions': sizes,
            'save_every': min(250, steps // 2), 'caption_dropout': .05,
            'notes': notes, 'warnings': dataset_notes(subject, images), 'sample_steps': info['sample_steps'], 'sample_cfg': info['guidance']}

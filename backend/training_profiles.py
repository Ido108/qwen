"""Transparent starting points, not a claim to infer optimal learning from image count."""
import math

VERSION = '2026-10-19-v1'


def recommend(arch, subject, images):
    from .training import ARCHS
    info = ARCHS[arch]
    count = max(1, len(images))
    turbo = bool(info.get('assistant_lora_path'))
    # The pinned toolkit example recommends 500–4000 total steps. These bounded
    # exposure targets are app policy, not undocumented model-author settings.
    exposures = 100 if subject == 'style' else 60
    cap = 2000 if turbo else 4000
    steps = int(min(cap, max(500, math.ceil(count * exposures / 50) * 50)))
    sides = [min(i.get('width') or 0, i.get('height') or 0) for i in images]
    known = sorted(s for s in sides if s > 0)
    median = known[len(known) // 2] if known else 1024
    sizes = [s for s in (512, 768, 1024) if s <= max(512, median)]
    notes = [f'{count} images; about {steps / count:.0f} training steps per image.']
    if count < 10: notes.append('A small dataset can repeat its poses and backgrounds. Check the saved samples before continuing.')
    if turbo: notes.append('Short training helps preserve Turbo speed. The training adapter is removed for samples and the final LoRA.')
    if median < 1024: notes.append('Image sizes follow the dataset. Larger settings cannot add missing detail.')
    return {'version': VERSION, 'steps': steps, 'rank': 32 if subject == 'style' or count > 40 else 16,
            'learning_rate': 5e-5 if count < 10 else 1e-4, 'resolutions': sizes,
            'save_every': min(250, steps // 2), 'caption_dropout': .05,
            'notes': notes, 'sample_steps': info['sample_steps'], 'sample_cfg': info['guidance']}

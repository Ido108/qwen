"""Local, queued dataset captions with durable per-image reuse."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
import time
import os
from pathlib import Path
from .common import DATA_DIR, APP_DIR, read_json, write_json, resolve_image

MODEL_ID = 'flux3-prompt'
VERSION = 'dataset-caption-v1'
CACHE = DATA_DIR / 'training-caption-cache'


def prompt(request):
    name = request.get('name', '').strip()
    trigger = request.get('trigger_word', '').strip()
    start = f'Use {json.dumps(trigger)} as the subject name. ' if trigger else ''
    if request.get('subject') == 'style':
        subject = ('Describe the depicted subjects, their actions, objects, composition and spatial relationships. '
                   'Keep the visual style itself implicit so the learned style word represents it. ')
        start = f'Begin with "In the style of {trigger},". ' if trigger else ''
    else:
        subject = ('Describe the main subject through its pose, expression, action, clothes, accessories and framing. '
                   'Keep its permanent identity features implicit so the subject word represents them. '
                   'Describe other visible people separately through their position and action. ')
    return ('Write one factual English caption for a diffusion LoRA training image, about 60 to 120 words. '
            'Use only clearly visible image content. State concrete details in natural sentences. '
            'Describe the setting, background, light and viewpoint. '+subject+start+
            f'The dataset name is {json.dumps(name)}; it identifies the training subject, not visible text. '
            'Return only the caption paragraph.')


def clean(text, request):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip().strip('"')
    if not text or len(text) > 8000:
        raise ValueError('The caption model returned no usable caption. Retry this image.')
    trigger = request.get('trigger_word', '').strip()
    if trigger and not re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)', text, re.I):
        text = (f'In the style of {trigger}, ' if request.get('subject') == 'style' else f'{trigger}, ') + text
    return text


def fingerprint(image_id, request):
    from .models import entry
    digest = hashlib.sha256()
    with resolve_image(image_id).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): digest.update(chunk)
    digest.update((VERSION + entry(MODEL_ID)['revision'] + prompt(request)).encode())
    return digest.hexdigest()


def run(request, directory, on_process, cancelled, update):
    """Spawn one caption worker for the batch; retry keeps every completed caption."""
    selected = list({i['id']: i for i in request['images'] if not i.get('caption', '').strip()}.values())
    previous = directory / 'captions.json'
    found = read_json(previous) if previous.exists() else {}
    # The cache key includes pixels, exact instructions, trigger and model revision.
    work = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for item in selected:
        key = fingerprint(item['id'], request)
        cache = CACHE / (key + '.json')
        if cache.exists(): found[item['id']] = read_json(cache)
        elif not (found.get(item['id'], {}).get('caption') and found[item['id']].get('key') == key):
            found.pop(item['id'], None)
            work.append({'id': item['id'], 'key': key})
    write_json(previous, found)
    if not work: return found
    from . import models
    models.ensure_installed([MODEL_ID], owner=directory.name, cancelled=cancelled,
        progress=lambda transfer: update(stage='downloading_caption_model', message='Downloading the caption model...', transfer=transfer))
    if cancelled(): return found
    spec = directory / 'caption-request.json'
    write_json(spec, {'request': request, 'items': work})
    with (directory / 'caption-worker.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'backend.training_captions', str(directory)], cwd=APP_DIR,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    on_process(process)
    while process.poll() is None:
        status = directory / 'caption-progress.json'
        if status.exists(): update(**read_json(status))
        time.sleep(.5)
    found = read_json(previous)
    if cancelled(): return found
    if process.returncode:
        raise RuntimeError('The caption model stopped. Completed captions remain available. Retry the missing captions.')
    return found


def worker(directory):
    spec = read_json(directory / 'caption-request.json'); request = spec['request']; items = spec['items']
    state = directory / 'caption-progress.json'
    def progress(**values): write_json(state, values)
    progress(stage='loading_caption_model', message='Loading the caption model...')
    from .flux3_runtime import activate
    activate(progress)
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload = True
    args.disable_pinned_memory = True
    args.disable_dynamic_vram = True
    import comfy_aimdo.control
    comfy_aimdo.control.init()
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_textgen import TextGenerate
    from .models import entry, local_path
    from .flux3_engine import PromptImages
    item = entry(MODEL_ID)
    weights, metadata = comfy.utils.load_safetensors(str(local_path(item, item['files'][0])))
    weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
    clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.FLUX3,
        model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
    del weights
    output = read_json(directory / 'captions.json')
    for index, item in enumerate(items):
        progress(stage='captioning', message=f'Writing caption {index + 1} of {len(items)}', step=index, total_steps=len(items))
        try:
            with Image.open(resolve_image(item['id'])) as image:
                photo = ImageOps.exif_transpose(image).convert('RGB')
                photo.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                tensor = torch.from_numpy(np.array(photo).astype(np.float32) / 255).unsqueeze(0)
            with torch.inference_mode():
                text = TextGenerate.execute(PromptImages(clip, [tensor]), 'Write the training caption for this image.', 512,
                    {'sampling_mode': 'off'}, thinking=False, use_default_template=True, mtp='off', system_prompt=prompt(request))[0]
            answer = {'caption': clean(text, request), 'source': 'automatic', 'key': item['key']}
            write_json(CACHE / (item['key'] + '.json'), answer)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            answer = {'caption': '', 'error': f'Caption failed for this image: {type(exc).__name__}. Retry or edit it.'}
        output[item['id']] = answer
        write_json(directory / 'captions.json', output)
    progress(stage='captioning', message='Captions ready.', step=len(items), total_steps=len(items))


if __name__ == '__main__':
    worker(Path(sys.argv[1]).resolve())

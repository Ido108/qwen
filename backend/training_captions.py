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

# Qwen-Image PE I2I (Qwen3.5-VL 9B, Heretic, Q8_0) through llama.cpp: on an RTX 5090 it captioned the user's photos in
# about 2 s each against about 10 s with the FLUX.3 prompt model in ComfyUI, followed the instructions, and refuses nothing.
MODEL_ID = 'pe-i2i-heretic-q8'
VERSION = 'dataset-caption-v3'
CACHE = DATA_DIR / 'training-caption-cache'


PROMPTS = APP_DIR / 'reference' / 'caption-prompts.json'


def prompts(request):
    """The caption wording for the training's base model, with its source (reference/caption-prompts.json)."""
    from .prompt_registry import apply_captions
    table = apply_captions(json.loads(PROMPTS.read_text(encoding='utf-8')))  # Settings > Prompts edits
    own = table.get(request.get('arch') or '')
    return own if own and own.get(request.get('subject', 'character')) else table['default']


def prompt(request):
    name = request.get('name', '').strip()
    trigger = request.get('trigger_word', '').strip()
    text = prompts(request)
    if 'intro' not in text:
        return text[request.get('subject', 'character')].format(name=json.dumps(name, ensure_ascii=False), trigger=trigger)
    style = request.get('subject') == 'style'
    start = text['trigger_style' if style else 'trigger_character'].format(trigger=json.dumps(trigger, ensure_ascii=False)) if trigger else ''
    return text['intro'] + text['style' if style else 'character'] + start + text['outro'].format(name=json.dumps(name, ensure_ascii=False))


def clean(text, request):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip().strip('"')
    if not text or len(text) > 8000:
        raise ValueError('The caption model returned no usable caption. Retry this image.')
    trigger = request.get('trigger_word', '').strip()
    if trigger:  # the instruction quotes the trigger; the caption must hold the bare word
        text = re.sub(r'["“”]' + re.escape(trigger) + r'["“”]', trigger, text)
    prefix = prompts(request).get('prefix')
    if prefix and request.get('subject') != 'style':
        # MOSS Serenity form: "photo of <trigger>, a man with ..."; the model writes only the sentence.
        text = re.sub(r'^(?:a\s+)?photo of\s+[^,]*,\s*', '', text, flags=re.I).strip()
        return prefix.format(trigger=trigger).replace('photo of , ', 'photo of ') + text[:1].lower() + text[1:]
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
    import base64
    import io
    from PIL import Image, ImageOps
    from .llama_runtime import Server
    from .models import entry, local_path
    item = entry(MODEL_ID)
    paths = [local_path(item, file) for file in item['files']]
    output = read_json(directory / 'captions.json')
    with Server(paths[0], paths[1]) as server:
        for index, item in enumerate(items):
            progress(stage='captioning', message=f'Writing caption {index + 1} of {len(items)}', step=index, total_steps=len(items))
            try:
                with Image.open(resolve_image(item['id'])) as image:
                    photo = ImageOps.exif_transpose(image).convert('RGB')
                photo.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                stream = io.BytesIO(); photo.save(stream, 'PNG')
                picture = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}}
                reply = server.chat({'messages': [{'role': 'system', 'content': prompt(request)},
                                                  {'role': 'user', 'content': [picture, {'type': 'text', 'text': 'Write the training caption for this image.'}]}],
                                     'temperature': 0.2, 'top_p': 0.9, 'max_tokens': 400, 'chat_template_kwargs': {'enable_thinking': False}})
                answer = {'caption': clean(reply['choices'][0]['message'].get('content') or '', request), 'source': 'automatic', 'key': item['key']}
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

"""One owned process per generation: cancellation releases all GPU and RAM allocations."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from .common import APP_DIR, DATA_DIR, MODELS_DIR, resolve_image, write_json

os.environ['HF_HOME'] = str(DATA_DIR / 'hf-cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

directory = Path(sys.argv[1])
job = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))

def update(**values):
    job.update(values)
    write_json(directory / 'status.json', job)
    print(json.dumps({k: v for k, v in values.items() if k != 'request'}, ensure_ascii=False), flush=True)

def rewrite_prompt(images):
    from .prompt_enhancer import enhance
    from .antigravity import selected_model
    return enhance(request['prompt'], images, request.get('enhancement_model') or selected_model(), update)

def run():
    update(status='running', stage='loading', message='Loading generation tools…', started_at=time.time())
    if request.get('kind') == 'video':
        from .video_worker import run as run_video
        return run_video(request, job, directory, update)
    import torch
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from .comfy_engine import generate
    from .models import entry
    from .image_settings import output_dimensions
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Restart after checking the graphics driver.')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(8)
    images = []
    for i, image_id in enumerate(request['images']):
        with Image.open(resolve_image(image_id)) as img:
            image = img.convert('RGBA')
            image.save(directory / f'input-{i+1}.png')
            images.append(image)
    rewrite = rewrite_prompt(images) if request['enhance'] else {}
    prompt = rewrite.get('positive_prompt') or request['prompt']
    width, height = output_dimensions(request, images, rewrite)
    render_width = math.ceil(width / 32) * 32
    render_height = math.ceil(height / 32) * 32
    update(stage='loading_model', message='Loading Qwen-Image-2.1…', width=width, height=height)
    image = generate(request,prompt,images,render_width,render_height,update)
    update(stage='saving', message='Saving PNG…')
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    destination = Path(request['output_dir']) / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '.png')
    metadata = PngInfo()
    selected = entry(request.get('model_id','official-bf16'))
    metadata.add_text('Model', selected['title'])
    metadata.add_text('Prompt', request['prompt'])
    metadata.add_text('Rewritten prompt', job.get('rewritten_prompt', ''))
    metadata.add_text('Seed', str(request['seed']))
    metadata.add_text('Parameters', json.dumps(request, ensure_ascii=False))
    image.save(destination, pnginfo=metadata)
    alpha_histogram=image.getchannel('A').histogram()
    transparent_fraction=sum(alpha_histogram[:128])/(image.width*image.height)
    visible_alpha_fraction=sum(alpha_histogram[:250])/(image.width*image.height)
    update(status='succeeded', stage='complete', message='Image saved.', output_path=str(destination),
        image_url=f'/api/jobs/{job["id"]}/image', finished_at=time.time(),
        elapsed_seconds=round(time.time() - job['started_at'], 1), width=image.width, height=image.height,
        has_transparency=visible_alpha_fraction>.001,transparent_pixel_fraction=round(transparent_fraction,5),render_size=[render_width, render_height],
        resized_to_requested=(render_width, render_height) != (width, height),
        actual_prompt=prompt,model_revision=selected['revision'])
    write_json(destination.with_suffix('.json'), job)

try:
    run()
except Exception as exc:
    traceback.print_exc()
    message = str(exc)
    if 'out of memory' in message.lower():
        message = ('Not enough free GPU memory. Close other AI apps, choose a smaller video size or shorter duration, and retry.'
                   if request.get('kind')=='video' else 'Not enough free GPU memory. Close other AI apps, choose a 1K size, and retry.')
    elif '1455' in message or 'paging file is too small' in message.lower():
        message = 'Windows ran out of available system memory. Close unused AI apps and retry. If this persists, increase the Windows paging file.'
    update(status='failed', stage='failed', message=message, finished_at=time.time())
    sys.exit(1)

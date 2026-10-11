"""Resident image worker; video jobs retain an isolated, single-use process."""
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

directory = None
job = {}
request = {}

def update(**values):
    now = time.time()
    if values.get('stage') and values['stage'] != job.get('stage'):
        if job.get('stage_started_at'):
            timings = job.setdefault('stage_seconds', {})
            stage = job['stage']
            timings[stage] = round(timings.get(stage, 0) + now - job['stage_started_at'], 3)
        values['stage_started_at'] = now
    if values.get('stage') == 'generating' and values.get('step') == 1:
        values['time_to_first_step_seconds'] = round(now - job['started_at'], 3)
    job.update(values)
    write_json(directory / 'status.json', job)
    print(json.dumps({k: v for k, v in values.items() if k != 'request'}, ensure_ascii=False), flush=True)

def rewrite_prompt(images):
    from .prompt_enhancer import enhance
    from .antigravity import selected_model
    return enhance(request['prompt'], images, request.get('enhancement_model') or selected_model(), update)

def run():
    update(status='running', stage='loading', message='Loading generation tools…', started_at=time.time())
    if request.get('kind') == 'multi-reference':
        from .ltx_worker import run as run_ltx
        return run_ltx(request, job, directory, update)
    if request.get('kind') in ('video','reference-video'):
        from .video_worker import run as run_video
        return run_video(request, job, directory, update)
    if request.get('kind') == 'upscale':
        from .upscale_worker import run as run_upscale
        return run_upscale(request, job, directory, update)
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
    tool = None
    paste_box = None
    if request.get('tool_id'):
        from .models import image_tool
        from .image_tools import prepare
        tool = image_tool(request['tool_id'])
        images, tool_size, paste_box = prepare(tool, request, images)
        for i, image in enumerate(images):
            image.save(directory / f'prepared-{i+1}.png')
        update(tool=tool['title'], prepared_inputs=[list(image.size) for image in images])
    rewrite = request.get('enhancement_result') or (rewrite_prompt(images) if request['enhance'] else {})
    prompt = rewrite.get('positive_prompt') or request['prompt']
    if tool and tool_size:
        width, height = tool_size
    else:
        # Edit tools follow the shape of image 1 (the picture being changed).
        width, height = output_dimensions(request, images, {'ratio_follow': '<image1>'} if tool else rewrite)
    render_width = math.ceil(width / 32) * 32
    render_height = math.ceil(height / 32) * 32
    update(stage='loading_model', message=f"Loading {entry(request.get('model_id','official-bf16'))['title']}…", width=width, height=height)
    image = generate(request,prompt,images,render_width,render_height,update)
    update(stage='saving', message='Saving PNG…')
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if paste_box:
        from .image_tools import paste_back
        image = paste_back(image, paste_box)
    destination = Path(request['output_dir']) / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '.png')
    metadata = PngInfo()
    selected = entry(request.get('model_id','official-bf16'))
    metadata.add_text('Model', selected['title'])
    metadata.add_text('Prompt', request['prompt'])
    metadata.add_text('Rewritten prompt', job.get('rewritten_prompt', ''))
    metadata.add_text('Seed', str(request['seed']))
    if tool:
        metadata.add_text('Tool', f"{tool['title']} ({request.get('lora_strength')})")
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

def execute(folder):
    global directory, job, request
    directory = Path(folder)
    job = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    try:
        update(worker_pid=os.getpid())
        run()
        return 0
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        if 'out of memory' in message.lower():
            if request.get('kind') in ('video','reference-video','upscale'):
                message = ('Not enough free GPU memory while processing the prompt and input images. '
                           'Close unused AI apps and retry.' if job.get('stage') == 'encoding' else
                           'Not enough free GPU memory while generating video. '
                           'Close unused AI apps, choose a smaller size or shorter duration, and retry.')
            else:
                message = 'Not enough free GPU memory. Close unused AI apps, choose a 1K size, and retry.'
        elif '1455' in message or 'paging file is too small' in message.lower():
            message = 'Windows ran out of available system memory. Close unused AI apps and retry. If this persists, increase the Windows paging file.'
        update(status='failed', stage='failed', message=message, finished_at=time.time())
        return 1


def main():
    if sys.argv[1] != '--persistent':
        return execute(sys.argv[1])
    # EOF also ends the worker if its owning server closes or crashes.
    for line in sys.stdin:
        command = json.loads(line)
        if command.get('stop'):
            return 0
        folder = Path(command['directory'])
        with (folder / 'worker.log').open('w', encoding='utf-8') as log:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            code = execute(folder)
            sys.stdout.flush()
            sys.stderr.flush()
            write_json(folder / 'worker.done.json', {'returncode': code, 'pid': os.getpid()})
        if code:
            # Discard any partially loaded models after an exception.
            return code
    return 0


if __name__ == '__main__':
    sys.exit(main())

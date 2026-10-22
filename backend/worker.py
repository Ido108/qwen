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
from .image_inputs import ImageInputs

os.environ['HF_HOME'] = str(DATA_DIR / 'hf-cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

directory = None
job = {}
request = {}
image_inputs = ImageInputs()

def update(**values):
    now = time.time()
    if 'sent_prompt' in values:
        from .sent_prompts import add
        values['sent_prompts'] = add(job, values.pop('sent_prompt'))
    stage=values.get('stage',job.get('stage'))
    if stage=='generating' and 'step' in values:
        step=values['step']
        previous=job.get('step',0)
        if job.get('stage')!='generating' or step<=previous:
            values.update(sampling_step_seconds=None,sampling_step_at=None)
        elif job.get('sampling_step_at') and step==previous+1:
            elapsed=now-job['sampling_step_at']
            average=job.get('sampling_step_seconds')
            values['sampling_step_seconds']=round(elapsed if not average else .6*elapsed+.4*average,3)
        if step>previous or values.get('sampling_step_at') is None and step>0:
            values['sampling_step_at']=now
    changed_stage=values.get('stage') and values['stage'] != job.get('stage')
    changed_phase=values.get('phase') is not None and values['phase'] != job.get('phase')
    if changed_stage or changed_phase:
        if job.get('stage_started_at'):
            timings = job.setdefault('stage_seconds', {})
            stage = job['stage']
            elapsed=round(now-job['stage_started_at'],3)
            timings[stage] = round(timings.get(stage, 0) + elapsed, 3)
            job.setdefault('stage_history',[]).append({'stage':stage,'phase':job.get('phase',''),
                'started_at':job['stage_started_at'],'seconds':elapsed})
        values['stage_started_at'] = now
    if values.get('stage') == 'generating' and values.get('step') == 1:
        values['time_to_first_step_seconds'] = round(now - job['started_at'], 3)
    job.update(values)
    write_json(directory / 'status.json', job)
    print(json.dumps({k: v for k, v in values.items() if k != 'request'}, ensure_ascii=False), flush=True)

def rewrite_prompt(images):
    from .enhancer_settings import route
    from .models import entry, is_zimage
    item = entry(request['model_id'])
    family = item.get('family', 'qwen-image-2.1')
    # Settings > Prompt enhancement: each architecture's own official enhancer unless the user chose another.
    chosen = route(family, images, is_zimage(item))
    if chosen['mode'] == 'none':
        return {}  # FLUX.3 rewrites with BFL's prompt model inside its own engine
    update(enhancer=chosen['label'], sent_prompt={'label': 'Enhancer used', 'text': f"{chosen['label']} (prompt: {chosen['spec']})"})
    from . import enhance_cache
    from .prompt_enhancer import prompts
    # Same request text, input pictures, enhancer and its prompt (including Settings edits) give the same key.
    key = enhance_cache.key(request['prompt'], chosen['label'], chosen['spec'], prompts().get(chosen['spec']) if chosen['spec'] else None,
                            request.get('text_encoder'), images=images)
    saved = enhance_cache.get(key)
    from .prompt_enhancer import CJK
    wrong_language = lambda result: bool(result) and not CJK.search(request['prompt']) and len(CJK.findall(result.get('positive_prompt') or '')) > 10
    if wrong_language(saved):
        saved = None  # an answer saved before the English retry existed
    if saved:
        update(stage='rewriting', message='Using the saved enhanced prompt (same prompt, images and enhancer)…', enhance_cached=True,
               rewrite_seconds=0, rewritten_prompt=saved['positive_prompt'],
               sent_prompt={'label': 'Enhance: rewritten prompt (saved from an earlier job)', 'text': saved['positive_prompt']})
        return saved
    result = run_enhancer(chosen, images, item, family)
    if result and result.get('positive_prompt') and not wrong_language(result):
        enhance_cache.put(key, result)
    return result

def run_enhancer(chosen, images, item, family):
    from .prompt_enhancer import enhance_local
    from .models import entry, is_zimage
    # Krea 2 and MOSS Image rewrite with their own text encoder, as their official templates do.
    picked_encoder = request.get('text_encoder') or 'auto'
    encoder = picked_encoder if picked_encoder != 'auto' else next((dep for dep in item.get('dependencies', []) if entry(dep).get('kind') == 'encoder'), None)
    return enhance_local(request['prompt'], images, update, family=family, redraw=is_zimage(item), seed=request.get('seed', 0), encoder_id=encoder,
                         spec=chosen['spec'], runner=chosen['runner'], fast=chosen.get('fast', False))

def image_engine(saved):
    """Krea 2 models run in krea_engine, Z-Image models in zimage_engine; Qwen-Image-2.1 models in comfy_engine."""
    from .models import entry, is_krea, is_zimage, is_flux3, is_moss
    item = entry(saved.get('model_id', 'official-bf16'))
    if is_moss(item):
        from .moss_engine import generate
    elif is_flux3(item):
        from .flux3_engine import generate
    elif is_krea(item):
        from .krea_engine import generate
    elif is_zimage(item):
        from .zimage_engine import generate
    else:
        from .comfy_engine import generate
    return generate

def run():
    update(status='running', stage='loading', message='Loading generation tools…', started_at=time.time())
    from . import sent_prompts
    sent_prompts.clear()  # a failed earlier job in this resident worker must not leak its encoder text
    if request.get('kind') == 'multi-reference':
        from .ltx_worker import run as run_ltx
        return run_ltx(request, job, directory, update)
    if request.get('kind') in ('video','reference-video'):
        from .video_worker import run as run_video
        return run_video(request, job, directory, update)
    if request.get('kind') == 'upscale':
        from .upscale_worker import run as run_upscale
        return run_upscale(request, job, directory, update)
    if request.get('kind') == 'image-upscale':
        from .image_upscale import run as run_image_upscale
        return run_image_upscale(request, job, directory, update)
    import torch
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from .models import entry
    from .image_settings import output_dimensions
    generate = image_engine(request)
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Restart after checking the graphics driver.')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(8)
    prepared_at = time.perf_counter()
    images, reused_inputs = [], 0
    for image_id in request['images']:
        image, reused = image_inputs.load(resolve_image(image_id))
        images.append(image)
        reused_inputs += int(reused)
    update(input_count=len(images), input_cache_hits=reused_inputs,
           input_read_seconds=round(time.perf_counter()-prepared_at, 3))
    tool = None
    paste_box = None
    if request.get('tool_id'):
        from .models import image_tool
        tool = image_tool(request['tool_id'])
        (images, tool_size, paste_box), reused = image_inputs.prepare(tool, request, images)
        update(tool=tool['title'], prepared_inputs=[list(image.size) for image in images],
               input_preparation_reused=reused)
    update(input_preparation_seconds=round(time.perf_counter()-prepared_at, 3))
    rewrite = request.get('enhancement_result') or (rewrite_prompt(images) if request['enhance'] else {})
    prompt = rewrite.get('positive_prompt') or request['prompt']
    from .lora_prompts import internal_prompt
    prompt, lora_words = internal_prompt(prompt, request)
    update(lora_prompt_words=lora_words)
    if request.get('swap_prompt_template'):
        update(sent_prompt={'label': 'Swap template', 'text': request['swap_prompt_template']})
    if tool and tool.get('prompt'):
        update(sent_prompt={'label': f"Tool prompt ({tool['title']})", 'text': tool['prompt']})
    update(sent_prompt={'label': 'Prompt sent to the image model (after enhancement, LoRA words and tool text)', 'text': prompt})
    if tool and tool_size:
        width, height = tool_size
    else:
        # Edit tools follow the shape of image 1 (the picture being changed).
        sizing = request
        # MOSS Image: 1 to 2 MP recommended, 4 MP is extrapolation (Comfy-Org template notes, OpenMOSS USERGUIDE).
        if entry(request['model_id']).get('family') in ('krea-2', 'moss-image', 'moss-image-base', 'moss-image-turbo') and not request['custom_size']:
            # Krea 2 Raw is trained up to 1K (krea-ai/krea-2 README); Turbo makes 1K to 2K.
            sizing = {**request, 'auto_resolution': min(request.get('auto_resolution', 2048), 1024)}
        # Z-Image and MOSS redraw image 1, so the output keeps its shape.
        follow_first = tool or (images and entry(request['model_id']).get('family', '').startswith(('z-image','flux3-klein','moss-image')))
        width, height = output_dimensions(sizing, images, {'ratio_follow': '<image1>'} if follow_first else rewrite)
    if images and entry(request['model_id']).get('family', '').startswith('krea-2') and (request.get('krea_reference_mode') or 'identity') == 'identity':
        # Krea 2 identity edit: the card says generate at 2 MP or less; the photo tokens double the sequence.
        from .krea_edit import identity_size
        if not request['custom_size'] and request.get('auto_resolution', 2048) <= 1024:
            # At 1K the photo's own shape (often near square) made Krea crop in close; 3:4 (4:3 for a landscape
            # photo) leaves room for the body (the user's choice, 2026-10-22).
            person = images[min(int(request.get('krea_person') or 0), len(images) - 1)]
            area, portrait = width * height, person.height >= person.width
            width, height = (int((area * 3 / 4) ** 0.5) // 16 * 16, int((area * 4 / 3) ** 0.5) // 16 * 16) if portrait else \
                            (int((area * 4 / 3) ** 0.5) // 16 * 16, int((area * 3 / 4) ** 0.5) // 16 * 16)
        width, height = identity_size(width, height)
    render_width = math.ceil(width / 32) * 32
    render_height = math.ceil(height / 32) * 32
    update(stage='loading_model', message=f"Loading {entry(request.get('model_id','official-bf16'))['title']}…", width=width, height=height)
    image = generate(request,prompt,images,render_width,render_height,update)
    from . import sent_prompts
    sent_prompts.flush(update, key=(request.get('model_id'), request.get('text_encoder'), prompt, tuple(request['images'])))
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
    metadata.add_text('Actual prompt', job.get('expanded_prompt') or prompt)
    metadata.add_text('LoRA prompt words', json.dumps(lora_words, ensure_ascii=False))
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
        actual_prompt=job.get('expanded_prompt') or prompt,model_revision=selected['revision'])
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
            elif request.get('kind') == 'image-upscale':
                message = 'Not enough free GPU memory while upscaling. Close unused AI apps, choose a smaller size, and retry.'
            else:
                message = 'Not enough free GPU memory. Close unused AI apps, choose a 1K size, and retry.'
        elif '1455' in message or 'paging file is too small' in message.lower():
            message = 'Windows ran out of available system memory. Close unused AI apps and retry. If this persists, increase the Windows paging file.'
        update(status='failed', stage='failed', message=message, finished_at=time.time())
        return 1


def warm(saved, log_path):
    """Load the model, text encoder and VAE onto the GPU with a tiny 1-step run, so the next job starts at once.
    Nothing is saved. A failure only means the next job loads normally."""
    started = time.time()
    with log_path.open('w', encoding='utf-8') as log:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            generate = image_engine(saved)
            generate({**saved, 'steps': 1, 'seed': 0, 'negative_prompt': '', 'tool_id': None, 'loras': [], 'reference_resolution': 1024},
                     'warm up', [], 256, 256, lambda **values: None)
            print(f'Warm-up done in {time.time() - started:.1f} s', flush=True)
        except Exception:
            traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    log_path.with_suffix('.done').write_text(str(time.time() - started), encoding='utf-8')


def main():
    if sys.argv[1] != '--persistent':
        return execute(sys.argv[1])
    # EOF also ends the worker if its owning server closes or crashes.
    for line in sys.stdin:
        command = json.loads(line)
        if command.get('stop'):
            return 0
        if command.get('warm'):
            warm(command['warm'], Path(command['log']))
            continue
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

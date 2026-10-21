"""Prompt enhancement with the official Qwen rewrite/output contract: on this computer (the default) or through Antigravity."""
import base64
import json
import io
import re
import sys
from .common import APP_DIR, VIDEO_UPLOADS_DIR
from . import antigravity


def enhance_reference(prompt, images, request, update):
    """Antigravity rewrite that preserves the Ref2VA trigger and reference order."""
    from PIL import Image
    model_id=request.get('enhancement_model') or antigravity.selected_model()
    trigger='Faceswap' if request['workflow']=='faceswap' else 'crossview' if request['workflow']=='crossview' else ''
    from .prompt_registry import REFERENCE_VIDEO, get
    system=get('video:reference',REFERENCE_VIDEO).replace('{trigger}',repr(trigger))
    parts=[]
    poster=VIDEO_UPLOADS_DIR/(request['source_video']+'.jpg')
    if poster.is_file():
        with Image.open(poster) as image:
            sample=image.convert('RGB')
            stream=io.BytesIO();sample.save(stream,'JPEG',quality=82)
            parts.extend([{'text':'Source video first frame (<Video 1>)'},
                {'inlineData':{'mimeType':'image/jpeg','data':base64.b64encode(stream.getvalue()).decode()}}])
    for index in range(len(request['reference_images'])):
        sample=images[f'ref_image_{index+1}'].copy()
        sample.thumbnail((768,768))
        stream=io.BytesIO();sample.save(stream,'JPEG',quality=82)
        parts.extend([{'text':f'Reference image <Picture {index+1}>'},
            {'inlineData':{'mimeType':'image/jpeg','data':base64.b64encode(stream.getvalue()).decode()}}])
    parts.append({'text':f'Duration: {request["frames"]/24:.2f} seconds. Brief: {prompt}'})
    _record(update, f'Antigravity {model_id}', [{'role':'system','content':system},{'role':'user','content':'\n'.join('[image]' if 'inlineData' in part else part['text'] for part in parts)}])
    update(stage='rewriting',message='Enhancing reference video prompt with Antigravity…',
           rewrite_provider='Antigravity',rewrite_model=model_id)
    result=antigravity.generate_text(model_id,system,parts).strip()
    if not result or len(result)>16000:
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    if trigger and not result.lower().startswith(trigger.lower()):
        result=f'{trigger}. {result}'
    update(rewritten_prompt=result,sent_prompt={'label':'Enhance: rewritten prompt','text':result})
    return result

def enhance_video(prompt, images, request, update):
    model_id=request.get('enhancement_model') or antigravity.selected_model()
    from .prompt_registry import VIDEO, get
    system=get('video:enhance',VIDEO)
    parts=[]
    for name,image in images.items():
        copy=image.copy();copy.thumbnail((1024,1024))
        stream=io.BytesIO();copy.save(stream,'PNG')
        parts.extend([{'text':name.replace('_',' ')},{'inlineData':{'mimeType':'image/png','data':base64.b64encode(stream.getvalue()).decode()}}])
    parts.append({'text':f'Duration: {request["frames"]/24:.2f} seconds. Audio: {request["audio"]}. Brief: {prompt}'})
    _record(update, f'Antigravity {model_id}', [{'role':'system','content':system},{'role':'user','content':'\n'.join('[image]' if 'inlineData' in part else part['text'] for part in parts)}])
    update(stage='rewriting',message='Enhancing video prompt with Antigravity…',rewrite_provider='Antigravity',rewrite_model=model_id)
    text=antigravity.generate_text(model_id,system,parts).strip()
    if not text or len(text)>30000:
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    update(rewritten_prompt=text,sent_prompt={'label':'Enhance: rewritten prompt','text':text})
    return text

LOCAL = 'local'
# Every system prompt with its official source and generation settings (a Settings editor can change this file).
PROMPTS_PATH = APP_DIR / 'reference' / 'prompt-enhancer-prompts.json'
# Qwen's official Qwen-Image-2.1 prompt rewriters (fine-tuned Qwen3.5-VL 9B), Heretic builds without refusals, run in llama.cpp.
LOCAL_MODELS = {'t2i': 'pe-t2i-heretic-q8', 'edit': 'pe-i2i-heretic-q8'}


def prompts():
    from .prompt_registry import apply_enhancer
    return apply_enhancer(json.loads(PROMPTS_PATH.read_text(encoding='utf-8')))


def spec_key(family, images, redraw=False):
    """Which official prompt applies: each architecture's own enhancer where one exists (Krea 2 and MOSS Image
    in their official ComfyUI templates, Z-Image in its official Space), Qwen's PE prompts otherwise."""
    family = family or ''
    if family.startswith('krea-2'):
        return 'krea-2-identity' if images else 'krea-2'
    if family.startswith('moss-image'):
        return 'moss-image'
    if family.startswith('z-image'):
        return 'z-image'
    return 'qwen-image-2.1-edit' if images and not redraw else 'qwen-image-2.1-t2i'


def runner_model(spec, runner):
    """The llama.cpp model for a prompt (Qwen PE T2I runs prompts written for a text encoder), or None for the encoder."""
    if runner == 'encoder':
        return None
    return spec['local_model'] if spec['local_model'] != 'encoder' else LOCAL_MODELS['t2i']


def local_model(images, family='qwen-image-2.1', redraw=False):
    """The llama.cpp model the chosen enhancer needs (Settings > Prompt enhancement), or None."""
    from .enhancer_settings import route
    chosen = route(family, images, redraw)
    return runner_model(prompts()[chosen['spec']], chosen['runner']) if chosen['mode'] == 'local' else None


def _record(update, title, chat):
    """The rewriter's input, message by message, for the result's "Sent to the model" details (images as [image])."""
    for message in chat:
        content = message['content']
        text = content if isinstance(content, str) else '\n'.join('[image]' if part.get('type') == 'image_url' else part.get('text', '') for part in content)
        update(sent_prompt={'label': f"Enhance with {title}: {message['role']} message", 'text': text})


def _finish(parsed, prompt, update):
    if not parsed['parse_ok'] or not parsed['positive_prompt'].strip():
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    if len(parsed['positive_prompt']) > 30000:
        raise RuntimeError('Prompt enhancement returned too much text. Try a shorter request.')
    # The image model's alpha instruction must survive rewriting verbatim.
    if re.search(r'RGBA image with transparency|alpha channel', prompt, re.I):
        value = parsed['positive_prompt']
        if not value.startswith('This is an RGBA image with transparency.'):
            value = 'This is an RGBA image with transparency. ' + value
        if 'The image has alpha channel and the background is transparent.' not in value:
            value += ' The image has alpha channel and the background is transparent.'
        parsed['positive_prompt'] = value
    update(rewritten_prompt=parsed['positive_prompt'], rewrite_ratio=parsed['wh_ratio'], ratio_follow=parsed['ratio_follow'],
           sent_prompt={'label': 'Enhance: rewritten prompt', 'text': parsed['positive_prompt']})
    return parsed


def parse(spec, text, task):
    """The answer after any thinking block: Qwen's JSON contract, or the plain prompt the other templates ask for."""
    sys.path.insert(0, str(APP_DIR / 'reference'))
    import pe_core
    _, answer = pe_core.split_thinking(text)
    if spec['output'] == 'caption_json':  # MOSS skill: {"caption": "..."}
        try:
            answer = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip()))['caption']
        except (ValueError, KeyError, TypeError):
            pass
    if spec['output'] == 'json' or answer.lstrip().startswith('{'):
        parsed = pe_core.parse_answer(answer, pe_core.get_profile(task))
        if parsed['parse_ok'] or spec['output'] == 'json':
            return parsed
    return {'positive_prompt': answer.strip().strip('"'), 'wh_ratio': '', 'ratio_follow': '', 'parse_ok': bool(answer.strip())}


def messages(spec, prompt, images=()):
    """The chat as the official source builds it: a system message, a prefix joined to the request
    (Krea's StringConcatenate), or a template with {prompt} (Z-Image's pe.py); images come before the text."""
    if spec['role'] == 'user_template':
        text = spec['text'].replace('{prompt}', prompt)
    elif spec['role'] == 'user_prefix':
        text = spec['text'] + prompt
    else:
        text = prompt
    parts = []
    for image in images:
        stream = io.BytesIO(); image.convert('RGB').save(stream, 'PNG')
        parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}})
    content = parts + [{'type': 'text', 'text': text}] if parts else text
    return ([{'role': 'system', 'content': spec['text']}] if spec['role'] == 'system' else []) + [{'role': 'user', 'content': content}]


def _free_comfy_gpu():
    """llama.cpp runs in its own process; ComfyUI does not see its memory, so resident image weights leave first."""
    if 'comfy.model_management' in sys.modules:
        memory = sys.modules['comfy.model_management']
        memory.unload_all_models()
        memory.soft_empty_cache()


def _encoder_rewrite(spec, prompt, encoder_id, moss, seed, update):
    """Krea 2 and MOSS Image: their official ComfyUI templates run TextGenerate on the image model's own text encoder."""
    _prepare_comfy(moss, update)
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as memory
    from comfy_extras.nodes_textgen import TextGenerate
    from . import sent_prompts
    sent_prompts.install(comfy.sd)
    from .models import entry, local_path
    item = entry(encoder_id)
    values, metadata = comfy.utils.load_safetensors(str(local_path(item, item['files'][0])))
    for tensor in values.values():
        storage = tensor.untyped_storage()
        if hasattr(storage, '_comfy_tensor_file_slice'):
            delattr(storage, '_comfy_tensor_file_slice')
    values, _ = comfy.utils.convert_old_quants(values, model_prefix='', metadata=metadata)
    clip_type = comfy.sd.CLIPType.MOSS_IMAGE if moss else comfy.sd.CLIPType.KREA2
    clip = comfy.sd.load_text_encoder_state_dicts([values], clip_type=clip_type, model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
    del values
    settings = spec['settings']
    sampling = {'sampling_mode': 'on', 'temperature': settings['temperature'], 'top_k': settings['top_k'], 'top_p': settings['top_p'],
                'min_p': settings['min_p'], 'repetition_penalty': settings['repetition_penalty'], 'seed': int(seed) & 0x7fffffff,
                'presence_penalty': settings['presence_penalty']}
    options = {'system_prompt': spec['text']} if spec['role'] == 'system' else {}
    text = spec['text'] + prompt if spec['role'] == 'user_prefix' else prompt
    try:
        with torch.inference_mode():
            result = TextGenerate.execute(clip, text, settings['max_tokens'], sampling, thinking=settings['thinking'],
                                          use_default_template=True, mtp='auto' if moss else 'off', **options)[0]
        sent_prompts.flush(update, "Enhance with the image model's text encoder: full input")
        return result
    finally:
        memory.unload_model_and_clones(clip.patcher)
        memory.soft_empty_cache()


def _prepare_comfy(moss, update):
    """The same ComfyUI setup the image engines run, done first: the rewriter runs before the engine.
    MOSS Image runs on the pinned FLUX.3 ComfyUI copy, every other image model on the vendored one."""
    import logging
    import torch
    if moss:
        from .flux3_runtime import activate
        activate(update)
    else:
        comfy_path = str(APP_DIR / 'vendor' / 'ComfyUI')
        if comfy_path not in sys.path:
            sys.path.insert(0, comfy_path)
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload = True
    args.disable_pinned_memory = True
    args.disable_dynamic_vram = True
    from .gpu_memory import configure_attention, configure_video_memory
    configure_attention(args)
    configure_video_memory(torch, args, update)
    if not logging.getLogger().handlers:
        logging.basicConfig(stream=sys.stdout, format='%(message)s')
    logging.getLogger().setLevel(logging.INFO)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')


def enhance_local(prompt, images, update=lambda **_: None, family='qwen-image-2.1', redraw=False, seed=0, encoder_id=None, spec=None, runner=None):
    """Rewrite on this computer with each architecture's official prompt and settings (prompt-enhancer-prompts.json)."""
    import time
    from .models import entry, local_path
    key = spec or spec_key(family, images, redraw)
    spec = prompts()[key]
    settings = spec['settings']
    task = 'edit' if key == 'qwen-image-2.1-edit' else 't2i'
    started = time.perf_counter()
    runner = runner or ('encoder' if spec['local_model'] == 'encoder' else 'llama')
    if runner == 'encoder':
        update(stage='rewriting', message='Enhancing prompt on this computer…', rewrite_provider='Local', rewrite_model=entry(encoder_id)['title'], rewrite_prompt=key)
        output = _encoder_rewrite(spec, prompt, encoder_id, key.startswith('moss-image'), seed, update)
        usage = {}
    else:
        from .llama_runtime import Server
        item = entry(runner_model(spec, runner))
        update(stage='rewriting', message='Enhancing prompt on this computer…', rewrite_provider='Local', rewrite_model=item['title'], rewrite_prompt=key)
        paths = [local_path(item, file) for file in item['files']]
        _free_comfy_gpu()
        # Z-Image's template takes text only; Qwen's edit rewriter sees the input images through its vision projector.
        pictures = images if key == 'qwen-image-2.1-edit' else []
        chat = messages(spec, prompt, pictures)
        _record(update, item['title'], chat)
        with Server(paths[0], paths[1] if len(paths) > 1 and pictures else None) as server:
            body = {'messages': chat, 'temperature': settings['temperature'], 'top_p': settings['top_p'],
                    'top_k': settings['top_k'], 'min_p': settings['min_p'], 'presence_penalty': settings['presence_penalty'],
                    'repeat_penalty': settings['repetition_penalty'], 'max_tokens': settings['max_tokens'], 'seed': int(seed) & 0x7fffffff,
                    'chat_template_kwargs': {'enable_thinking': settings['thinking']}}
            reply = server.chat(body)
        message = reply['choices'][0]['message']
        reasoning = message.get('reasoning_content') or ''
        output = (f'<think>\n{reasoning}\n</think>\n' if reasoning else '') + (message.get('content') or '')
        usage = {'rewrite_tokens': reply.get('usage', {}).get('completion_tokens'),
                 'rewrite_tokens_per_second': round((reply.get('timings') or {}).get('predicted_per_second', 0), 1)}
    update(rewrite_seconds=round(time.perf_counter() - started, 3), **usage)
    return _finish(parse(spec, output, task), prompt, update)


def enhance(prompt, images, model_id, update=lambda **_: None, redraw=False, family='qwen-image-2.1', spec=None):
    """Antigravity (Gemini) with the same official prompts. redraw: Z-Image redraws image 1 (image to image)."""
    key = spec or spec_key(family, images, redraw)
    data = prompts()
    spec = data[key]
    task = 'edit' if key == 'qwen-image-2.1-edit' else 't2i'
    chat = messages(spec, prompt)
    system = spec['text'] if spec['role'] == 'system' else ''
    if spec['output'] == 'json':
        system += '\n' + data['cloud-json-reminder']['text']
    parts = []
    if key == 'qwen-image-2.1-edit':
        for index, image in enumerate(images, 1):
            copy = image.convert('RGBA')
            copy.thumbnail((1024, 1024))
            stream = io.BytesIO(); copy.save(stream, 'PNG')
            parts.extend([{'text': f'<image{index}>'}, {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(stream.getvalue()).decode()}}])
    parts.append({'text': chat[-1]['content']})
    _record(update, f'Antigravity {model_id}', ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': '\n'.join('[image]' if 'inlineData' in part else part['text'] for part in parts)}])
    update(stage='rewriting', message='Enhancing prompt with Antigravity…', rewrite_provider='Antigravity', rewrite_model=model_id, rewrite_prompt=key)
    text = antigravity.generate_text(model_id, system, parts)
    return _finish(parse(spec, text, task), prompt, update)

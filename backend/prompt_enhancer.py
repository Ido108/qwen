"""Prompt enhancement on this computer with each architecture's official rewrite prompt and output contract."""
import base64
import json
import io
import re
import sys
from .common import APP_DIR, VIDEO_UPLOADS_DIR
from . import enhance_cache


def enhance_reference(prompt, images, request, update):
    """Rewrite that preserves the Ref2VA trigger and reference order."""
    from PIL import Image
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
    result=video_text(system,parts,update,'Enhancing reference video prompt…').strip()
    if not result or len(result)>16000:
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    if trigger and not result.lower().startswith(trigger.lower()):
        result=f'{trigger}. {result}'
    update(rewritten_prompt=result,sent_prompt={'label':'Enhance: rewritten prompt','text':result})
    return result

def enhance_video(prompt, images, request, update):
    from .prompt_registry import VIDEO, get
    system=get('video:enhance',VIDEO)
    parts=[]
    for name,image in images.items():
        copy=image.copy();copy.thumbnail((1024,1024))
        stream=io.BytesIO();copy.save(stream,'PNG')
        parts.extend([{'text':name.replace('_',' ')},{'inlineData':{'mimeType':'image/png','data':base64.b64encode(stream.getvalue()).decode()}}])
    parts.append({'text':f'Duration: {request["frames"]/24:.2f} seconds. Audio: {request["audio"]}. Brief: {prompt}'})
    text=video_text(system,parts,update,'Enhancing video prompt…').strip()
    if not text or len(text)>30000:
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    update(rewritten_prompt=text,sent_prompt={'label':'Enhance: rewritten prompt','text':text})
    return text

# Video prompts on this computer: Gemma 4 12B (Heretic, uncensored), the model family LTX 2.5 uses as its text encoder
# and official enhancer, with its own vision projector. No thinking pass, as in the official LTX node; sampling from
# the official LTX TextGenerate node defaults.
GEMMA_MODEL = 'gemma4-12b-heretic-q6'


def video_models():
    """(model id, vision model id) for video prompts: Gemma 4 by default, Qwen PE when chosen in Settings > Prompts."""
    from .enhancer_settings import choice
    return ('pe-t2i-heretic-q6', 'pe-i2i-heretic-q6') if choice('video') == 'qwen-pe' else (GEMMA_MODEL, GEMMA_MODEL)
VIDEO_SETTINGS = {'temperature': 0.7, 'top_k': 64, 'top_p': 0.95, 'min_p': 0.05, 'repeat_penalty': 1.05, 'max_tokens': 1024}


def video_text(system, parts, update, message):
    """A video rewrite on this computer. parts: text and inlineData pictures in order; they carry the pictures,
    duration and brief, so they also key the saved answer."""
    from .models import entry, local_path
    from .llama_runtime import Server
    model_id, vision_id = video_models()
    model = entry(model_id)
    update(stage='rewriting', message=message, rewrite_provider='Local', rewrite_model=model['title'])
    content = [{'type': 'image_url', 'image_url': {'url': f"data:{part['inlineData']['mimeType']};base64,{part['inlineData']['data']}"}}
               if 'inlineData' in part else {'type': 'text', 'text': part['text']} for part in parts]
    chat = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
    _record(update, model['title'], [chat[0], {'role': 'user', 'content': _parts_text(parts)}])
    name = enhance_cache.key('local-video', model_id, VIDEO_SETTINGS, system, parts)
    saved = enhance_cache.get(name)
    if saved:
        update(enhance_cached=True, message='Using the saved enhanced prompt (same prompt, images and enhancer)…')
        return saved
    import time
    started = time.perf_counter()
    pictures = any('inlineData' in part for part in parts)
    vision = entry(vision_id)
    _free_comfy_gpu()
    with Server(local_path(model, model['files'][0]), local_path(vision, vision['files'][1]) if pictures else None) as server:
        reply = server.chat({'messages': chat, **VIDEO_SETTINGS, 'chat_template_kwargs': {'enable_thinking': False}})
    text = re.sub(r'<think>.*?(?:</think>|$)', '', reply['choices'][0]['message'].get('content') or '', flags=re.S).strip()
    update(rewrite_seconds=round(time.perf_counter() - started, 3), rewrite_tokens=reply.get('usage', {}).get('completion_tokens'))
    if text:
        enhance_cache.put(name, text)
    return text


def _parts_text(parts):
    return '\n'.join('[image]' if 'inlineData' in part else part['text'] for part in parts)

LOCAL = 'local'
FAST_OPENING = '{"rewritten_prompt": "'
CJK = re.compile(r'[一-鿿]')
# Shot sizes a user can ask for (English and Hebrew), and the phrases of a close crop.
SHOTS = [('full body', re.compile(r'full[- ]?(body|length)|head[- ]to[- ](toe|feet)|גוף מלא', re.I)),
         ('wide shot', re.compile(r'wide[- ]shot|long[- ]shot|וייד', re.I)),
         ('medium shot', re.compile(r'medium[- ]shot|mid[- ]shot|cowboy[- ]shot|waist[- ]up|knees?[- ]up|מדיום', re.I))]
CLOSE = re.compile(r'close[- ]?up|chest[- ]up|head[- ]and[- ]shoulders|קלוז', re.I)
# An answer that is planning notes instead of the prompt (Krea's template on Qwen PE without thinking, 1 of 3 seeds
# for a Hebrew request, 2026-10-22).
PLANNING = re.compile(r'\s*#*\s*(reasoning|thinking|analysis)\b', re.I)


def asked_shot(prompt):
    """The shot size the request names, unless it also asks for a close-up."""
    if CLOSE.search(prompt):
        return None
    return next((name for name, pattern in SHOTS if pattern.search(prompt)), None)
# Every system prompt with its official source and generation settings (a Settings editor can change this file).
PROMPTS_PATH = APP_DIR / 'reference' / 'prompt-enhancer-prompts.json'
# Qwen's official Qwen-Image-2.1 prompt rewriters (fine-tuned Qwen3.5-VL 9B), Heretic builds without refusals, run in llama.cpp.
LOCAL_MODELS = {'t2i': 'pe-t2i-heretic-q6', 'edit': 'pe-i2i-heretic-q6'}


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
    if runner == 'gemma':
        return GEMMA_MODEL
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


MENTION = re.compile(r'<image\d+>')
MENTION_PHRASE = re.compile(r'\bthe ([\w\- ]{1,40}?) from (<image\d+>)', re.I)


def missing_mentions(prompt, answer):
    """The image references of the request (<image1>) that the rewritten prompt lost."""
    return [ref for ref in dict.fromkeys(MENTION.findall(prompt)) if ref not in answer]


def keep_mentions(prompt, answer):
    """Put back each lost image reference with its label: "Use the woman from <image1>." """
    missing = missing_mentions(prompt, answer)
    if not missing:
        return answer
    phrases = {ref: f'the {label} from {ref}' for label, ref in MENTION_PHRASE.findall(prompt)}
    return answer.rstrip() + ' ' + ' '.join(f'Use {phrases.get(ref, ref)}.' for ref in missing)


def _finish(parsed, prompt, update):
    if not parsed['parse_ok'] or not parsed['positive_prompt'].strip():
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    parsed['positive_prompt'] = keep_mentions(prompt, parsed['positive_prompt'])
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


def _free_vram():
    """Free GPU memory in bytes as the driver reports it (0 where this process has no CUDA)."""
    torch = sys.modules.get('torch')
    try:
        return torch.cuda.mem_get_info()[0] if torch and torch.cuda.is_available() else 0
    except RuntimeError:
        return 0


def _image_model_loaded():
    memory = sys.modules.get('comfy.model_management')
    return bool(memory and getattr(memory, 'current_loaded_models', None))


KEEP_FREE = 8 * 1024**3  # room left for the next encode and sampling when the rewriter stays loaded


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


def enhance_local(prompt, images, update=lambda **_: None, family='qwen-image-2.1', redraw=False, seed=0, encoder_id=None, spec=None, runner=None, fast=False):
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
        from .llama_runtime import is_resident, release, shared
        item = entry(runner_model(spec, runner))
        if item['id'] in LOCAL_MODELS.values() and not key.startswith('qwen-image-2.1'):
            # Another architecture's prompt on Qwen PE: its own model card sampling (presence penalty 1.5). Z-Image's
            # values, set for Qwen3-Max, looped this 9B model until the token cap (18,767 characters, 2026-10-21).
            own = prompts()['qwen-image-2.1-t2i']['settings']
            settings = {**settings, **{name: own[name] for name in ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty')}}
        update(stage='rewriting', message='Enhancing prompt on this computer…', rewrite_provider='Local', rewrite_model=item['title'], rewrite_prompt=key)
        paths = [local_path(item, file) for file in item['files']]
        marks = [time.perf_counter()]
        pictures = images if key == 'qwen-image-2.1-edit' else []
        files = (paths[0], paths[1] if len(paths) > 1 and pictures else None)
        warm = is_resident(*files)
        # Files plus about 5 GB for its context and vision buffers: with 2 GB next to a loaded image model the start took 20 s.
        needed = sum(f.get('bytes') or 0 for f in item['files'][:2 if files[1] else 1]) + 5 * 1024**3
        free_before, loaded_before = _free_vram(), _image_model_loaded()
        if not warm and free_before < needed:
            _free_comfy_gpu()
        marks.append(time.perf_counter())
        # Z-Image's template takes text only; Qwen's edit rewriter sees the input images through its vision projector.
        chat = messages(spec, prompt, pictures)
        # Fast: no thinking pass, and Qwen's answer starts at its JSON contract. Without the opening the edit rewriter
        # still wrote about 4,000 tokens of plain-text analysis first (thinking off alone saved nothing for edits).
        opening = FAST_OPENING if fast and key.startswith('qwen-image-2.1') else ''
        if opening:
            chat = chat + [{'role': 'assistant', 'content': opening}]
        _record(update, item['title'], chat)
        server = shared(*files)
        marks.append(time.perf_counter())
        body = {'messages': chat, 'temperature': settings['temperature'], 'top_p': settings['top_p'],
                'top_k': settings['top_k'], 'min_p': settings['min_p'], 'presence_penalty': settings['presence_penalty'],
                'repeat_penalty': settings['repetition_penalty'], 'max_tokens': settings['max_tokens'], 'seed': int(seed) & 0x7fffffff,
                'chat_template_kwargs': {'enable_thinking': settings['thinking'] and not fast}}
        if not key.startswith('qwen-image-2.1'):
            # DRY stops repeated phrases: with presence penalty alone Z-Image's prompt still looped 2 of 6 times.
            # The cap: Qwen PE did not stop on Z-Image's prompt (21,436 characters); the text encoders read about
            # 512 tokens, so a longer answer only costs time. It is cut at its last full sentence below.
            # llama.cpp counts the request's own words in these penalties, so presence 1.5 and DRY from 2 tokens pushed
            # the answer away from "medium shot" (kept 7 of 12; presence 0 with DRY from 4 tokens: 9 of 12, no loops).
            body.update(presence_penalty=0.0, dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=4, max_tokens=min(settings['max_tokens'], 800))
        reply = server.chat(body)
        if key.startswith('qwen-image-2.1') and not CJK.search(prompt) and len(CJK.findall(reply['choices'][0]['message'].get('content') or '')) > 10:
            # Qwen PE answered some Hebrew edit requests in Chinese (4 of 12, Q6 and Q8 alike, by request, not seed);
            # asking again with this note gave English in 4 of 4 (2026-10-22).
            retry = messages(spec, prompt + ' (Write the rewritten prompt in English.)', pictures) + chat[len(chat) - (1 if opening else 0):]
            reply = server.chat({**body, 'messages': retry})
            update(sent_prompt={'label': 'Enhance: asked again for English (the first answer was in Chinese)', 'text': prompt + ' (Write the rewritten prompt in English.)'})
        if not key.startswith('qwen-image-2.1') and PLANNING.match(reply['choices'][0]['message'].get('content') or ''):
            note = ' (Write only the final image prompt paragraph.)'
            retry = messages(spec, prompt + note, pictures) + chat[len(chat) - (1 if opening else 0):]
            reply = server.chat({**body, 'messages': retry})
            update(sent_prompt={'label': 'Enhance: asked again for the prompt only (the first answer was planning notes)', 'text': prompt + note})
        shot = asked_shot(prompt)
        if shot and CLOSE.search(reply['choices'][0]['message'].get('content') or ''):
            # The request named a shot size and the answer described a close-up (Krea's template on Qwen PE, 2026-10-22).
            note = f' (Framing: {shot}. Do not describe a close-up.)'
            retry = messages(spec, prompt + note, pictures) + chat[len(chat) - (1 if opening else 0):]
            reply = server.chat({**body, 'messages': retry})
            update(sent_prompt={'label': f'Enhance: asked again for a {shot} (the first answer was a close-up)', 'text': prompt + note})
        missing = missing_mentions(prompt, reply['choices'][0]['message'].get('content') or '')
        if missing:
            # The rewriter dropped an image reference ("the woman from <image1>" became "her"), so the image model no
            # longer knows which picture is meant (2026-10-22). Ask once more; _finish adds any still missing.
            note = f" (Keep {', '.join(missing)} exactly as written.)"
            retry = messages(spec, prompt + note, pictures) + chat[len(chat) - (1 if opening else 0):]
            reply = server.chat({**body, 'messages': retry})
            update(sent_prompt={'label': 'Enhance: asked again to keep the image references', 'text': prompt + note})
        marks.append(time.perf_counter())
        # Stay loaded only beside an image model that is already in the GPU, with room for its next encode and steps.
        kept = _image_model_loaded() and _free_vram() >= KEEP_FREE
        if not kept:
            release()
        marks.append(time.perf_counter())
        update(rewrite_parts={'free_gpu': round(marks[1] - marks[0], 2), 'start_model': round(marks[2] - marks[1], 2),
                              'write': round(marks[3] - marks[2], 2), 'stop_model': round(marks[4] - marks[3], 2),
                              'was_loaded': warm, 'kept_loaded': kept, 'free_gb_before': round(free_before / 1024**3, 1),
                              'image_model_loaded_before': loaded_before, 'needed_gb': round(needed / 1024**3, 1)})
        message = reply['choices'][0]['message']
        reasoning = message.get('reasoning_content') or ''
        content = message.get('content') or ''
        if reply['choices'][0].get('finish_reason') == 'length' and not key.startswith('qwen-image-2.1'):
            content = content[:max(content.rfind('. '), content.rfind('.\n')) + 1] or content
        if opening and not content.lstrip().startswith('{'):
            content = opening + content
        output = (f'<think>\n{reasoning}\n</think>\n' if reasoning else '') + content
        usage = {'rewrite_tokens': reply.get('usage', {}).get('completion_tokens'),
                 'rewrite_tokens_per_second': round((reply.get('timings') or {}).get('predicted_per_second', 0), 1)}
    update(rewrite_seconds=round(time.perf_counter() - started, 3), **usage)
    return _finish(parse(spec, output, task), prompt, update)

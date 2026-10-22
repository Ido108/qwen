"""Official FLUX.3 Klein flow-map inference and local prompt upsampling."""
from __future__ import annotations
import gc
import importlib.util
import io
import json
import math
import re
import time
import sys
from .common import APP_DIR
from .models import entry, local_path, required_image
from .image_cache import ConditioningCache, retain_cpu_weights, retain_lora_clone

_components={}
_conditioning=ConditioningCache()


def prompt_identifiers(request,prompt):
    """Keep requested LoRA trigger words as learned concepts, not written text."""
    from .lora_prompts import required_words, contains_word
    return sorted({word for record in required_words(request) for word in record['words']
                   if contains_word(prompt, word)})


def prompt_without_identifiers(prompt,words):
    # The caption model can mistake an invented trigger for a brand name, even
    # with instructions. Keep those tokens outside it and restore them only in
    # the image-model caption. Keep a trigger used as explicitly quoted text.
    for word in sorted(words,key=len,reverse=True):
        if re.search(r'["\u201c\u201d\u0027]'+re.escape(word)+r'["\u201c\u201d\u0027]',prompt,re.I):continue
        prompt=re.sub(r'(?<!\w)'+re.escape(word)+r'(?!\w)','',prompt,flags=re.I)
    return prompt.strip(' ,.;') or 'Create an image in the selected style.'


def prompt_reference(image,count):
    """black-forest-labs/flux-3-image-klein pup.policy_reference: RGB over white, LANCZOS to the
    upsampler's pixel budget for this many references (1 MP, 724 px square, then 512 px square), JPEG q92.
    Only the prompt model sees this copy; the image model keeps the full reference."""
    from PIL import Image
    if image.mode in ('RGBA','LA','PA') or (image.mode=='P' and 'transparency' in image.info):
        rgba=image.convert('RGBA')
        image=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba)
    image=image.convert('RGB')
    cap={1:1024*1024,2:724*724}.get(count,512*512)
    floor=lambda v:max(32,int(math.floor(v/32))*32)
    w,h=max(32,round(image.width/32)*32),max(32,round(image.height/32)*32)
    if w*h>cap:
        beta=math.sqrt(image.width*image.height/cap)
        w,h=floor(image.width/beta),floor(image.height/beta)
    if w*h>cap:
        if w==32:h=floor(cap/w)
        else:w=floor(cap/h)
    if image.size!=(w,h):image=image.resize((w,h),Image.Resampling.LANCZOS)
    buffer=io.BytesIO()
    image.save(buffer,format='JPEG',quality=92)
    return Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')


def prompt_sampling(images,seed):
    """Plain greedy decoding. The official workflows sample with top_k 1 and presence penalty 1.5 (t2i); on this
    runtime (no CUDA graphs, see the aimdo note) that path and MTP drafting both measured slower on an RTX 5090:
    greedy 28 tokens/s, top_k 1 + penalty + MTP 18-20, greedy + MTP 11-21 (2026-10-21)."""
    return {'sampling_mode':'off'}


def canvas_line(width,height):
    """The optional first line the official t2i system prompt reads; a square canvas needs none."""
    if width==height:return ''
    d=math.gcd(width,height)
    return f'TARGET CANVAS: {width//d}:{height//d}\n'


class PromptImages:
    """Pass each reference separately, preserving differing aspect ratios."""
    def __init__(self,clip,images):self.clip,self.images=clip,images
    def __getattr__(self,name):return getattr(self.clip,name)
    def tokenize(self,text,**kwargs):
        kwargs.pop('image',None)
        return self.clip.tokenize(text,images=self.images,**kwargs)


def generate(request,prompt,images,width,height,progress):
    from .flux3_runtime import activate
    activate(progress)
    import torch
    import numpy as np
    from PIL import Image
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload=True
    args.disable_pinned_memory=True
    args.disable_dynamic_vram=True
    from .gpu_memory import configure_video_memory
    configure_video_memory(torch,args,progress)
    # ComfyUI reports weight placement ("loaded partially ... offloaded") at INFO level.
    import logging
    if not logging.getLogger().handlers:logging.basicConfig(stream=sys.stdout,format='%(message)s')
    logging.getLogger().setLevel(logging.INFO)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():raise RuntimeError('Could not initialize the model file reader.')
    import comfy.sd
    from . import sent_prompts
    sent_prompts.install(comfy.sd)  # the exact encoder input, for the result details
    import comfy.sample
    import comfy.utils
    import comfy.model_base
    import comfy.model_management as memory
    from comfy_extras.nodes_textgen import TextGenerate
    from comfy_extras.nodes_edit_model import ReferenceMultiLatent
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_extras.nodes_model_advanced import ModelSamplingFlux
    memory.EXTRA_RESERVED_VRAM=args.reserve_vram*1024**3
    torch.cuda.reset_peak_memory_stats()
    selected=entry(request['model_id'])
    from .res4lyf_runtime import ensure_sampler
    ensure_sampler(request.get('sampler') or selected['generation_defaults'].get('sampler', 'euler'))
    model_id,encoder_id,vae_id=required_image(request)[:3]
    key=(model_id,encoder_id,vae_id)
    if _components and _components['key']!=key:raise RuntimeError('Start a new image worker when changing FLUX.3 models.')
    reused=bool(_components)
    progress(model_reused=reused,engine='ComfyUI FLUX.3',selected_model=selected['title'],text_encoder_id=encoder_id,vae_id=vae_id)

    def weights(model_id):
        item=entry(model_id)
        values,metadata=comfy.utils.load_safetensors(str(local_path(item,item['files'][0])))
        for tensor in values.values():
            storage=tensor.untyped_storage()
            if hasattr(storage,'_comfy_tensor_file_slice'):delattr(storage,'_comfy_tensor_file_slice')
        return values,metadata

    def load_clip(model_id):
        values,metadata=weights(model_id)
        values,_=comfy.utils.convert_old_quants(values,model_prefix='',metadata=metadata)
        clip=comfy.sd.load_text_encoder_state_dicts([values],clip_type=comfy.sd.CLIPType.FLUX3,
            model_options={'initial_device':torch.device('cpu')},disable_dynamic=True)
        retain_cpu_weights(clip.patcher)
        return clip

    defaults=selected['generation_defaults']
    cfg=float(request.get('cfg') or defaults['cfg'])
    tensors={f'image{i+1}':torch.from_numpy(np.array(im.convert('RGB')).astype(np.float32)/255).unsqueeze(0)
             for i,im in enumerate(images)}
    identifiers=prompt_identifiers(request,prompt)
    canvas='' if images else canvas_line(width,height)
    condition_key=_conditioning.fingerprint('flux3|'+canvas+prompt+'|'+json.dumps(identifiers),request.get('negative_prompt','')+'|'+str(cfg),images)
    cached=_conditioning.get(condition_key)
    if not reused:
        progress(stage='loading_prompt_model',message='Loading the local prompt model…')
        # The official workflows run the INT8 prompt model; W4A8 stays the smaller fallback.
        from .models import is_installed
        prompt_model='flux3-prompt-int8' if is_installed('flux3-prompt-int8') else 'flux3-prompt'
        pup=load_clip(prompt_model)
        _components.update(key=key,pup=pup,prompt_model=prompt_model)
    progress(prompt_model=_components['prompt_model'])
    if cached is None:
        progress(stage='rewriting',message='Preparing the image description on this computer…')
        from .prompt_registry import apply_flux3
        system=apply_flux3(json.loads((APP_DIR/'reference/flux3-system-prompts.json').read_text(encoding='utf-8')))
        caption_request=prompt_without_identifiers(prompt,identifiers)
        text=('Reference images follow in order (the first image, the second image, ...).\nUser request: '+caption_request+'\nOutput the caption.') if images else canvas+caption_request
        references=[torch.from_numpy(np.array(prompt_reference(im,len(images))).astype(np.float32)/255).unsqueeze(0) for im in images]
        with torch.inference_mode():
            # MTP drafts tokens with the checkpoint's own head; at top_k 1 the accepted text is the greedy text.
            expanded=TextGenerate.execute(PromptImages(_components['pup'],references),text,3584,prompt_sampling(images,int(request['seed'])),
                thinking=False,use_default_template=True,mtp='off',
                system_prompt=system['i2i' if images else 't2i'])[0].strip()
        if not expanded:raise RuntimeError('The local prompt model returned no image description. Try again.')
        missing=[word for word in identifiers if word.casefold() not in expanded.casefold()]
        if missing:expanded=' '.join(missing)+'. '+expanded
        from .lora_prompts import internal_prompt
        expanded, _ = internal_prompt(expanded, request)
    else:expanded=cached[2]
    sent_prompts.flush(progress,'FLUX.3 prompt model (BFL upsampler): full input',key=('flux3-pup',condition_key))
    progress(rewritten_prompt=expanded,expanded_prompt=expanded,conditioning_reused=cached is not None,
             sent_prompt={'label':'Description written by the FLUX.3 prompt model','text':expanded})
    if 'clip' not in _components:
        progress(stage='loading_encoder',message='Loading the image text encoder…')
        clip=load_clip(encoder_id)
        values,metadata=weights(vae_id)
        vae=comfy.sd.VAE(sd=values,metadata=metadata)
        del values
        retain_cpu_weights(vae.patcher)
        _components.update(clip=clip,vae=vae)
    clip,vae=_components['clip'],_components['vae']
    progress(stage='encoding',message='Using saved prompt and image data.' if cached else 'Encoding the description and reference images…')
    with torch.inference_mode():
        if cached is None:
            positive=clip.encode_from_tokens_scheduled(clip.tokenize(expanded))
            negative=clip.encode_from_tokens_scheduled(clip.tokenize(request.get('negative_prompt',''))) if cfg>1 else positive
            if tensors:
                positive,negative,_=ReferenceMultiLatent.execute(positive,{'resolution':'resize','megapixels':1.0},
                    negative=negative,vae=vae,images=tensors)
            _conditioning.put(condition_key,(positive,negative,expanded))
        else:positive,negative,_=cached
    progress(stage='loading_model',message='Using the loaded image model.' if 'model' in _components else 'Loading FLUX.3 image model…')
    if 'model' not in _components:
        options={}
        if selected['format']=='GGUF':
            if 'studio_gguf' not in sys.modules:
                folder=APP_DIR/'vendor/ComfyUI-GGUF'
                spec=importlib.util.spec_from_file_location('studio_gguf',folder/'__init__.py',submodule_search_locations=[str(folder)])
                module=importlib.util.module_from_spec(spec);sys.modules['studio_gguf']=module;spec.loader.exec_module(module)
            from studio_gguf import loader as gguf_loader
            from studio_gguf.dequant import dequantize_tensor,is_quantized
            from studio_gguf.ops import GGMLOps
            from studio_gguf.nodes import GGUFModelPatcher
            filename=local_path(selected,selected['files'][0])
            from .civitai import inspect
            if inspect(filename)['arch']!='flux3-klein':raise RuntimeError('This GGUF file is not a FLUX.3 image model.')
            # This vendor loader predates FLUX.3. Its tensor loader is generic;
            # admit the file's architecture only after checking its tensor layout.
            reader=gguf_loader.gguf.GGUFReader(str(filename))
            architecture=gguf_loader.get_field(reader,'general.architecture',str)
            del reader
            if architecture:gguf_loader.IMG_ARCH_LIST.add(architecture)
            values,extra=gguf_loader.gguf_sd_loader(str(filename))
            metadata=extra.get('metadata',{})
            for name,value in list(values.items()):
                if len(value.shape)==1 and is_quantized(value):values[name]=dequantize_tensor(value,dtype=torch.float32)
            operations=GGMLOps();operations.Linear.dequant_dtype='target'
            options={'custom_operations':operations,'dtype':torch.bfloat16}
        else:values,metadata=weights(model_id)
        model=comfy.sd.load_diffusion_model_state_dict(values,model_options=options,metadata=metadata,disable_dynamic=True)
        if selected['format']=='GGUF' and model is not None:model=GGUFModelPatcher.clone(model)
        del values
        if model is None or not isinstance(model.model,comfy.model_base.Flux3):
            raise RuntimeError('This file is not a supported FLUX.3 image model.')
        retain_cpu_weights(model)
        _components['model']=model
        gc.collect()
    model=_components['model']
    applied=[]
    lora_checks=[]
    for choice in request.get('loras') or []:
        item=entry(choice['id']);strength=float(choice.get('strength',1))
        from .lora_loading import read_lora_weights, load_model_lora
        lora=read_lora_weights(local_path(item,item['files'][0]))
        # Quantized bases keep their weights: a merged LoRA would be requantized and mostly lost.
        model,check=load_model_lora(model,lora,strength,item['title'],runtime=selected['format'] not in ('BF16','FP16','FP32','GGUF'))
        lora_checks.append({'id':item['id'],'title':item['title'],**check})
        applied.append(f"{item['title']} ({strength:g})")
    if applied:retain_lora_clone(model,_components['model'])
    progress(lora_checks=lora_checks)
    alpha=float(defaults.get('alpha',16.0))
    model=ModelSamplingFlux().patch(model,math.log(alpha),math.log(alpha),width,height)[0]
    retain_lora_clone(model,_components['model'])
    # Preserve the flow-map model's target-time conditioning. The pinned Comfy
    # sampler supplies the target time to the flow-map DiT.
    latent=EmptyFlux2LatentImage.execute(width,height,1)[0]['samples']
    noise=comfy.sample.prepare_noise(latent,request['seed'])
    steps=int(request['steps']);sampler=request.get('sampler') or defaults.get('sampler','euler');scheduler=request.get('scheduler') or defaults.get('scheduler','simple')
    from .image_preview import LiveImagePreview
    preview=LiveImagePreview(model.model.latent_format,progress)
    def callback(step,x0,x,total):
        progress(stage='generating',message='Generating image…',step=step+1,total_steps=total)
        preview.publish(x0,step,total)
    progress(stage='generating',message='Generating image…',step=0,cfg=cfg,alpha=alpha,sampling=f'{sampler}/{scheduler}',sampler=sampler,scheduler=scheduler)
    start=time.perf_counter()
    with torch.inference_mode():
        samples=comfy.sample.sample(model,noise,steps,cfg,sampler,scheduler,positive,negative,latent,
            callback=callback,seed=request['seed'])
        progress(stage='decoding',message='Decoding image…',sampling_seconds=round(time.perf_counter()-start,3))
        pixels=vae.decode(samples)
        array=(pixels[0].float().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated()/1024**3,2),true_cfg_scale=cfg,applied_lora=', '.join(applied) or None)
    return Image.fromarray(array[...,:3]).convert('RGBA')

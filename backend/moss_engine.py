"""MOSS Image (OpenMOSS, 5B single-stream DiT) text to image on the pinned FLUX.3 ComfyUI runtime.

That runtime (Comfy-Org/ComfyUI 6607f36, see flux3_runtime) has native MOSS-Image support: comfy/ldm/moss_image,
model_base.MOSSImage, CLIPType.MOSS_IMAGE (stock Qwen3.5 9B, last hidden state) and the FLUX.2 latent format.
The official ComfyUI template (Comfy-Org/workflow_templates templates/image_moss_image_t2i.json) is
UNETLoader + CLIPLoader(type moss_image) + VAELoader + EmptyFlux2LatentImage + KSampler; the model's own shift is 5.
"""
from __future__ import annotations
import gc
import importlib.util
import json
import sys
import time
from .common import APP_DIR
from .models import entry, local_path, required_image
from .image_cache import ConditioningCache, retain_cpu_weights, retain_lora_clone

_components={}
_conditioning=ConditioningCache()
DEFAULT_SHIFT=5.0  # supported_models.MOSSImage sampling_settings


def moss_denoise(request,images):
    """1 without an input image; with one, the user's image strength (denoise) or 0.7, as Z-Image's image to image."""
    return float(request.get('denoise') or 0.7) if images else 1.0


def generate(request,prompt,images,width,height,progress):
    # MOSS has no reference-image input (OpenMOSS; pleiades7/MOSS-Image-CharRef-PoC: "a text-to-image model with no
    # reference-image input"). An input image is redrawn the standard way: VAE encode, then partial denoise.
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
    from .gpu_memory import configure_attention, configure_video_memory
    attention=configure_attention(args)
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
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    memory.EXTRA_RESERVED_VRAM=args.reserve_vram*1024**3
    torch.cuda.reset_peak_memory_stats()
    selected=entry(request['model_id'])
    defaults=selected['generation_defaults']
    sampler=request.get('sampler') or defaults.get('sampler','euler');scheduler=request.get('scheduler') or defaults.get('scheduler','simple')
    from .res4lyf_runtime import ensure_sampler
    ensure_sampler(sampler)
    model_id,encoder_id,vae_id=required_image(request)[:3]
    key=(model_id,encoder_id,vae_id)
    if _components and _components['key'][1:]!=key[1:]:raise RuntimeError('Start a new image worker when changing the MOSS Image encoder or decoder.')
    if _components and _components['key']!=key:
        # Another MOSS checkpoint (SFT, Base, Turbo) with the same encoder and decoder: keep them and the saved prompt data.
        memory.unload_model_and_clones(_components.pop('model'))
        _components['key']=key
        gc.collect();memory.soft_empty_cache()
    reused=bool(_components)
    if not reused:_components['key']=key
    # A BF16 model runs from its INT8 ConvRot copy when there is one (int8_copies); a loaded model keeps what it loaded.
    from .int8_copies import active as int8_copy
    copy=_components.get('int8_copy') if 'model' in _components else int8_copy(selected)
    if copy:progress(int8_copy=str(copy))
    progress(model_reused=reused,engine='ComfyUI MOSS Image',selected_model=selected['title'],text_encoder_id=encoder_id,vae_id=vae_id,attention=attention)

    def weights(item_id,path=None):
        item=entry(item_id)
        values,metadata=comfy.utils.load_safetensors(str(path or local_path(item,item['files'][0])))
        for tensor in values.values():
            storage=tensor.untyped_storage()
            if hasattr(storage,'_comfy_tensor_file_slice'):delattr(storage,'_comfy_tensor_file_slice')
        return values,metadata

    cfg=float(request.get('cfg') or defaults['cfg'])
    # A model can name its own negative (MOSS Serenity: "bad ugly blurry"); one the user typed wins.
    negative_text=request.get('negative_prompt','').strip() or defaults.get('negative_prompt','')
    if negative_text and cfg>1:progress(negative_prompt_used=negative_text)
    condition_key=_conditioning.fingerprint('moss|'+prompt,negative_text+'|'+str(cfg>1),[])
    cached=_conditioning.get(condition_key)
    if 'clip' not in _components:
        progress(stage='loading_encoder',message='Loading the MOSS Image text encoder and decoder…')
        started=time.perf_counter()
        from .file_prefetch import prefetch
        model_files=([copy] if copy else [local_path(selected,f) for f in selected['files']]) if 'model' not in _components else []
        prefetch(local_path(entry(encoder_id),entry(encoder_id)['files'][0]),*model_files)
        values,metadata=weights(encoder_id)
        values,_=comfy.utils.convert_old_quants(values,model_prefix='',metadata=metadata)
        clip=comfy.sd.load_text_encoder_state_dicts([values],clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
            model_options={'initial_device':torch.device('cpu')},disable_dynamic=True)
        del values
        retain_cpu_weights(clip.patcher)
        values,metadata=weights(vae_id)
        vae=comfy.sd.VAE(sd=values,metadata=metadata)
        del values
        retain_cpu_weights(vae.patcher)
        _components.update(clip=clip,vae=vae)
        progress(encoder_load_seconds=round(time.perf_counter()-started,3))
    clip,vae=_components['clip'],_components['vae']
    progress(stage='encoding',message='Using saved prompt data.' if cached else 'Encoding the prompt…',conditioning_reused=cached is not None)
    started=time.perf_counter()
    with torch.inference_mode():
        if cached is None:
            # MOSSImageTokenizer applies the official chat template (user turn, open think block).
            positive=clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            negative=clip.encode_from_tokens_scheduled(clip.tokenize(negative_text)) if cfg>1 else positive
            _conditioning.put(condition_key,(positive,negative))
        else:positive,negative=cached
    progress(encoding_seconds=round(time.perf_counter()-started,3))
    progress(stage='loading_model',message='Using the loaded image model.' if 'model' in _components else 'Loading MOSS Image model…')
    started=time.perf_counter()
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
            if inspect(filename)['arch']!='moss-image':raise RuntimeError('This GGUF file is not a MOSS Image model.')
            # The vendored loader predates MOSS Image; its tensor loader is generic. Admit the file's
            # architecture only after the tensor layout check above.
            reader=gguf_loader.gguf.GGUFReader(str(filename))
            architecture=gguf_loader.get_field(reader,'general.architecture',str)
            del reader
            if not architecture:raise RuntimeError('This GGUF file has no architecture name. Use a ComfyUI GGUF (for example realrebelai/MOSS-Image_GGUFs).')
            gguf_loader.IMG_ARCH_LIST.add(architecture)
            values,extra=gguf_loader.gguf_sd_loader(str(filename))
            metadata=extra.get('metadata',{})
            for name,value in list(values.items()):
                if len(value.shape)==1 and is_quantized(value):values[name]=dequantize_tensor(value,dtype=torch.float32)
            operations=GGMLOps();operations.Linear.dequant_dtype='target'
            options={'custom_operations':operations,'dtype':torch.bfloat16}
        else:values,metadata=weights(model_id,copy)
        model=comfy.sd.load_diffusion_model_state_dict(values,model_options=options,metadata=metadata,disable_dynamic=True)
        if selected['format']=='GGUF' and model is not None:model=GGUFModelPatcher.clone(model)
        del values
        if model is None or not isinstance(model.model,comfy.model_base.MOSSImage):
            raise RuntimeError('This file is not a supported MOSS Image model.')
        retain_cpu_weights(model)
        _components['model']=model
        _components['int8_copy']=copy
        gc.collect()
    progress(diffusion_load_seconds=round(time.perf_counter()-started,3))
    model=_components['model']
    applied,lora_checks=[],[]
    # A Turbo entry built from a speed LoRA (MOSS Serenity Turbo: the DMD LoRA at 1.0) applies it first.
    speed=defaults.get('speed_lora')
    choices=([speed] if speed else [])+[choice for choice in request.get('loras') or [] if not speed or choice['id']!=speed['id']]
    for choice in choices:
        item=entry(choice['id']);strength=float(choice.get('strength',1))
        from .lora_loading import read_lora_weights, load_model_lora
        lora=read_lora_weights(local_path(item,item['files'][0]))
        # Quantized bases keep their weights: a merged LoRA would be requantized and mostly lost.
        model,check=load_model_lora(model,lora,strength,item['title'],runtime=bool(copy) or selected['format'] not in ('BF16','FP16','FP32','GGUF'))
        lora_checks.append({'id':item['id'],'title':item['title'],**check})
        applied.append(f"{item['title']} ({strength:g})")
    if applied:retain_lora_clone(model,_components['model'])
    progress(lora_checks=lora_checks)
    shift=float(defaults.get('shift',DEFAULT_SHIFT))
    if shift!=DEFAULT_SHIFT:
        # The template's note: change the shift with ModelSamplingAuraFlow (ModelSamplingSD3 uses another scale).
        model=ModelSamplingAuraFlow().patch_aura(model,shift)[0]
        retain_lora_clone(model,_components['model'])
    if cfg>1:
        # The model card (uncond_skip_layers=[22]) and the official template (SkipLayerGuidanceDiTSimple, single
        # layer 22, 0 to 50%) skip layer 22 on the negative branch only.
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        model=SkipLayerGuidanceDiTSimple.execute(model,0.0,0.5,'','22')[0]
        retain_lora_clone(model,_components['model'])
        progress(skip_layer_guidance='single 22, 0-50%')
    denoise=moss_denoise(request,images)
    if images:
        # VAEEncode of image 1, cropped to the output shape: the layout, colors and pose stay, the prompt redraws.
        from PIL import ImageOps
        source=ImageOps.fit(images[0].convert('RGB'),(width,height),Image.Resampling.LANCZOS)
        progress(stage='encoding',message='Processing the input image…')
        with torch.inference_mode():
            latent=vae.encode(torch.from_numpy(np.array(source).astype(np.float32)/255.0).unsqueeze(0))
        latent=comfy.sample.fix_empty_latent_channels(model,latent)
    else:
        latent=EmptyFlux2LatentImage.execute(width,height,1)[0]['samples']
    noise=comfy.sample.prepare_noise(latent,request['seed'])
    steps=int(request['steps'])
    def callback(step,x0,x,total):
        progress(stage='generating',message='Generating image…',step=step+1,total_steps=total)
    progress(stage='generating',message='Generating image…',step=0,cfg=cfg,shift=shift,sampling=f'{sampler}/{scheduler}',sampler=sampler,scheduler=scheduler,denoise=denoise,input_image_used=bool(images))
    start=time.perf_counter()
    with torch.inference_mode():
        samples=comfy.sample.sample(model,noise,steps,cfg,sampler,scheduler,positive,negative,latent,denoise=denoise,callback=callback,seed=request['seed'])
        progress(stage='decoding',message='Decoding image…',sampling_seconds=round(time.perf_counter()-start,3))
        from .lora_runtime import execution_report
        progress(lora_execution=execution_report(model))
        pixels=vae.decode(samples)
        array=(pixels[0].float().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated()/1024**3,2),true_cfg_scale=cfg,applied_lora=', '.join(applied) or None)
    return Image.fromarray(array[...,:3]).convert('RGBA')

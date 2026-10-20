"""Pinned ComfyUI Qwen 2.1 workflow, run headlessly inside the owned image process."""
from __future__ import annotations
import gc
import importlib.util
import json
import re
from pathlib import Path
import sys
import time
from .common import APP_DIR, MODELS_DIR
from .models import entry, local_path, required_image
from .image_cache import ConditioningCache, PositiveOnlyEncoder, retain_cpu_weights, retain_lora_clone

_components = {}
_conditioning = ConditioningCache()
FUSED_MLP = re.compile(r'([._])img_mlp([._])gate_up([._])')


def split_fused_mlp(weights):
    """LoRAs trained on the fused Qwen-Image-2.1 MLP (ai-toolkit trains the Comfy-Org file, whose img_mlp.gate_up
    holds gate_layer and proj on top of each other) as LoRAs for the two halves, which is how ComfyUI addresses
    them in both the fused and the split (GGUF) model. Up/B rows split in half; down/A and alpha apply to both."""
    if not any(FUSED_MLP.search(key) for key in weights):
        return weights
    out = {}
    for key, value in weights.items():
        if not FUSED_MLP.search(key):
            out[key] = value
            continue
        gate, proj = (FUSED_MLP.sub(lambda m, name=name: f'{m[1]}img_mlp{m[2]}{name}{m[3]}', key) for name in ('gate_layer', 'proj'))
        if re.search(r'lora_(?:B|up)[._]', key):
            half = value.shape[0] // 2
            out[gate], out[proj] = value[:half], value[half:]
        else:
            out[gate] = out[proj] = value
    return out

def generate(request, prompt, images, width, height, progress):
    import torch
    import numpy as np
    from PIL import Image

    comfy_path=str(APP_DIR/'vendor'/'ComfyUI')
    if comfy_path not in sys.path:
        sys.path.insert(0,comfy_path)
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    # Use synchronous transfers. The initial Diffusers CPU-offload path produced
    # a native Windows access violation. Large GGUF weights stay quantized.
    args.disable_async_offload=True
    args.disable_pinned_memory=True
    args.disable_dynamic_vram=True
    from .gpu_memory import configure_video_memory
    configure_video_memory(torch, args, progress)
    # Initialize only the file-mapping library, not its GPU allocator/hooks.
    # safetensors' copy-on-write mapping reserves the whole file against the
    # Windows commit limit (error 1455). ComfyUI's read-only loader avoids that.
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')
    import comfy.sd
    import comfy.sample
    import comfy.utils
    import comfy.model_management as memory
    # A resident process must refresh the budget when other apps change VRAM use.
    memory.EXTRA_RESERVED_VRAM = args.reserve_vram * 1024**3
    torch.cuda.reset_peak_memory_stats()
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21

    def read_weights(path):
        weights,metadata=comfy.utils.load_safetensors(str(path))
        # Keep read-only mmap lifetime references, but use ordinary synchronous
        # Torch transfers. Direct AIMDO device IO requires its GPU allocator.
        for tensor in weights.values():
            storage=tensor.untyped_storage()
            if hasattr(storage,'_comfy_tensor_file_slice'):
                delattr(storage,'_comfy_tensor_file_slice')
        return weights,metadata

    selected=entry(request.get('model_id','official-bf16'))
    from .res4lyf_runtime import ensure_sampler
    ensure_sampler(request.get('sampler') or selected['generation_defaults'].get('sampler', 'euler'))
    model_id,encoder_id,vae_id=required_image(request)[:3]
    encoder=entry(encoder_id)
    vae_entry=entry(vae_id)
    cache_key=(selected['id'],encoder_id,vae_id)
    if _components and _components['key'] != cache_key:
        raise RuntimeError('Start a new image worker when changing model or encoder.')
    reused=bool(_components)
    started=time.perf_counter()
    progress(stage='loading_model',message='Using loaded image models.' if reused else 'Loading text encoder and VAE…',
             model_reused=reused,
             selected_model=selected['title'],text_encoder_id=encoder_id,vae_id=vae_id)
    if reused:
        vae,clip=_components['vae'],_components['clip']
    else:
        vae_weights,_=read_weights(local_path(vae_entry,vae_entry['files'][0]))
        vae=comfy.sd.VAE(sd=vae_weights)
        del vae_weights
        retain_cpu_weights(vae.patcher)
        encoder_weights,encoder_metadata=read_weights(local_path(encoder,encoder['files'][0]))
        encoder_weights,_=comfy.utils.convert_old_quants(encoder_weights,model_prefix='',metadata=encoder_metadata)
        # Build from the read-only CPU mapping, then let ComfyUI load/offload the
        # encoder for inference. Eager CUDA construction can exhaust Windows commit
        # memory even when nvidia-smi reports plenty of free VRAM.
        clip=comfy.sd.load_text_encoder_state_dicts([encoder_weights],clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
            model_options={'initial_device':torch.device('cpu')},disable_dynamic=True)
        del encoder_weights,encoder_metadata
        retain_cpu_weights(clip.patcher)
    progress(encoder_load_seconds=round(time.perf_counter()-started,3))
    negative=request['negative_prompt'].strip()
    defaults=selected['generation_defaults']
    turbo=bool(defaults.get('sigmas'))
    # A CFG the user typed, else the model's own CFG (Turbo: 1, or what its Civitai page asks for); otherwise CFG 2 only with a negative prompt.
    cfg=float(request['cfg']) if request.get('cfg') else float(defaults['cfg']) if defaults.get('cfg') else 1.0 if turbo else 2.0 if negative else 1.0
    started=time.perf_counter()
    reference_resolution=int(request.get('reference_resolution',1024))
    conditioning_key=_conditioning.fingerprint(f'{reference_resolution}|{cfg>1}|{prompt}',negative,images)
    cached=_conditioning.get(conditioning_key)
    progress(stage='encoding',message='Using saved prompt and image data.' if cached else
             ('Processing prompt and input images…' if images else 'Processing prompt…'),step=0,
             conditioning_reused=cached is not None)
    if cached is None:
        image_tensors={f'image_{i+1}':torch.from_numpy(np.array(img.convert('RGBA')).astype(np.float32)/255.0).unsqueeze(0) for i,img in enumerate(images)}
        with torch.inference_mode():
            encoded=TextEncodeQwenImage21.execute(clip=PositiveOnlyEncoder(clip,bool(negative) or cfg>1),prompt=prompt,negative_prompt=negative,vae=vae,resolution=reference_resolution,images=image_tensors)
        positive,negative_conditioning=encoded[0],encoded[1]
        _conditioning.put(conditioning_key,(positive,negative_conditioning))
        del encoded,image_tensors
    else:
        positive,negative_conditioning=cached
    progress(encoding_seconds=round(time.perf_counter()-started,3))
    started=time.perf_counter()
    progress(stage='loading_model',message='Using loaded image model.' if reused else 'Loading image model…')
    if reused:
        model=_components['model']
    elif selected['format']=='GGUF':
        module_dir=APP_DIR/'vendor'/'ComfyUI-GGUF'
        if 'studio_gguf' not in sys.modules:
            spec=importlib.util.spec_from_file_location('studio_gguf',module_dir/'__init__.py',submodule_search_locations=[str(module_dir)])
            package=importlib.util.module_from_spec(spec)
            sys.modules['studio_gguf']=package
            spec.loader.exec_module(package)
        from studio_gguf.loader import gguf_sd_loader
        from studio_gguf.dequant import dequantize_tensor, is_quantized
        from studio_gguf.ops import GGMLOps
        from studio_gguf.nodes import GGUFModelPatcher
        weights,extra=gguf_sd_loader(str(local_path(selected,selected['files'][0])))
        fixed=[]
        for name,value in list(weights.items()):
            # Q8_0 encodes a 128-value RMSNorm vector as 136 bytes. RMSNorm is
            # not a Linear layer: decode these tiny vectors before model detection.
            if len(value.shape)==1 and is_quantized(value):
                decoded=dequantize_tensor(value,dtype=torch.float32)
                if tuple(decoded.shape)!=tuple(value.shape):
                    raise RuntimeError(f'Incorrect GGUF normalization shape: {name}')
                weights[name]=decoded
                fixed.append(name)
        operations=GGMLOps()
        operations.Linear.dequant_dtype='target'
        model=comfy.sd.load_diffusion_model_state_dict(weights,model_options={'custom_operations':operations,'dtype':torch.bfloat16},metadata=extra.get('metadata',{}),disable_dynamic=True)
        if model is None:
            raise RuntimeError('This GGUF file is not a compatible Qwen-Image-2.1 checkpoint.')
        model=GGUFModelPatcher.clone(model)
        progress(gguf_norm_vectors_decoded=len(fixed),gguf_quantization=selected['title'].split(' · ')[-1])
    elif selected.get('layout')=='comfy':
        # Single-file ComfyUI checkpoint (Comfy-Org Turbo BF16).
        weights,metadata=read_weights(local_path(selected,selected['files'][0]))
        model=comfy.sd.load_diffusion_model_state_dict(weights,model_options={'dtype':torch.bfloat16},metadata=metadata,disable_dynamic=True)
        if model is None:
            raise RuntimeError(f"Could not load {selected['title']}.")
    else:
        index_path=MODELS_DIR/'Qwen-Image-2.1'/'transformer'/'diffusion_pytorch_model.safetensors.index.json'
        weight_map=json.loads(index_path.read_text())['weight_map']
        weights={}
        for shard in sorted(set(weight_map.values())):
            weights.update(read_weights(index_path.parent/shard)[0])
        if set(weights)!=set(weight_map):
            raise RuntimeError('The official image model is missing weights.')
        model=comfy.sd.load_diffusion_model_state_dict(weights,model_options={'dtype':torch.bfloat16},disable_dynamic=True)
        if model is None:
            raise RuntimeError('Could not load the official Qwen-Image-2.1 model.')
    if not reused:
        del weights
        retain_cpu_weights(model)
        _components.update(key=cache_key,clip=clip,vae=vae,model=model)
        gc.collect()
    progress(diffusion_load_seconds=round(time.perf_counter()-started,3))
    # Tool LoRA first, then the user's LoRAs; each one is patched onto a clone.
    stack=[]
    if request.get('tool_id'):
        from .models import image_tool
        stack.append((image_tool(request['tool_id'])['lora'],float(request.get('lora_strength',1.0))))
    stack+=[(lora['id'],float(lora.get('strength',1.0))) for lora in request.get('loras') or []]
    tool_lora=None
    lora_checks=[]
    if stack:
        started=time.perf_counter()
        loras=_components.setdefault('loras',{})
        for stale in [key for key in loras if key not in {lora_id for lora_id,_ in stack}]:
            del loras[stale]  # keep only this job's LoRAs in RAM
        applied=[]
        for lora_id,strength in stack:
            lora=entry(lora_id)
            if lora_id not in loras:
                loras[lora_id]=split_fused_mlp(comfy.utils.load_torch_file(str(local_path(lora,lora['files'][0])),safe_load=True))
            from .lora_loading import load_model_lora
            model,check=load_model_lora(model,loras[lora_id],strength,lora['title'],runtime=True)
            lora_checks.append({'id':lora_id,'title':lora['title'],**check})
            applied.append(f"{lora['title']} ({strength:g})")
        retain_lora_clone(model,_components['model'])
        tool_lora=', '.join(applied)
        progress(lora=tool_lora,lora_strength=stack[0][1],lora_load_seconds=round(time.perf_counter()-started,3))
    progress(lora_checks=lora_checks)
    # The upstream automatic cache can copy large multi-image prefixes to host
    # RAM. That path caused a native Windows access violation. The supported
    # GPU policy keeps the lossless cache in VRAM when its memory checks allow,
    # otherwise recomputes the prefix without host cache transfers.
    model.model_options.setdefault('transformer_options', {})['qwen_image21_cache']={'device':'gpu','dtype':'default'}
    progress(prefix_cache_device='gpu')
    # Official Qwen 2.1 workflow: 64-channel, 16x spatial latent, Euler + simple.
    latent=torch.zeros((1,64,height//16,width//16),device=memory.intermediate_device())
    noise=comfy.sample.prepare_noise(latent,request['seed'])
    sampler,scheduler=request.get('sampler') or defaults.get('sampler','euler'),request.get('scheduler') or defaults.get('scheduler','simple')
    def callback(step,x0,x,total):
        progress(stage='generating',message='Generating image…',step=step+1,total_steps=total)
    progress(stage='generating',message='Generating image…',step=0,sampling='turbo-sigmas' if turbo else f'{sampler}-{scheduler}',cfg=cfg)
    with torch.inference_mode():
        if turbo:
            import comfy.samplers
            sigmas=defaults['sigmas']
            if request['steps']!=len(sigmas)-1:
                # A custom step count follows the same trained curve, stretched to the new length.
                sigmas=np.interp(np.linspace(0,1,request['steps']+1),np.linspace(0,1,len(sigmas)),sigmas).tolist()
            sigmas=torch.tensor(sigmas,dtype=torch.float32)
            samples=comfy.sample.sample_custom(model,noise,cfg,comfy.samplers.sampler_object(defaults.get('sampler','euler')),sigmas,
                                               positive,negative_conditioning,latent,callback=callback,seed=request['seed'])
        else:
            samples=comfy.sample.sample(model,noise,request['steps'],cfg,sampler,scheduler,positive,negative_conditioning,
                                        latent,callback=callback,seed=request['seed'])
        del noise,latent
        from .lora_runtime import execution_report
        progress(lora_execution=execution_report(model))
        progress(stage='decoding',message='Decoding image…')
        pixels=vae.decode(samples)
    array=(pixels[0].float().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
    if array.shape[-1]!=4:
        raise RuntimeError('The Qwen 2.1 VAE did not return an RGBA image.')
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated()/1024**3,2),engine='ComfyUI',true_cfg_scale=cfg,applied_lora=tool_lora)
    return Image.fromarray(array)

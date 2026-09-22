"""Pinned ComfyUI Qwen 2.1 workflow, run headlessly inside the owned image process."""
from __future__ import annotations
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
from .common import APP_DIR, MODELS_DIR
from .models import entry, local_path, required
from .image_cache import ConditioningCache, PositiveOnlyEncoder, retain_cpu_weights

_components = {}
_conditioning = ConditioningCache()

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
    encoder_id=required(selected['id'],request.get('text_encoder','auto'))[1]
    encoder=entry(encoder_id)
    vae_entry=entry('vae')
    cache_key=(selected['id'],encoder_id)
    if _components and _components['key'] != cache_key:
        raise RuntimeError('Start a new image worker when changing model or encoder.')
    reused=bool(_components)
    started=time.perf_counter()
    progress(stage='loading_model',message='Using loaded image models.' if reused else 'Loading text encoder and VAE…',
             model_reused=reused,
             selected_model=selected['title'],text_encoder_id=encoder_id,vae_id='vae')
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
    started=time.perf_counter()
    conditioning_key=_conditioning.fingerprint(prompt,negative,images)
    cached=_conditioning.get(conditioning_key)
    progress(stage='encoding',message='Using saved prompt and image data.' if cached else
             ('Processing prompt and input images…' if images else 'Processing prompt…'),step=0,
             conditioning_reused=cached is not None)
    if cached is None:
        image_tensors={f'image_{i+1}':torch.from_numpy(np.array(img.convert('RGBA')).astype(np.float32)/255.0).unsqueeze(0) for i,img in enumerate(images)}
        with torch.inference_mode():
            encoded=TextEncodeQwenImage21.execute(clip=PositiveOnlyEncoder(clip,bool(negative)),prompt=prompt,negative_prompt=negative,vae=vae,resolution=1024,images=image_tensors)
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
    # The upstream automatic cache can copy large multi-image prefixes to host
    # RAM. That path caused a native Windows access violation. The supported
    # GPU policy keeps the lossless cache in VRAM when its memory checks allow,
    # otherwise recomputes the prefix without host cache transfers.
    model.model_options.setdefault('transformer_options', {})['qwen_image21_cache']={'device':'gpu','dtype':'default'}
    progress(prefix_cache_device='gpu')
    # Official Qwen 2.1 workflow: 64-channel, 16x spatial latent, Euler + simple.
    latent=torch.zeros((1,64,height//16,width//16),device=memory.intermediate_device())
    noise=comfy.sample.prepare_noise(latent,request['seed'])
    cfg=2.0 if negative else 1.0
    def callback(step,x0,x,total):
        progress(stage='generating',message='Generating image…',step=step+1,total_steps=total)
    progress(stage='generating',message='Generating image…',step=0)
    with torch.inference_mode():
        samples=comfy.sample.sample(model,noise,request['steps'],cfg,'euler','simple',positive,negative_conditioning,
                                    latent,callback=callback,seed=request['seed'])
        del noise,latent
        progress(stage='decoding',message='Decoding image…')
        pixels=vae.decode(samples)
    array=(pixels[0].float().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
    if array.shape[-1]!=4:
        raise RuntimeError('The Qwen 2.1 VAE did not return an RGBA image.')
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated()/1024**3,2),engine='ComfyUI',true_cfg_scale=cfg)
    return Image.fromarray(array)

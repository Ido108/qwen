"""Z-Image (Turbo and Base), run headlessly with ComfyUI inside the owned image process.

Matches the official ComfyUI workflows (Comfy-Org/workflow_templates image_z_image_turbo, image_z_image and
utility_z_image_turbo_2k_upscaler):
- Qwen3 4B text encoder (CLIPLoader type 'lumina2'), Flux VAE ae.safetensors (16-channel, 8x latent).
- ModelSamplingAuraFlow shift 3, then KSampler: Turbo 8 steps, CFG 1, res_multistep, simple;
  Base 30 to 50 steps, CFG 3 to 5 with a negative prompt.
- Z-Image has no edit model (Tongyi lists Z-Image-Edit as "to be released"). An input image is used as in the
  official upscale workflow: VAEEncode, then KSampler with denoise below 1, so the result keeps the layout,
  colors and pose of the image and redraws the details. Lower denoise keeps more of the image.
"""
from __future__ import annotations
import gc
import importlib.util
import sys
import time
from .common import APP_DIR
from .models import entry, local_path, required_image
from .image_cache import ConditioningCache, retain_cpu_weights, retain_lora_clone

_components = {}
_conditioning = ConditioningCache()
DEFAULT_DENOISE = 0.7


def zimage_cfg(request, defaults):
    """A CFG the user typed, else the model default (Turbo 1, Base 4). ComfyUI convention: 1 means no guidance."""
    return float(request['cfg']) if request.get('cfg') else float(defaults.get('cfg', 1.0))


def zimage_denoise(request, images):
    """1 without an input image; with one, the user's image strength (denoise) or the default 0.7."""
    if not images:
        return 1.0
    return float(request.get('denoise') or DEFAULT_DENOISE)


def generate(request, prompt, images, width, height, progress):
    import torch
    import numpy as np

    comfy_path = str(APP_DIR / 'vendor' / 'ComfyUI')
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    # Same process settings as the Qwen engine: synchronous transfers, no pinned or dynamic VRAM.
    args.disable_async_offload = True
    args.disable_pinned_memory = True
    args.disable_dynamic_vram = True
    from .gpu_memory import configure_attention, configure_video_memory
    attention = configure_attention(args)
    configure_video_memory(torch, args, progress)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')
    import comfy.sd
    from . import sent_prompts
    sent_prompts.install(comfy.sd)  # the exact encoder input, for the result details
    import comfy.sample
    import comfy.samplers
    import comfy.supported_models
    import comfy.utils
    import comfy.model_management as memory
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    from PIL import Image, ImageOps
    memory.EXTRA_RESERVED_VRAM = args.reserve_vram * 1024**3
    torch.cuda.reset_peak_memory_stats()

    def read_weights(path):
        weights, metadata = comfy.utils.load_safetensors(str(path))
        for tensor in weights.values():
            storage = tensor.untyped_storage()
            if hasattr(storage, '_comfy_tensor_file_slice'):
                delattr(storage, '_comfy_tensor_file_slice')
        return weights, metadata

    selected = entry(request['model_id'])
    from .res4lyf_runtime import ensure_sampler
    ensure_sampler(request.get('sampler') or selected['generation_defaults'].get('sampler', 'res_multistep'))
    model_id, encoder_id, vae_id = required_image(request)[:3]
    encoder, vae_entry = entry(encoder_id), entry(vae_id)
    cache_key = (selected['id'], encoder_id, vae_id)
    if _components and _components['key'] != cache_key:
        if _components['key'][1:] != cache_key[1:]:
            raise RuntimeError('Start a new image worker when changing model or encoder.')
        # Another checkpoint with the same encoder and VAE: keep them and the saved prompt data
        # (they do not depend on the image model) and replace only the image model.
        memory.unload_model_and_clones(_components.pop('model'))
        _components.pop('loras', None)
        _components['key'] = cache_key
        gc.collect()
        memory.soft_empty_cache()
    reused = bool(_components)
    model_reused = 'model' in _components
    started = time.perf_counter()
    progress(stage='loading_model', message='Using loaded image models.' if reused else 'Loading text encoder and VAE…',
             model_reused=reused, selected_model=selected['title'], text_encoder_id=encoder_id, vae_id=vae_id, engine='ComfyUI Z-Image', attention=attention)
    if reused:
        vae, clip = _components['vae'], _components['clip']
    else:
        vae_weights, vae_metadata = read_weights(local_path(vae_entry, vae_entry['files'][0]))
        vae = comfy.sd.VAE(sd=vae_weights, metadata=vae_metadata)
        del vae_weights
        retain_cpu_weights(vae.patcher)
        encoder_weights, encoder_metadata = read_weights(local_path(encoder, encoder['files'][0]))
        encoder_weights, _ = comfy.utils.convert_old_quants(encoder_weights, model_prefix='', metadata=encoder_metadata)
        # CLIPLoader type 'lumina2' with a Qwen3 4B file is the Z-Image text encoder.
        clip = comfy.sd.load_text_encoder_state_dicts([encoder_weights], clip_type=comfy.sd.CLIPType.LUMINA2,
                                                       model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
        del encoder_weights, encoder_metadata
        retain_cpu_weights(clip.patcher)
    progress(encoder_load_seconds=round(time.perf_counter() - started, 3))

    defaults = selected['generation_defaults']
    cfg = zimage_cfg(request, defaults)
    negative_text = request.get('negative_prompt', '').strip()
    started = time.perf_counter()
    conditioning_key = _conditioning.fingerprint(f'zimage|{cfg > 1}|{prompt}', negative_text, [])
    cached = _conditioning.get(conditioning_key)
    progress(stage='encoding', message='Using saved prompt data.' if cached else 'Processing prompt…', step=0, conditioning_reused=cached is not None)
    if cached is None:
        with torch.inference_mode():
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            if cfg > 1:
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
            else:
                # Unused at CFG 1 (ComfyUI skips it), as ConditioningZeroOut in the official Turbo workflow.
                negative = [[torch.zeros_like(c[0]), {k: v for k, v in c[1].items() if k != 'pooled_output'}] for c in positive]
        _conditioning.put(conditioning_key, (positive, negative))
    else:
        positive, negative = cached
    progress(encoding_seconds=round(time.perf_counter() - started, 3))

    started = time.perf_counter()
    progress(stage='loading_model', message='Using loaded image model.' if model_reused else 'Loading image model…', diffusion_model_reused=model_reused)
    if model_reused:
        model = _components['model']
    elif selected['format'] == 'GGUF':
        module_dir = APP_DIR / 'vendor' / 'ComfyUI-GGUF'
        if 'studio_gguf' not in sys.modules:
            spec = importlib.util.spec_from_file_location('studio_gguf', module_dir / '__init__.py', submodule_search_locations=[str(module_dir)])
            package = importlib.util.module_from_spec(spec)
            sys.modules['studio_gguf'] = package
            spec.loader.exec_module(package)
        from studio_gguf.loader import gguf_sd_loader
        from studio_gguf.dequant import dequantize_tensor, is_quantized
        from studio_gguf.ops import GGMLOps
        from studio_gguf.nodes import GGUFModelPatcher
        weights, extra = gguf_sd_loader(str(local_path(selected, selected['files'][0])))
        for name, value in list(weights.items()):
            # Normalization vectors are not Linear layers: decode them before model detection.
            if len(value.shape) == 1 and is_quantized(value):
                weights[name] = dequantize_tensor(value, dtype=torch.float32)
        operations = GGMLOps()
        operations.Linear.dequant_dtype = 'target'
        model = comfy.sd.load_diffusion_model_state_dict(weights, model_options={'custom_operations': operations, 'dtype': torch.bfloat16},
                                                         metadata=extra.get('metadata', {}), disable_dynamic=True)
        if model is None:
            raise RuntimeError('This GGUF file is not a compatible Z-Image checkpoint.')
        model = GGUFModelPatcher.clone(model)
    else:
        weights, metadata = read_weights(local_path(selected, selected['files'][0]))
        # UNETLoader with weight_dtype 'default': INT8, NVFP4 and FP8 files keep their own quantization.
        model = comfy.sd.load_diffusion_model_state_dict(weights, model_options={}, metadata=metadata, disable_dynamic=True)
        if model is None:
            raise RuntimeError(f"Could not load {selected['title']}.")
    if type(model.model.model_config) is not comfy.supported_models.ZImage:
        raise RuntimeError(f"{selected['title']} is not a Z-Image model.")
    if not model_reused:
        del weights
        retain_cpu_weights(model)
        _components.update(key=cache_key, clip=clip, vae=vae, model=model)
        gc.collect()
    progress(diffusion_load_seconds=round(time.perf_counter() - started, 3))

    stack = [(lora['id'], float(lora.get('strength', 1.0))) for lora in request.get('loras') or []]
    applied_loras = None
    lora_checks = []
    if stack:
        started = time.perf_counter()
        loras = _components.setdefault('loras', {})
        for stale in [key for key in loras if key not in {lora_id for lora_id, _ in stack}]:
            del loras[stale]
        applied = []
        for lora_id, strength in stack:
            lora = entry(lora_id)
            if lora_id not in loras:
                from .lora_loading import read_lora_weights
                loras[lora_id] = read_lora_weights(local_path(lora, lora['files'][0]))
            from .lora_loading import load_model_lora
            # Quantized bases (INT8, FP8, NVFP4) keep their weights: a merged LoRA would be requantized and mostly lost.
            model, check = load_model_lora(model, loras[lora_id], strength, lora['title'], runtime=selected['format'] not in ('BF16', 'FP16', 'FP32', 'GGUF'))
            lora_checks.append({'id': lora_id, 'title': lora['title'], **check})
            applied.append(f"{lora['title']} ({strength:g})")
        retain_lora_clone(model, _components['model'])
        applied_loras = ', '.join(applied)
        progress(lora=applied_loras, lora_strength=stack[0][1], lora_load_seconds=round(time.perf_counter() - started, 3))
    progress(lora_checks=lora_checks)

    # ModelSamplingAuraFlow (official shift 3); it clones the model, so the cached model stays unpatched.
    shift = float(defaults.get('shift', 3.0))
    model = ModelSamplingAuraFlow().patch_aura(model, shift)[0]
    sampler = request.get('sampler') or defaults.get('sampler', 'res_multistep')
    scheduler = request.get('scheduler') or defaults.get('scheduler', 'simple')
    steps = int(request['steps'])
    denoise = zimage_denoise(request, images)

    with torch.inference_mode():
        if images:
            # VAEEncode of image 1, cropped to the output shape (the output size follows image 1).
            source = ImageOps.fit(images[0].convert('RGB'), (width, height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.array(source).astype(np.float32) / 255.0).unsqueeze(0)
            progress(stage='encoding', message='Processing the input image…')
            latent = vae.encode(pixels)
        else:
            latent = torch.zeros((1, 16, height // 8, width // 8), device=memory.intermediate_device())
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    noise = comfy.sample.prepare_noise(latent, request['seed'])

    from .image_preview import LiveImagePreview
    preview = LiveImagePreview(model.model.latent_format, progress)
    def callback(step, x0, x, total):
        progress(stage='generating', message='Generating image…', step=step + 1, total_steps=total)
        preview.publish(x0, step, total)
    progress(stage='generating', message='Generating image…', step=0, sampling=f'{sampler}-{scheduler}', sampler=sampler, scheduler=scheduler, cfg=cfg, shift=shift,
             denoise=denoise, input_image_used=bool(images))
    started = time.perf_counter()
    with torch.inference_mode():
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent,
                                      denoise=denoise, callback=callback, seed=request['seed'])
        sampling_seconds = time.perf_counter() - started
        del noise, latent
        progress(stage='decoding', message='Decoding image…', sampling_seconds=round(sampling_seconds, 3),
                 seconds_per_step=round(sampling_seconds / steps, 3))
        pixels = vae.decode(samples)
    pixels = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
    array = (pixels[0].float().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2), engine='ComfyUI Z-Image', true_cfg_scale=cfg, applied_lora=applied_loras)
    return Image.fromarray(array[..., :3]).convert('RGBA')

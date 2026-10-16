"""Krea 2 (Raw and Turbo), run headlessly with ComfyUI inside the owned image process.

Matches the official ComfyUI workflows (Comfy-Org/workflow_templates image_krea2_turbo_t2i and
image_krea2_turbo_int8_image_style_reference) and the official Krea sampler (krea-ai/krea-2 sampling.py):
- Qwen3-VL 4B text encoder, CLIP type 'krea2' (12 tapped hidden layers), Qwen-Image VAE (16-channel, 8x latent).
- Euler on a flow-matching schedule t = linspace(1, 0, steps + 1), time-shifted with mu:
  Turbo pins mu = 1.15; Raw takes mu from the image size, 0.5 at 256 px up to 1.15 at 1280 px.
- Turbo runs without guidance (CFG 1). Raw uses Krea CFG 3.5 = ComfyUI CFG 4.5 with an empty negative prompt.
- Input images are style references: the official workflow encodes them with TextEncodeQwenImageEditPlus,
  sets the reference method 'index_timestep_zero' and applies the Krea 2 style reference LoRA.
"""
from __future__ import annotations
import gc
import importlib.util
import math
import sys
import time
from .common import APP_DIR
from .models import KREA_STYLE_LORA, entry, local_path, required_image
from .image_cache import ConditioningCache, retain_cpu_weights, retain_lora_clone

_components = {}
_conditioning = ConditioningCache()


def krea_mu(defaults, width, height):
    """The official time shift: a fixed mu, or mu interpolated in image tokens (16 px per token)."""
    if defaults.get('mu') is not None:
        return float(defaults['mu'])
    (y1, y2), (low, high) = defaults.get('mu_range', (0.5, 1.15)), defaults.get('mu_resolution', (256, 1280))
    x1, x2 = (low // 16) ** 2, (high // 16) ** 2
    tokens = math.ceil(height / 16) * math.ceil(width / 16)
    return y1 + (y2 - y1) / (x2 - x1) * (tokens - x1)


def krea_sigmas(steps, mu):
    """sampling.timesteps(): exp(mu) / (exp(mu) + (1 / t - 1)) on t = linspace(1, 0, steps + 1)."""
    sigmas = []
    for i in range(steps + 1):
        t = 1 - i / steps
        sigmas.append(0.0 if t <= 0 else math.exp(mu) / (math.exp(mu) + (1 / t - 1)))
    return sigmas


def krea_cfg(request, defaults):
    """A CFG the user typed, else the model default (Turbo 1, Raw 4.5). ComfyUI convention: 1 means no guidance."""
    return float(request['cfg']) if request.get('cfg') else float(defaults.get('cfg', 1.0))


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
    from .gpu_memory import configure_video_memory
    configure_video_memory(torch, args, progress)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')
    import comfy.sd
    import comfy.model_base
    import comfy.sample
    import comfy.samplers
    import comfy.model_sampling
    import comfy.utils
    import comfy.model_management as memory
    import node_helpers
    from PIL import Image
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
    model_id, encoder_id, vae_id = required_image(request)[:3]
    encoder, vae_entry = entry(encoder_id), entry(vae_id)
    cache_key = (selected['id'], encoder_id, vae_id)
    if _components and _components['key'] != cache_key:
        raise RuntimeError('Start a new image worker when changing model or encoder.')
    reused = bool(_components)
    started = time.perf_counter()
    progress(stage='loading_model', message='Using loaded image models.' if reused else 'Loading text encoder and VAE…',
             model_reused=reused, selected_model=selected['title'], text_encoder_id=encoder_id, vae_id=vae_id, engine='ComfyUI Krea 2')
    if reused:
        vae, clip = _components['vae'], _components['clip']
    else:
        vae_weights, vae_metadata = read_weights(local_path(vae_entry, vae_entry['files'][0]))
        vae = comfy.sd.VAE(sd=vae_weights, metadata=vae_metadata)
        del vae_weights
        retain_cpu_weights(vae.patcher)
        encoder_weights, encoder_metadata = read_weights(local_path(encoder, encoder['files'][0]))
        encoder_weights, _ = comfy.utils.convert_old_quants(encoder_weights, model_prefix='', metadata=encoder_metadata)
        # CLIPLoader type 'krea2': the Qwen3-VL 4B 12-layer tap. Built on the CPU mapping, loaded by ComfyUI.
        clip = comfy.sd.load_text_encoder_state_dicts([encoder_weights], clip_type=comfy.sd.CLIPType.KREA2,
                                                       model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
        del encoder_weights, encoder_metadata
        retain_cpu_weights(clip.patcher)
    progress(encoder_load_seconds=round(time.perf_counter() - started, 3))

    defaults = selected['generation_defaults']
    cfg = krea_cfg(request, defaults)
    negative_text = request.get('negative_prompt', '').strip()
    started = time.perf_counter()
    conditioning_key = _conditioning.fingerprint(f'krea2|{cfg > 1}|{prompt}', negative_text, images)
    cached = _conditioning.get(conditioning_key)
    progress(stage='encoding', message='Using saved prompt and image data.' if cached else
             ('Processing prompt and style images…' if images else 'Processing prompt…'), step=0, conditioning_reused=cached is not None)
    if cached is None:
        with torch.inference_mode():
            if images:
                # Official style reference workflow: Qwen edit-style encoding of 1 to 3 images (384 px for
                # the vision encoder, 1 MP reference latents), then the index_timestep_zero method.
                from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
                tensors = {f'image{i + 1}': torch.from_numpy(np.array(img.convert('RGB')).astype(np.float32) / 255.0).unsqueeze(0)
                           for i, img in enumerate(images[:3])}
                positive = TextEncodeQwenImageEditPlus.execute(clip=clip, prompt=prompt, vae=vae, **tensors)[0]
                positive = node_helpers.conditioning_set_values(positive, {'reference_latents_method': 'index_timestep_zero'})
            else:
                positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            negative = None
            if cfg > 1:
                # Krea's own sampler encodes the negative prompt text ('' by default), not zeros.
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
                if images:
                    extra = {key: positive[0][1][key] for key in ('reference_latents', 'reference_latents_method') if key in positive[0][1]}
                    negative = node_helpers.conditioning_set_values(negative, extra)
            else:
                # Unused at CFG 1 (ComfyUI skips it), as ConditioningZeroOut in the official workflow.
                negative = [[torch.zeros_like(c[0]), {k: v for k, v in c[1].items() if k != 'pooled_output'}] for c in positive]
        _conditioning.put(conditioning_key, (positive, negative))
    else:
        positive, negative = cached
    progress(encoding_seconds=round(time.perf_counter() - started, 3))

    started = time.perf_counter()
    progress(stage='loading_model', message='Using loaded image model.' if reused else 'Loading image model…')
    if reused:
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
            raise RuntimeError('This GGUF file is not a compatible Krea 2 checkpoint.')
        model = GGUFModelPatcher.clone(model)
    else:
        weights, metadata = read_weights(local_path(selected, selected['files'][0]))
        # UNETLoader with weight_dtype 'default': FP8, NVFP4, MXFP8 and INT8 files keep their own quantization.
        model = comfy.sd.load_diffusion_model_state_dict(weights, model_options={}, metadata=metadata, disable_dynamic=True)
        if model is None:
            raise RuntimeError(f"Could not load {selected['title']}.")
    if not isinstance(model.model, comfy.model_base.Krea2):
        raise RuntimeError(f"{selected['title']} is not a Krea 2 model.")
    if not reused:
        del weights
        retain_cpu_weights(model)
        _components.update(key=cache_key, clip=clip, vae=vae, model=model)
        gc.collect()
    progress(diffusion_load_seconds=round(time.perf_counter() - started, 3))

    # The style reference LoRA for input images (official strength 1), then the user's LoRAs.
    stack = [(KREA_STYLE_LORA, 1.0)] if images else []
    stack += [(lora['id'], float(lora.get('strength', 1.0))) for lora in request.get('loras') or []]
    applied_loras = None
    if stack:
        started = time.perf_counter()
        loras = _components.setdefault('loras', {})
        for stale in [key for key in loras if key not in {lora_id for lora_id, _ in stack}]:
            del loras[stale]
        applied = []
        for lora_id, strength in stack:
            lora = entry(lora_id)
            if lora_id not in loras:
                loras[lora_id] = comfy.utils.load_torch_file(str(local_path(lora, lora['files'][0])), safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, loras[lora_id], strength, 0)
            applied.append(f"{lora['title']} ({strength:g})")
        retain_lora_clone(model, _components['model'])
        applied_loras = ', '.join(applied)
        progress(lora=applied_loras, lora_strength=stack[0][1], lora_load_seconds=round(time.perf_counter() - started, 3))

    mu = krea_mu(defaults, width, height)
    sampler = defaults.get('sampler', 'euler')
    scheduler = defaults.get('scheduler', 'simple')

    steps = int(request['steps'])
    if scheduler == 'simple':
        sigmas = torch.tensor(krea_sigmas(steps, mu), dtype=torch.float32)  # exact official schedule
    else:
        # A Civitai page's own scheduler, on the same flow time shift (ComfyUI ModelSamplingFlux with shift mu).
        # The shift only shapes these sigmas, so the loaded model needs no sampling patch.
        sampling = comfy.model_sampling.ModelSamplingFlux(model.model.model_config)
        sampling.set_parameters(shift=mu)
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps).float()

    latent = torch.zeros((1, 16, height // 8, width // 8), device=memory.intermediate_device())
    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    noise = comfy.sample.prepare_noise(latent, request['seed'])

    def callback(step, x0, x, total):
        progress(stage='generating', message='Generating image…', step=step + 1, total_steps=total)
    progress(stage='generating', message='Generating image…', step=0, sampling=f'{sampler}-{scheduler}', cfg=cfg, shift_mu=round(mu, 4))
    started = time.perf_counter()
    with torch.inference_mode():
        samples = comfy.sample.sample_custom(model, noise, cfg, comfy.samplers.sampler_object(sampler), sigmas,
                                             positive, negative, latent, callback=callback, seed=request['seed'])
        sampling_seconds = time.perf_counter() - started
        del noise, latent
        progress(stage='decoding', message='Decoding image…', sampling_seconds=round(sampling_seconds, 3),
                 seconds_per_step=round(sampling_seconds / steps, 3))
        pixels = vae.decode(samples)
    pixels = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
    array = (pixels[0].float().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2), engine='ComfyUI Krea 2', true_cfg_scale=cfg, applied_lora=applied_loras)
    return Image.fromarray(array[..., :3]).convert('RGBA')

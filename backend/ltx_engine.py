"""Pinned LTX-2.5 MSR V2 native AV workflow, with optional spatial refinement."""
from __future__ import annotations
import gc
import importlib.util
import logging
import sys
from .common import APP_DIR
from .models import entry, local_path
from .ltx_settings import SIGMAS, REFINE_SIGMAS


def load_extension(name, directory):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate(request, images, voices, progress, preview_directory=None):
    import numpy as np
    import torch
    sys.path.insert(0, str(APP_DIR / 'vendor/ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload = args.disable_pinned_memory = args.disable_dynamic_vram = True
    from .gpu_memory import configure_video_memory, GIB
    configure_video_memory(torch, args, progress)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')
    import comfy.sd
    import comfy.utils
    import comfy.sample
    import comfy.samplers
    import comfy.model_management as memory
    import folder_paths
    from comfy_extras.nodes_lt import (EmptyLTXVLatentVideo, LTXVConditioning,
        LTXVConcatAVLatent, LTXVSeparateAVLatent, LTXVCropGuides)
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEEncode, LTXVAudioVAEDecode
    from nodes import VAEDecodeTiled
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    def read_weights(model_id):
        item = entry(model_id)
        weights, metadata = comfy.utils.load_safetensors(str(local_path(item, item['files'][0])))
        for tensor in weights.values():
            storage = tensor.untyped_storage()
            if hasattr(storage, '_comfy_tensor_file_slice'):
                delattr(storage, '_comfy_tensor_file_slice')
        return weights, metadata

    def release():
        gc.collect()
        memory.cleanup_models()
        memory.soft_empty_cache()

    total_steps = 11 if request['refine'] else 8
    progress(stage='loading_encoder', message='Loading the LTX text encoder...', total_steps=total_steps)
    weights, metadata = read_weights('ltx25-gemma-int8')
    weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
    clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.LTXV,
        model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
    del weights, metadata
    clip.cond_stage_model.memory_estimation_function = lambda tokens, device=None: 3 * GIB
    progress(stage='encoding_text', message='Encoding the video description...')
    with torch.inference_mode():
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(request['prompt']))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize('subtitles, watermark, worst quality, blurry, jittery, distorted, inconsistent appearance'))
    del clip
    release()
    positive, negative = LTXVConditioning.execute(positive, negative, request['fps'])
    progress(stage='loading_video_vae', message='Loading the video and audio encoders...')
    weights, metadata = read_weights('ltx25-video-vae')
    vae = comfy.sd.VAE(sd=weights, metadata=metadata)
    del weights
    weights, metadata = read_weights('ltx25-audio-vae')
    audio_vae = comfy.sd.VAE(sd=weights, metadata=metadata)
    del weights, metadata
    vae.throw_exception_if_invalid()
    audio_vae.throw_exception_if_invalid()
    refs = {name: torch.from_numpy(np.asarray(img.convert('RGB'), dtype=np.float32) / 255).unsqueeze(0)
            for name, img in images.items()}
    audio_refs = {}
    with torch.inference_mode():
        for name, voice in voices.items():
            progress(stage='encoding_audio', message=f'Encoding voice reference {name[-1]}...')
            audio_refs[name] = LTXVAudioVAEEncode.execute(voice, audio_vae)[0]
    latent = EmptyLTXVLatentVideo.execute(request['width'], request['height'], request['frames'], 1)[0]
    audio_latent = LTXVEmptyLatentAudio.execute(request['frames'], request['fps'], 1, audio_vae)[0]
    progress(stage='loading_model', message='Loading LTX 2.5...')
    selected = entry(request['model_id'])
    if selected['format'] == 'GGUF':
        load_extension('qwen_ltx_gguf', APP_DIR / 'vendor/ComfyUI-GGUF')
        from qwen_ltx_gguf.loader import gguf_sd_loader
        from qwen_ltx_gguf.ops import GGMLOps
        from qwen_ltx_gguf.nodes import GGUFModelPatcher
        weights, extra = gguf_sd_loader(str(local_path(selected, selected['files'][0])))
        operations = GGMLOps()
        operations.Linear.dequant_dtype = 'target'
        model = comfy.sd.load_diffusion_model_state_dict(weights,
            model_options={'custom_operations': operations, 'dtype': torch.bfloat16},
            metadata=extra.get('metadata', {}), disable_dynamic=True)
        if model is not None:
            model = GGUFModelPatcher.clone(model)
    else:
        weights, metadata = read_weights(selected['id'])
        model = comfy.sd.load_diffusion_model_state_dict(weights,
            model_options={'dtype': torch.bfloat16}, metadata=metadata, disable_dynamic=True)
    del weights
    if model is None:
        raise RuntimeError('This file is not a compatible LTX 2.5 model.')
    load_extension('qwen_ltx_msr', APP_DIR / 'vendor/LTX25-MSR')
    from qwen_ltx_msr.nodes import ComfyUILTX25MSRICLoRALoader, ComfyUILTX25MSRMultiReferenceGuide
    lora = entry('ltx25-msr-v2')
    lora_path = local_path(lora, lora['files'][0])
    folder_paths.add_model_folder_path('loras', str(lora_path.parent))
    model, parameters = ComfyUILTX25MSRICLoRALoader.execute(model, lora_path.name, request['lora_strength'])
    progress(model_format=selected['format'], model_weight_gb=round(model.model_size()/GIB, 2),
             text_encoder_id='ltx25-gemma-int8', vae_id='ltx25-video-vae', audio_vae_id='ltx25-audio-vae',
             lora_id='ltx25-msr-v2', actual_steps=total_steps, sampler='euler_ancestral', guidance=1.0)
    release()
    from .video_preview import LiveVideoPreview
    preview = LiveVideoPreview(preview_directory, model.model.latent_format, progress, request['frames']/request['fps'])
    progress(preview_url=None, preview_step=0, preview_unavailable=not preview.enabled)

    def sample_pass(video, audio, pos, neg, sigmas, offset):
        progress(stage='encoding_references', message='Encoding reference images...')
        with torch.inference_mode():
            pos, neg, guided = ComfyUILTX25MSRMultiReferenceGuide.execute(pos, neg, vae, video,
                strength=request['reference_strength'], reference_frames=str(request['reference_frames']),
                use_tiled_encode=True, tile_size=256, tile_overlap=64,
                msr_parameters=parameters, **refs, **audio_refs)
            joint = LTXVConcatAVLatent.execute(guided, audio)[0]
            # AVref prepends voice slots through native conditioning/model patches.
            # Preserve the guide's full joint noise mask; never denoise reference slots.
            noise = comfy.sample.prepare_noise(joint['samples'], request['seed'])
            def callback(step, estimate, noisy, total):
                progress(stage='generating', message='Refining video...' if offset else 'Generating video and audio...',
                         step=offset+step+1, total_steps=total_steps)
                video_estimate = estimate.unbind()[0] if estimate.is_nested else estimate
                # The native guide crop removes reference slots from the END.
                target_count = video['samples'].shape[2]
                preview.publish(video_estimate[:, :, :target_count], offset+step+1)
            progress(stage='generating', message='Refining video...' if offset else 'Generating video and audio...',
                     step=offset, total_steps=total_steps)
            output = comfy.sample.sample_custom(model, noise, 1.0,
                comfy.samplers.sampler_object('euler_ancestral'), torch.tensor(sigmas, dtype=torch.float32),
                pos, neg, joint['samples'], noise_mask=joint.get('noise_mask'), callback=callback, seed=request['seed'])
            sampled_video, sampled_audio = LTXVSeparateAVLatent.execute({**joint, 'samples': output})
            pos, neg, sampled_video = LTXVCropGuides.execute(pos, neg, sampled_video)
            return sampled_video, sampled_audio, pos, neg

    with torch.inference_mode():
        latent, audio_latent, positive, negative = sample_pass(latent, audio_latent, positive, negative, SIGMAS, 0)
        if request['refine']:
            from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
            from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
            progress(stage='upscaling_latent', message='Increasing video resolution...')
            upscale = entry('ltx25-spatial-upscaler')
            path = local_path(upscale, upscale['files'][0])
            folder_paths.add_model_folder_path('latent_upscale_models', str(path.parent))
            upsampler = LatentUpscaleModelLoader.execute(path.name)[0]
            latent = LTXVLatentUpsampler.execute(latent, upsampler, vae)[0]
            del upsampler
            release()
            latent, audio_latent, positive, negative = sample_pass(latent, audio_latent, positive, negative, REFINE_SIGMAS, 8)
        del model, positive, negative, parameters, refs, audio_refs
        release()
        progress(stage='decoding', message='Decoding video frames...')
        pixels = VAEDecodeTiled().decode(vae, latent, 512, 64, 64, 16)[0]
        pixels = pixels[:request['frames']].float().cpu()
        del vae, latent
        release()
        audio = None
        if request['audio']:
            progress(stage='decoding', message='Decoding audio...')
            audio = LTXVAudioVAEDecode.execute(audio_latent, audio_vae)[0]
            audio['waveform'] = audio['waveform'].float().cpu()
    return pixels, audio

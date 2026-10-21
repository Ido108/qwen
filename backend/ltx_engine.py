"""LTX-2.5 native AV inference with pinned ComfyUI nodes.

generate():       Reference images (MSR V2 IC-LoRA), with optional spatial refinement.
generate_video(): Create video (text, first frame, first and last frame) with the
                  official ComfyUI LTX-2.5 distilled workflows, without the MSR LoRA.
"""
from __future__ import annotations
import gc
import importlib.util
import logging
import sys
from .common import APP_DIR
from .models import entry, local_path
from .ltx_settings import (SIGMAS, REFINE_SIGMAS, LTX_VIDEO_NEGATIVE, LTX_FIRST_FRAME_STRENGTH,
    LTX_KEYFRAME_STRENGTH, LTX_REFINE_SEED, LTX_FPS)

MSR_NEGATIVE = 'subtitles, watermark, worst quality, blurry, jittery, distorted, inconsistent appearance'


def load_extension(name, directory):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _start_runtime(progress):
    import torch
    sys.path.insert(0, str(APP_DIR / 'vendor/ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload = args.disable_pinned_memory = args.disable_dynamic_vram = True
    from .gpu_memory import configure_video_memory
    configure_video_memory(torch, args, progress)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')
    logging.basicConfig(level=logging.INFO, format='%(message)s')


def _read_weights(model_id):
    import comfy.utils
    item = entry(model_id)
    weights, metadata = comfy.utils.load_safetensors(str(local_path(item, item['files'][0])))
    for tensor in weights.values():
        storage = tensor.untyped_storage()
        if hasattr(storage, '_comfy_tensor_file_slice'):
            delattr(storage, '_comfy_tensor_file_slice')
    return weights, metadata


def _path(model_id):
    item = entry(model_id)
    return local_path(item, item['files'][0])


def _prefetch(selected, request, *extra):
    """Large sequential reads into the file cache, in the order the job needs the files."""
    from .file_prefetch import prefetch
    loras = [choice['id'] for choice in _with_distill_lora(selected, request.get('loras')) or []]
    prefetch(_path('ltx25-gemma-int8'), _path('ltx25-video-vae'), _path('ltx25-audio-vae'), _path(selected['id']),
             *(_path(model_id) for model_id in [*extra, *loras]))


def _release():
    import comfy.model_management as memory
    gc.collect()
    memory.cleanup_models()
    memory.soft_empty_cache()


def _encode_text(prompt, negative_text, fps, progress):
    import torch
    import comfy.sd
    from . import sent_prompts
    sent_prompts.install(comfy.sd)  # the exact encoder input, for the result details
    import comfy.utils
    from comfy_extras.nodes_lt import LTXVConditioning
    from .gpu_memory import GIB
    weights, metadata = _read_weights('ltx25-gemma-int8')
    weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
    clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.LTXV,
        model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
    del weights, metadata
    clip.cond_stage_model.memory_estimation_function = lambda tokens, device=None: 3 * GIB
    progress(stage='encoding_text', message='Encoding the video description...')
    with torch.inference_mode():
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
    del clip
    _release()
    return LTXVConditioning.execute(positive, negative, fps)


def _load_vaes():
    import comfy.sd
    weights, metadata = _read_weights('ltx25-video-vae')
    vae = comfy.sd.VAE(sd=weights, metadata=metadata)
    del weights
    weights, metadata = _read_weights('ltx25-audio-vae')
    audio_vae = comfy.sd.VAE(sd=weights, metadata=metadata)
    del weights, metadata
    vae.throw_exception_if_invalid()
    audio_vae.throw_exception_if_invalid()
    return vae, audio_vae


def _load_transformer(selected):
    import torch
    import comfy.sd
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
        weights, metadata = _read_weights(selected['id'])
        model = comfy.sd.load_diffusion_model_state_dict(weights,
            model_options={'dtype': torch.bfloat16}, metadata=metadata, disable_dynamic=True)
    del weights
    if model is None:
        raise RuntimeError('This file is not a compatible LTX 2.5 model.')
    return model


def _with_distill_lora(selected, loras):
    """A dev checkpoint runs with the official distilled LoRA first, at the strength its page asks for."""
    if selected.get('distill_lora_strength'):
        return [{'id': 'ltx25-distill-lora', 'strength': selected['distill_lora_strength']}, *(loras or [])]
    return loras


def _apply_user_loras(model, loras, progress):
    """The user's LoRAs (for example from Civitai) go on top of the built-in ones, with the same
    key, size and attachment checks as image LoRAs; a LoRA that attaches nothing stops the job."""
    from .lora_loading import read_lora_weights, load_model_lora
    checks = []
    for choice in loras or []:
        item = entry(choice['id'])
        strength = float(choice.get('strength', 1.0))
        model, check = load_model_lora(model, read_lora_weights(local_path(item, item['files'][0])), strength, item['title'])
        checks.append({'id': choice['id'], 'title': item['title'], **check})
        progress(user_lora=choice['id'], user_lora_strength=strength)
    progress(lora_checks=checks)
    return _kitchen_attention(model, progress)


def _kitchen_attention(model, progress):
    """Comfy Kitchen INT8 attention, as the H3 engine uses; PyTorch attention when the kernel is missing."""
    import comfy.ldm.modules.attention as attention
    if not attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE:
        return model
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    progress(attention='comfy kitchen int8')
    return ModelAttentionBackend.execute(model, 'comfy kitchen attention')[0]


def _drop_cleared_guides(conditioning):
    keys = ('keyframe_idxs', 'guide_attention_entries')
    return [[tensor, {k: v for k, v in values.items() if not (k in keys and v is None)}]
            for tensor, values in conditioning]


def _load_upsampler(progress):
    import folder_paths
    from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
    progress(stage='upscaling_latent', message='Increasing video resolution...')
    upscale = entry('ltx25-spatial-upscaler')
    path = local_path(upscale, upscale['files'][0])
    folder_paths.add_model_folder_path('latent_upscale_models', str(path.parent))
    return LatentUpscaleModelLoader.execute(path.name)[0]


def _decode(vae, audio_vae, latent, audio_latent, frames, want_audio, progress):
    from nodes import VAEDecodeTiled
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode
    progress(stage='decoding', message='Decoding video frames...')
    # 32 GB holds larger tiles: 768 px covers a 1280x704 frame in 2 x 1 tiles instead of 3 x 2.
    pixels = VAEDecodeTiled().decode(vae, latent, 768, 96, 64, 16)[0]
    pixels = pixels[:frames].float().cpu()
    del vae, latent
    _release()
    audio = None
    if want_audio:
        progress(stage='decoding_audio', message='Decoding audio...')
        audio = LTXVAudioVAEDecode.execute(audio_latent, audio_vae)[0]
        audio['waveform'] = audio['waveform'].float().cpu()
    return pixels, audio


def generate(request, images, voices, progress, preview_directory=None):
    import numpy as np
    import torch
    _start_runtime(progress)
    import comfy.sample
    import comfy.samplers
    import folder_paths
    from comfy_extras.nodes_lt import (EmptyLTXVLatentVideo, LTXVConcatAVLatent, LTXVSeparateAVLatent, LTXVCropGuides)
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEEncode
    from .gpu_memory import GIB

    total_steps = 11 if request['refine'] else 8
    _prefetch(entry(request['model_id']), request, 'ltx25-msr-v2', *(['ltx25-spatial-upscaler'] if request['refine'] else []))
    progress(stage='loading_encoder', message='Loading the LTX text encoder...', total_steps=total_steps)
    positive, negative = _encode_text(request['prompt'], MSR_NEGATIVE, request['fps'], progress)
    progress(stage='loading_video_vae', message='Loading the video and audio encoders...')
    vae, audio_vae = _load_vaes()
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
    model = _load_transformer(selected)
    load_extension('qwen_ltx_msr', APP_DIR / 'vendor/LTX25-MSR')
    from qwen_ltx_msr.nodes import ComfyUILTX25MSRICLoRALoader, ComfyUILTX25MSRMultiReferenceGuide
    lora = entry('ltx25-msr-v2')
    lora_path = local_path(lora, lora['files'][0])
    folder_paths.add_model_folder_path('loras', str(lora_path.parent))
    model, parameters = ComfyUILTX25MSRICLoRALoader.execute(model, lora_path.name, request['lora_strength'])
    model = _apply_user_loras(model, _with_distill_lora(selected, request.get('loras')), progress)
    progress(model_format=selected['format'], model_weight_gb=round(model.model_size()/GIB, 2),
             text_encoder_id='ltx25-gemma-int8', vae_id='ltx25-video-vae', audio_vae_id='ltx25-audio-vae',
             lora_id='ltx25-msr-v2', actual_steps=total_steps, sampler='euler_ancestral', scheduler='ltx-distilled', guidance=1.0)
    _release()
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
            # The crop marks the guide keys as None; the MSR guide of the refinement pass
            # appends to them, so drop the cleared keys instead of passing None along.
            pos, neg = _drop_cleared_guides(pos), _drop_cleared_guides(neg)
            return sampled_video, sampled_audio, pos, neg

    with torch.inference_mode():
        latent, audio_latent, positive, negative = sample_pass(latent, audio_latent, positive, negative, SIGMAS, 0)
        if request['refine']:
            from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
            upsampler = _load_upsampler(progress)
            latent = LTXVLatentUpsampler.execute(latent, upsampler, vae)[0]
            del upsampler
            _release()
            latent, audio_latent, positive, negative = sample_pass(latent, audio_latent, positive, negative, REFINE_SIGMAS, 8)
        del model, positive, negative, parameters, refs, audio_refs
        _release()
        return _decode(vae, audio_vae, latent, audio_latent, request['frames'], request['audio'], progress)


def generate_video(request, prompt, images, progress, preview_directory=None):
    """Create video with LTX 2.5: text, first frame, or first and last frame.

    Text and first frame follow the official two-pass workflow: 8 distilled steps at
    half size, 2x latent upscale, then 3 refinement steps at full size. First and last
    frame follows the official single full-size pass with keyframe guides at 0.7.
    """
    import numpy as np
    import torch
    _start_runtime(progress)
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    from comfy_extras.nodes_lt import (EmptyLTXVLatentVideo, LTXVConcatAVLatent, LTXVSeparateAVLatent,
        LTXVCropGuides, LTXVImgToVideoInplace, LTXVAddGuide, LTXVPreprocess)
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio
    from .gpu_memory import GIB
    from .ltx_settings import ltx_video_two_pass
    from types import SimpleNamespace

    width, height, frames, fps = request['width'], request['height'], request['frames'], LTX_FPS
    keyframes = 'last_frame' in images
    two_pass = ltx_video_two_pass(SimpleNamespace(last_frame=keyframes))
    total_steps = len(SIGMAS) - 1 + (len(REFINE_SIGMAS) - 1 if two_pass else 0)
    _prefetch(entry(request['model_id']), request, *(['ltx25-spatial-upscaler'] if two_pass else []))
    progress(stage='loading_encoder', message='Loading the LTX text encoder...', total_steps=total_steps,
             text_encoder_id='ltx25-gemma-int8', vae_id='ltx25-video-vae', audio_vae_id='ltx25-audio-vae',
             fps=fps, frames=frames, width=width, height=height)
    positive, negative = _encode_text(prompt, LTX_VIDEO_NEGATIVE, fps, progress)
    progress(stage='loading_video_vae', message='Loading the video and audio encoders...')
    vae, audio_vae = _load_vaes()

    def to_tensor(image):
        return torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32) / 255).unsqueeze(0)

    pass_width, pass_height = (width // 2, height // 2) if two_pass else (width, height)
    first = last = None
    with torch.inference_mode():
        if keyframes:
            # Official: scale each keyframe to the output size with a centre crop, then LTXVPreprocess(18).
            def keyframe(image):
                pixels = comfy.utils.common_upscale(to_tensor(image).movedim(-1, 1), width, height,
                                                    'nearest-exact', 'center').movedim(1, -1)
                return LTXVPreprocess.execute(pixels, 18)[0]
            first, last = keyframe(images['first_frame']), keyframe(images['last_frame'])
        elif 'first_frame' in images:
            # Official: scale the longer side to 1536 px (Lanczos), then LTXVPreprocess(18).
            pixels = to_tensor(images['first_frame'])
            h, w = pixels.shape[1], pixels.shape[2]
            scale = 1536 / max(w, h)
            pixels = comfy.utils.common_upscale(pixels.movedim(-1, 1), max(1, round(w*scale)), max(1, round(h*scale)),
                                                'lanczos', 'disabled').movedim(1, -1)
            first = LTXVPreprocess.execute(pixels, 18)[0]
        latent = EmptyLTXVLatentVideo.execute(pass_width, pass_height, frames, 1)[0]
        audio_latent = LTXVEmptyLatentAudio.execute(frames, fps, 1, audio_vae)[0]
        if keyframes:
            progress(stage='encoding_references', message='Encoding the first and last frames...')
            positive, negative, latent = LTXVAddGuide.execute(positive, negative, vae, latent, first, 0, LTX_KEYFRAME_STRENGTH)
            positive, negative, latent = LTXVAddGuide.execute(positive, negative, vae, latent, last, -1, LTX_KEYFRAME_STRENGTH)
        elif first is not None:
            progress(stage='encoding_references', message='Encoding the first frame...')
            latent = LTXVImgToVideoInplace.execute(vae, first, latent, LTX_FIRST_FRAME_STRENGTH[0])[0]

    progress(stage='loading_model', message='Loading LTX 2.5...')
    selected = entry(request['model_id'])
    model = _load_transformer(selected)
    model = _apply_user_loras(model, _with_distill_lora(selected, request.get('loras')), progress)
    # Official FLF uses SamplerEulerAncestral(eta=0, s_noise=1); T2V/I2V use KSamplerSelect(euler_ancestral).
    sampler_name = 'euler_ancestral'
    sampler = (comfy.samplers.ksampler(sampler_name, {'eta': 0.0, 's_noise': 1.0}) if keyframes
               else comfy.samplers.sampler_object(sampler_name))
    progress(model_format=selected['format'], model_weight_gb=round(model.model_size()/GIB, 2),
             actual_steps=total_steps, sampler=sampler_name + (' (eta 0)' if keyframes else ''), scheduler='ltx-distilled', guidance=1.0,
             first_pass_width=pass_width, first_pass_height=pass_height, ltx_two_pass=two_pass, engine='ComfyUI')
    _release()
    from .video_preview import LiveVideoPreview
    preview = LiveVideoPreview(preview_directory, model.model.latent_format, progress, frames/fps)
    progress(preview_url=None, preview_step=0, preview_unavailable=not preview.enabled)

    def sample_pass(video, audio, sigmas, offset, seed):
        message = 'Refining video at full size...' if offset else 'Generating video and audio...'
        joint = LTXVConcatAVLatent.execute(video, audio)[0]
        noise = comfy.sample.prepare_noise(joint['samples'], seed)
        target_count = latent_frames
        def callback(step, estimate, noisy, total):
            progress(stage='generating', message=message, step=offset+step+1, total_steps=total_steps)
            video_estimate = estimate.unbind()[0] if estimate.is_nested else estimate
            preview.publish(video_estimate[:, :, :target_count], offset+step+1)
        progress(stage='generating', message=message, step=offset, total_steps=total_steps)
        output = comfy.sample.sample_custom(model, noise, 1.0, sampler, torch.tensor(sigmas, dtype=torch.float32),
            positive, negative, joint['samples'], noise_mask=joint.get('noise_mask'), callback=callback, seed=seed)
        return LTXVSeparateAVLatent.execute({**joint, 'samples': output})

    latent_frames = (frames - 1) // 8 + 1
    with torch.inference_mode():
        video, audio_latent = sample_pass(latent, audio_latent, SIGMAS, 0, request['seed'])
        if keyframes:
            positive, negative, video = LTXVCropGuides.execute(positive, negative, video)
        if two_pass:
            from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
            upsampler = _load_upsampler(progress)
            video = LTXVLatentUpsampler.execute(video, upsampler, vae)[0]
            del upsampler
            _release()
            if first is not None:
                video = LTXVImgToVideoInplace.execute(vae, first, video, LTX_FIRST_FRAME_STRENGTH[1])[0]
            video, audio_latent = sample_pass(video, audio_latent, REFINE_SIGMAS, len(SIGMAS) - 1, LTX_REFINE_SEED)
        del model, positive, negative
        _release()
        pixels, audio = _decode(vae, audio_vae, video, audio_latent, frames, request.get('audio', True), progress)
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated() / GIB, 2), actual_steps=total_steps)
    return pixels, audio

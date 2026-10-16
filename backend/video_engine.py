"""MiniMax H3 / FastH3 joint video and audio inference with pinned ComfyUI nodes."""
from __future__ import annotations
import gc
import importlib.util
import logging
import sys
from .common import APP_DIR
from .models import entry, local_path


def generate(request, prompt, images, progress, preview_directory=None):
    progress(stage='initializing_video', message='Initializing video processing…')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import numpy as np
    import torch
    sys.path.insert(0, str(APP_DIR / 'vendor/ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload = True
    args.disable_pinned_memory = True
    args.disable_dynamic_vram = True
    from .gpu_memory import configure_video_memory, GIB
    configure_video_memory(torch, args, progress)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():
        raise RuntimeError('Could not initialize the model file reader.')
    import comfy.sd
    import comfy.sample
    import comfy.utils
    import comfy.model_management as memory
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide, MiniMaxH3SigmaShift
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    from comfy_extras.nodes_audio import vae_decode_audio

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

    selected = entry(request['model_id'])
    reference = request.get('kind') == 'reference-video'
    if reference and request['workflow']=='crossview' and 'warp_video' not in images:
        from .camera_warp import build_camera_guide
        images['warp_video']=build_camera_guide(images['source_video'],request,progress)
        release()
    fast = selected['family'] == 'fasth3'
    turbo = not fast and request['quality'] == 'turbo'
    steps = request['steps'] if reference else (8 if fast or turbo else request['steps'])
    from .reference_sampling import reference_sampling
    sampling = reference_sampling(selected, request['quality'], request.get('steps')) if reference else None
    # Official H3 and FastH3 workflows use BasicGuider (CFG 1). Applying the
    # conventional video CFG 6 burns image-conditioned detail on this model.
    cfg = 1.0
    progress(stage='loading_encoder', message='Loading the video text encoder…', total_steps=steps,
             text_encoder_id='h3-encoder', vae_id='h3-video-vae', audio_vae_id='h3-audio-vae',
             fps=24, frames=request['frames'], width=request['width'], height=request['height'])
    weights, metadata = read_weights('h3-encoder')
    weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
    clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
    del weights, metadata
    # The 32B conditioning encoder has no activation estimate upstream. Reserve
    # its working space so ComfyUI streams some weights from RAM when needed.
    # This preserves the prompt, reference image resolution and video settings.
    encoding_scratch = (4 if images or reference else 2) * GIB
    clip.cond_stage_model.memory_estimation_function = lambda tokens, device=None: encoding_scratch
    progress(encoder_working_memory_gb=encoding_scratch/GIB)
    vae = None
    from .video_conditioning import SingleUseEncoder, ProgressVAE
    if images or reference:
        progress(stage='loading_video_vae', message='Loading the video encoder…')
        weights, _ = read_weights('h3-video-vae')
        vae = ProgressVAE(comfy.sd.VAE(sd=weights), progress)
        del weights
    tensors = {name: torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32) / 255).unsqueeze(0)
               for name, image in images.items() if hasattr(image,'convert')}
    encoder = SingleUseEncoder(clip, release, progress)
    del clip
    progress(stage='preparing_conditioning', message='Preparing video conditioning…')
    audio_vae = None
    voices = images.get('ref_audios') or {}
    if reference and (images.get('source_audio') is not None or voices):
        progress(stage='loading_audio_vae', message='Loading the audio encoder…')
        weights, _ = read_weights('h3-audio-vae')
        audio_vae = ProgressVAE(comfy.sd.VAE(sd=weights), progress, audio=True)
        del weights
    with torch.inference_mode():
        if reference:
            references={f'ref_image_{i+1}':tensors[f'ref_image_{i+1}'] for i in range(len(request['reference_images']))}
            # Reference images only (official r2v template) or with a source video.
            source=images.get('source_video')
            video_audios={'ref_video_audio_1':images['source_audio']} if source is not None and images.get('source_audio') is not None else None
            positive, latent=MiniMaxH3ReferenceToVideo.execute(encoder,prompt,
                request['width'],request['height'],request['frames'],
                ref_image_size=request['reference_image_size'],vae=vae,audio_vae=audio_vae,
                ref_images=references,ref_videos={'ref_video_1':source} if source is not None else None,
                ref_video_audios=video_audios,ref_audios=voices or None)
            if request['workflow']=='crossview':
                positive=MiniMaxH3AddGuide.execute(positive,latent,0,vae=vae,image=images['warp_video'])[0]
        else:
            positive, latent = MiniMaxH3ImageToVideo.execute(encoder, vae, prompt,
                request['width'], request['height'], request['frames'], **tensors)
        negative = []
    del encoder, tensors, vae, audio_vae
    release()
    progress(conditioning_cuda_allocated_gb=round(torch.cuda.memory_allocated()/GIB,2),
             conditioning_cuda_free_gb=round(torch.cuda.mem_get_info()[0]/GIB,2))
    progress(stage='loading_model', message='Loading the video model…')
    if selected['format'] == 'GGUF':
        module_dir = APP_DIR / 'vendor/ComfyUI-GGUF'
        spec = importlib.util.spec_from_file_location('qwen_video_gguf', module_dir / '__init__.py', submodule_search_locations=[str(module_dir)])
        package = importlib.util.module_from_spec(spec)
        sys.modules['qwen_video_gguf'] = package
        spec.loader.exec_module(package)
        from qwen_video_gguf import loader as gguf_loader
        # The pinned molbal converter uses this equivalent architecture label;
        # ComfyUI still detects and validates the actual H3 tensor structure.
        gguf_loader.IMG_ARCH_LIST.add('minimax_h3')
        from qwen_video_gguf.loader import gguf_sd_loader
        from qwen_video_gguf.dequant import dequantize_tensor, is_quantized
        from qwen_video_gguf.ops import GGMLOps
        from qwen_video_gguf.nodes import GGUFModelPatcher
        weights, extra = gguf_sd_loader(str(local_path(selected, selected['files'][0])))
        if fast and not any('to_gate_compress.weight' in name for name in weights):
            raise RuntimeError('This FastH3 file is missing its required sparse attention weights.')
        for name, tensor in list(weights.items()):
            if len(tensor.shape) == 1 and is_quantized(tensor):
                weights[name] = dequantize_tensor(tensor, dtype=torch.float32)
        operations = GGMLOps()
        operations.Linear.dequant_dtype = 'target'
        model = comfy.sd.load_diffusion_model_state_dict(weights,
            model_options={'custom_operations': operations, 'dtype': torch.bfloat16},
            metadata=extra.get('metadata', {}), disable_dynamic=True)
        if model is not None:
            model = GGUFModelPatcher.clone(model)
    else:
        weights, metadata = read_weights(selected['id'])
        if fast and not any('to_gate_compress.weight' in name for name in weights):
            raise RuntimeError('This FastH3 file is missing its required sparse attention weights.')
        model = comfy.sd.load_diffusion_model_state_dict(weights, model_options={'dtype': torch.bfloat16}, metadata=metadata, disable_dynamic=True)
    if model is None:
        raise RuntimeError('This file is not a compatible MiniMax H3 video model.')
    del weights
    turbo_lora = sampling.get('lora_id') if reference else ('h3-turbo' if turbo else None)
    if turbo_lora:
        lora, _ = read_weights(turbo_lora)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0)
        del lora
    if reference and request['workflow'] in ('faceswap','crossview'):
        lora_id='h3-ref-faceswap' if request['workflow']=='faceswap' else 'h3-ref-crossview'
        strength=1.0 if request['workflow']=='faceswap' else 0.8
        lora, _ = read_weights(lora_id)
        model, _ = comfy.sd.load_lora_for_models(model,None,lora,strength,0)
        del lora
    # The user's LoRAs (for example from Civitai) go on top of the built-in ones.
    for choice in request.get('loras') or []:
        lora, _ = read_weights(choice['id'])
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, float(choice.get('strength', 1.0)), 0)
        del lora
        progress(user_lora=choice['id'], user_lora_strength=float(choice.get('strength', 1.0)))
    video_shift = sampling['video_shift'] if sampling else (10.0 if fast else 12.0)
    audio_shift = sampling['audio_shift'] if sampling else 3.0
    model = MiniMaxH3SigmaShift.execute(model, video_shift, audio_shift)[0]
    model = ModelAttentionBackend.execute(model, 'comfy kitchen attention')[0]
    if fast:
        import comfy_kitchen as ck
        if not ck.sol_attn_is_available(torch.device('cuda')):
            raise RuntimeError('FastH3 requires the CUDA sparse attention kernel. Run Start Qwen.bat to repair the runtime.')
        model = BlockSparseAttention.execute(model, {'selection': 'vsa', 'keep_percent': 10.0},
            0.2, 1.0, min_tokens=12288, extra_tokens=256, verbose=True)[0]
        progress(attention='VSA-H3', sparse_kernel_available=True, video_sigma_shift=10.0, audio_sigma_shift=3.0)
    release()
    samples = latent['samples']
    streams=samples.unbind() if samples.is_nested else [samples]
    packed_elements=sum(stream[0].numel() for stream in streams)
    from .reference_memory import account_for_reference_memory
    reference_elements = account_for_reference_memory(model.model, positive) if reference else 0
    progress(model_format=selected['format'],
             reference_latent_elements=reference_elements,video_sigma_shift=video_shift,audio_sigma_shift=audio_shift,
             model_weight_gb=round(model.model_size()/GIB,2),
             inference_estimate_gb=round(model.model.memory_required([1,1,packed_elements])/GIB,2))
    del streams
    noise = comfy.sample.prepare_noise(samples, request['seed'])
    from .video_preview import LiveVideoPreview
    preview = LiveVideoPreview(preview_directory, model.model.latent_format, progress, request['frames']/24)
    progress(preview_url=None,preview_step=0,preview_unavailable=not preview.enabled)
    def callback(step, x0, x, total):
        progress(stage='generating', message='Generating video and audio…', step=step + 1, total_steps=total)
        if step==0:
            progress(model_gpu_weight_gb=round(model.loaded_size()/GIB,2),
                     model_offloaded_weight_gb=round((model.model_size()-model.loaded_size())/GIB,2))
        preview.publish(x0,step+1)
    sampler=sampling['sampler'] if sampling else 'res_multistep'
    scheduler=sampling['scheduler'] if sampling else 'simple'
    progress(stage='generating', message='Generating video and audio…', step=0, guidance=cfg, sampler=sampler, scheduler=scheduler)
    with torch.inference_mode():
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler, scheduler,
            positive, negative, samples, callback=callback, seed=request['seed'])
        del model, positive, negative, noise, latent
        release()
        progress(stage='decoding', message='Decoding video frames…')
        weights, _ = read_weights('h3-video-vae')
        vae = comfy.sd.VAE(sd=weights)
        del weights
        video_latent = samples.unbind()[0]
        pixels = vae.decode(video_latent)
        if pixels.ndim == 5:
            pixels = pixels.reshape(-1, *pixels.shape[-3:])
        pixels = pixels[:request['frames']].float().cpu()
        del vae, video_latent
        release()
        audio = None
        if request.get('audio', request.get('output_audio')=='generated'):
            progress(stage='decoding', message='Decoding audio…')
            weights, _ = read_weights('h3-audio-vae')
            vae = comfy.sd.VAE(sd=weights)
            del weights
            audio = vae_decode_audio(vae, {'samples': samples})
            audio['waveform'] = audio['waveform'].float().cpu()
            del vae
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2), engine='ComfyUI', actual_steps=steps)
    return pixels, audio

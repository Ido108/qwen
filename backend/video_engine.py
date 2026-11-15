"""MiniMax H3 / FastH3 joint video and audio inference with pinned ComfyUI nodes."""
from __future__ import annotations
import gc
import importlib.util
import logging
import sys
import time
from .common import APP_DIR, DATA_DIR
from .models import entry, local_path

CONDITIONING_DIR = DATA_DIR / 'cache' / 'h3-conditioning'


def _runtime(progress):
    """ComfyUI set up for H3 in this worker process, and the helpers every pass uses."""
    progress(stage='initializing_video', message='Initializing video processing…')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
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
    from . import sent_prompts
    sent_prompts.install(comfy.sd)  # the exact encoder input, for the result details
    import comfy.utils
    import comfy.model_management as memory

    def path_of(model_id):
        item = entry(model_id)
        return local_path(item, item['files'][0])

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

    return path_of, read_weights, release


# A long replacement keeps the video VAE in RAM between its parts instead of reading it again for each part.
_SHARED_VAE = {}


def _shared_video_vae(read_weights):
    import comfy.sd
    if 'video' not in _SHARED_VAE:
        weights, _ = read_weights('h3-video-vae')
        _SHARED_VAE['video'] = comfy.sd.VAE(sd=weights)
    return _SHARED_VAE['video']


def generate(request, prompt, images, progress, preview_directory=None):
    path_of, read_weights, release = _runtime(progress)
    import torch
    import comfy.model_management as memory
    from .gpu_memory import GIB
    selected = entry(request['model_id'])
    reference = request.get('kind') == 'reference-video'
    if reference and request['workflow']=='crossview' and 'warp_video' not in images:
        from .camera_warp import build_camera_guide
        images['warp_video']=build_camera_guide(images['source_video'],request,progress)
        release()
    fast = selected['family'] == 'fasth3'
    from .video_sampling import video_sampling
    sampling = request.get('sampling') or request.get('sampling_profile') or video_sampling(selected, request.get('quality'), request.get('steps'))
    from .res4lyf_runtime import ensure_sampler
    ensure_sampler(sampling['sampler'])
    steps = sampling['steps']
    # Official H3 and FastH3 workflows use BasicGuider (CFG 1). Applying the
    # conventional video CFG 6 burns image-conditioned detail on this model.
    cfg = 1.0
    from .video_conditioning import conditioning_key, load_conditioning
    from .file_prefetch import prefetch
    keyframes = {name: image for name, image in images.items() if hasattr(image, 'convert')}
    # Create video conditioning does not depend on the seed: a repeat reuses the saved
    # encoder output and skips loading the 32B encoder. Reference video is always encoded.
    cache_key = None if reference else conditioning_key({'encoder': entry('h3-encoder')['files'][0], 'vae': entry('h3-video-vae')['files'][0],
        'prompt': prompt, 'width': request['width'], 'height': request['height'], 'frames': request['frames']}, keyframes)
    cached = load_conditioning(CONDITIONING_DIR, cache_key) if cache_key else None
    # A part of a long replacement arrives with its conditioning already encoded (ReferenceEncoder).
    precomputed = images.get('precomputed_positive') if reference else None
    lora_ids = [sampling.get('lora_id')] + (['h3-ref-faceswap' if request['workflow'] == 'faceswap' else 'h3-ref-crossview']
        if reference and request['workflow'] in ('faceswap', 'crossview') else []) + [choice['id'] for choice in request.get('loras') or []]
    # Large sequential reads into the file cache, in the order the job needs the files.
    prefetch(*(path_of(i) for i in ([] if cached is not None or precomputed is not None else ['h3-encoder']) + (['h3-video-vae'] if images or reference else [])
               + (['h3-audio-vae'] if reference else [])), path_of(selected['id']), *(path_of(i) for i in lora_ids if i),
             path_of('h3-video-vae'), path_of('h3-audio-vae'))
    progress(stage='loading_encoder', message='Using saved prompt and keyframe data.' if cached is not None else 'Loading the video text encoder…',
             total_steps=steps, text_encoder_id='h3-encoder', vae_id='h3-video-vae', audio_vae_id='h3-audio-vae',
             conditioning_reused=cached is not None, fps=24, frames=request['frames'], width=request['width'], height=request['height'])
    decoder = None
    if precomputed is not None:
        positive, latent, negative, decoder = _from_precomputed(request, precomputed, images, read_weights, progress)
    elif cached is not None:
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        positive, latent, negative = cached, _empty_av_latent(request['width'], request['height'], request['frames'])[0], []
    else:
        positive, latent, negative, decoder = _encode(request, prompt, images, reference, keyframes, read_weights, release, progress, cache_key)
    release()
    progress(conditioning_cuda_allocated_gb=round(torch.cuda.memory_allocated()/GIB,2),
             conditioning_cuda_free_gb=round(torch.cuda.mem_get_info()[0]/GIB,2))
    progress(stage='loading_model', message='Loading the video model…')
    model = _load_model(selected, fast, read_weights)
    from .lora_loading import read_lora_weights, load_model_lora
    checks = []
    def apply(lora_id, strength):
        nonlocal model
        item = entry(lora_id)
        model, check = load_model_lora(model, read_lora_weights(local_path(item, item['files'][0])), strength, item['title'])
        checks.append({'id': lora_id, 'title': item['title'], **check})
    if sampling.get('lora_id'):
        apply(sampling['lora_id'], 1.0)
    if reference and request['workflow'] in ('faceswap','crossview'):
        apply('h3-ref-faceswap' if request['workflow']=='faceswap' else 'h3-ref-crossview', 1.0 if request['workflow']=='faceswap' else 0.8)
    # The user's LoRAs (for example from Civitai) go on top of the built-in ones.
    for choice in request.get('loras') or []:
        apply(choice['id'], float(choice.get('strength', 1.0)))
        progress(user_lora=choice['id'], user_lora_strength=float(choice.get('strength', 1.0)))
    progress(lora_checks=checks)
    return _sample_and_decode(request, selected, fast, sampling, steps, cfg, model, positive, negative, latent, decoder,
                              reference, read_weights, release, progress, preview_directory)


def _from_precomputed(request, positive, images, read_weights, progress):
    """Conditioning of one part encoded ahead (ReferenceEncoder); adds the guides that depend on the previous part."""
    import torch
    import comfy.model_management as memory
    from .video_timing import gpu_load_stage
    from .video_conditioning import ProgressVAE
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent, MiniMaxH3AddGuide
    latent = _empty_av_latent(request['width'], request['height'], request['frames'])[0]
    shared = _shared_video_vae(read_weights)
    shared.qwen_timing = lambda stage, message: gpu_load_stage(memory, progress, torch.cuda.synchronize,
        'loading_vae_gpu', 'Moving video encoder to GPU…', stage, message)
    vae = ProgressVAE(shared, progress)
    with torch.no_grad():
        if request['workflow'] == 'crossview':
            positive = MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, image=images['warp_video'])[0]
        if images.get('continuity_guide') is not None:
            positive = MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, image=images['continuity_guide'])[0]
    memory.unload_model_and_clones(shared.patcher)
    return positive, latent, [], shared


class _RecordingClip:
    """Stands in for the text encoder while the VAE pass runs MiniMaxH3ReferenceToVideo: it keeps the
    encoder input (prompt and reference items) for the encoder pass and returns an empty conditioning."""
    def __init__(self):
        self.prompt, self.items = None, None

    def tokenize(self, prompt, minimax_ref_items=None, **kwargs):
        self.prompt, self.items = prompt, minimax_ref_items
        return None

    def encode_from_tokens_scheduled(self, tokens):
        import torch
        return [[torch.zeros(1, 1, 1), {}]]


class ReferenceEncoder:
    """Encodes the parts of a long replacement with one load of each encoder, in two passes.

    MiniMaxH3ReferenceToVideo encodes the references with the VAE and then the prompt with the 32B text encoder.
    Run once per part, it loads the text encoder for every part. Here pass 1 (prepare) runs the node's VAE work
    for every part with a recording stand-in for the text encoder, and pass 2 (encode_all) loads the text encoder
    once for every part's prompt. The text encoder never shares the GPU with a VAE, so it is never partly evicted,
    and it is dropped at the end, never copied back to the CPU (that copy can crash on Windows, see
    SingleUseEncoder). Each part gets the conditioning MiniMaxH3ReferenceToVideo gives for the same inputs.
    """
    def __init__(self, progress):
        self.progress = progress
        self.path_of, self.read_weights, self.release = _runtime(progress)
        self.audio_vae = None

    def prepare(self, request, prompt, images):
        import numpy as np
        import torch
        import comfy.sd
        import comfy.model_management as memory
        from .video_timing import gpu_load_stage
        from .video_conditioning import ProgressVAE
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        progress = self.progress
        shared = _shared_video_vae(self.read_weights)
        shared.qwen_timing = lambda stage, message: gpu_load_stage(memory, progress, torch.cuda.synchronize,
            'loading_vae_gpu', 'Moving video encoder to GPU…', stage, message)
        vae = ProgressVAE(shared, progress)
        voices = images.get('ref_audios') or {}
        audio_vae = None
        if images.get('source_audio') is not None or voices:
            if self.audio_vae is None:
                weights, _ = self.read_weights('h3-audio-vae')
                self.audio_vae = comfy.sd.VAE(sd=weights)
                del weights
            self.audio_vae.qwen_timing = lambda stage, message: gpu_load_stage(memory, progress, torch.cuda.synchronize,
                'loading_audio_encoder_gpu', 'Moving audio encoder to GPU…', stage, message)
            audio_vae = ProgressVAE(self.audio_vae, progress, audio=True)
        references = {f'ref_image_{i + 1}': torch.from_numpy(np.asarray(images[f'ref_image_{i + 1}'].convert('RGB'), dtype=np.float32) / 255).unsqueeze(0)
                      for i in range(len(request['reference_images']))}
        source = images.get('source_video')
        soundtrack = {'ref_video_audio_1': images['source_audio']} if source is not None and images.get('source_audio') is not None else None
        clip = _RecordingClip()
        with torch.no_grad():
            positive, _ = MiniMaxH3ReferenceToVideo.execute(clip, prompt, request['width'], request['height'], request['frames'],
                ref_image_size=request['reference_image_size'], vae=vae, audio_vae=audio_vae, ref_images=references,
                ref_videos={'ref_video_1': source} if source is not None else None,
                ref_video_audios=soundtrack, ref_audios=voices or None)
        return {'prompt': clip.prompt, 'items': clip.items, 'refs': positive[0][1].get('minimax_refs'), 'frames': request['frames']}

    def encode_all(self, units):
        """The conditioning of every prepared part, with one load of the text encoder."""
        import torch
        import comfy.sd
        import comfy.utils
        import comfy.model_management as memory
        import node_helpers
        from .gpu_memory import GIB
        from .video_timing import gpu_load_stage
        progress = self.progress
        # The VAEs wait in RAM; the text encoder gets the GPU to itself.
        memory.unload_model_and_clones(_shared_video_vae(self.read_weights).patcher)
        if self.audio_vae is not None:
            memory.unload_model_and_clones(self.audio_vae.patcher)
        self.release()
        progress(stage='loading_encoder', message='Loading the video text encoder…')
        weights, metadata = self.read_weights('h3-encoder')
        weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
        clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
        del weights, metadata
        scratch = 4 * GIB
        clip.cond_stage_model.memory_estimation_function = lambda tokens, device=None: scratch
        positives = []
        try:
            with torch.no_grad():
                for index, unit in enumerate(units):
                    progress(stage='encoding_prompt', preparing_part=index + 1,
                             message=f'Encoding the prompt and references of part {index + 1} of {len(units)}…')
                    tokens = clip.tokenize(unit['prompt'], minimax_ref_items=unit['items'])
                    if index == 0:
                        with gpu_load_stage(memory, progress, torch.cuda.synchronize, 'loading_encoder_gpu', 'Moving text encoder to GPU…',
                                            'encoding_prompt', 'Encoding prompt and visual references on GPU…'):
                            cond = clip.encode_from_tokens_scheduled(tokens)
                    else:
                        cond = clip.encode_from_tokens_scheduled(tokens)
                    if unit['refs']:
                        cond = node_helpers.conditioning_set_values(cond, {'minimax_refs': unit['refs']})
                    positives.append(cond)
        finally:
            # Dropped, never copied back to the CPU (see SingleUseEncoder).
            clip = None
            self.release()
        return positives


def _encode(request, prompt, images, reference, keyframes, read_weights, release, progress, cache_key):
    import numpy as np
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as memory
    from .gpu_memory import GIB
    from .video_timing import gpu_load_stage
    from .video_conditioning import SingleUseEncoder, ProgressVAE, save_conditioning
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide
    weights, metadata = read_weights('h3-encoder')
    weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
    clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options={'initial_device': torch.device('cpu')}, disable_dynamic=True)
    clip.qwen_timing=lambda:gpu_load_stage(memory,progress,torch.cuda.synchronize,
        'loading_encoder_gpu','Moving text encoder to GPU…','encoding_prompt','Encoding prompt and visual references on GPU…')
    del weights, metadata
    # The 32B conditioning encoder has no activation estimate upstream. Reserve
    # its working space so ComfyUI streams some weights from RAM when needed.
    # This preserves the prompt, reference image resolution and video settings.
    encoding_scratch = (4 if images or reference else 2) * GIB
    clip.cond_stage_model.memory_estimation_function = lambda tokens, device=None: encoding_scratch
    progress(encoder_working_memory_gb=encoding_scratch/GIB)
    vae = None
    if images or reference:
        progress(stage='loading_video_vae', message='Loading the video encoder…')
        weights, _ = read_weights('h3-video-vae')
        vae = ProgressVAE(comfy.sd.VAE(sd=weights), progress)
        vae.vae.qwen_timing=lambda stage,message:gpu_load_stage(memory,progress,torch.cuda.synchronize,
            'loading_vae_gpu','Moving video encoder to GPU…',stage,message)
        del weights
    tensors = {name: torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32) / 255).unsqueeze(0)
               for name, image in keyframes.items()}
    encoder = SingleUseEncoder(clip, release, progress)
    del clip
    progress(stage='preparing_conditioning', message='Preparing video conditioning…')
    audio_vae = None
    voices = images.get('ref_audios') or {}
    if reference and (images.get('source_audio') is not None or voices):
        progress(stage='loading_audio_vae', message='Loading the audio encoder…')
        weights, _ = read_weights('h3-audio-vae')
        audio_vae = ProgressVAE(comfy.sd.VAE(sd=weights), progress, audio=True)
        audio_vae.vae.qwen_timing=lambda stage,message:gpu_load_stage(memory,progress,torch.cuda.synchronize,
            'loading_audio_encoder_gpu','Moving audio encoder to GPU…',stage,message)
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
            # A long face swap anchors the last frames of the previous part at the start of this one (long_faceswap.py).
            if images.get('continuity_guide') is not None:
                positive=MiniMaxH3AddGuide.execute(positive,latent,0,vae=vae,image=images['continuity_guide'])[0]
        else:
            positive, latent = MiniMaxH3ImageToVideo.execute(encoder, vae, prompt,
                request['width'], request['height'], request['frames'], **tensors)
        negative = []
    if cache_key:
        save_conditioning(CONDITIONING_DIR, cache_key, positive)
    # Keep the video VAE for decoding instead of reading it from disk again; it waits in RAM, not VRAM.
    decoder = vae.vae if vae is not None else None
    if decoder is not None:
        memory.unload_model_and_clones(decoder.patcher)
    del encoder, tensors, vae, audio_vae
    return positive, latent, negative, decoder


def _load_model(selected, fast, read_weights):
    import torch
    import comfy.sd
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
    return model


def _sample_and_decode(request, selected, fast, sampling, steps, cfg, model, positive, negative, latent, decoder,
                       reference, read_weights, release, progress, preview_directory):
    import torch
    import comfy.sd
    import comfy.sample
    import comfy.samplers
    import comfy.model_management as memory
    from .video_timing import gpu_load_stage
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    from comfy_extras.nodes_audio import vae_decode_audio
    from .gpu_memory import GIB
    video_shift = sampling['video_shift']
    audio_shift = sampling['audio_shift']
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
    preview_total=0.0
    def callback(step, x0, x, total):
        nonlocal preview_total
        # A callback may run before queued CUDA work finishes. Wait here so its
        # timestamp means a completed step, not merely submitted GPU work.
        torch.cuda.synchronize()
        progress(stage='generating', message='Generating video and audio…' if request.get('audio',True) else 'Generating video…', step=step + 1, total_steps=total)
        if step==0:
            progress(model_gpu_weight_gb=round(model.loaded_size()/GIB,2),
                     model_offloaded_weight_gb=round((model.model_size()-model.loaded_size())/GIB,2))
        preview_started=time.monotonic()
        preview.publish(x0,step+1)
        preview_total+=time.monotonic()-preview_started
        progress(preview_seconds_total=round(preview_total,3))
    sampler=sampling['sampler']
    scheduler=sampling['scheduler']
    from .video_sampling import scheduler_sigmas
    sigmas = scheduler_sigmas(sampling, model, MiniMaxH3SigmaShift.execute, comfy.samplers.calculate_sigmas)
    progress(stage='generating', message='Generating video and audio…' if request.get('audio',True) else 'Generating video…', step=0, guidance=cfg, sampler=sampler, scheduler=scheduler)
    with torch.inference_mode():
        with gpu_load_stage(memory,progress,torch.cuda.synchronize,'loading_model_gpu',
                'Moving video model to GPU and applying LoRAs…','generating',
                'Generating video and audio…' if request.get('audio',True) else 'Generating video…'):
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler, scheduler,
                positive, negative, samples, sigmas=sigmas, callback=callback, seed=request['seed'])
        progress(stage='releasing_model',message='Releasing video model for decoding…')
        del model, positive, negative, noise, latent
        release()
        progress(stage='loading_video_decoder',message='Preparing video decoder…')
        if decoder is not None:
            vae, decoder = decoder, None
        else:
            weights, _ = read_weights('h3-video-vae')
            vae = comfy.sd.VAE(sd=weights)
            del weights
        video_latent = samples.unbind()[0]
        progress(stage='decoding',message='Decoding video frames…')
        with gpu_load_stage(memory,progress,torch.cuda.synchronize,'loading_decoder_gpu',
                'Moving video decoder to GPU…','decoding','Decoding video frames…'):
            pixels = vae.decode(video_latent)
        if pixels.ndim == 5:
            pixels = pixels.reshape(-1, *pixels.shape[-3:])
        pixels = pixels[:request['frames']].float().cpu()
        del vae, video_latent
        release()
        audio = None
        if request.get('audio', request.get('output_audio')=='generated'):
            progress(stage='loading_audio_decoder', message='Preparing audio decoder…')
            weights, _ = read_weights('h3-audio-vae')
            vae = comfy.sd.VAE(sd=weights)
            del weights
            progress(stage='decoding_audio', message='Decoding audio…')
            with gpu_load_stage(memory,progress,torch.cuda.synchronize,'loading_audio_decoder_gpu',
                    'Moving audio decoder to GPU…','decoding_audio','Decoding audio…'):
                audio = vae_decode_audio(vae, {'samples': samples})
            audio['waveform'] = audio['waveform'].float().cpu()
            del vae
    progress(peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2), engine='ComfyUI', actual_steps=steps)
    return pixels, audio

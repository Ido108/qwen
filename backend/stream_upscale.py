"""FlashVSR and SwiftVR adapters with bounded input, exact frame count and playable previews."""
from __future__ import annotations
import importlib.util
import math
import sys
from .common import APP_DIR
from .models import entry, local_path


def checkpoint(model_id):
    item=entry(model_id)
    return local_path(item,item['files'][0])


def frame_batches(source, size, overlap=0):
    import av
    import numpy as np
    buffered=[]
    prefix=0
    with av.open(str(source)) as media:
        for frame in media.decode(video=0):
            buffered.append(frame.to_ndarray(format='rgb24'))
            if len(buffered)==size:
                yield np.stack(buffered),prefix
                buffered=buffered[-overlap:] if overlap else []
                prefix=len(buffered)
    if len(buffered)>prefix:
        yield np.stack(buffered),prefix


class Output:
    def __init__(self,directory,dimensions,fps,update):
        from .seedvr2_cli import H264Writer
        self.directory,self.dimensions,self.fps,self.update=directory,dimensions,fps,update
        self.writer=H264Writer(directory/'upscaled_raw.mp4',*dimensions,fps)
        self.count=0
        self.chunks=0

    def write(self,frames):
        import numpy as np
        import torch
        from .seedvr2_cli import publish_chunk
        if frames is None or len(frames)==0:return
        # RGB THWC float. Transfer one decoded frame at a time; never duplicate a full 4K clip.
        for frame in frames:
            array=frame.detach().clamp(0,1).mul(255).byte().cpu().numpy()
            if (array.shape[1],array.shape[0])!=self.dimensions:
                from PIL import Image
                array=np.asarray(Image.fromarray(array).resize(self.dimensions,Image.Resampling.LANCZOS))
            self.writer.write(np.ascontiguousarray(array[...,::-1]))
            self.count+=1
        self.chunks+=1
        try:
            details=publish_chunk(frames,self.directory,self.chunks,float(self.fps))
            if details:self.update(**details)
        except Exception as error:
            print(f'Optional preview unavailable: {error}',flush=True)
            self.update(preview_unavailable=True)
        self.update(output_frames_done=self.count,chunk=self.chunks)

    def close(self):self.writer.release()


def load_flash(request, selected, update):
    import torch
    sys.path.insert(0,str(APP_DIR/'vendor/ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.disable_async_offload=True
    args.disable_pinned_memory=True
    args.disable_dynamic_vram=True
    from .gpu_memory import configure_video_memory
    configure_video_memory(torch,args,update)
    import comfy_aimdo.control
    if not comfy_aimdo.control.init():raise RuntimeError('Could not initialize the model file reader.')
    path=APP_DIR/'vendor/FlashVSR-Stock'
    spec=importlib.util.spec_from_file_location('qwen_flashvsr',path/'__init__.py',submodule_search_locations=[str(path)])
    package=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=package
    spec.loader.exec_module(package)
    from qwen_flashvsr import nodes
    import folder_paths
    # The model and its support files can sit on different drives (D:\models library first, the data folder second).
    for model_id in (selected['id'],'flashvsr-prompt','flashvsr-lq-proj-in','flashvsr-tcdecoder-fp16'):
        folder_paths.add_model_folder_path('flashvsr',str(checkpoint(model_id).parent),is_default=model_id==selected['id'])
    model=nodes.FlashVSRModelLoader().load(checkpoint(selected['id']).name)[0]
    prompt=nodes.FlashVSRPromptLoader().load(checkpoint('flashvsr-prompt').name)[0]
    lq=nodes.FlashVSRLQLoader().load(checkpoint('flashvsr-lq-proj-in').name,'bf16')[0]
    if request.get('flash_decoder')=='full':
        import comfy.sd,comfy.utils
        decoder=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(checkpoint('flashvsr-wan-vae')),safe_load=True))
    else:
        decoder=nodes.FlashVSRTCDecoderLoader().load(checkpoint('flashvsr-tcdecoder-fp16').name,'bf16')[0]
    return nodes,model,prompt,lq,decoder


def flash(request,selected,source,output,update):
    import torch
    nodes,model,prompt,lq,decoder=load_flash(request,selected,update)
    import comfy.sample
    import comfy.utils
    import comfy.model_management as memory
    from comfy_extras.nodes_custom_sampler import Guider_Basic
    size=request.get('flash_chunk_size',33)
    overlap=min(8,size-1)
    from .gpu_memory import dedicated_memory_snapshot
    # 24 GB or more: attention windows sized from free VRAM (0). The decoder keeps 1-frame groups: on an RTX 5090
    # (100 frames to 1080x1928) query chunk 0 took the sampling from 309 s to 92 s, while 4-frame decoder groups
    # took decoding from 13 s to over 398 s with the GPU at 7% (2026-10-21).
    large=(dedicated_memory_snapshot().get('free_gb') or 0)>=24
    query_chunk=0 if large else 1
    output_chunk,decode_batch=1,1
    update(flash_query_block_chunk=query_chunk,flash_temporal_batch=decode_batch)
    for index,(batch,skip) in enumerate(frame_batches(source,size,overlap),1):
        update(stage='encoding',message=f'Preparing frames · section {index}',step=0,total_steps=0)
        images=torch.from_numpy(batch).float().div_(255)
        prepared,latent,_,_,_=nodes.FlashVSRPrepareVideo().prepare(images,output.dimensions[0]/batch.shape[2])
        prepared.output_width,prepared.output_height=output.dimensions
        patched,runtime=nodes.FlashVSRApply().apply(model,lq,prepared,request.get('flash_conditioning',1.0))
        sampler,sigmas=nodes.FlashVSRStreamingSampler().build(runtime,
            request.get('flash_sampling','streaming_faithful_lowvram'),request.get('flash_sparse_ratio',2.0),
            request.get('flash_local_range',11),query_block_chunk=query_chunk,new_latent_frames=2,
            cache_format=request.get('flash_cache_format','int8'),cache_residency_backend='cpu')
        guider=Guider_Basic(patched)
        guider.set_conds(prompt)
        noise=comfy.sample.prepare_noise(latent['samples'],request['seed']+index-1)
        update(stage='upscaling',message=f'Upscaling frames · section {index}',step=0,total_steps=1)
        with torch.inference_mode():
            comfy.utils.set_progress_bar_global_hook(lambda value,total,preview=None,**kwargs: update(
                message=f'Upscaling frames · section {index} · {value}/{total} batches',step=value,total_steps=total))
            try:
                samples=guider.sample(noise,latent['samples'],sampler,sigmas,disable_pbar=False,seed=request['seed']+index-1)
            finally:
                comfy.utils.set_progress_bar_global_hook(None)
            update(stage='decoding',message=f'Decoding frames · section {index}',step=0,total_steps=0)
            if request.get('flash_decoder')=='full':
                decoded=decoder.decode_tiled(samples.to(memory.intermediate_device()),tile_x=64,tile_y=64,overlap=8,tile_t=8,overlap_t=2)
                if decoded.ndim==5:decoded=decoded.flatten(0,1)
                decoded=nodes.FlashVSRCropFrames().crop(decoded,prepared)[0]
            else:
                decoded=nodes.FlashVSRTCDecode().decode({'samples':samples.to(memory.intermediate_device())},
                    decoder,prepared,output_chunk_size=output_chunk,temporal_batch_size=decode_batch)[0]
            if request.get('flash_color','off')!='off':
                from qwen_flashvsr.color import apply_color_correction
                decoded=apply_color_correction(decoded,prepared,method=request['flash_color'],chunk_size=1,compute_device='cuda',inplace=True)
            if len(decoded)!=len(batch):raise RuntimeError('FlashVSR returned an incomplete section. Source frames were kept.')
            output.write(decoded[skip:])
        del images,prepared,latent,patched,runtime,sampler,guider,samples,noise,decoded


def load_swift(request,selected,update):
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0,str(APP_DIR/'vendor/SwiftVR'))
    from swiftvr import SwiftVRPipeline
    from swiftvr.models import ReAE,WanTransformer3DModel
    root=checkpoint(selected['id']).parent.parent
    # from_pretrained reads config.json next to the weights; a group split across D:\models and the data folder
    # can hold it on the other drive.
    config=checkpoint('swiftvr-config');weights_config=root/'transformer'/config.name
    if config!=weights_config and not weights_config.is_file():
        import shutil
        shutil.copy2(config,weights_config)
    transformer=WanTransformer3DModel.from_pretrained(str(root),subfolder='transformer',
        torch_dtype=torch.bfloat16,local_files_only=True)
    reae=ReAE(str(checkpoint('swiftvr-reae')))
    prompt=load_file(str(checkpoint('swiftvr-prompt-embedding')))['prompt_emb'][0]
    return SwiftVRPipeline(reae,transformer,prompt).to('cuda',dtype='bfloat16',attention_backend='sdpa',torch_compile=False)


def swift(request,selected,source,output,update):
    import torch
    pipe=load_swift(request,selected,update)
    # The paper uses 24. Smaller multiples of four bound activations on consumer GPUs.
    size=request.get('clip_len',8)
    session=pipe.stream(clip_len=size,resolution=output.dimensions,dit_overlap=0)
    seen=0
    last=None
    def save(result):
        if result is not None:
            frames=result[0].permute(0,2,3,1)
            output.write(frames[:max(0,seen-output.count)])
    for index,(batch,_) in enumerate(frame_batches(source,size),1):
        seen+=len(batch)
        last=torch.from_numpy(batch[-1:]).clone()
        update(stage='upscaling',message=f'Upscaling frames · section {index}',step=0,total_steps=0)
        save(session.step(torch.from_numpy(batch)))
    if last is not None:
        # ReAE removes three causal head frames; supply the matching three-frame
        # tail, flush temporal padding and trim to ALL input frames (not 4k+1).
        trim=int(pipe.reae.frames_to_trim)
        if trim:save(session.step(last.repeat(trim,1,1,1)))
        save(session.flush())
    if output.count!=seen:
        raise RuntimeError(f'SwiftVR returned {output.count} of {seen} frames. Source frames were kept.')


def run(request,selected,source,directory,update,dimensions):
    import av
    import torch
    if not torch.cuda.is_available():raise RuntimeError('This upscaler needs an NVIDIA GPU. Choose RunPod in Settings on Mac.')
    with av.open(str(source)) as media:
        stream=media.streams.video[0]
        fps=stream.average_rate or stream.guessed_rate
        if not fps:raise ValueError('The source video has no valid frame rate.')
        frames=stream.frames
    update(stage='loading_model',message=f'Loading {selected["model_group"]}…',fps=float(fps),frames=frames,
           actual_steps=1,model_revision=selected['revision'],seed_used=selected['family']!='swiftvr')
    output=Output(directory,dimensions,fps,update)
    try:
        with torch.inference_mode():
            (flash if selected['family']=='flashvsr' else swift)(request,selected,source,output,update)
    finally:
        output.close()
    if not output.count:raise RuntimeError('No video frames were produced.')
    update(frames=output.count,total_chunks=output.chunks,step=0,total_steps=0)

"""Run the pinned SeedVR2 CLI in the existing queue and keep source audio."""
from __future__ import annotations

import os
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .common import APP_DIR, MODELS_DIR, resolve_video, write_json
from .models import entry, local_path

CHUNK=re.compile(r'Chunk\s+(\d+)\s*/\s*(\d+)',re.IGNORECASE)
BATCH=re.compile(r'(Encoding|Upscaling|Decoding|Post-processing) batch\s+(\d+)\s*/\s*(\d+)',re.IGNORECASE)
PHASE={'encoding':('encoding','Encoding source frames'),
       'upscaling':('upscaling','Upscaling frames'),
       'decoding':('decoding','Decoding upscaled frames'),
       'post-processing':('saving','Preparing output frames')}


def run(request, job, directory, update):
    import av
    selected=entry(request['model_id'])
    from .upscale_settings import output_dimensions, seed_memory_options, seed_vae_options
    source=resolve_video(request['source_video'])
    with av.open(str(source)) as media:
        stream=media.streams.video[0]
        width,height=output_dimensions(stream.width,stream.height,request)
        source_size=dict(source_width=stream.width,source_height=stream.height)
    update(width=width,height=height,**source_size,actual_steps=1,model_id=selected['id'],model_format=selected['format'])
    if selected['family'] in ('flashvsr','swiftvr'):
        from .stream_upscale import run as run_streaming
        run_streaming(request,selected,source,directory,update,(width,height))
        return finish_output(request,job,directory,update)
    model_dir,model_name=seed_model_dir(selected)
    raw=directory/'upscaled_raw.mp4'
    short=min(width,height)
    cmd=[sys.executable,'-u','-m','backend.seedvr2_cli',str(source),
         '--output',str(raw),'--output_format','mp4',
         '--model_dir',str(model_dir),'--dit_model',model_name,
         '--resolution',str(short),
         '--batch_size',str(request['batch_size']),
         '--chunk_size',str(request['chunk_size']),
         '--temporal_overlap',str(request['temporal_overlap']),
         '--seed',str(request['seed']),'--uniform_batch_size',
         '--color_correction',request.get('color_correction','lab')]
    from .gpu_memory import dedicated_memory_snapshot
    gpu=dedicated_memory_snapshot()
    free=gpu.get('free_gb')
    memory=seed_memory_options(selected,{**request,'resolution':short},free)
    fast=seed_vae_options(short,free)+memory
    # No --compile_dit: on Windows its first compile held batch 1 for over 6 minutes on one CPU core with the
    # GPU idle (2026-10-21, 3B FP16, 1080p), against 63 s for the whole uncompiled DiT stage.
    safe=seed_vae_options(short,free,conservative=True)+seed_memory_options(selected,{**request,'resolution':short,'memory_mode':'low_memory'},free)
    if gpu.get('available'):update(dedicated_free_gb=free,gpu_total_gb=gpu['total_gb'])
    environment=os.environ.copy()
    environment['HF_HUB_OFFLINE']='1'
    environment['PYTHONUTF8']='1'
    environment['QWEN_UPSCALE_WIDTH']=str(width)
    environment['QWEN_UPSCALE_HEIGHT']=str(height)
    update(stage='loading_model',message='Loading SeedVR2 upscaling model…',
           model_id=selected['id'],model_format=selected['format'],upscale_options=' '.join(fast))
    code=run_seed(cmd+fast,environment,update)
    if code and fast!=safe and request.get('memory_mode','auto')=='auto':
        # Larger batches, untiled VAE and compile are estimates; one retry uses the memory-saving path.
        raw.unlink(missing_ok=True)
        update(stage='loading_model',message='Retrying with GPU memory saving…',upscale_options=' '.join(safe),upscale_retry=True)
        code=run_seed(cmd+safe,environment,update)
    if code:
        raise RuntimeError(f'SeedVR2 stopped with code {code}. See the job log for details.')
    if not raw.is_file() or raw.stat().st_size==0:
        raise RuntimeError('SeedVR2 finished without an output video.')

    finish_output(request,job,directory,update)


def seed_model_dir(selected):
    """The folder and file name of a SeedVR2 DiT, with its VAE beside it; shared by the video and image upscalers."""
    model_name=Path(selected['files'][0]['path']).name
    model_dir=local_path(selected,selected['files'][0]).parent
    # SeedVR2 reads the DiT and its VAE from one folder; the D:\models library and the data folder can each hold one of them.
    vae=entry('seedvr2-vae');vae_path=local_path(vae,vae['files'][0])
    if vae_path.parent!=model_dir and not (model_dir/vae_path.name).is_file():
        shutil.copy2(vae_path,model_dir/(vae_path.name+'.part'));(model_dir/(vae_path.name+'.part')).replace(model_dir/vae_path.name)
    cli=APP_DIR/'vendor'/'SeedVR2'/'inference_cli.py'
    if not cli.is_file():
        raise RuntimeError('The SeedVR2 inference package is missing. Run Start Qwen.bat again.')
    return model_dir,model_name


def run_seed(cmd, environment, update, phases=PHASE):
    process=subprocess.Popen(cmd,cwd=APP_DIR,env=environment,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             encoding='utf-8',errors='replace',bufsize=1,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        for line in process.stdout:
            clean=line.strip()
            if clean:
                print(clean,flush=True)
            if clean.startswith('QWEN_PREVIEW '):
                update(**json.loads(clean[len('QWEN_PREVIEW '):]))
            match=CHUNK.search(clean)
            if match:
                index,total=map(int,match.groups())
                update(message=f'Processing clip {index} of {total}…',chunk=index,total_chunks=total)
            batch=BATCH.search(clean)
            if batch:
                name,index,total=batch.group(1).lower(),int(batch.group(2)),int(batch.group(3))
                stage,label=phases[name]
                update(stage=stage,message=f'{label} · {index} of {total}',
                       step=index,total_steps=total)
        return process.wait()
    finally:
        process.stdout.close()


def finish_output(request, job, directory, update, *, keep_raw=False):
    """Finish an already upscaled clip; also usable for recovery without inference."""
    import av
    from PIL import Image
    from .video_audio import add_source_audio
    raw=directory/'upscaled_raw.mp4'
    source=resolve_video(request['source_video'])
    selected=entry(request['model_id'])
    update(stage='saving',message='Saving the upscaled video…',step=0,total_steps=0)
    destination=Path(request['output_dir'])/(time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+job['id'][:8]+'_upscaled.mp4')
    temporary=destination.with_name(destination.stem+'.partial.mp4')
    try:
        add_source_audio(raw,source,temporary,update,preserve_audio=request['preserve_audio'])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    update(stage='saving',message='Preparing video preview…')
    with av.open(str(destination)) as media:
        video=media.streams.video[0]
        frame=next(media.decode(video))
        width,height=frame.width,frame.height
        fps=float(video.average_rate) if video.average_rate else 24.0
        frames=int(video.frames or 0)
        duration=media.duration/1_000_000 if media.duration else (frames/fps if frames else 0)
        has_audio=bool(media.streams.audio)
        poster=frame.to_image().convert('RGB')
        poster.thumbnail((960,540),Image.Resampling.LANCZOS)
        poster.save(directory/'poster.png')
    update(status='succeeded',stage='complete',message='Upscaled video saved.',
           output_path=str(destination),video_url=f'/api/jobs/{job["id"]}/video',
           poster_url=f'/api/jobs/{job["id"]}/poster',finished_at=time.time(),
           elapsed_seconds=round(time.time()-job['started_at'],1),
           width=width,height=height,fps=fps,frames=frames,
           duration_seconds=round(duration,3),has_audio=has_audio,
           model_revision=selected['revision'])
    write_json(destination.with_suffix('.json'),job)
    if not keep_raw:
        raw.unlink(missing_ok=True)

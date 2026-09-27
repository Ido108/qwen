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
    from .upscale_settings import output_dimensions, seed_memory_options
    source=resolve_video(request['source_video'])
    with av.open(str(source)) as media:
        stream=media.streams.video[0]
        width,height=output_dimensions(stream.width,stream.height,request)
    update(width=width,height=height,actual_steps=1,model_id=selected['id'],model_format=selected['format'])
    if selected['family'] in ('flashvsr','swiftvr'):
        from .stream_upscale import run as run_streaming
        run_streaming(request,selected,source,directory,update,(width,height))
        return finish_output(request,job,directory,update)
    model_name=Path(selected['files'][0]['path']).name
    model_dir=local_path(selected,selected['files'][0]).parent
    cli=APP_DIR/'vendor'/'SeedVR2'/'inference_cli.py'
    if not cli.is_file():
        raise RuntimeError('The SeedVR2 inference package is missing. Run Setup.bat again.')
    raw=directory/'upscaled_raw.mp4'
    cmd=[sys.executable,'-u','-m','backend.seedvr2_cli',str(source),
         '--output',str(raw),'--output_format','mp4',
         '--model_dir',str(model_dir),'--dit_model',model_name,
         '--resolution',str(min(width,height)),
         '--batch_size',str(request['batch_size']),
         '--chunk_size',str(request['chunk_size']),
         '--temporal_overlap',str(request['temporal_overlap']),
         '--seed',str(request['seed']),'--uniform_batch_size',
         '--color_correction',request.get('color_correction','lab')]
    if min(width,height)>=1080:
        cmd.extend(['--vae_encode_tiled','--vae_decode_tiled'])
    from .gpu_memory import dedicated_memory_snapshot
    gpu=dedicated_memory_snapshot()
    cmd.extend(seed_memory_options(selected,{**request,'resolution':min(width,height)},gpu.get('free_gb')))
    if gpu.get('available'):update(dedicated_free_gb=gpu['free_gb'],gpu_total_gb=gpu['total_gb'])
    environment=os.environ.copy()
    environment['HF_HUB_OFFLINE']='1'
    environment['PYTHONUTF8']='1'
    environment['QWEN_UPSCALE_WIDTH']=str(width)
    environment['QWEN_UPSCALE_HEIGHT']=str(height)
    update(stage='loading_model',message='Loading SeedVR2 upscaling model…',
           model_id=selected['id'],model_format=selected['format'])
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
                stage,label=PHASE[name]
                update(stage=stage,message=f'{label} · {index} of {total}',
                       step=index,total_steps=total)
        code=process.wait()
    finally:
        process.stdout.close()
    if code:
        raise RuntimeError(f'SeedVR2 stopped with code {code}. See the job log for details.')
    if not raw.is_file() or raw.stat().st_size==0:
        raise RuntimeError('SeedVR2 finished without an output video.')

    finish_output(request,job,directory,update)


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

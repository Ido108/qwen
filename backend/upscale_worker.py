"""Run the pinned SeedVR2 CLI in the existing queue and keep source audio."""
from __future__ import annotations

import os
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
    from PIL import Image

    selected=entry(request['model_id'])
    model_name=Path(selected['files'][0]['path']).name
    model_dir=local_path(selected,selected['files'][0]).parent
    source=resolve_video(request['source_video'])
    cli=APP_DIR/'vendor'/'SeedVR2'/'inference_cli.py'
    if not cli.is_file():
        raise RuntimeError('The SeedVR2 inference package is missing. Run Setup.bat again.')
    raw=directory/'upscaled_raw.mp4'
    cmd=[sys.executable,'-u',str(cli),str(source),
         '--output',str(raw),'--output_format','mp4',
         '--model_dir',str(model_dir),'--dit_model',model_name,
         '--resolution',str(request['resolution']),
         '--batch_size',str(request['batch_size']),
         '--chunk_size',str(request['chunk_size']),
         '--temporal_overlap',str(request['temporal_overlap']),
         '--seed',str(request['seed']),'--uniform_batch_size']
    if request['resolution']>=1080:
        cmd.extend(['--vae_encode_tiled','--vae_decode_tiled'])
    if request['resolution']>=2160 or selected['id']=='seedvr2-7b-fp8':
        cmd.extend(['--dit_offload_device','cpu','--blocks_to_swap',
                    '20' if selected['id']=='seedvr2-7b-fp8' else '16'])
    environment=os.environ.copy()
    environment['HF_HUB_OFFLINE']='1'
    environment['PYTHONUTF8']='1'
    update(stage='loading_model',message='Loading SeedVR2 upscaling model…',
           model_id=selected['id'],model_format=selected['format'])
    process=subprocess.Popen(cmd,cwd=cli.parent,env=environment,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             encoding='utf-8',errors='replace',bufsize=1,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        for line in process.stdout:
            clean=line.strip()
            if clean:
                print(clean,flush=True)
            match=CHUNK.search(clean)
            if match:
                index,total=map(int,match.groups())
                update(message=f'Processing clip {index} of {total}…')
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

    update(stage='saving',message='Saving the upscaled video…')
    destination=Path(request['output_dir'])/(time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+job['id'][:8]+'_upscaled.mp4')
    temporary=destination.with_name(destination.stem+'.partial.mp4')
    try:
        if request['preserve_audio']:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            result=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y',
                '-i',str(raw),'-i',str(source),'-map','0:v:0','-map','1:a:0?',
                '-c:v','copy','-c:a','aac','-b:a','192k','-af','apad','-shortest',
                '-movflags','+faststart',str(temporary)],
                capture_output=True,text=True,encoding='utf-8',errors='replace')
            if result.returncode:
                raise RuntimeError('Could not preserve source audio: '+result.stderr[-900:])
        else:
            shutil.copyfile(raw,temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)

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

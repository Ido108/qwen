"""Keep the pinned SeedVR2 CLI intact; add playable output and completed-chunk previews."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class H264Writer:
    def __init__(self,path,width,height,fps,use_10bit=False):
        import imageio_ffmpeg
        target_width=int(os.environ.get('QWEN_UPSCALE_WIDTH',width))
        target_height=int(os.environ.get('QWEN_UPSCALE_HEIGHT',height))
        resize=['-vf',f'scale={target_width}:{target_height}:flags=lanczos,setsar=1'] if (target_width,target_height)!=(width,height) else ['-vf','setsar=1']
        self.process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
            '-an',*resize,'-c:v','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','12','-movflags','+faststart',str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    def isOpened(self):return self.process.poll() is None
    def write(self,frame):
        if not self.isOpened():raise RuntimeError('The video encoder stopped before the output was complete.')
        self.process.stdin.write(frame.tobytes())
    def release(self):
        self.process.stdin.close()
        try:code=self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait();raise RuntimeError('Saving the upscaled frames timed out.')
        if code:raise RuntimeError('Could not save the upscaled frames.')


def publish_chunk(frames,directory,chunk,fps):
    """Small animated samples from decoded output; never another model pass."""
    from PIL import Image
    count=len(frames)
    if not count:return None
    indexes=sorted({round(i*(count-1)/max(1,min(12,count)-1)) for i in range(min(12,count))})
    pictures=[]
    for index in indexes:
        array=frames[index].detach().float().cpu().clamp(0,1).mul(255).byte().numpy()
        picture=Image.fromarray(array);picture.thumbnail((512,512));pictures.append(picture)
    temporary=directory/'live-preview.tmp'
    pictures[0].save(temporary,format='WEBP',save_all=True,append_images=pictures[1:],
                     duration=max(80,min(600,round(count/fps/len(pictures)*1000))),loop=0,quality=75,method=0)
    temporary.replace(directory/'live-preview.webp')
    details={'preview_url':f'/api/jobs/{directory.name}/live-preview',
        'preview_chunk':chunk,'preview_updated_at':time.time(),'preview_unavailable':False}
    print('QWEN_PREVIEW '+json.dumps(details),flush=True)
    return details


def main():
    cli=Path(__file__).resolve().parents[1]/'vendor/SeedVR2/inference_cli.py'
    sys.path.insert(0,str(cli.parent))
    spec=importlib.util.spec_from_file_location('qwen_seedvr2_inference',cli)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from src.utils.model_registry import MODEL_REGISTRY, ModelInfo
    from .models import catalog
    for item in catalog():
        if item.get('kind')!='upscale' or item.get('family')!='seedvr2':continue
        file=item['files'][0]
        name=Path(file['path']).name
        MODEL_REGISTRY[name]=ModelInfo(repo=item['repo'],size='7B' if '7b' in name else '3B',
            precision=item.get('quantization','fp16'),variant='sharp' if 'sharp' in name else None,sha256=file.get('sha256'))
    # Force the bundled encoder, without relying on a globally installed ffmpeg.
    original=module.save_frames_to_video
    module.FFMPEGVideoWriter=H264Writer
    count=0
    def save(frames,output_path,fps=30.0,**kwargs):
        nonlocal count
        kwargs['video_backend']='ffmpeg'
        writer=original(frames,output_path,fps,**kwargs)
        count+=1
        try:publish_chunk(frames,Path(output_path).parent,count,fps)
        except Exception as error:
            print('QWEN_PREVIEW '+json.dumps({'preview_unavailable':True}),flush=True)
            print(f'Optional preview unavailable: {error}',flush=True)
        return writer
    module.save_frames_to_video=save
    module.main()


if __name__=='__main__':main()

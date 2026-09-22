"""Inspect a generated MP4 and save a local report/contact sheet for visual QA."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw

def validate(path, destination):
    destination.mkdir(parents=True,exist_ok=True)
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate)
        result=dict(file=path.name,bytes=path.stat().st_size,width=stream.width,height=stream.height,fps=fps,video_codec=stream.codec_context.name)
        frames=[];means=[];deltas=[];previous=None;times=[]
        for frame in container.decode(video=0):
            array=frame.to_ndarray(format='rgb24')
            means.append(float(array.mean()));times.append(float(frame.time))
            if previous is not None:
                deltas.append(float(np.abs(array.astype(np.float32)-previous).mean()))
            previous=array.astype(np.float32)
            preview=Image.fromarray(array);preview.thumbnail((320,240));frames.append(preview)
        result.update(frames=len(frames),duration_seconds=len(frames)/fps,
            black_frames=sum(m<2 for m in means),min_frame_mean=min(means),max_frame_mean=max(means),
            max_mean_brightness_jump=max(abs(b-a) for a,b in zip(means,means[1:])),
            max_frame_delta=max(deltas),mean_motion_delta=float(np.mean(deltas)),
            frame_clock_max_error=max(abs(t-i/fps) for i,t in enumerate(times)))
    with av.open(str(path)) as container:
        result['has_audio']=bool(container.streams.audio)
        if container.streams.audio:
            stream=container.streams.audio[0]
            result.update(audio_codec=stream.codec_context.name,audio_rate=stream.codec_context.sample_rate,
                audio_duration=float(stream.duration*stream.time_base),audio_channels=stream.codec_context.channels)
            wave=np.concatenate([f.to_ndarray() for f in container.decode(audio=0)],axis=1)
            result.update(audio_rms=float(np.sqrt((wave**2).mean())),audio_peak=float(np.abs(wave).max()),
                audio_video_duration_difference=abs(result['audio_duration']-result['duration_seconds']))
    sheet=Image.new('RGB',(960,540),'white');draw=ImageDraw.Draw(sheet)
    for i,index in enumerate(np.linspace(0,len(frames)-1,6,dtype=int)):
        x=i%3*320;y=i//3*270;sheet.paste(frames[index],(x,y));draw.text((x+8,y+246),f'{index/fps:.2f}s',fill='black')
    sheet.save(destination/(path.stem+'-contact-sheet.jpg'))
    (destination/(path.stem+'-validation.json')).write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('video',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();validate(args.video,args.output)

"""Run a bounded real-GPU Ref2VA smoke test with an uploaded source and image."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import uuid

from backend.common import ReferenceVideoRequest, resolve_video
from backend.models import entry, is_installed, required_reference
from backend.reference_timing import probe_video_duration, resolve_reference_clip
from backend.video_worker import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='h3-ref-turbo-int8')
    parser.add_argument('--source',required=True,help='Existing uploaded video ID')
    parser.add_argument('--image',help='Existing uploaded reference image ID (required for faceswap)')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workflow',choices=['faceswap','reference','crossview'],default='faceswap')
    parser.add_argument('--quality',choices=['standard','turbo'])
    parser.add_argument('--width',type=int,default=768)
    parser.add_argument('--height',type=int,default=448)
    parser.add_argument('--mapping',action='store_true',help='Detect and assign the source person to the reference image')
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--seconds',type=float,default=22/24)
    parser.add_argument('--extra-reserve-gb',type=float,default=0,
                        help='Test low-VRAM offloading without changing production settings')
    args=parser.parse_args()
    quality=args.quality or ('turbo' if args.model=='h3-ref-turbo-int8' else 'standard')
    if args.workflow!='crossview' and not args.image:
        parser.error('--image is required for faceswap and reference')
    references=[args.image] if args.image and args.workflow!='crossview' else []
    prompt={'faceswap':'Faceswap. Replace the main person with <Picture 1>; keep motion and scene.',
            'reference':'Follow the source video motion and timing with <Picture 1>.',
            'crossview':'crossview. Show the same scene from a slightly different camera angle.'}[args.workflow]
    if args.extra_reserve_gb:
        from backend import gpu_memory
        normal_configure=gpu_memory.configure_video_memory
        def constrained_configure(torch, comfy_args, progress):
            normal_configure(torch,comfy_args,progress)
            comfy_args.reserve_vram += args.extra_reserve_gb
            progress(test_reserve_vram_gb=round(comfy_args.reserve_vram,2))
        gpu_memory.configure_video_memory=constrained_configure
    if not all(is_installed(name) for name in required_reference(ReferenceVideoRequest(
        model_id=args.model,source_video=args.source,reference_images=references,
        workflow=args.workflow,quality=quality,prompt=prompt,width=args.width,height=args.height))):
        raise RuntimeError('This model or one of its required files is not installed.')
    args.output.mkdir(parents=True,exist_ok=True)
    people=[]
    assignments={}
    if args.mapping:
        if args.workflow!='faceswap':
            parser.error('--mapping requires --workflow faceswap')
        from backend.reference_people import analyze_people
        people=analyze_people(resolve_video(args.source),args.start,args.start+args.seconds)
        if not people:
            raise RuntimeError('No person was detected in the selected segment.')
        assignments={people[0]['id']:args.image}
        prompt='Faceswap. Replace @Person1 with @Ref1; keep the original motion and scene.'
    request=ReferenceVideoRequest(model_id=args.model,source_video=args.source,
        source_start_seconds=args.start,source_end_seconds=args.start+args.seconds,
        reference_images=references,source_people=people,person_assignments=assignments,
        workflow=args.workflow,source_audio=False,output_audio='mute',
        quality=quality,
        prompt=prompt,width=args.width,height=args.height,
        output_dir=str(args.output))
    resolve_reference_clip(request,probe_video_duration(resolve_video(args.source)),
                           entry(args.model).get('generation_defaults',{}).get('max_frames',362))
    data=request.model_dump()
    data.update(kind='reference-video',steps=8 if quality=='turbo' else entry(args.model)['generation_defaults']['steps'],seed=12345)
    job_id=uuid.uuid4().hex
    job={'id':job_id,'started_at':time.time()}
    directory=args.output/job_id
    directory.mkdir()
    def update(**values):
        job.update(values)
        if values.get('stage'):
            print(values['stage'],values.get('step',''),values.get('message',''),flush=True)
    run(data,job,directory,update)
    print('SAVED',job['output_path'],job['frames'],job['width'],job['height'],flush=True)


if __name__=='__main__':
    main()

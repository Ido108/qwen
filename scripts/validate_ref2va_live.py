"""Optional short CUDA validation of the real Ref2VA FaceSwap path.

Run explicitly from the repository root with the application's Python venv.
Creates fresh validation inputs and a result on the E model drive.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import time
import uuid

import numpy as np
import torch
from PIL import Image

from backend.common import DATA_DIR, JOBS_DIR, UPLOADS_DIR, VIDEO_UPLOADS_DIR, write_json
from backend.video_worker import save_mp4


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workflow',choices=('faceswap','crossview'),default='faceswap')
    parser.add_argument('--replace-first',action='store_true',help='Run FaceSwap before CrossView and check the intermediate video.')
    parser.add_argument('--model',choices=('h3-ref-turbo-int8','h3-ref-eros-q4','h3-ref-eros-q8'),default='h3-ref-turbo-int8')
    args=parser.parse_args()
    if args.replace_first and args.workflow!='crossview':
        parser.error('--replace-first requires --workflow crossview')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available')
    run_id=uuid.uuid4().hex
    folder=DATA_DIR/'validation'/f'ref2va-{run_id[:8]}'
    folder.mkdir(parents=True)
    input_id=run_id+'.mp4'
    source=VIDEO_UPLOADS_DIR/input_id
    reference_id=run_id+'.jpg'
    shutil.copy2('E:/CONTACT-IMAGES/Aba.jpg',UPLOADS_DIR/reference_id)
    original=Image.open('E:/CONTACT-IMAGES/Jo.jpg').convert('RGB')
    width,height=original.size
    edge=min(width,height)
    frames=[]
    for index in range(22):
        left=round((width-edge)*index/21) if width>edge else 0
        top=round((height-edge)*index/21) if height>edge else 0
        frame=original.crop((left,top,left+edge,top+edge)).resize((256,256),Image.Resampling.LANCZOS)
        frames.append(torch.from_numpy(np.asarray(frame,dtype=np.float32)/255))
    save_mp4(source,torch.stack(frames),None)
    turbo=args.model=='h3-ref-turbo-int8'
    steps=8 if turbo else 25
    request={'kind':'reference-video','workflow':args.workflow,'model_id':args.model,
             'source_video':input_id,'reference_images':[reference_id] if args.workflow=='faceswap' or args.replace_first else [],
             'replace_subjects_first':args.replace_first,
             'replacement_prompt':'Faceswap. Replace the person with the reference identity, keeping the movement and clothing.',
             'warp_video':None,
             'camera_azimuth':-30,'camera_elevation':10,'camera_distance':1.0,
             'camera_fov':50,'camera_vertical_shift':0.0,'camera_aim_lock':False,
             'prompt':('Faceswap. The person looks at the camera, natural lighting.' if args.workflow=='faceswap'
                       else 'crossview. The same person and room from a camera left of the source.'),
             'width':256,'height':256,'frames':22,'steps':steps,'quality':'turbo' if turbo else 'standard',
             'seed':42,'randomize_seed':False,'reference_image_size':'match',
             'source_audio':False,'output_audio':'mute','enhance':False,'output_dir':str(folder)}
    job={'id':run_id,'kind':'reference-video','status':'queued','message':'Queued',
         'stage':'queued','created_at':time.time(),'seed':42,'request':request,
         'rewritten_prompt':'','step':0,'total_steps':steps*(2 if args.replace_first else 1)}
    write_json(folder/'request.json',request)
    write_json(folder/'status.json',job)
    print(f'Validation folder: {folder}',flush=True)
    from backend.worker import execute
    raise SystemExit(execute(folder))


if __name__=='__main__':
    main()

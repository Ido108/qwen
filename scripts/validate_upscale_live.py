"""Optional SeedVR2 CUDA validation using a previously uploaded short video."""
from __future__ import annotations
import argparse
import time
import uuid

from backend.common import DATA_DIR, write_json
from backend.worker import execute


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source_video',help='Video ID in the app uploads/videos directory')
    parser.add_argument('--model',default='seedvr2-3b-fp8')
    parser.add_argument('--resolution',type=int,default=480)
    parser.add_argument('--batch',type=int,default=5)
    parser.add_argument('--overlap',type=int,default=4)
    args=parser.parse_args()
    run_id=uuid.uuid4().hex
    directory=DATA_DIR/'validation'/f'upscale-{run_id[:8]}'
    directory.mkdir(parents=True)
    request={'kind':'upscale','model_id':args.model,'source_video':args.source_video,
             'resolution':args.resolution,'batch_size':args.batch,'chunk_size':100,'temporal_overlap':args.overlap,
             'seed':100,'preserve_audio':False,'output_dir':str(directory)}
    job={'id':run_id,'kind':'upscale','status':'queued','message':'Queued',
         'stage':'queued','created_at':time.time(),'seed':100,'request':request,
         'rewritten_prompt':'','step':0,'total_steps':0}
    write_json(directory/'request.json',request)
    write_json(directory/'status.json',job)
    print(f'Validation folder: {directory}',flush=True)
    raise SystemExit(execute(directory))


if __name__=='__main__':
    main()

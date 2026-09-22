"""Download only the chosen models and their required companions; never delete files."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from backend import models

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('models',nargs='*',default=['gguf-q8_0','gguf-q4_k_m'])
args=parser.parse_args()
for model_id in args.models:
    print('Preparing:',models.entry(model_id)['title'],flush=True)
    models.start_download(model_id)
    previous=None
    while True:
        with models.lock:
            status=dict(models.transfer)
        current=(status.get('stage'),status.get('file'),status.get('completed_bytes',0)//(256*1024**2))
        if current!=previous:
            print(status.get('stage'),status.get('file',''),f"{status.get('completed_bytes',0)/1024**3:.2f} / {status.get('total_bytes',0)/1024**3:.2f} GB",flush=True)
            previous=current
        if status['status']!='running':
            if status['status']!='succeeded':
                raise SystemExit(status.get('message','Download failed.'))
            break
        time.sleep(.5)
print('Selected models and matching encoders / VAE are verified.',flush=True)

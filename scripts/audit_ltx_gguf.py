"""Bounded remote headers plus native CPU/meta shape detection, never inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.models import catalog
from scripts.audit_ref2va_gguf import header


def inspect(item):
    file=item['files'][0]
    url=f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{file['path']}"
    with urlopen(Request(url,headers={'Range':'bytes=0-1048575'}),timeout=60) as response:
        assert response.status==206
        assert response.headers['Content-Range'].endswith('/'+str(file['bytes']))
        data=response.read(1048576)
    metadata,tensors=header(data)
    assert metadata['general.architecture']=='ltxv'
    config=json.loads(metadata['config'])['transformer']
    assert config['num_layers']==48 and config['in_channels']==128
    assert config['use_audio_video_cross_attention'] and config['use_keyframes_abs_pos_embedding']
    # Only shape-only meta tensors are constructed; not a single checkpoint weight.
    import torch
    from comfy.model_detection import model_config_from_unet
    shapes={key:torch.empty(value['shape'],device='meta') for key,value in tensors.items()}
    native=model_config_from_unet(shapes,'',metadata=metadata)
    assert type(native).__name__=='LTXAV'
    assert native.unet_config['caption_channels']==3840
    return {'id':item['id'],'repo':item['repo'],'revision':item['revision'],'file':file['path'],
            'bytes':file['bytes'],'sha256':file['sha256'],'header_sha256':hashlib.sha256(data).hexdigest(),
            'architecture':metadata['general.architecture'],'native_model':'LTXAV',
            'tensor_count':len(tensors),'blocks':config['num_layers'],'video_channels':config['in_channels'],
            'metadata_config_preserved':True,
            'shape_signature':hashlib.sha256(json.dumps({k:v['shape'] for k,v in tensors.items()},sort_keys=True).encode()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'vendor/ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu=True
    import comfy.model_detection
    items=[m for m in catalog() if m['kind']=='multi-reference' and m['format']=='GGUF']
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(inspect,items))
    assert len({m['shape_signature'] for m in results})==1
    args.output.write_text(json.dumps({'checked_on':'2026-09-26','scope':'Remote headers and CPU/meta model detection only; no model weights or inference','models':results},indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(results)} LTX GGUF headers, matching tensor shapes and native AV model configuration. No inference.')

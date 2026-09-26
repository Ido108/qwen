"""Check pinned remote GGUF tensor headers without downloading weights or using CUDA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import sys
from urllib.request import Request, urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models import catalog


def header(data):
    offset=0
    def read(fmt):
        nonlocal offset
        size=struct.calcsize(fmt)
        result=struct.unpack_from(fmt,data,offset);offset+=size
        return result[0] if len(result)==1 else result
    def string():
        nonlocal offset
        count=read('<Q');result=data[offset:offset+count].decode();offset+=count
        return result
    def value(kind):
        if kind==8:return string()
        if kind==9:
            subtype,count=read('<IQ');return [value(subtype) for _ in range(count)]
        return read({0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[kind])
    magic,version,count,nmeta=read('<4sIQQ')
    assert magic==b'GGUF' and version==3
    metadata={}
    for _ in range(nmeta):
        key=string();metadata[key]=value(read('<I'))
    tensors={}
    for _ in range(count):
        name=string();ndim=read('<I');shape=[read('<Q') for _ in range(ndim)]
        quant,_=read('<IQ')
        tensors[name]={'shape':metadata.get('comfy.gguf.orig_shape.'+name,list(reversed(shape))),'quant':quant}
    return metadata,tensors


def inspect(item):
    file=item['files'][0]
    url=f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{file['path']}"
    with urlopen(Request(url,headers={'Range':'bytes=0-131071'}),timeout=60) as response:
        if response.status!=206:raise ValueError('Server did not honor the bounded header request')
        assert response.headers['Content-Range'].endswith('/'+str(file['bytes']))
        data=response.read(131072)
    metadata,tensors=header(data)
    assert metadata['general.architecture'] in ('wan','minimax_h3')
    assert len([key for key in tensors if key.startswith('blocks.') and key.endswith('.attn.qkv_proj.weight')])==50
    assert tensors['video_patch_proj.weight']['shape'][1]==96
    assert tensors['final_layer.video_out.weight']['shape'][0]==96
    assert tensors['audio_patch_proj.weight']['shape'][1]==32
    assert tensors['final_layer.audio_out.weight']['shape'][0]==32
    is_pruned='adaln_t_table' in tensors
    if is_pruned:assert tensors['adaln_t_table']['shape'][1]==8
    return {'id':item['id'],'repo':item['repo'],'revision':item['revision'],'file':file['path'],
            'bytes':file['bytes'],'sha256':file['sha256'],'header_sha256':hashlib.sha256(data).hexdigest(),
            'architecture':metadata['general.architecture'],'tensors':len(tensors),'pruned':is_pruned,
            'video_channels':24,'audio_channels':32,'blocks':50}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    items=[item for item in catalog() if item['id'].startswith(('h3-ref-pruned-','h3-ref-full-'))]
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(inspect,items))
    args.output.write_text(json.dumps({'checked_on':'2026-09-26','scope':'Remote headers only; no model inference',
                                     'models':results},indent=2)+'\n',encoding='utf-8')
    print(f'Checked {len(results)} pinned GGUF headers: H3 architecture, shapes, channels, blocks and file sizes.')

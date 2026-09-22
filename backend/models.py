from __future__ import annotations
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from .common import APP_DIR, DATA_DIR, MODELS_DIR, write_json

os.environ['HF_HOME']=str(DATA_DIR/'hf-cache')
os.environ['HF_XET_CACHE']=str(DATA_DIR/'hf-cache'/'xet')

CATALOG_PATH=APP_DIR/'reference'/'model-catalog.json'
INSTALL_PATH=DATA_DIR/'installed-models.json'
lock=threading.RLock()
transfer={'status':'idle'}
cancel_event=threading.Event()

def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding='utf-8'))

def entry(model_id):
    for item in catalog():
        if item['id']==model_id:
            return item
    raise ValueError('Unknown model.')

def installed_manifest():
    try:
        return json.loads(INSTALL_PATH.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return {}

def local_path(item,file):
    path=(MODELS_DIR/item['directory']/file['path']).resolve()
    if not path.is_relative_to(MODELS_DIR.resolve()):
        raise ValueError('Invalid model path.')
    return path

def is_installed(model_id):
    item=entry(model_id)
    saved=installed_manifest().get(model_id,{})
    return saved.get('verified') and saved.get('revision')==item['revision'] and all(
        local_path(item,file).is_file() and local_path(item,file).stat().st_size==file['bytes']
        for file in item['files'])

def required(model_id,encoder='auto'):
    item=entry(model_id)
    if item['kind']!='image':
        raise ValueError('Select an image model.')
    if encoder not in ('auto','encoder-int8','encoder-bf16'):
        raise ValueError('Unknown text encoder.')
    default_encoder='encoder-int8' if item['format']=='GGUF' else 'encoder-bf16'
    return [model_id,encoder if encoder!='auto' else default_encoder,'vae']

def state():
    with lock:
        items=[]
        for item in catalog():
            installed=bool(is_installed(item['id']))
            disk_bytes=sum(local_path(item,file).stat().st_size for file in item.get('removal_files',item['files']) if local_path(item,file).is_file())
            items.append({k:v for k,v in item.items() if k not in ('files','removal_files','directory','revision')}|{
                'installed':installed,
                'disk_bytes':disk_bytes,
                'ready':installed and all(is_installed(d) for d in item['dependencies'])})
        return {'models':items,'transfer':dict(transfer),'free_bytes':shutil.disk_usage(DATA_DIR).free}

class DownloadCancelled(Exception): pass

def install(model_id):
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm
    item=entry(model_id)
    if item.get('kind')=='enhancer':
        raise ValueError('Prompt enhancement now uses Antigravity. Local prompt models are no longer downloaded.')
    queue=[]
    def add(model):
        for dep in model['dependencies']:
            add(entry(dep))
        if model['id'] not in [m['id'] for m in queue] and not is_installed(model['id']):
            queue.append(model)
    add(item)
    total=sum(m['bytes'] for m in queue)
    completed=0
    try:
        if shutil.disk_usage(DATA_DIR).free<total+2*1024**3:
            raise RuntimeError('Not enough free space on the model drive.')
        for model in queue:
            if cancel_event.is_set(): raise DownloadCancelled()
            with lock:
                transfer.update(model_id=model['id'],title=model['title'],stage='downloading',total_bytes=total,completed_bytes=completed)
            model_done=0
            for file in model['files']:
                if cancel_event.is_set(): raise DownloadCancelled()
                base=completed+model_done
                class Progress(tqdm):
                    def update(self,n=1):
                        if cancel_event.is_set(): raise DownloadCancelled()
                        result=super().update(n)
                        with lock:
                            transfer.update(completed_bytes=min(total,base+int(self.n)),file=file['path'])
                        return result
                target=local_path(model,file)
                invalid=target.with_name(target.name+'.verification-failed')
                path=hf_hub_download(model['repo'],file['path'],revision=model['revision'],local_dir=MODELS_DIR/model['directory'],tqdm_class=Progress,force_download=invalid.exists())
                with lock:
                    transfer.update(stage='verifying',file=file['path'])
                if Path(path).stat().st_size!=file['bytes']:
                    invalid.write_text('Incomplete file; force a fresh download on retry.',encoding='utf-8')
                    raise RuntimeError('Incomplete model file. Retry the download.')
                if file.get('sha256'):
                    digest=hashlib.sha256()
                    with open(path,'rb') as handle:
                        while chunk:=handle.read(8*1024*1024):
                            if cancel_event.is_set(): raise DownloadCancelled()
                            digest.update(chunk)
                    if digest.hexdigest()!=file['sha256']:
                        invalid.write_text('Checksum mismatch; force a fresh download on retry.',encoding='utf-8')
                        raise RuntimeError('Model checksum verification failed. Retry to replace the incomplete file.')
                invalid.unlink(missing_ok=True)
                model_done+=file['bytes']
                with lock:
                    transfer.update(stage='downloading',completed_bytes=completed+model_done)
            completed+=model['bytes']
            with lock:
                saved=installed_manifest()
                saved[model['id']]={'revision':model['revision'],'verified':True}
                write_json(INSTALL_PATH,saved)
        with lock:
            transfer.update(status='succeeded',stage='complete',completed_bytes=total)
    except DownloadCancelled:
        with lock:
            transfer.update(status='cancelled',message='Download cancelled. Run it again to resume.')
    except Exception as exc:
        with lock:
            transfer.update(status='failed',message=str(exc))

def start_download(model_id):
    if entry(model_id).get('kind')=='enhancer':
        raise ValueError('Prompt enhancement uses Antigravity. Local prompt models are no longer downloaded.')
    entry(model_id)
    with lock:
        if transfer.get('status')=='running':
            raise ValueError('A model is already downloading.')
        transfer.clear()
        transfer.update(status='running',requested_id=model_id,model_id=model_id,stage='preparing',completed_bytes=0,total_bytes=0)
        cancel_event.clear()
        threading.Thread(target=install,args=(model_id,),daemon=True).start()
    return {'ok':True}

def remove(model_id):
    item=entry(model_id)
    with lock:
        if transfer.get('status')=='running':
            raise ValueError('Wait for the download to finish before removing a model.')
        # Only the pinned, allowlisted files can be removed. Generated images never live here.
        for file in item.get('removal_files',item['files']):
            path=local_path(item,file)
            if path.is_file():
                path.unlink()
        saved=installed_manifest()
        saved.pop(model_id,None)
        write_json(INSTALL_PATH,saved)
    return {'ok':True}

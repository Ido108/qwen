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
EXTENSION_CATALOG_PATH=APP_DIR/'reference'/'reference-video-catalog.json'
INSTALL_PATH=DATA_DIR/'installed-models.json'
lock=threading.RLock()
transfer={'status':'idle'}
cancel_event=threading.Event()
active_download=None

def catalog():
    items=json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    if EXTENSION_CATALOG_PATH.is_file():
        items.extend(json.loads(EXTENSION_CATALOG_PATH.read_text(encoding='utf-8')))
    return items

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

def required_video(request):
    item=entry(request.model_id)
    if item['kind']!='video':
        raise ValueError('Select a video model.')
    if request.mode not in item['modes']:
        raise ValueError('Choose MiniMax H3 for image-to-video. FastH3 V2 is distilled for text-to-video.')
    if request.mode=='image' and not request.first_frame:
        raise ValueError('Add a first frame for image-to-video.')
    if request.mode=='text' and (request.first_frame or request.last_frame):
        raise ValueError('Choose image-to-video to use input images.')
    if request.width*request.height>1344*768:
        raise ValueError('The video size must not exceed 1344 × 768 pixels.')
    if not 0.25 <= request.width/request.height <= 4:
        raise ValueError('MiniMax H3 supports output aspect ratios from 1:4 to 4:1.')
    ids=[item['id'],*item['dependencies']]
    if request.quality=='turbo' and item['family']=='minimax-h3':
        ids.append('h3-turbo')
    return ids


def required_reference(request):
    item=entry(request.model_id)
    if item['kind']!='reference-video' or item['family']!='minimax-h3-ref2va':
        raise ValueError('Choose a MiniMax H3 Ref2VA model for this workflow.')
    if request.workflow=='faceswap' and not request.reference_images:
        raise ValueError('Add at least one reference image for subject replacement.')
    if request.workflow=='crossview' and request.reference_images and not request.replace_subjects_first:
        raise ValueError('Enable Replace subjects first to use identity images in Camera view.')
    if request.workflow=='crossview' and request.replace_subjects_first and not request.reference_images:
        raise ValueError('Add reference images for Replace subjects first.')
    if request.width*request.height>1344*768:
        raise ValueError('The video size must not exceed 1344 × 768 pixels.')
    if not 0.25 <= request.width/request.height <= 4:
        raise ValueError('MiniMax H3 supports output aspect ratios from 1:4 to 4:1.')
    ids=[item['id'],*item['dependencies']]
    if request.workflow=='faceswap':
        ids.append('h3-ref-faceswap')
    if request.workflow=='crossview':
        ids.append('h3-ref-crossview')
        if request.replace_subjects_first:
            ids.append('h3-ref-faceswap')
        if not request.warp_video:
            ids.append('moge2-geometry')
    from .reference_sampling import reference_sampling
    profile = reference_sampling(item, request.quality, request.steps)
    if profile.get('lora_id'):
        ids.append(profile['lora_id'])
    return ids


def required_upscale(request):
    item=entry(request.model_id)
    if item['kind']!='upscale' or item['family']!='seedvr2':
        raise ValueError('Choose a SeedVR2 upscaling model.')
    if request.temporal_overlap>=request.chunk_size:
        raise ValueError('Temporal overlap must be smaller than the chunk size.')
    if request.temporal_overlap>=request.batch_size:
        raise ValueError('Temporal overlap must be smaller than the frame batch size.')
    if request.batch_size>request.chunk_size:
        raise ValueError('Frame batch size must not exceed the chunk size.')
    return [item['id'],*item['dependencies']]


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

def install(model_id, *, include_dependencies=True):
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
    from huggingface_hub.errors import GatedRepoError
    from tqdm.auto import tqdm
    # The app keeps its model cache on the roomy data drive. Hugging Face's
    # normal CLI token remains in the user's home cache and is read only for
    # gated downloads; it is never copied into app data or logged.
    token=os.environ.get('HF_TOKEN')
    if not token:
        default_token=Path.home()/'.cache'/'huggingface'/'token'
        if default_token.is_file():
            token=default_token.read_text(encoding='utf-8').strip() or None
    items=[entry(name) for name in ([model_id] if isinstance(model_id,str) else model_id)]
    if any(item.get('kind')=='enhancer' for item in items):
        raise ValueError('Prompt enhancement now uses Antigravity. Local prompt models are no longer downloaded.')
    queue=[]
    def add(model):
        if include_dependencies:
            for dep in model['dependencies']:
                add(entry(dep))
        if model['id'] not in [m['id'] for m in queue] and not is_installed(model['id']):
            queue.append(model)
    for item in items:
        add(item)
    total=sum(m['bytes'] for m in queue)
    completed=0
    try:
        # Check access before downloading any of the accompanying large files.
        for model in queue:
            if model['repo'] == 'Lightricks/LTX-2.5':
                if cancel_event.is_set(): raise DownloadCancelled()
                with lock:
                    transfer.update(model_id=model['id'], title=model['title'], stage='checking_access')
                get_hf_file_metadata(hf_hub_url(model['repo'], model['files'][0]['path'], revision=model['revision']), token=token)
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
                path=hf_hub_download(model['repo'],file['path'],revision=model['revision'],local_dir=MODELS_DIR/model['directory'],tqdm_class=Progress,force_download=invalid.exists(),token=token)
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
    except GatedRepoError:
        with lock:
            transfer.update(status='failed',message=(
                f'Access to {model["repo"]} is restricted. Open '
                f'https://huggingface.co/{model["repo"]} and accept the model terms '
                'with the Hugging Face account used on this computer, then retry.'
            ))
    except Exception as exc:
        with lock:
            transfer.update(status='failed',message=str(exc))

def _download_running_locked():
    return transfer.get('status')=='running' or (active_download is not None and not active_download['done'].is_set())


def _begin_download_locked(ids, *, owner=None, include_dependencies=True):
    """Caller holds lock; retain each result even if the next download starts."""
    global active_download
    task={'ids':list(ids),'owner':owner,'done':threading.Event(),'result':None}
    active_download=task
    transfer.clear()
    transfer.update(status='running',requested_id=ids[0],model_id=ids[0],stage='preparing',completed_bytes=0,total_bytes=0)
    cancel_event.clear()
    def run():
        try:
            install(ids,include_dependencies=include_dependencies)
        except Exception as exc:
            with lock:
                transfer.update(status='failed',message=str(exc))
        finally:
            with lock:
                task['result']=dict(transfer)
                task['done'].set()
    threading.Thread(target=run,daemon=True).start()
    return task


def ensure_installed(ids, *, owner, cancelled=lambda:False, progress=lambda state:None):
    """Download an exact validated dependency set before a queued job starts.

    Manual and automatic transfers share the same lock/cache. A cancelled job
    stops only its own download; another job can resume the cached partial file.
    """
    ids=list(dict.fromkeys(ids))
    while True:
        if cancelled():
            raise DownloadCancelled('Download cancelled.')
        with lock:
            missing=[name for name in ids if not is_installed(name)]
            if not missing:
                return
            if _download_running_locked():
                task=active_download
            else:
                task=_begin_download_locked(missing,owner=owner,include_dependencies=False)
        if task is None:
            progress({'stage':'waiting_for_download','message':'Waiting for the current model download'})
            time.sleep(.2)
            continue
        while not task['done'].is_set():
            if cancelled():
                with lock:
                    if active_download is task and task['owner']==owner:
                        cancel_event.set()
                raise DownloadCancelled('Download cancelled.')
            with lock:
                current=dict(transfer) if active_download is task else dict(task['result'] or {})
            if task['owner']==owner or set(task['ids']).intersection(ids):
                progress(current)
            else:
                progress({'stage':'waiting_for_download','message':'Waiting for the current model download'})
            task['done'].wait(.2)
        result=task['result']
        relevant=task['owner']==owner or (task['owner'] is None and bool(set(task['ids']).intersection(ids)))
        if relevant and result['status']=='cancelled':
            raise DownloadCancelled('Model download cancelled. Retry this job to resume.')
        if relevant and result['status']=='failed':
            raise RuntimeError(result.get('message','Model download failed. Retry this job to resume.'))
        if task['owner']==owner and any(not is_installed(name) for name in ids):
            raise RuntimeError('A required model file could not be verified. Retry this job to resume.')


def start_download(model_id):
    if entry(model_id).get('kind')=='enhancer':
        raise ValueError('Prompt enhancement uses Antigravity. Local prompt models are no longer downloaded.')
    entry(model_id)
    with lock:
        if _download_running_locked():
            raise ValueError('A model is already downloading.')
        _begin_download_locked([model_id])
    return {'ok':True}

def remove(model_id):
    item=entry(model_id)
    with lock:
        if _download_running_locked():
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

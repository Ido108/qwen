"""Create a pinned, allowlisted catalogue and checksum-verify already installed models."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ['QWEN_DATA_DIR'])
api = HfApi()
catalog = []
for model_id, name, revision, title, kind in [
    ('official-bf16','Qwen-Image-2.1','790c92633540aa0cb11d9abf19eb46d861714758','Qwen-Image-2.1 · BF16','image'),
    ('pe-t2i','Qwen-Image-2.1-PE-T2I','f3ed7985c788ad75b3ab7223e0c4c51e2a43545b','Prompt enhancement · Text to image','enhancer'),
    ('pe-i2i','Qwen-Image-2.1-PE-I2I','72927bc08afc99b7888ceb7d7d51a12db3700bbd','Prompt enhancement · Image editing','enhancer'),
]:
    info=api.model_info('Qwen/'+name, revision=revision, files_metadata=True)
    files=[{'path':s.rfilename,'bytes':s.size,'sha256':s.lfs.sha256 if s.lfs else None} for s in info.siblings if not s.rfilename.startswith('assets/') and s.rfilename!='.gitattributes']
    catalog.append({'id':model_id,'title':title,'kind':kind,'format':'BF16','repo':info.id,'revision':revision,'directory':name,'files':files,'bytes':sum(f['bytes'] for f in files),'dependencies':['encoder-bf16','vae'] if kind=='image' else []})
repo='abenzerps/Qwen-Image-2.1-Uncensored-GGUF'
revision='e92386d3b86fd77ae1650534815ab3f6b85d46ba'
info=api.model_info(repo, revision=revision,files_metadata=True)
by_name={s.rfilename:s for s in info.siblings}
for quant in ['Q8_0','Q6_K','Q5_K_M','Q4_K_M','Q4_0']:
    name=f'qwen-image-2.1-{quant}.gguf'
    item=by_name[name]
    catalog.append({'id':'gguf-'+quant.lower(),'title':'Qwen-Image-2.1 Uncensored · '+quant,'kind':'image','format':'GGUF','repo':repo,'revision':revision,'directory':'abenzerps-GGUF','files':[{'path':name,'bytes':item.size,'sha256':item.lfs.sha256}],'bytes':item.size,'dependencies':['encoder-int8','vae']})
for model_id,name,title,precision,kind in [
    ('encoder-int8','text_encoders/qwen3vl_8b_int8_convrot.safetensors','Qwen3-VL 8B · INT8','INT8','encoder'),
    ('encoder-bf16','text_encoders/qwen3vl_8b_bf16.safetensors','Qwen3-VL 8B · BF16','BF16','encoder'),
    ('vae','vae/qwen_image_2.1_vae_bf16.safetensors','Qwen-Image-2.1 VAE','BF16','vae'),
]:
    item=by_name[name]
    catalog.append({'id':model_id,'title':title,'kind':kind,'format':precision,'repo':repo,'revision':revision,'directory':'abenzerps-GGUF','files':[{'path':name,'bytes':item.size,'sha256':item.lfs.sha256}],'bytes':item.size,'dependencies':[]})
for item in catalog:
    if item['kind']=='image':
        item['generation_defaults']={'steps':40,'source':'https://github.com/Comfy-Org/workflow_templates/blob/371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4/templates/image_qwen_image_2_1_t2i.json',
            'edit_source':'https://github.com/Comfy-Org/workflow_templates/blob/371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4/templates/image_qwen_image_2_1_image_edit.json',
            'note':'The model card uses 40 steps (num_inference_steps=40). The ComfyUI template starts at 25 and notes that the official pipeline uses about 40-50.'}
(ROOT/'reference'/'model-catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
installed={}
original=json.loads((DATA/'model-manifest.json').read_text())
for model in catalog:
    if model['directory'] in original:
        installed[model['id']]={'revision':model['revision'],'verified':True}
        continue
    paths=[(DATA/'models'/model['directory']/f['path'],f) for f in model['files']]
    if not all(p.is_file() and p.stat().st_size==f['bytes'] for p,f in paths):
        continue
    for path,item in paths:
        with path.open('rb') as handle:
            digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=item['sha256']:
            raise RuntimeError(f'Checksum mismatch: {path}')
    installed[model['id']]={'revision':model['revision'],'verified':True}
    print('Verified',model['id'],flush=True)
(DATA/'installed-models.json').write_text(json.dumps(installed,indent=2),encoding='utf-8')
print('Catalog:',len(catalog),'Installed:',list(installed),flush=True)

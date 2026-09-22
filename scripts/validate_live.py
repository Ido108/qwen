"""Run an explicitly chosen, real CUDA workflow through the running app API."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import httpx
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('case',choices=['ten-images','enhance-edit','contact-single','contact-pair'])
parser.add_argument('--url',default='http://127.0.0.1:47821')
parser.add_argument('--contact-dir',type=Path,default=Path('E:/CONTACT-IMAGES'))
args=parser.parse_args()
client=httpx.Client(base_url=args.url,timeout=30)
config=client.get('/api/config').raise_for_status().json()
if config['active_job']:
    raise SystemExit('Wait for the active image to finish.')
client.headers['X-Studio-Token']=config['token']
body={'model_id':'gguf-q4_k_m','text_encoder':'auto','custom_size':True,'width':1024,'height':1024,
      'steps':40,'seed':1234,'randomize_seed':False,'output_dir':config['output_dir']}
originals={}
if args.case=='ten-images':
    examples=client.get('/api/examples').raise_for_status().json()
    example=max(examples,key=lambda e:len(e['inputs']))
    assert len(example['inputs'])==10
    body.update(prompt=example['prompt'],images=['example:'+i for i in example['inputs']],enhance=False)
elif args.case=='enhance-edit':
    image=client.post('/api/jobs/514ecb0d3e524fcbb770671bf27715f3/reuse').raise_for_status().json()
    body.update(prompt='Keep the strawberry and its green leaves unchanged. Place it on a pale blue ceramic plate on a light wooden table, with soft natural daylight.',
                images=[image['id']],enhance=True)
else:
    names=['ziv.jpg'] if args.case=='contact-single' else ['ziv.jpg','tomer.jpg']
    multipart=[]
    for name in names:
        source=args.contact_dir/name
        raw=source.read_bytes()
        originals[str(source)]=hashlib.sha256(raw).hexdigest()
        multipart.append(('files',(name,raw,'image/jpeg')))
    uploaded=client.post('/api/uploads',files=multipart).raise_for_status().json()
    assert [item['name'] for item in uploaded]==names
    body.update(images=[item['id'] for item in uploaded],enhance=False)
    if args.case=='contact-single':
        body.update(model_id='gguf-q8_0',prompt='Replace only the plain gray background with an elegant softly blurred green garden in natural daylight. Preserve the exact same man, facial identity, hairstyle, beard, expression, head angle, black shirt and framing from the input image. Make the lighting on the person consistent with the garden. Photorealistic portrait, no text.')
    else:
        body.update(width=1184,height=864,prompt='Create one realistic photograph of the two men sitting side by side at an outdoor cafe table. The man from image 1 is on the left wearing his black shirt. The man from image 2 is on the right wearing his gray shirt. Preserve each man\'s distinct facial identity, hairstyle, beard and appearance accurately from his reference image. Both are facing the camera with natural relaxed expressions. Frame both men from the waist up, with two cups of coffee on the table, soft daylight and a softly blurred cafe background. Exactly two people, no text.')
job=client.post('/api/generate',json=body).raise_for_status().json()
print(args.case,job['id'],flush=True)
last=None
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    job=client.get('/api/jobs/'+job['id']).raise_for_status().json()
    state=(job['status'],job.get('stage'),job.get('step'),job.get('tokens',0)//128)
    if state!=last:
        print(state,job['message'],flush=True)
        last=state
    if job['status'] in ('succeeded','failed','cancelled'):
        break
    time.sleep(2)
else:
    client.post('/api/jobs/'+job['id']+'/cancel').raise_for_status()
    raise SystemExit('Validation exceeded 30 minutes and was cancelled.')
target=Path(config['data_dir'])/'validation'
target.mkdir(exist_ok=True)
(target/(args.case+'.json')).write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
if job['status']!='succeeded': raise SystemExit(job['message'])
assert job['width']==body['width'] and job['height']==body['height']
assert job['request']['images']==body['images']
if body['enhance']: assert job.get('rewritten_prompt')
image=client.get(job['image_url']+'?download=true').raise_for_status()
assert image.content.startswith(b'\x89PNG\r\n\x1a\n')
with Image.open(io.BytesIO(image.content)) as result:
    result.load()
    assert result.size==(body['width'],body['height'])
    assert result.mode=='RGBA'
for filename,digest in originals.items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest, 'An original input image changed.'
if originals:
    evidence={'job_id':job['id'],'source_sha256':originals,'output_sha256':hashlib.sha256(image.content).hexdigest(),
              'upload_order':body['images'],'output_size':[job['width'],job['height']],'originals_unchanged':True,
              'enhance':body['enhance'],'model_id':body['model_id'],'visual_review':'pending'}
    (target/(args.case+'-evidence.json')).write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',args.case,job['elapsed_seconds'],'seconds',flush=True)

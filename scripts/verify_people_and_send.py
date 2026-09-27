"""Isolated browser checks. All API calls are mocked, including every mutation."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit,parse_qs
from urllib.request import urlopen
from cloakbrowser import launch

parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--url',default='http://127.0.0.1:47830')
parser.add_argument('--api-url',default='http://127.0.0.1:47821')
args=parser.parse_args()
root=args.data_dir
out=root/'validation/people-and-send';out.mkdir(parents=True,exist_ok=True)
fixture=next((root/'uploads').glob('*.png'))
with urlopen(args.api_url+'/api/config',timeout=4) as response:config=json.load(response)
config['active_job']=None
source={'id':'a'*32+'.mp4','name':'Generated portrait.mp4','url':'/api/video-fixture','poster_url':'/api/fixture','bytes':100,'width':1080,'height':1928,'fps':24,'duration_seconds':5.167,'has_audio':True}
state={'uploads':0,'submitted':[],'sent':[],'error':False}
jobs=[]
for kind,model in [('video','fasth3-q4_k_m'),('reference-video','h3-ref-base-int8'),('multi-reference','ltx25-msr-q4_k_m'),('upscale','seedvr2-3b-fp8')]:
 jobs.append({'id':kind,'kind':kind,'status':'succeeded','stage':'complete','message':'Video saved.','created_at':100,'started_at':110,'finished_at':175,'elapsed_seconds':65,'seed':17,'actual_steps':8,'total_steps':8,'total_chunks':2 if kind=='upscale' else None,'width':1080,'height':1928,'fps':24,'duration_seconds':5.167,'poster_url':'/api/fixture','video_url':'/api/video-fixture','request':{'prompt':'Test video','model_id':model,'source_video':source['id'],'reference_images':[],'images':[],'seed':17,'randomize_seed':False}})
people=[{'id':'person-1','time':.2,'box':[.1,.1,.2,.2],'hits':4,'preview':'/api/fixture'}]

def route(r):
 u=urlsplit(r.request.url);p=u.path
 if p=='/api/config':return r.fulfill(json=config)
 if p=='/api/session':return r.fulfill(json={'token':config['token']})
 if p=='/api/models':return r.fulfill(json=config['catalog'])
 if p=='/api/queue':return r.fulfill(json={'active':None,'pending':[],'recent':[],'paused':False})
 if p=='/api/library':return r.fulfill(json={'items':jobs,'total':len(jobs)})
 if p=='/api/history':return r.fulfill(json=[j for j in jobs if j['kind']==parse_qs(u.query).get('kind',['image'])[0]])
 if p=='/api/gpu-memory':return r.fulfill(json={'available':True,'free_gb':24,'total_gb':24})
 if p.endswith('/metadata') and '/video-uploads/' in p:return r.fulfill(json={**source,'id':p.split('/')[-2]})
 if p.endswith('/people'):return r.fulfill(json={'people':people})
 if p.endswith('/reuse-video'):
  state['sent'].append((p,parse_qs(u.query)))
  if state['error']:return r.fulfill(status=422,json={'detail':'Saved video is unavailable.'})
  return r.fulfill(json=source)
 if p=='/api/uploads':
  state['uploads']+=1;image_id=f"view-{state['uploads']}.png"
  return r.fulfill(json=[{'id':image_id,'url':'/api/uploads/'+image_id,'name':image_id}])
 if p=='/api/reference-video/generate':
  data=r.request.post_data_json;state['submitted'].append(data)
  return r.fulfill(json={**jobs[1],'id':'queued-test','status':'queued','stage':'queued','request':data})
 if p=='/api/jobs/queued-test':return r.fulfill(json={**jobs[1],'id':'queued-test','request':state['submitted'][-1]})
 if p=='/api/fixture' or p.startswith('/api/uploads/') or p.endswith(('/poster','/timeline')):return r.fulfill(path=str(fixture),content_type='image/png')
 if p=='/api/video-fixture':return r.fulfill(body=b'',content_type='video/mp4')
 if r.request.method not in ('GET','HEAD','OPTIONS'):raise AssertionError('Unexpected mutation '+p)
 return r.fulfill(json=[] if p=='/api/examples' else {})

browser=launch(headless=True,args=['--disable-gpu'])
try:
 page=browser.new_page(viewport={'width':1440,'height':900});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.route('**/api/**',route)
 page.add_init_script("localStorage.setItem('qwen-reference-draft',"+json.dumps(json.dumps({'source_video':source['id'],'reference_people':[],'reference_images':[]}))+');')
 page.goto(args.url+'/#video-to-video',wait_until='networkidle')
 page.get_by_role('button',name='Replacement for Source 1',exact=True).wait_for()
 def add(button):
  with page.expect_file_chooser() as chooser:page.get_by_role('button',name=button,exact=True).click()
  chooser.value.set_files(str(fixture))
 def draft():return page.evaluate("JSON.parse(localStorage.getItem('qwen-reference-draft'))")
 def wait_uploads(number):page.wait_for_function('(n)=>JSON.parse(localStorage.getItem("qwen-reference-draft")).reference_images.length===n',arg=number)
 add('Add person');wait_uploads(1)
 add('Add images to Person 1');wait_uploads(2)
 add('Add person');wait_uploads(3)
 data=draft();first_id=data['reference_people'][0]['id'];second_id=data['reference_people'][1]['id']
 assert data['reference_people'][0]['images']==['view-1.png','view-2.png']
 assert data['person_assignments']=={'person-1':first_id}
 choice=page.get_by_role('button',name='Replacement for Source 1',exact=True);choice.click()
 menu=page.get_by_role('listbox',name='Choose replacement for Source 1')
 assert menu.get_by_role('option',name='Person 1 2 images').locator('img').count()==1
 menu.get_by_role('option',name='Person 2 1 image').click()
 assert choice.get_by_alt_text('Selected Person 2').is_visible()
 page.get_by_role('button',name='Add to queue',exact=True).click()
 page.wait_for_function('()=>document.querySelector(".reference-job-progress")!==null')
 assert state['submitted'][-1]['reference_people']==data['reference_people']
 assert state['submitted'][-1]['reference_images']==['view-1.png','view-2.png','view-3.png']
 assert state['submitted'][-1]['person_assignments']=={'person-1':second_id}
 assert state['submitted'][-1]['prompt']==''
 page.locator('.reference-inputs > .reference-extra-instructions > summary').click()
 prompt=page.get_by_label('Additional instructions (optional)',exact=True)
 prompt.fill('Use @');page.get_by_role('option',name='@Person1 Uploaded person 1 (2 images)').click()
 assert '@Person1' in prompt.input_value()
 prompt.fill('Preserve @Person2.')
 page.get_by_role('button',name='Remove Person 1',exact=True).click()
 from playwright.sync_api import expect
 expect(prompt).to_have_value('Preserve @Person1.')
 assert prompt.input_value()=='Preserve @Person1.',draft()
 assert draft()['person_assignments']=={'person-1':second_id}
 assert draft()['reference_people'][0]['id']==second_id
 # Open all views; removal updates only that person's group.
 add('Add images to Person 1');wait_uploads(2)
 page.get_by_role('button',name='View images for Person 1',exact=True).click()
 page.get_by_role('button',name='Remove view 2',exact=True).click()
 assert draft()['reference_images']==['view-3.png']
 page.screenshot(path=str(out/'people-desktop.png'),full_page=True)
 # Every video-producing workflow provides every destination.
 destinations=['faceswap','crossview','reference','upscale']
 tabs=['Create video','Video to video','Reference images','Upscale']
 for tab in tabs:
  for target in destinations:
   page.get_by_role('button',name='Video',exact=True).click();page.get_by_role('tab',name=tab,exact=True).click()
   control=page.locator('main:not([hidden])').get_by_label('Send video to',exact=True)
   control.wait_for();control.select_option(target)
   expected='Upscale' if target=='upscale' else 'Video to video'
   page.wait_for_function('(label)=>[...document.querySelectorAll(".video-subtabs [role=tab],[role=tab]")].some(e=>e.textContent===label&&e.getAttribute("aria-selected")==="true")',arg=expected)
   if target=='upscale':
    expect(page.locator('.upscale-page .reference-video-drop strong')).to_have_text(source['name'])
   else:
    page.get_by_role('tab',name={'faceswap':'Replace subjects','crossview':'Camera view','reference':'Reference-guided'}[target],exact=True).wait_for()
    page.wait_for_function('(v)=>{const p=JSON.parse(localStorage.getItem("qwen-reference-draft"));return p.workflow===v.target&&p.source_video===v.id}',arg={'target':target,'id':source['id']})
    assert abs(draft()['source_end_seconds']-source['duration_seconds'])<.05
    assert draft()['reference_people'][0]['id']==second_id
 assert len(state['sent'])==16
 assert len(state['submitted'])==1,'Send to must not generate'
 # Gallery closes its overlay and opens the requested destination.
 page.get_by_role('button',name='Gallery',exact=True).click();page.locator('.gallery-preview').first.click()
 dialog=page.get_by_role('dialog',name='Generation details',exact=True)
 dialog.get_by_label('Send video to',exact=True).select_option('upscale')
 page.wait_for_function('()=>location.hash==="#upscale"');assert not dialog.is_visible()
 # Failure stays in the source tab and is visible beside the dropdown.
 state['error']=True
 control=page.locator('.upscale-page').get_by_label('Send video to',exact=True)
 control.select_option('crossview');page.get_by_role('alert').filter(has_text='Saved video is unavailable.').wait_for()
 assert page.url.endswith('#upscale') and control.is_enabled()
 state['error']=False
 control.select_option('faceswap');page.wait_for_function('()=>location.hash==="#video-to-video"')
 for width in (1440,768,390):
  page.set_viewport_size({'width':width,'height':900})
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  page.screenshot(path=str(out/f'people-send-{width}.png'),full_page=True)
 assert not errors,errors
 print('PASS: grouped people, all views, group selection, tags, renumbering, removal, 16 workflow transfers, gallery transfer, failure/retry and responsive layout. No real generation submitted.')
except Exception:
 page.screenshot(path=str(out/'failure.png'),full_page=True)
 raise
finally:browser.close()

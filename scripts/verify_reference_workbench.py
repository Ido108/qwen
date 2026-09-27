import argparse,json,time,urllib.request,sys
from pathlib import Path
from urllib.parse import urlsplit
from cloakbrowser import launch

parser=argparse.ArgumentParser(description='UI checks with every generation request intercepted; no inference.')
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--source-id',required=True)
parser.add_argument('--reference-id',required=True)
parser.add_argument('--api-url',default='http://127.0.0.1:47821')
parser.add_argument('--url',default='http://127.0.0.1:47830')
args=parser.parse_args()
ROOT=args.data_dir;OUT=ROOT/'validation/reference-workbench-qa'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=args.source_id;REF=args.reference_id
for fixture in (ROOT/'uploads'/REF,ROOT/'uploads/videos'/SOURCE):
    if not fixture.is_file():raise FileNotFoundError(f'UI fixture does not exist: {fixture}')

def get(path):
    with urllib.request.urlopen(args.api_url.rstrip('/')+path) as response:return json.load(response)
config=get('/api/config');config['active_job']=None;models=get('/api/models')
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models import catalog
by_id={m['id']:m for m in models['models']}
models['models']=[{**m,**{k:v for k,v in by_id.get(m['id'],{}).items() if k in ('installed','ready','disk_bytes')},'ready':False,'installed':False} for m in catalog()]
metadata=json.loads((ROOT/'uploads/videos'/f'{SOURCE}.json').read_text(encoding='utf-8'))
metadata.update(id=SOURCE,url=f'/api/video-uploads/{SOURCE}',poster_url=f'/api/video-uploads/{SOURCE}/poster')
people=[{'id':'person-1','time':.17,'box':[.54,.14,.23,.17],'hits':4,'preview':f'/api/uploads/{REF}'}]
state={'job':None,'requests':[],'session':config['token'],'refreshes':0,'rejected':0}
errors=[]
def handle(route):
    path=urlsplit(route.request.url).path
    def reply(value,status=200):route.fulfill(status=status,json=value)
    if path=='/api/config':return reply(config)
    if path=='/api/session':
        state['refreshes']+=1
        return reply({'token':state['session']})
    if route.request.method not in ('GET','HEAD','OPTIONS') and route.request.headers.get('x-studio-token')!=state['session']:
        state['rejected']+=1
        return reply({'code':'session_expired','detail':'Refresh the app and try again.'},403)
    if path=='/api/models':return reply(models)
    if path=='/api/queue':return reply({'active':None,'pending':[],'recent':[],'paused':False})
    if path=='/api/history':return reply([])
    if path=='/api/gpu-memory':return reply({'available':True,'free_gb':24,'total_gb':24})
    if path.endswith('/metadata') and '/video-uploads/' in path:return reply(metadata)
    if path.endswith('/people'):return reply({'people':people})
    if path==f'/api/uploads/{REF}':return route.fulfill(path=str(ROOT/'uploads'/REF),content_type='image/png')
    if path.startswith(f'/api/video-uploads/{SOURCE}'):
        suffix=path[len(f'/api/video-uploads/{SOURCE}'):]
        file=ROOT/'uploads/videos'/(SOURCE+('.jpg' if suffix=='/poster' else '.timeline-v2.webp' if suffix=='/timeline' else ''))
        return route.fulfill(path=str(file))
    if path=='/api/reference-video/generate':
        request=route.request.post_data_json;state['requests'].append(request)
        state['job']={'id':'ui-check','kind':'reference-video','status':'running','stage':'generating','message':'Generating video and audio…','started_at':time.time()-20,'stage_started_at':time.time()-5,'created_at':time.time(),'step':0,'total_steps':8,'request':request,'seed':0}
        return reply(state['job'],202)
    if path=='/api/jobs/ui-check':return reply(state['job'])
    if path=='/api/jobs/ui-check/live-preview':return route.fulfill(path=str(ROOT/'uploads'/REF),content_type='image/png')
    # No mutation can escape this test context.
    if route.request.method!='GET':return reply({'detail':'Blocked in UI verification'},409)
    if path.startswith('/api/'):
        if path in ['/api/examples']:return reply([])
        return reply({})
    route.continue_()

browser=launch(headless=True,args=['--disable-gpu'])
try:
    page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.route('**/api/**',handle)
    draft={'source_video':SOURCE,'source_start_seconds':0,'source_end_seconds':14.042,'reference_images':[REF],'prompt':'Faceswap. Replace the main person with the referenced person. Keep the original movement, clothing, camera, lighting and background.'}
    page.add_init_script('localStorage.setItem("qwen-reference-draft",'+json.dumps(json.dumps(draft))+');')
    page.add_init_script("const d=Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype,'currentTime');Object.defineProperty(HTMLMediaElement.prototype,'currentTime',{...d,set(value){window.__lastSeek=value;d.set.call(this,value);}});")
    page.goto(args.url.rstrip('/')+'/#video-to-video',wait_until='networkidle')
    page.get_by_role('button',name='Replacement for Source 1',exact=True).wait_for()
    assert page.locator('.person-reference-picker').count()==0
    choice=page.get_by_role('button',name='Replacement for Source 1',exact=True)
    choice.scroll_into_view_if_needed()
    assert choice.get_by_alt_text('Selected Person 1').is_visible()
    # Reproduce the user's 403 after a server restart on the actual Scan button.
    state['session']='restarted-server-session'
    rejected_before=state['rejected']
    with page.expect_response(lambda response:response.url.endswith('/people') and response.status==200):
        page.get_by_role('button',name='Scan selected clip').click()
    assert state['rejected']==rejected_before+1 and state['refreshes']>=1
    assert choice.get_by_alt_text('Selected Person 1').is_visible()
    assert page.get_by_text('Refresh the app and try again.',exact=True).count()==0
    choice.click()
    menu=page.get_by_role('listbox',name='Choose replacement for Source 1')
    assert menu.get_by_role('option',name='Person 1 1 image',exact=True).locator('img').count()==1
    page.screenshot(path=str(OUT/'reference-menu.png'),full_page=True)
    menu.get_by_role('option',name='Keep person',exact=True).click()
    assert choice.inner_text().strip()=='Keep person'
    choice.press('ArrowDown')
    page.get_by_role('option',name='Keep person',exact=True).press('ArrowDown')
    page.get_by_role('option',name='Person 1 1 image',exact=True).press('Enter')
    assert choice.get_by_alt_text('Selected Person 1').is_visible()
    assert page.locator('.reference-source-editor .clip-timeline').count()==1
    model=page.locator('#reference-model')
    page.evaluate('window.scrollTo(0,0)')
    page.locator('.reference-workbench .reference-inputs').evaluate('(element)=>element.scrollTop=0')
    assert model.bounding_box()['y'] < page.get_by_role('heading',name='Source video',exact=True).bounding_box()['y']
    prompt=page.get_by_label('Additional instructions (optional)',exact=True)
    assert prompt.input_value()=='' and not prompt.is_visible()
    page.screenshot(path=str(OUT/'desktop.png'),full_page=True)
    button=page.get_by_role('button',name='Add to queue',exact=True)
    assert button.bounding_box()['y']+button.bounding_box()['height']<900
    button.click()
    page.get_by_role('heading',name='Live preview',exact=True).wait_for()
    assert state['requests'][-1]['prompt']==''
    assert state['requests'][-1]['person_assignments']=={'person-1':'ref-person-legacy-1'}
    assert page.locator('.reference-output video').count()==0
    assert page.get_by_text('0 / 8 steps',exact=True).is_visible()
    assert page.get_by_text('The first preview appears after the first generation step finishes.',exact=True).is_visible()
    state['job'].update(step=1,preview_step=1,preview_url='/api/jobs/ui-check/live-preview',preview_updated_at=time.time())
    page.get_by_alt_text('Approximate video generated at the latest completed step').wait_for()
    assert button.is_enabled()
    button.click()
    assert len(state['requests'])==2
    state['job'].update(stage='downloading_models',message='Downloading model files',download_completed_bytes=5_000_000_000,download_total_bytes=10_000_000_000)
    page.get_by_text('Downloading model files',exact=True).wait_for()
    assert button.is_enabled()
    assert page.get_by_role('dialog',name='Settings',exact=True).count()==0 or not page.get_by_role('dialog',name='Settings',exact=True).is_visible()
    state['job'].update(stage='generating')
    details=page.locator('.reference-inputs > .reference-extra-instructions')
    details.locator('summary').click()
    prompt.fill('Keep @')
    page.get_by_role('option',name='@Person1 Uploaded person 1 (1 image)').click()
    assert '@Person1' in prompt.input_value()
    prompt.fill('Preserve glasses.')
    page.get_by_role('tab',name='Camera view',exact=True).click()
    assert prompt.input_value()==''
    button.click()
    assert state['requests'][-1]['workflow']=='crossview'
    assert state['requests'][-1]['person_assignments']=={}
    assert state['requests'][-1]['reference_images']==[]
    disclosure=page.locator('details').filter(has=page.locator('summary',has_text='Replace subjects before changing camera view'))
    disclosure.locator('summary').click()
    page.get_by_label('Replace subjects first',exact=True).check()
    replacement=page.get_by_label('Additional replacement instructions (optional)',exact=True)
    assert replacement.input_value()=='' and not replacement.is_visible()
    button.click()
    assert state['requests'][-1]['replace_subjects_first']
    assert state['requests'][-1]['replacement_prompt']==''
    assert state['requests'][-1]['person_assignments']=={'person-1':'ref-person-legacy-1'}
    page.get_by_role('tab',name='Replace subjects',exact=True).click()
    assert prompt.input_value()=='Preserve glasses.'
    prompt.fill('')
    # Trim the actual source player through the integrated controls.
    trim=page.get_by_label('Clip start in seconds')
    trim.fill('2');trim.press('Tab')
    assert float(trim.input_value())==2
    assert page.evaluate('window.__lastSeek')==2
    assert page.locator('.reference-source-editor video').count()==1
    assert page.get_by_text('The clip changed. Scan it again to update the people.',exact=True).is_visible()
    button.click()
    assert page.get_by_text('Scan the selected clip again before creating a video.',exact=True).is_visible()
    people[0]['time']=2.17
    page.get_by_role('button',name='Scan selected clip').click()
    page.get_by_text('Finding people…',exact=True).wait_for(state='hidden')
    button.click()
    assert state['requests'][-1]['source_start_seconds']==2
    page.locator('.reference-workbench').get_by_label('Aspect ratio',exact=True).select_option('1:1')
    assert page.locator('.reference-resolution-result strong').inner_text().split(' × ')[0]==page.locator('.reference-resolution-result strong').inner_text().split(' × ')[1]
    page.locator('.reference-workbench').get_by_label('Resolution (px)',exact=True).select_option('draft')
    page.get_by_role('tab',name='Reference-guided',exact=True).click()
    assert prompt.input_value()==''
    button.click()
    assert state['requests'][-1]['workflow']=='reference'
    # Two distinct assignments, no mirrored second picker.
    page.get_by_role('tab',name='Replace subjects',exact=True).click()
    people.append({'id':'person-2','time':3.5,'box':[.1,.1,.2,.2],'hits':3,'preview':f'/api/uploads/{REF}'})
    page.get_by_role('button',name='Scan selected clip').click()
    second=page.get_by_role('button',name='Replacement for Source 2',exact=True)
    second.wait_for();second.click()
    page.get_by_role('listbox',name='Choose replacement for Source 2').get_by_role('option',name='Person 1 1 image',exact=True).click()
    choice.click();page.get_by_role('listbox',name='Choose replacement for Source 1').get_by_role('option',name='Keep person').click()
    button.click()
    assert state['requests'][-1]['person_assignments']=={'person-1':None,'person-2':'ref-person-legacy-1'}
    state['job'].update(step=2,preview_step=2,preview_url='/api/jobs/ui-check/live-preview',preview_updated_at=time.time())
    page.get_by_alt_text('Approximate video generated at the latest completed step').wait_for()
    advanced=page.locator('.reference-workbench .reference-advanced');advanced.locator('summary').click()
    for model_id,default in [(m['id'],m['generation_defaults']['steps']) for m in catalog() if m.get('family')=='minimax-h3-ref2va']:
        model.select_option(model_id)
        assert page.locator('#reference-steps').get_attribute('placeholder')==str(default)
        assert page.locator('#reference-steps').input_value()==''
    model.select_option('h3-ref-base-int8')
    page.locator('#reference-quality').select_option('turbo')
    assert page.locator('#reference-steps').get_attribute('placeholder')=='8'
    page.locator('#reference-steps').fill('12');button.click()
    assert state['requests'][-1]['steps']==12
    page.locator('.reference-steps-input button').click()
    assert page.locator('#reference-steps').input_value()==''
    advanced.locator('summary').click()
    for width,height in [(1280,720),(768,850),(390,844)]:
        page.set_viewport_size({'width':width,'height':height})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        progress_box=page.locator('.reference-workbench .reference-job-progress').bounding_box()
        settings_box=page.locator('.reference-output-settings').bounding_box()
        assert progress_box['y']+progress_box['height'] <= settings_box['y']+1, (width,progress_box,settings_box)
        page.screenshot(path=str(OUT/f'viewport-{width}.png'),full_page=True)
    assert not errors,errors
    print('PASS: session renewal after server restart, selection, thumbnails, keyboard, optional prompts, separate stages, queueing, live preview, trim, source ratio, responsive layout; no generation submitted')
    print(json.dumps({'pageHeight':page.evaluate('document.documentElement.scrollHeight'),'viewport':900,'submit':page.get_by_role('button',name='Add to queue',exact=True).bounding_box(),'errors':errors}))
except Exception:
    page.screenshot(path=str(OUT/'failure.png'),full_page=True)
    raise
finally:browser.close()

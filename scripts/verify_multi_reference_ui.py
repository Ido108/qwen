"""Browser interactions with all API mutations intercepted; never inference."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
from cloakbrowser import launch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.models import catalog
parser=argparse.ArgumentParser(description='Mocked browser checks; no generation request reaches the app.')
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--source-id',required=True)
parser.add_argument('--reference-id',required=True)
parser.add_argument('--api-url',default='http://127.0.0.1:47821')
parser.add_argument('--url',default='http://127.0.0.1:47830')
args=parser.parse_args()
DATA=args.data_dir
OUT=DATA/'validation/multi-reference-ui';OUT.mkdir(parents=True,exist_ok=True)
REF=args.reference_id
SOURCE=args.source_id
for fixture in (DATA/'uploads'/REF,DATA/'uploads/videos'/SOURCE):
    if not fixture.is_file():raise FileNotFoundError(f'UI fixture does not exist: {fixture}')
with urlopen(args.api_url.rstrip('/')+'/api/config') as response:config=json.load(response)
models={'models':[dict(m,installed=False,ready=False) for m in catalog()],'remote':False,'free_bytes':1e12,'transfer':{'status':'idle'}}
config.update(catalog=models,active_job=None)
state={'jobs':{},'submitted':[],'images':0}
errors=[]


def handle(route):
    path=urlsplit(route.request.url).path
    def reply(value,status=200):route.fulfill(status=status,json=value)
    if path=='/api/config':return reply(config)
    if path=='/api/session':return reply({'token':config['token']})
    if path=='/api/models':return reply(models)
    if path=='/api/queue':return reply({'active':None,'pending':[],'recent':[],'paused':False})
    if path in ('/api/history','/api/examples'):return reply([])
    if path=='/api/library':return reply({'items':[],'total':0})
    if path=='/api/gpu-memory':return reply({'available':True,'total_gb':24,'free_gb':24})
    if path=='/api/uploads':
        assert b'name="files"' in route.request.post_data_buffer
        state['images']+=1
        return reply([{'id':REF,'url':f'/api/uploads/{REF}','name':'Reference.png'}])
    if path==f'/api/uploads/{REF}':return route.fulfill(path=str(DATA/'uploads'/REF),content_type='image/png')
    if path.endswith('/metadata'):
        return reply({'id':SOURCE,'url':f'/api/video-uploads/{SOURCE}','name':'Source.mp4','poster_url':f'/api/uploads/{REF}','width':2160,'height':3840,'duration_seconds':14.042,'fps':24,'has_audio':True})
    if path==f'/api/video-uploads/{SOURCE}':return route.fulfill(path=str(DATA/'uploads/videos'/SOURCE))
    if path=='/api/audio-uploads':
        assert b'name="file"' in route.request.post_data_buffer
        return reply({'id':'voice.wav','name':'Voice.wav','url':'/api/audio-uploads/voice.wav'})
    if path=='/api/audio-uploads/voice.wav':return route.fulfill(status=200,body=b'',content_type='audio/wav')
    if path in ['/api/multi-reference/generate','/api/generate','/api/video/generate','/api/upscale/generate']:
        kind={'/api/multi-reference/generate':'multi-reference','/api/generate':'image','/api/video/generate':'video','/api/upscale/generate':'upscale'}[path]
        request=route.request.post_data_json
        state['submitted'].append((kind,request))
        job={'id':str(len(state['submitted'])),'kind':kind,'request':request,'status':'running','stage':'downloading_models','message':'Downloading model files','download_completed_bytes':5e9,'download_total_bytes':10e9,'created_at':time.time(),'started_at':time.time(),'stage_started_at':time.time(),'seed':0,'step':0,'total_steps':11 if request.get('refine') else 8}
        state['jobs'][job['id']]=job
        return reply(job,202)
    if path.startswith('/api/jobs/'):
        parts=path.split('/')
        if len(parts)>4 and parts[4]=='live-preview':return route.fulfill(path=str(DATA/'uploads'/REF),content_type='image/png')
        return reply(state['jobs'][parts[3]])
    # No API mutation or unspecified request reaches the running app.
    return reply({})


browser=launch(headless=True,args=['--disable-gpu'])
try:
    page=browser.new_page(viewport={'width':1440,'height':900})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.route('**/api/**',handle)
    drafts={'image-studio-draft-v1':{'prompt':'A chair','enhance':False},'qwen-video-draft':{'prompt':'A tree in wind','enhance':False},'qwen-upscale-draft':{'source_video':SOURCE}}
    page.add_init_script(';'.join('localStorage.setItem('+json.dumps(key)+','+json.dumps(json.dumps(value))+')' for key,value in drafts.items()))
    page.goto(args.url.rstrip('/')+'/#multi-reference',wait_until='networkidle')
    area=page.locator('.msr-workbench')
    area.get_by_role('heading',name='Reference images to video').wait_for()
    file=str(DATA/'uploads'/REF)
    area.locator('input[type=file][multiple]').set_input_files(file)
    area.get_by_alt_text('Image 1',exact=True).wait_for()
    area.locator('.msr-tag',has_text='Image 1').click()
    assert area.get_by_label('Describe the video').input_value()=='Image 1'
    area.get_by_label('Describe the video').fill('Image 1 walks through the scene.')
    area.locator('input[type=file][accept="image/png,image/jpeg,image/webp"]:not([multiple])').set_input_files(file)
    area.get_by_alt_text('Background reference').wait_for()
    assert state['images']==2
    area.get_by_label('LTX model',exact=True).select_option('ltx25-msr-q8_0')
    area.get_by_label('LTX output size').select_option('custom')
    area.get_by_label('Width (px)',exact=True).fill('736')
    area.get_by_label('Height (px)',exact=True).fill('416')
    area.locator('.msr-options>summary').click()
    area.get_by_label('Double width and height (2×)',exact=True).check()
    area.get_by_label('Random seed',exact=True).uncheck()
    area.get_by_label('Seed',exact=True).fill('713')
    area.locator('.msr-voices>summary').click()
    area.locator('.msr-voice input[type=file]').first.set_input_files({'name':'voice.wav','mimeType':'audio/wav','buffer':b'fixture'})
    area.locator('audio').wait_for()
    area.locator('.msr-options>summary').click()
    area.locator('.msr-voices>summary').click()
    submit=area.get_by_role('button',name='Add to queue',exact=True)
    assert submit.is_enabled()
    submit.click()
    area.get_by_text('Downloading model files',exact=True).wait_for()
    assert state['submitted'][-1][1]['refine']
    assert state['submitted'][-1][1]['background']==REF
    assert state['submitted'][-1][1]['audio_references']==['voice.wav']
    assert state['submitted'][-1][1]['seed']==713
    assert submit.is_enabled()
    assert not page.locator('dialog[open]').count()
    page.screenshot(path=str(OUT/'downloading.png'),full_page=True)
    state['jobs']['1'].update(stage='generating',step=1,preview_url='/api/jobs/1/live-preview',preview_updated_at=time.time())
    area.get_by_alt_text('Live generation preview').wait_for()
    submit.click()
    assert len(state['submitted'])==2
    for width,height in [(1440,900),(1280,720),(768,850),(390,844)]:
        page.set_viewport_size({'width':width,'height':height})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width
        if width>=1000:
            box=submit.bounding_box();assert box['y']+box['height']<=height,(width,box)
        page.screenshot(path=str(OUT/f'viewport-{width}.png'),full_page=True)
    page.set_viewport_size({'width':1440,'height':900})
    # All existing generation tabs permit submitting uninstalled models too.
    page.get_by_role('button',name='Images',exact=True).click()
    page.locator('main:not([hidden])').get_by_role('button',name='Generate image',exact=True).click()
    assert state['submitted'][-1][0]=='image'
    page.get_by_role('button',name='Video',exact=True).click()
    page.get_by_role('tab',name='Create video',exact=True).click()
    page.locator('main:not([hidden])').get_by_role('button',name='Generate video',exact=True).click()
    assert state['submitted'][-1][0]=='video'
    page.get_by_role('tab',name='Upscale',exact=True).click()
    page.locator('main:not([hidden])').get_by_role('button',name='Add to queue',exact=True).click()
    assert state['submitted'][-1][0]=='upscale'
    assert not errors,errors
    print('PASS: real image/voice upload contracts, GGUF selection, custom size, refinement, seed, download progress, live preview, continuing queue, all generation tabs with missing models, responsive layout. No inference.')
except Exception:
    page.screenshot(path=str(OUT/'failure.png'),full_page=True)
    raise
finally:browser.close()

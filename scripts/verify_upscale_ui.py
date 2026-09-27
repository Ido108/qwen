"""UI-only contracts. Every API, upload and generation is intercepted."""
import argparse,json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit,parse_qs
from cloakbrowser import launch
from playwright.sync_api import expect

p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:47830');p.add_argument('--api-url',default='http://127.0.0.1:47821');args=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.data_dir/'validation/upscale-ui';out.mkdir(parents=True,exist_ok=True)
fixture=next((args.data_dir/'uploads').glob('*.png'))
with urlopen(args.api_url+'/api/config',timeout=4) as r:config=json.load(r)
config['active_job']=None
entries=json.loads((root/'reference/model-catalog.json').read_text(encoding='utf-8'))+json.loads((root/'reference/reference-video-catalog.json').read_text(encoding='utf-8'))
config['catalog']['models']=[{**m,'installed':False,'ready':False} for m in entries]
source={'id':'a'*32+'.mp4','name':'Source.mp4','url':'/api/input-fixture','poster_url':'/api/fixture','width':1080,'height':1920,'duration_seconds':5,'fps':24,'has_audio':True}
previous={'id':'previous','kind':'upscale','status':'succeeded','stage':'complete','request':{'model_id':'seedvr2-3b-fp8'},'seed':12,'elapsed_seconds':60,'width':1080,'height':1920,'duration_seconds':5,'video_url':'/api/result-fixture','poster_url':'/api/fixture','total_chunks':2}
active={**previous,'id':'active','status':'running','stage':'upscaling','message':'Upscaling frames','step':2,'total_steps':6,'started_at':1,'stage_started_at':1,'preview_chunk':1,'preview_url':'/api/live-fixture','preview_updated_at':1}
state={'active':None,'submitted':[],'current':dict(active)}
def route(r):
 u=urlsplit(r.request.url);path=u.path
 if path=='/api/config':return r.fulfill(json=config)
 if path=='/api/session':return r.fulfill(json={'token':config['token']})
 if path=='/api/models':return r.fulfill(json=config['catalog'])
 if path=='/api/queue':return r.fulfill(json={'active':state['active'],'pending':[],'recent':[],'paused':False})
 if path=='/api/history':return r.fulfill(json=[previous] if parse_qs(u.query).get('kind')==['upscale'] else [])
 if path.endswith('/metadata'):return r.fulfill(json=source)
 if path=='/api/upscale/generate':
  state['submitted'].append(r.request.post_data_json)
  state['active']=state['current']
  return r.fulfill(json=state['current'])
 if path=='/api/jobs/active':return r.fulfill(json=state['current'])
 if path in ('/api/fixture','/api/live-fixture'):return r.fulfill(path=str(fixture),content_type='image/png')
 if path.endswith('-fixture'):return r.fulfill(body=b'',content_type='video/mp4')
 if r.request.method not in ('GET','HEAD','OPTIONS'):raise AssertionError('Unexpected mutation '+path)
 return r.fulfill(json=[] if path=='/api/examples' else {})
browser=launch(headless=True,args=['--disable-gpu'])
try:
 page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.route('**/api/**',route)
 page.add_init_script("localStorage.setItem('qwen-upscale-draft',"+json.dumps(json.dumps({'source_video':source['id']}))+');')
 page.goto(args.url+'/#upscale',wait_until='networkidle')
 area=page.locator('.upscale-page')
 expect(area.locator('.upscale-dimensions strong')).to_have_text('1080 × 1920')
 assert area.locator('#upscale-model option').count()>=25
 assert area.locator('#upscale-model optgroup').count()>=5
 assert area.locator('#upscale-model').bounding_box()['y']<area.get_by_role('heading',name='Input video').bounding_box()['y']
 area.get_by_text('Advanced settings',exact=False).first.click()
 area.locator('#upscale-model').select_option('seedvr2-7b-sharp-q8_0')
 expect(area.locator('#upscale-batch')).to_have_value('5')
 expect(area.locator('#upscale-overlap')).to_have_value('1')
 area.locator('#upscale-model').select_option('flashvsr11-int8')
 expect(area.locator('#upscale-flash-strength')).to_have_value('1')
 expect(area.locator('#upscale-flash-sparse')).to_have_value('2')
 assert not area.locator('#upscale-batch').count()
 area.locator('#upscale-model').select_option('swiftvr-bf16')
 expect(area.locator('#upscale-clip')).to_have_value('8')
 assert not area.locator('#upscale-flash-strength').count()
 area.locator('#upscale-size-mode').select_option('scale');area.locator('#upscale-size').select_option('2')
 expect(area.locator('.upscale-dimensions strong')).to_have_text('2160 × 3840')
 area.locator('#upscale-size-mode').select_option('short_side');area.locator('#upscale-size').select_option('720')
 expect(area.locator('.upscale-dimensions strong')).to_have_text('720 × 1280')
 area.locator('#upscale-model').select_option('seedvr2-3b-q4')
 area.get_by_text('Advanced settings',exact=False).first.click()
 for width in (1440,768,390):
  page.set_viewport_size({'width':width,'height':1000})
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  page.screenshot(path=str(out/f'upscale-{width}.png'),full_page=True)
 page.set_viewport_size({'width':1280,'height':720})
 page.evaluate('window.scrollTo(0,0)')
 submit=area.get_by_role('button',name='Add to queue').bounding_box()
 assert submit['y']+submit['height']<=720,submit
 page.screenshot(path=str(out/'upscale-1280-720.png'),full_page=True)
 page.set_viewport_size({'width':1440,'height':1000})
 area.get_by_role('button',name='Add to queue').click()
 expect(area.get_by_role('heading',name='Live preview')).to_be_visible()
 preview=area.get_by_alt_text('Upscaled frames from the latest completed section')
 expect(preview).to_have_attribute('src','/api/live-fixture?v=1')
 state['current']['preview_updated_at']=2;state['current']['preview_chunk']=2
 expect(preview).to_have_attribute('src','/api/live-fixture?v=2',timeout=5000)
 area.get_by_role('button',name='Previous result').click()
 expect(area.locator('video[src="/api/result-fixture"]')).to_be_visible()
 area.get_by_role('button',name='Live preview',exact=True).click()
 area.get_by_role('button',name='Add to queue').click()
 expect(preview).to_be_visible()
 assert len(state['submitted'])==2 and state['submitted'][0]['model_id']=='seedvr2-3b-q4'
 page.reload(wait_until='networkidle')
 expect(page.locator('.upscale-page').get_by_role('heading',name='Live preview')).to_be_visible()
 state['current'].update(status='succeeded',stage='complete',message='Saved',finished_at=10);state['active']=None
 expect(page.locator('.upscale-page').get_by_role('heading',name='Upscaled video')).to_be_visible(timeout=6000)
 expect(page.locator('.upscale-page .run-details')).to_be_visible()
 assert not errors,errors
 print('PASS: all model families, exact output ratios, model defaults, responsive layout, live preview revisions, previous result, queue while running, reconnect and completed details. No inference.')
except:
 page.screenshot(path=str(out/'failure.png'),full_page=True);raise
finally:browser.close()

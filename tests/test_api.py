"""Validation and persistence contracts. Actual CUDA inference is verified separately."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

temporary = tempfile.TemporaryDirectory(prefix='image-studio-tests-')
os.environ['QWEN_DATA_DIR'] = temporary.name
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from backend.common import GenerationRequest, VideoRequest, JOBS_DIR, UPLOADS_DIR, resolve_image, read_json, write_json
from backend.server import app

def png():
    stream = io.BytesIO()
    Image.new('RGBA', (32, 24), (255, 0, 0, 100)).save(stream, 'PNG')
    return stream.getvalue()

class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.headers = {'X-Studio-Token': cls.client.get('/api/config').json()['token']}

    def test_library_delete_removes_files_and_keeps_running_jobs(self):
        from backend import server
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'JOBS_DIR', Path(folder) / 'jobs'):
            jobs_dir = Path(folder) / 'jobs'
            made = {}
            for name, status in (('a', 'succeeded'), ('b', 'succeeded'), ('c', 'succeeded'), ('busy', 'running')):
                output = Path(folder) / f'{name}.png'
                output.write_bytes(b'png')
                output.with_suffix('.json').write_text('{}')
                (jobs_dir / name).mkdir(parents=True)
                write_json(jobs_dir / name / 'status.json', {'id': name, 'status': status, 'kind': 'image', 'created_at': time.time(),
                                                              'output_path': str(output), 'request': {'prompt': f'prompt {name}', 'model_id': 'gguf-q8_0'}})
                made[name] = output
            with patch.object(server, 'active_job', 'busy'):
                result = self.client.post('/api/library/delete', json={'ids': ['a', 'busy']}, headers=self.headers).json()
                self.assertEqual(result['deleted'], 1)
                self.assertFalse(made['a'].exists() or made['a'].with_suffix('.json').exists() or (jobs_dir / 'a').exists())
                self.assertTrue(made['busy'].exists() and (jobs_dir / 'busy').exists())
                # Select all with a search deletes only the matching finished results.
                self.assertEqual(self.client.post('/api/library/delete', json={'all': True, 'search': 'prompt b'}, headers=self.headers).json()['deleted'], 1)
                self.assertTrue(made['c'].exists() and not made['b'].exists())
                self.assertEqual(self.client.post('/api/library/delete', json={'all': True}, headers=self.headers).json()['deleted'], 1)
                self.assertFalse(made['c'].exists())
                self.assertTrue(made['busy'].exists())

    def test_library_delete_works_while_the_model_worker_holds_the_log_open(self):
        from backend import server
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'JOBS_DIR', Path(folder) / 'jobs'):
            directory = Path(folder) / 'jobs' / 'held'
            directory.mkdir(parents=True)
            output = Path(folder) / 'held.png'
            output.write_bytes(b'png')
            (directory / 'input-1.png').write_bytes(b'png')
            write_json(directory / 'status.json', {'id': 'held', 'status': 'succeeded', 'kind': 'image', 'created_at': time.time(),
                                                   'output_path': str(output), 'request': {'prompt': 'p', 'model_id': 'gguf-q8_0'}})
            with (directory / 'worker.log').open('a') as log:
                result = self.client.post('/api/library/delete', json={'ids': ['held']}, headers=self.headers).json()
                self.assertEqual((result['deleted'], result['failed']), (1, []))
                self.assertFalse(output.exists() or (directory / 'status.json').exists() or (directory / 'input-1.png').exists())
                self.assertEqual(self.client.get('/api/library').json()['total'], 0)

    def test_local_access_and_session_guard(self):
        self.assertEqual(self.client.post('/api/open-output').status_code, 403)
        self.assertEqual(self.client.get('/api/config', headers={'host':'evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/uploads', headers={**self.headers, 'Origin':'https://evil.example'}).status_code, 403)

    def test_lora_prompt_settings_persist_and_validate_the_target(self):
        from backend import lora_preferences, models
        with tempfile.TemporaryDirectory() as folder, patch.object(lora_preferences, 'PATH', Path(folder) / 'settings.json'), patch('backend.civitai.user_catalog', return_value=[{'id':'qa-ink','kind':'image-lora','title':'Ink'}]):
            path = '/api/loras/qa-ink/settings'
            self.assertEqual(self.client.put(path, json={'prompt_addition':'Paper.'}).status_code, 403)
            response = self.client.put(path, json={'prompt_addition':'  Paper texture.  '}, headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.client.get(path).json()['prompt_addition'], 'Paper texture.')
            self.assertEqual(models.entry('qa-ink')['prompt_addition'], 'Paper texture.')
            self.assertEqual(self.client.put(path, json={'prompt_addition':'x'*4001}, headers=self.headers).status_code, 422)
            self.assertEqual(self.client.put('/api/loras/gguf-q8_0/settings', json={'prompt_addition':'x'}, headers=self.headers).status_code, 400)
            self.assertEqual(self.client.get('/api/loras/missing/settings').status_code, 400)
            self.client.put(path, json={'prompt_addition':''}, headers=self.headers)
            self.assertEqual(self.client.get(path).json()['prompt_addition'], '')

    def test_phone_heic_photo_becomes_a_readable_upload_and_person(self):
        from pillow_heif import register_heif_opener
        register_heif_opener(thumbnails=False)
        stream = io.BytesIO()
        Image.new('RGB', (48, 64), (41, 133, 207)).save(stream, format='HEIF')
        raw = stream.getvalue()
        result = self.client.post('/api/uploads', headers=self.headers,
                                  files=[('files', ('IMG_1001.HEIC', raw, 'image/heic'))])
        self.assertEqual(result.status_code, 200, result.text)
        photo = result.json()[0]
        self.assertEqual((photo['width'], photo['height']), (48, 64))
        self.assertEqual(photo['name'], 'IMG_1001.HEIC')
        with Image.open(UPLOADS_DIR / photo['id']) as stored:
            self.assertEqual(stored.format, 'PNG')
            self.assertEqual(stored.size, (48, 64))
        self.assertEqual(self.client.get(photo['thumb_url']).status_code, 200)
        person = self.client.post('/api/people', headers=self.headers,
                                 json={'name': 'Phone upload test', 'image_ids': [photo['id']]})
        self.assertEqual(person.status_code, 200, person.text)
        self.assertEqual(person.json()['image_ids'], [photo['id']])
        # Re-selecting the same phone file must not make another stored image.
        repeated = self.client.post('/api/uploads', headers=self.headers,
                                    files=[('files', ('IMG_1001.HEIF', raw, 'application/octet-stream'))])
        self.assertEqual(repeated.json()[0]['id'], photo['id'])

    def test_phone_jpeg_orientation_and_invalid_batch(self):
        source = Image.new('RGB', (40, 24), (40, 180, 80))
        exif = source.getexif()
        exif[274] = 6
        stream = io.BytesIO()
        source.save(stream, format='JPEG', exif=exif)
        result = self.client.post('/api/uploads', headers=self.headers,
                                  files=[('files', ('portrait.jpg', stream.getvalue(), 'image/jpeg'))])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual((result.json()[0]['width'], result.json()[0]['height']), (24, 40))
        before = set(UPLOADS_DIR.iterdir())
        invalid = self.client.post('/api/uploads', headers=self.headers, files=[
            ('files', ('new.png', png(), 'image/png')),
            ('files', ('broken.heic', b'not an image', 'image/heic'))])
        self.assertEqual(invalid.status_code, 400)
        self.assertIn('Could not read this image', invalid.json()['detail'])
        self.assertEqual(set(UPLOADS_DIR.iterdir()), before, 'A bad batch must not save partial uploads')

    def test_expired_session_has_retry_code_without_running_handler(self):
        with patch('backend.server.resolve_video') as resolve:
            response=self.client.post('/api/video-uploads/'+'a'*32+'.mp4/people',json={},headers={'X-Studio-Token':'old'})
            self.assertEqual(response.status_code,403)
            self.assertEqual(response.json()['code'],'session_expired')
            resolve.assert_not_called()
        session=self.client.get('/api/session')
        self.assertEqual(session.json()['token'],self.headers['X-Studio-Token'])
        self.assertEqual(session.headers['cache-control'],'no-store')
        self.assertEqual(self.client.get('/api/session',headers={'host':'evil.example'}).status_code,403)
        wrong_origin=self.client.post('/api/uploads',headers={'Origin':'https://evil.example'})
        self.assertNotIn('code',wrong_origin.json())

    def test_atomic_status_write_survives_a_windows_reader(self):
        target=Path(temporary.name)/'status-retry.json'
        write_json(target,{'step':1})
        reader=target.open('rb')
        timer=threading.Timer(.08,reader.close)
        timer.start()
        try:
            write_json(target,{'step':2})
            self.assertEqual(json.loads(target.read_text()),{'step':2})
        finally:
            timer.join();reader.close();target.unlink()

    def test_recent_uploads_dedupe_list_and_thumbnail(self):
        def image(color):
            stream = io.BytesIO(); Image.new('RGB', (40, 30), color).save(stream, 'PNG'); return stream.getvalue()
        first = self.client.post('/api/uploads', files=[('files', ('a.png', image('blue'), 'image/png'))], headers=self.headers).json()[0]
        time.sleep(0.02)
        other = self.client.post('/api/uploads', files=[('files', ('b.png', image('green'), 'image/png'))], headers=self.headers).json()[0]
        time.sleep(0.02)
        again = self.client.post('/api/uploads', files=[('files', ('copy.png', image('blue'), 'image/png'))], headers=self.headers).json()[0]
        self.assertEqual(first['id'], again['id'])
        self.assertNotEqual(first['id'], other['id'])
        listing = self.client.get('/api/uploads?limit=200').json()
        ids = [item['id'] for item in listing['items']]
        self.assertEqual(ids.count(first['id']), 1)
        self.assertLess(ids.index(first['id']), ids.index(other['id']))  # re-adding moves it to the top
        self.assertFalse(any(name.startswith('.') for name in ids))
        thumb = self.client.get(f"/api/uploads/{first['id']}/thumb")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.headers['content-type'], 'image/webp')
        self.assertEqual(self.client.get('/api/uploads/../../x.png/thumb').status_code, 404)

    def test_uploads_delete_removes_every_copy_and_keeps_people_and_busy_pictures(self):
        from backend import server
        def image(color):
            stream = io.BytesIO(); Image.new('RGB', (41, 31), color).save(stream, 'PNG'); return stream.getvalue()
        post = lambda name, color: self.client.post('/api/uploads', files=[('files', (name, image(color), 'image/png'))], headers=self.headers).json()[0]['id']
        gone, person_photo, busy_photo = post('red.png', 'red'), post('teal.png', 'teal'), post('navy.png', 'navy')
        copy = server.UPLOADS_DIR / ('copy-' + gone)  # an older copy saved under another name
        copy.write_bytes((server.UPLOADS_DIR / gone).read_bytes())
        busy_job = 'f' * 32
        (server.JOBS_DIR / busy_job).mkdir(parents=True, exist_ok=True)
        write_json(server.JOBS_DIR / busy_job / 'status.json', {'id': busy_job, 'status': 'running', 'request': {'images': [busy_photo]}})
        with patch.object(server.people, 'load', return_value=[{'image_ids': [person_photo]}]), patch.object(server, 'active_job', busy_job):
            result = self.client.post('/api/uploads/delete', json={'ids': [gone, person_photo, busy_photo]}, headers=self.headers).json()
        self.assertEqual(result['deleted'], 2)
        self.assertFalse((server.UPLOADS_DIR / gone).exists() or copy.exists())
        self.assertTrue((server.UPLOADS_DIR / person_photo).exists() and (server.UPLOADS_DIR / busy_photo).exists())

    def test_image_upload_preserves_alpha(self):
        response = self.client.post('/api/uploads', files=[('files', ('alpha.png', png(), 'image/png'))], headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        image = response.json()[0]
        loaded = Image.open(io.BytesIO(self.client.get(image['url']).content))
        self.assertEqual(loaded.mode, 'RGBA')
        self.assertEqual(loaded.getchannel('A').getextrema(), (100,100))

    def test_status_read_retries_windows_replacement_sharing_race(self):
        with patch.object(Path, 'read_text', side_effect=[PermissionError(13, 'Sharing violation'), '{"step":2}']) as read:
            self.assertEqual(read_json(Path('status.json')), {'step': 2})
            self.assertEqual(read.call_count, 2)

    def test_audio_reference_upload_and_invalid_file_cleanup(self):
        import wave
        from backend.common import AUDIO_UPLOADS_DIR, resolve_audio
        stream=io.BytesIO()
        with wave.open(stream,'wb') as audio:
            audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000)
            audio.writeframes(b'\x00\x00'*1600)
        response=self.client.post('/api/audio-uploads',files={'file':('voice.wav',stream.getvalue(),'audio/wav')},headers=self.headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.get(response.json()['url']).content,stream.getvalue())
        before=set(AUDIO_UPLOADS_DIR.iterdir())
        bad=self.client.post('/api/audio-uploads',files={'file':('bad.wav',b'not audio','audio/wav')},headers=self.headers)
        self.assertEqual(bad.status_code,400)
        self.assertEqual(before,set(AUDIO_UPLOADS_DIR.iterdir()))
        with self.assertRaises(ValueError):
            resolve_audio('../voice.wav')

    def test_invalid_batch_leaves_no_partial_files(self):
        before = set(UPLOADS_DIR.iterdir())
        response = self.client.post('/api/uploads', files=[('files',('a.png',png(),'image/png')),('files',('bad.png',b'not an image','image/png'))], headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(before, set(UPLOADS_DIR.iterdir()))

    def test_ten_image_boundary(self):
        files = [('files',(f'{i}.png',png(),'image/png')) for i in range(10)]
        self.assertEqual(self.client.post('/api/uploads',files=files,headers=self.headers).status_code,200)
        self.assertEqual(self.client.post('/api/uploads',files=files+[files[0]],headers=self.headers).status_code,400)
        with self.assertRaises(ValidationError):
            GenerationRequest(prompt='image',images=['x']*11)

    def test_seed_prompt_and_dimension_validation(self):
        for kwargs in ({'prompt':'  '},{'prompt':'x','seed':-1},{'prompt':'x','seed':2147483648},{'prompt':'x','width':257},{'prompt':'x','width':3000}):
            with self.assertRaises(ValidationError):
                GenerationRequest(**kwargs)
        self.assertEqual(GenerationRequest(prompt=' x ',width=1000).prompt,'x')

    def test_empty_image_prompt_stays_empty_when_queued(self):
        from backend import server
        image=self.client.post('/api/uploads',files={'files':('a.png',png(),'image/png')},headers=self.headers).json()[0]['id']
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'JOBS_DIR',Path(folder)),patch.object(server,'pending_jobs',[]),patch.object(server,'start_next_locked'),patch.object(server,'save_queue_locked'):
            response=self.client.post('/api/generate',json={'prompt':' ','images':[image],'enhance':True,'auto_resolution':1024,'output_dir':folder},headers=self.headers)
            self.assertEqual(response.status_code,202,response.text)
            r=response.json()['request']
            self.assertEqual(r['prompt'],'')
            self.assertFalse(r['enhance'])

    def test_path_traversal_cannot_read_files(self):
        for image_id in ('../secret.png', 'example:../../package.json', str(Path(temporary.name)/'secret.png')):
            with self.assertRaises(ValueError):
                resolve_image(image_id)

    def test_turbo_steps_default_custom_and_tools_validated(self):
        from backend import server
        upload = self.client.post('/api/uploads', files=[('files', ('a.png', png(), 'image/png'))], headers=self.headers).json()[0]['id']
        with tempfile.TemporaryDirectory() as folder,              patch.object(server,'JOBS_DIR',Path(folder)),              patch.object(server,'active_job',None),              patch.object(server,'threading',SimpleNamespace(Thread=Mock())):
            def post(**values):
                server.active_job=None
                return self.client.post('/api/generate',json={'prompt':'An apple','enhance':False,'auto_resolution':1024,'output_dir':folder,**values},headers=self.headers)
            for steps,expected in ((None,(8,'model')),(12,(12,'custom'))):
                response=post(model_id='turbo-gguf-q8_0',**({'steps':steps} if steps else {}))
                self.assertEqual(response.status_code,202,response.text)
                saved=json.loads((Path(folder)/response.json()['id']/'request.json').read_text())
                self.assertEqual((saved['steps'],saved['steps_mode']),expected)
            # Wrong input count, wrong model family, bad outpaint options, unknown tool.
            self.assertIn('needs 2 input images',post(model_id='gguf-q8_0',tool_id='outfit-swap',images=[upload]).json()['detail'])
            self.assertIn('Turbo',post(model_id='gguf-q8_0',tool_id='character-swap',images=[upload,upload]).json()['detail'])
            self.assertEqual(post(model_id='gguf-q8_0',tool_id='outpaint',images=[upload],tool_options={'direction':'up','amount':'50'}).status_code,400)
            self.assertEqual(post(model_id='gguf-q8_0',tool_id='nope',images=[upload]).status_code,400)
            response=post(model_id='gguf-q8_0',tool_id='restore',images=[upload],enhance=True,lora_strength=0.5,vae_id='vae-texture-fix')
            self.assertEqual(response.status_code,202,response.text)
            saved=json.loads((Path(folder)/response.json()['id']/'request.json').read_text())
            self.assertEqual((saved['enhance'],saved['lora_strength'],saved['reference_resolution'],saved['vae_id']),(False,0.5,0,'vae-texture-fix'))
        from backend.models import required_image
        self.assertEqual(required_image({'model_id':'turbo-bf16','tool_id':'restore'}),['turbo-bf16','encoder-int8','vae-texture-fix','lora-restore'])
        self.assertEqual(post_status:=self.client.post('/api/generate',json={'prompt':'x','lora_strength':3},headers=self.headers).status_code,422)

    def test_swap_queue_stores_server_prompt_and_face_choices(self):
        from backend import server, image_people
        upload = self.client.post('/api/uploads', files=[('files', ('a.png', png(), 'image/png'))], headers=self.headers).json()[0]['id']
        scan = {'revision': 'a'*24, 'faces': [{'id': 'face-1', 'box': [.1,.2,.2,.3]}, {'id': 'face-2', 'box': [.6,.2,.2,.3]}]}
        selection = {'scene_image': upload, 'reference_image': upload, 'scene_revision': 'a'*24,
                     'reference_revision': 'a'*24, 'assignments': {'face-1': None, 'face-2': 'face-1'}}
        with tempfile.TemporaryDirectory() as folder, patch.object(image_people, 'analyze_image', return_value=scan), \
             patch.object(server, 'enqueue_locked') as enqueue:
            body = {'prompt': 'Replace everyone with a random person', 'model_id': 'turbo-gguf-q8_0',
                    'tool_id': 'head-swap', 'images': [upload,upload], 'enhance': True, 'output_dir': folder}
            self.assertEqual(self.client.post('/api/generate', json=body, headers=self.headers).status_code, 400)
            self.assertFalse(enqueue.called)
            response = self.client.post('/api/generate', json={**body, 'swap_selection': selection, 'swap_extra': 'Keep the earrings.'}, headers=self.headers)
            self.assertEqual(response.status_code, 202, response.text)
            saved = enqueue.call_args.args[1]
            self.assertEqual(saved['swap_selection'], selection)
            self.assertIn('Keep person 1 in <image1>', saved['prompt'])
            self.assertIn('person 2 in <image1> on the right', saved['prompt'])
            self.assertNotIn('random person', saved['prompt'])
            self.assertFalse(saved['enhance'])
            self.assertEqual(saved['steps'], 8)
            self.assertEqual(self.client.get('/api/image-people', params={'image_id': upload}).json(), scan)
        self.assertEqual(self.client.get('/api/image-people', params={'image_id': '../secret.png'}).status_code, 400)

    def test_tool_preparation_sizes(self):
        from backend.image_tools import prepare, outpaint_canvas, paste_back
        from backend.models import image_tool
        big = Image.new('RGBA', (4000, 3000), 'red')
        images, size, box = prepare(image_tool('restore'), {'auto_resolution': 2048}, [big])
        self.assertEqual(size, images[0].size)
        self.assertTrue(size[0] % 32 == 0 and size[1] % 32 == 0 and 1_800_000 < size[0] * size[1] < 2_200_000)
        images, size, box = prepare(image_tool('outfit-swap'), {'auto_resolution': 1024}, [big, Image.new('RGBA', (3000, 3000))])
        self.assertLessEqual(size[0] * size[1], 1024 * 1024 * 1.1)
        self.assertLessEqual(images[1].width * images[1].height, 1024 * 1024 * 1.1)
        self.assertIsNone(prepare(image_tool('angles'), {}, [big])[1])
        canvas, box = outpaint_canvas(Image.new('RGBA', (400, 300), 'red'), 'right', '100', 1024 * 1024)
        self.assertEqual(canvas.getpixel((5, canvas.height // 2))[:3], (255, 0, 0))
        self.assertEqual(canvas.getpixel((canvas.width - 5, canvas.height // 2))[:3], (128, 128, 128))
        result = paste_back(Image.new('RGBA', canvas.size, 'blue'), box)
        x, y, original = box
        self.assertEqual(result.getpixel((x + 5, y + original.height // 2))[:3], (255, 0, 0))  # left edge touches the border: kept
        self.assertEqual(result.getpixel((canvas.width - 5, canvas.height // 2))[:3], (0, 0, 255))  # generated area untouched

    def test_image_steps_and_auto_resolution_validation(self):
        for values in ({'steps':0},{'steps':81},{'steps':12.5},{'auto_resolution':1536}):
            response=self.client.post('/api/generate',json={'prompt':'An apple',**values},headers=self.headers)
            self.assertEqual(response.status_code,422,response.text)
        self.assertIsNone(GenerationRequest(prompt='An apple').steps)

    def test_generation_persists_model_default_or_custom_steps(self):
        from backend import server
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(server,'JOBS_DIR',Path(folder)), \
             patch.object(server,'active_job',None), \
             patch.object(server.models,'required',return_value=[]), \
             patch.object(server.models,'entry',return_value={'generation_defaults':{'steps':24}}), \
             patch.object(server,'threading',SimpleNamespace(Thread=Mock())):
            for steps,expected,mode in [(None,24,'model'),(12,12,'custom')]:
                server.active_job=None
                # speed_lora off: the model's own default (on Qwen Standard Create runs Pruna 8-step by default)
                response=self.client.post('/api/generate',json={'prompt':'An apple','enhance':False,'steps':steps,'auto_resolution':1024,'output_dir':folder,'speed_lora':False},headers=self.headers)
                self.assertEqual(response.status_code,202,response.text)
                job=response.json()
                saved=json.loads((Path(folder)/job['id']/'request.json').read_text())
                self.assertEqual((job['total_steps'],saved['steps'],saved['steps_mode'],saved['auto_resolution']),(expected,expected,mode,1024))

    def test_missing_model_is_accepted_into_queue(self):
        from backend import server
        with tempfile.TemporaryDirectory() as folder, patch.multiple(server,JOBS_DIR=Path(folder),active_job=None,pending_jobs=[]), \
             patch.object(server.models,'is_installed',return_value=False),patch.object(server,'threading',SimpleNamespace(Thread=Mock())):
            response=self.client.post('/api/generate',json={'prompt':'A flower','enhance':False,'output_dir':folder},headers=self.headers)
            self.assertEqual(response.status_code,202,response.text)
            self.assertEqual(response.json()['status'],'queued')

    def test_video_duration_and_canvas_constraints(self):
        from backend.models import required_video
        for values in ({'frames':120},{'width':769},{'frames':379},{'prompt':' '}):
            with self.assertRaises(ValidationError):
                VideoRequest(**({'prompt':'A beach'}|values))
        with self.assertRaises(ValueError):
            required_video(VideoRequest(prompt='A beach',width=1344,height=1344))
        self.assertIn('h3-turbo',required_video(VideoRequest(prompt='A beach',model_id='h3-gguf-q4',quality='turbo')))
        self.assertNotIn('h3-turbo',required_video(VideoRequest(prompt='A beach')))

    def test_video_input_mode_and_missing_weights(self):
        for values in ({'model_id':'fasth3-q4_k_m','mode':'image','first_frame':'missing.png'},
                       {'model_id':'h3-gguf-q4','mode':'image'},
                       {'model_id':'h3-gguf-q4','first_frame':'../private.png'}):
            response=self.client.post('/api/video/generate',json={'prompt':'A beach',**values},headers=self.headers)
            self.assertEqual(response.status_code,400,response.text)
        from backend import server
        with tempfile.TemporaryDirectory() as folder, patch.multiple(server,JOBS_DIR=Path(folder),active_job=None,pending_jobs=[]), \
             patch.object(server.models,'is_installed',return_value=False),patch.object(server,'threading',SimpleNamespace(Thread=Mock())):
            response=self.client.post('/api/video/generate',json={'prompt':'A beach','output_dir':folder},headers=self.headers)
            self.assertEqual(response.status_code,202,response.text)

    def test_video_history_and_download_are_separate_from_images(self):
        job_id='b'*32;directory=JOBS_DIR/job_id;directory.mkdir(exist_ok=True)
        output=Path(temporary.name)/'finished.mp4';output.write_bytes(b'test-video-container')
        write_json(directory/'status.json',{'id':job_id,'kind':'video','status':'succeeded','created_at':2,'output_path':str(output)})
        try:
            self.assertNotIn(job_id,[j['id'] for j in self.client.get('/api/history').json()])
            self.assertIn(job_id,[j['id'] for j in self.client.get('/api/history?kind=video').json()])
            response=self.client.get(f'/api/jobs/{job_id}/video?download=true')
            self.assertEqual(response.content,b'test-video-container')
            self.assertIn('attachment',response.headers['content-disposition'])
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/image').status_code,404)
            self.assertEqual(self.client.post(f'/api/jobs/{job_id}/reuse',headers=self.headers).status_code,404)
        finally:
            (directory/'status.json').unlink();directory.rmdir();output.unlink()

    def test_original_png_download_and_metadata(self):
        job_id='a'*32
        directory=JOBS_DIR/job_id
        directory.mkdir(exist_ok=True)
        output=Path(temporary.name)/'finished.png'
        output.write_bytes(png())
        job={'id':job_id,'status':'succeeded','output_path':str(output),'seed':42,'created_at':1}
        write_json(directory/'status.json',job)
        downloaded=self.client.get(f'/api/jobs/{job_id}/image?download=true')
        self.assertEqual(downloaded.content,png())
        self.assertIn('attachment',downloaded.headers['content-disposition'])
        self.assertEqual(self.client.get(f'/api/jobs/{job_id}/metadata').json()['seed'],42)
        preview = self.client.get(f'/api/jobs/{job_id}/thumbnail')
        self.assertEqual(preview.status_code, 200)
        with Image.open(io.BytesIO(preview.content)) as picture:
            self.assertEqual(picture.size, (32, 24))
            self.assertEqual(picture.getchannel('A').getextrema(), (100, 100))
        reused=self.client.post(f'/api/jobs/{job_id}/reuse',headers=self.headers).json()
        self.assertEqual(self.client.get(reused['url']).content,png())
        (directory/'status.json').unlink()
        (directory/'thumbnail.webp').unlink()
        directory.rmdir()

if __name__=='__main__':
    unittest.main(verbosity=2)

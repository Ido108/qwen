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

    def test_local_access_and_session_guard(self):
        self.assertEqual(self.client.post('/api/open-output').status_code, 403)
        self.assertEqual(self.client.get('/api/config', headers={'host':'evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/uploads', headers={**self.headers, 'Origin':'https://evil.example'}).status_code, 403)

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

    def test_path_traversal_cannot_read_files(self):
        for image_id in ('../secret.png', 'example:../../package.json', str(Path(temporary.name)/'secret.png')):
            with self.assertRaises(ValueError):
                resolve_image(image_id)

    def test_turbo_steps_are_fixed_and_tools_validated(self):
        from backend import server
        upload = self.client.post('/api/uploads', files=[('files', ('a.png', png(), 'image/png'))], headers=self.headers).json()[0]['id']
        with tempfile.TemporaryDirectory() as folder,              patch.object(server,'JOBS_DIR',Path(folder)),              patch.object(server,'active_job',None),              patch.object(server,'threading',SimpleNamespace(Thread=Mock())):
            def post(**values):
                server.active_job=None
                return self.client.post('/api/generate',json={'prompt':'An apple','enhance':False,'auto_resolution':1024,'output_dir':folder,**values},headers=self.headers)
            response=post(model_id='turbo-gguf-q8_0',steps=30)
            self.assertEqual(response.status_code,202,response.text)
            saved=json.loads((Path(folder)/response.json()['id']/'request.json').read_text())
            self.assertEqual((saved['steps'],saved['steps_mode']),(8,'model'))
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
        self.assertEqual(required_image({'model_id':'turbo-bf16','tool_id':'restore'}),['turbo-bf16','encoder-bf16','vae','lora-restore'])
        self.assertEqual(post_status:=self.client.post('/api/generate',json={'prompt':'x','lora_strength':3},headers=self.headers).status_code,422)

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
                response=self.client.post('/api/generate',json={'prompt':'An apple','enhance':False,'steps':steps,'auto_resolution':1024,'output_dir':folder},headers=self.headers)
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

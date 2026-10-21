"""LoRA training: ai-toolkit job configs, progress from the trainer log, registration of the result and the API."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QWEN_DATA_DIR', tempfile.mkdtemp(prefix='training-tests-'))
from fastapi.testclient import TestClient
from PIL import Image
from backend import civitai, models, training
from tests.test_krea import safetensors

REQUEST = {'name': 'Knit Elephant!', 'arch': 'z-image-turbo', 'subject': 'character', 'trigger_word': 'knitelph', 'steps': 300, 'rank': 16,
           'learning_rate': 1e-4, 'resolutions': [512, 768], 'save_every': 100, 'sample_prompts': ['knitelph on a chair'],
           'images': [{'id': 'a', 'caption': 'knitelph, a toy'}]}


class ConfigTests(unittest.TestCase):
    def test_each_architecture_uses_the_ai_toolkit_defaults(self):
        expected = {'flux3-klein-base': ('flux3','black-forest-labs/flux-3-image-klein-base','shift',None,['flux3-klein','flux3-klein-base']),
                    'qwen-image-2.1': ('qwen_image_2', 'Comfy-Org/Qwen-Image-2.1', 'shift', None, ['qwen-image-2.1']),
                    'qwen-image-2.1-turbo': ('qwen_image_2', 'Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors',
                                             'shift', 'ostris/qwen_image_2_turbo_training_adapter', ['qwen-image-2.1-turbo']),
                    'krea-2': ('krea2', 'krea/Krea-2-Raw', 'linear', None, ['krea-2']),
                    'krea-2-turbo': ('krea2', 'krea/Krea-2-Turbo', 'linear', 'ostris/krea2_turbo_training_adapter', ['krea-2-turbo']),
                    'z-image-turbo': ('zimage', 'Tongyi-MAI/Z-Image-Turbo', 'weighted', 'ostris/zimage_turbo_training_adapter', ['z-image-turbo']),
                    'z-image': ('zimage', 'Tongyi-MAI/Z-Image', 'weighted', None, ['z-image'])}
        self.assertEqual(set(training.ARCHS), set(expected))
        for key, (arch, path, timestep, adapter, families) in expected.items():
            config = training.build_config({**REQUEST, 'arch': key}, Path('data'), Path('out'))
            process = config['config']['process'][0]
            selected_path=process['model']['name_or_path']
            if arch=='flux3':
                self.assertEqual(Path(selected_path),Path(process['model']['model_kwargs']['model_path']).parent)
                selected_path=path
            self.assertEqual((process['model']['arch'], selected_path, process['train']['timestep_type']), (arch, path, timestep), key)
            self.assertEqual(process['model'].get('assistant_lora_path', '').startswith(adapter) if adapter else 'assistant_lora_path' not in process['model'], True, key)
            self.assertEqual(training.ARCHS[key]['families'], families)
            # 32 GB GPU: Z-Image trains unquantized; only Krea 2 keeps CPU-first loading; captions are cached once except on FLUX.3.
            self.assertEqual(process['model']['quantize'], arch!='zimage', key)
            self.assertEqual(process['model']['low_vram'], arch=='krea2', key)
            # FLUX.3's own runner reuses the prompt; cached embeddings measured twice as slow there.
            self.assertEqual(process['train']['cache_text_embeddings'], arch!='flux3', key)

    def test_request_values_reach_the_job(self):
        config = training.build_config(REQUEST, Path('data'), Path('out'))
        process = config['config']['process'][0]
        self.assertEqual(config['config']['name'], 'Knit_Elephant')
        self.assertEqual((process['network']['linear'], process['network']['linear_alpha'], process['train']['steps'], process['train']['lr']), (16, 16, 300, 1e-4))
        self.assertEqual((process['trigger_word'], process['datasets'][0]['resolution'], process['save']['save_every']), ('knitelph', [512, 768], 100))
        self.assertEqual(process['sample']['samples'], [{'prompt': 'knitelph on a chair'}])
        self.assertEqual((process['sample']['guidance_scale'], process['sample']['sample_steps']), (1.0, 9))  # Turbo samples at CFG 1
        self.assertFalse(process['train']['disable_sampling'])
        self.assertTrue(training.build_config({**REQUEST, 'sample_prompts': []}, Path('d'), Path('o'))['config']['process'][0]['train']['disable_sampling'])


class ProgressTests(unittest.TestCase):
    def test_reads_step_loss_and_time_left(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'worker.log'
            log.write_bytes(('Running 1 job\nWARNING:torchao.kernel.intmm:Warning: Detected no triton\n'
                             'Knit_Elephant:  40%|####      | 120/300 [01:10<01:45,  1.71it/s, lr: 1.0e-04 loss: 3.351e-01]\r'
                             'Knit_Elephant:  41%|####      | 123/300 [01:12<01:43,  1.71it/s, lr: 1.0e-04 loss: 3.783e-01]\r'
                             '- 0.0000s avg - log_to_tensorboard, num = 10\n').encode())
            found = training.read_progress(log, 300)
            self.assertEqual((found['step'], found['loss'], found['eta_seconds']), (123, 0.3783, 103))
            self.assertEqual(found['trainer_message'], 'Running 1 job')  # timing tables and warnings are skipped
            # A progress bar of another length (a download, the samples) is not the training step.
            log.write_text('Generating Samples:  50%|#####     | 1/2 [00:05<00:05,  5.87s/it]\r', encoding='utf-8')
            self.assertNotIn('step', training.read_progress(log, 300))

    def test_samples_and_final_file(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            samples = out / 'Knit_Elephant' / 'samples'
            samples.mkdir(parents=True)
            for name in ('1792244344575__000000000_0.jpg', '1792244534126__000000100_0.jpg'):
                (samples / name).write_bytes(b'jpg')
            self.assertEqual(len(training.sample_files(out, 'Knit Elephant!')), 2)
            self.assertIsNone(training.final_lora(out, 'Knit Elephant!'))
            (out / 'Knit_Elephant' / 'Knit_Elephant.safetensors').write_bytes(b'x')
            self.assertEqual(training.final_lora(out, 'Knit Elephant!').name, 'Knit_Elephant.safetensors')


class FusedMlpTests(unittest.TestCase):
    def test_a_lora_for_the_fused_qwen_mlp_becomes_two_exact_halves(self):
        import torch
        from backend.comfy_engine import split_fused_mlp
        a, b = torch.randn(16, 3072), torch.randn(2 * 12288, 16)
        weights = {'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight': a,
                   'diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight': b,
                   'lora_unet_transformer_blocks_1_img_mlp_gate_up.lora_down.weight': a,
                   'lora_unet_transformer_blocks_1_img_mlp_gate_up.lora_up.weight': b,
                   'lora_unet_transformer_blocks_1_img_mlp_gate_up.alpha': torch.tensor(16.0),
                   'diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight': a}
        out = split_fused_mlp(weights)
        prefix = 'diffusion_model.transformer_blocks.0.img_mlp.'
        # gate_layer is the first half of gate_up, proj the second (comfy/lora.py maps them the same way).
        full = b @ a
        self.assertTrue(torch.equal(out[prefix + 'gate_layer.lora_B.weight'] @ out[prefix + 'gate_layer.lora_A.weight'], full[:12288]))
        self.assertTrue(torch.equal(out[prefix + 'proj.lora_B.weight'] @ out[prefix + 'proj.lora_A.weight'], full[12288:]))
        self.assertEqual(out['lora_unet_transformer_blocks_1_img_mlp_proj.alpha'], 16.0)
        self.assertEqual(out['lora_unet_transformer_blocks_1_img_mlp_gate_layer.lora_up.weight'].shape[0], 12288)
        self.assertFalse(any('gate_up' in key for key in out))
        self.assertIn('diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight', out)
        # A LoRA without the fused layer is returned as it is.
        plain = {'diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight': a}
        self.assertIs(split_fused_mlp(plain), plain)


class RegisterTests(unittest.TestCase):
    def test_a_trained_lora_joins_the_lora_list_of_its_type(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'trained.safetensors'
            safetensors(source, {'diffusion_model.layers.0.attention.to_q.lora_A.weight': [16, 3840]})
            with patch.multiple(civitai, USER_CATALOG=root / 'user.json'), patch.object(models, 'INSTALL_PATH', root / 'installed.json'), \
                    patch.object(training, 'TRAINED_DIR', root / 'trained'), patch.object(models, 'MODELS_DIR', root):
                entry = training.register('6f2b791379e145daa9058dde6fc6e66b', REQUEST, source)
                self.assertEqual((entry['id'], entry['kind'], entry['source'], entry['families'], entry['trained_words']),
                                 ('trained-lora-6f2b791379e1', 'image-lora', 'trained', ['z-image-turbo'], ['knitelph']))
                self.assertTrue((root / 'trained' / entry['files'][0]['path']).is_file())
                self.assertTrue(models.is_installed(entry['id']))
                self.assertEqual(models.required('zimage-turbo-bf16', loras=[entry['id']])[-1], entry['id'])
                with self.assertRaisesRegex(ValueError, 'does not work'):
                    models.required('zimage-base-bf16', loras=[entry['id']])
                # A file of another architecture is refused, not added.
                safetensors(source, {'transformer_blocks.0.attn.to_q.lora_A.default.weight': [16, 4096]})
                with self.assertRaisesRegex(RuntimeError, 'not a Z-Image Turbo LoRA'):
                    training.register('0' * 32, REQUEST, source)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.server import app
        cls.client = TestClient(app)
        cls.headers = {'X-Studio-Token': cls.client.get('/api/config').json()['token']}

    def test_state_and_refusals(self):
        state = self.client.get('/api/training').json()
        self.assertEqual([a['id'] for a in state['archs']], list(training.ARCHS))
        self.assertTrue(next(a for a in state['archs'] if a['id'] == 'krea-2')['gate'].startswith('https://huggingface.co/krea/'))
        body = {**REQUEST, 'images': [{'id': 'a' * 32 + '.png', 'caption': ''}]}
        with patch.object(training, 'setup_status', return_value={'ready': False, 'status': 'idle'}):
            self.assertEqual(self.client.post('/api/training/start', json=body, headers=self.headers).status_code, 409)
        self.assertEqual(self.client.post('/api/training/start', json={**body, 'arch': 'sdxl'}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post('/api/training/start', json={**body, 'images': []}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post('/api/training/captions', json={'images': ['a']}, headers=self.headers).status_code, 400)

    def test_start_queues_a_training_job(self):
        from backend import server
        stream = io.BytesIO()
        Image.new('RGB', (64, 64), 'blue').save(stream, 'PNG')
        upload = self.client.post('/api/uploads', files={'files': ('a.png', stream.getvalue(), 'image/png')}, headers=self.headers).json()[0]
        # Its own jobs folder: other test modules leave partial jobs in the shared one.
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'JOBS_DIR', Path(folder)),                 patch.object(training, 'setup_status', return_value={'ready': True, 'status': 'ready'}), patch.object(server, 'start_next_locked'),                 patch.object(server, 'save_queue_locked'):
            job = self.client.post('/api/training/start', json={**REQUEST, 'images': [{'id': upload['id'], 'caption': 'x'}]}, headers=self.headers).json()
            self.assertEqual((job['kind'], job['status'], job['total_steps']), ('training', 'queued', 300))
            self.assertNotIn('thumbnail_url', job)
            listed = self.client.get('/api/training').json()['jobs']
            self.assertEqual(listed[0]['id'], job['id'])
            # Trainings are not images: the gallery leaves them out.
            self.assertNotIn(job['id'], [j['id'] for j in self.client.get('/api/library').json()['items']])
            self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/cancel", headers=self.headers).json(), {'ok': True})
            with server.active_lock:
                if job['id'] in server.pending_jobs:
                    server.pending_jobs.remove(job['id'])

    def test_zip_upload_and_caption_queue_contract_without_gpu(self):
        from backend import server, training_data
        from tests.test_training_dataset import picture, archive
        raw=archive([('photos/a.png',picture()),('photos/a.txt','Existing caption')])
        with tempfile.TemporaryDirectory() as folder, patch.object(training_data,'ROOT',Path(folder)/'imports'), patch.object(server,'JOBS_DIR',Path(folder)/'jobs'), patch.object(server,'pending_jobs',[]), patch.object(server,'start_next_locked'), patch.object(server,'save_queue_locked'):
            server.JOBS_DIR.mkdir()
            response=self.client.post('/api/training/imports',json={'files':[{'name':'photos.zip','size':len(raw)}]},headers=self.headers)
            self.assertEqual(response.status_code,200,response.text);ident=response.json()['id']
            sent=self.client.put(f'/api/training/imports/{ident}/files/0?offset=0',content=raw,headers={**self.headers,'Content-Type':'application/octet-stream'})
            self.assertEqual(sent.json()['offset'],len(raw))
            result=self.client.post(f'/api/training/imports/{ident}/finish',headers=self.headers)
            self.assertEqual(result.status_code,200,result.text);item=result.json()['items'][0]
            self.assertEqual(item['caption'],'Existing caption')
            self.assertFalse(training_data.directory(ident).exists())
            result=self.client.post('/api/training/captions',json={'images':[item['id'],item['id']],'name':'Cup','trigger_word':'cupx','subject':'style'},headers=self.headers)
            self.assertEqual(result.status_code,202,result.text)
            job=result.json();self.assertEqual((job['kind'],job['total_steps']),('captioning',1))
            self.assertNotIn('thumbnail_url',job)
            self.assertEqual(self.client.get('/api/training/captions/'+job['id']).json()['captions'],{})

    def test_recommended_training_uses_actual_image_sizes(self):
        from backend import server
        from tests.test_training_dataset import picture
        item=self.client.post('/api/uploads',files={'files':('tiny.png',picture(),'image/png')},headers=self.headers).json()[0]
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'JOBS_DIR',Path(folder)),patch.object(server,'pending_jobs',[]),patch.object(server,'start_next_locked'),patch.object(server,'save_queue_locked'),patch.object(training,'setup_status',return_value={'ready':True}):
            response=self.client.post('/api/training/start',json={**REQUEST,'settings_mode':'recommended','steps':19999,'images':[{'id':item['id'],'caption':'User text'}]},headers=self.headers)
            self.assertEqual(response.status_code,202,response.text)
            r=response.json()['request']
            self.assertEqual(r['steps'],500)
            self.assertEqual(r['resolutions'],[512])
            self.assertEqual(r['images'][0]['caption'],'User text')
            self.assertTrue(r['profile_version'])


if __name__ == '__main__':
    unittest.main()

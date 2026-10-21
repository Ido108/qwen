"""Image upscale routes and geometry. No weights, CUDA work or downloads."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

temporary = tempfile.TemporaryDirectory(prefix='image-upscale-tests-')
os.environ['QWEN_DATA_DIR'] = temporary.name
from fastapi.testclient import TestClient
from PIL import Image
from backend import image_upscale, server
from backend.common import UPLOADS_DIR, write_json
from backend.image_upscale import VAE_TILES, ImageUpscaleRequest, output_dimensions, required
from backend.server import app


def upload(size=(600, 400), color=(20, 120, 200, 255)):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = f'{size[0]}x{size[1]}-{color[3]}.png'
    Image.new('RGBA', size, color).save(UPLOADS_DIR / name)
    return name


class ImageUpscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.headers = {'X-Studio-Token': cls.client.get('/api/config').json()['token']}

    def test_geometry_scales_both_sides_and_refuses_too_large_or_smaller_results(self):
        p = ImageUpscaleRequest(source_image='a.png').model_dump()
        self.assertEqual(output_dimensions(1024, 768, p), (2048, 1536))
        self.assertEqual(output_dimensions(1024, 768, {**p, 'scale': 4}), (4096, 3072))
        self.assertEqual(output_dimensions(1536, 2688, {**p, 'size_mode': 'long_side', 'long_side': 4096}), (2341, 4096))
        with self.assertRaisesRegex(ValueError, 'smaller size'):
            output_dimensions(2688, 1536, {**p, 'scale': 4})
        with self.assertRaisesRegex(ValueError, 'larger than the source'):
            output_dimensions(2048, 1024, {**p, 'size_mode': 'long_side', 'long_side': 2048})
        with self.assertRaisesRegex(ValueError, '8 times'):
            output_dimensions(300, 200, {**p, 'size_mode': 'long_side', 'long_side': 4096})

    def test_only_seedvr2_models_and_their_vae(self):
        self.assertEqual(required({'model_id': 'seedvr2-3b-fp16'}), ['seedvr2-3b-fp16', 'seedvr2-vae'])
        self.assertIn('seedvr2-vae', required({'model_id': 'seedvr2-7b-sharp-fp16'}))
        for model_id in ('flashvsr11-int8', 'swiftvr-bf16', 'uc-int8-convrot'):
            with self.subTest(model=model_id), self.assertRaises(ValueError):
                required({'model_id': model_id})

    def test_memory_options_keep_3b_fp16_on_the_gpu_and_swap_only_when_short(self):
        from backend.models import entry
        from backend.upscale_settings import seed_memory_options
        request = {'resolution': 4096, 'batch_size': 1}
        self.assertEqual(seed_memory_options(entry('seedvr2-3b-fp16'), request, 29.9), [])
        self.assertIn('--blocks_to_swap', seed_memory_options(entry('seedvr2-3b-fp16'), {**request, 'memory_mode': 'low_memory'}, 29.9))
        self.assertIn('--vae_decode_tiled', VAE_TILES)

    def test_plan_route_reports_source_and_result_size(self):
        name = upload()
        result = self.client.get(f'/api/image-upscale/plan?source_image={name}&scale=4', headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {'source_width': 600, 'source_height': 400, 'width': 2400, 'height': 1600})
        long = self.client.get(f'/api/image-upscale/plan?source_image={name}&size_mode=long_side&long_side=3000', headers=self.headers).json()
        self.assertEqual((long['width'], long['height']), (3000, 2000))
        # A size that is too large still reports the source size, with the reason.
        large = self.client.get(f'/api/image-upscale/plan?source_image={name}&size_mode=long_side&long_side=8000', headers=self.headers).json()
        self.assertEqual((large['source_width'], large['width']), (600, None))
        self.assertIn('8 times', large['error'])
        self.assertEqual(self.client.get('/api/image-upscale/plan?source_image=missing.png', headers=self.headers).status_code, 400)
        self.assertEqual(self.client.get('/api/image-upscale/plan', headers=self.headers).status_code, 400)

    def test_generate_queues_a_local_image_job_from_an_input_or_a_gallery_result(self):
        name = upload((512, 512))
        captured = []
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(server, 'enqueue_locked', side_effect=lambda job, data: captured.append((job, data))):
            job = self.client.post('/api/image-upscale/generate', headers=self.headers,
                                   json={'source_image': name, 'scale': 2, 'output_dir': folder}).json()
            self.assertEqual((job['kind'], job['width'], job['height'], job['source_width']), ('image-upscale', 1024, 1024, 512))
            data = captured[-1][1]
            self.assertEqual((data['execution'], data['images'], data['model_id']), ('local', [name], 'seedvr2-3b-fp16'))

            # A finished gallery image: the route copies it to the uploads and records it as the input.
            jobs = Path(folder) / 'jobs'
            source_id = 'a' * 32
            output = Path(folder) / 'result.png'
            Image.new('RGB', (300, 200), (200, 10, 10)).save(output)
            (jobs / source_id).mkdir(parents=True)
            write_json(jobs / source_id / 'status.json', {'id': source_id, 'kind': 'image', 'status': 'succeeded',
                                                         'created_at': time.time(), 'output_path': str(output), 'request': {}})
            with patch.object(image_upscale, 'JOBS_DIR', jobs):
                result = self.client.post('/api/image-upscale/generate', headers=self.headers,
                                          json={'source_job': source_id, 'size_mode': 'long_side', 'long_side': 1200, 'output_dir': folder})
                self.assertEqual(result.status_code, 202, result.text)
                self.assertEqual((result.json()['width'], result.json()['height']), (1200, 800))
                copied = captured[-1][1]['source_image']
                self.assertTrue((UPLOADS_DIR / copied).is_file())
                with Image.open(UPLOADS_DIR / copied) as picture:
                    self.assertEqual(picture.size, (300, 200))
                # A video or an unfinished job is not an image source.
                write_json(jobs / source_id / 'status.json', {'id': source_id, 'kind': 'upscale', 'status': 'succeeded',
                                                             'created_at': time.time(), 'output_path': str(output), 'request': {}})
                self.assertEqual(self.client.post('/api/image-upscale/generate', headers=self.headers,
                                                  json={'source_job': source_id}).status_code, 400)
            refused = self.client.post('/api/image-upscale/generate', headers=self.headers,
                                       json={'source_image': name, 'model_id': 'flashvsr11-int8'})
            self.assertEqual(refused.status_code, 400)
            self.assertEqual(self.client.post('/api/image-upscale/generate', headers=self.headers,
                                              json={'source_image': name, 'scale': 3}).status_code, 422)

    def test_upscaled_images_are_image_results_in_history_library_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'JOBS_DIR', Path(folder)):
            job_id = 'b' * 32
            output = Path(folder) / 'up.png'
            Image.new('RGB', (64, 48), (0, 200, 0)).save(output)
            (Path(folder) / job_id).mkdir()
            write_json(Path(folder) / job_id / 'status.json', {
                'id': job_id, 'kind': 'image-upscale', 'status': 'succeeded', 'created_at': time.time(), 'output_path': str(output),
                'width': 64, 'height': 48, 'source_width': 32, 'source_height': 24,
                'request': {'model_id': 'seedvr2-3b-fp16', 'prompt': '', 'images': []}})
            self.assertEqual([j['id'] for j in self.client.get('/api/history', headers=self.headers).json()], [job_id])
            self.assertEqual(self.client.get('/api/library?kind=image', headers=self.headers).json()['total'], 1)
            self.assertEqual(self.client.get('/api/library?kind=video', headers=self.headers).json()['total'], 0)
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/image', headers=self.headers).status_code, 200)
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/display', headers=self.headers).status_code, 200)
            reused = self.client.post(f'/api/jobs/{job_id}/reuse', headers=self.headers).json()
            self.assertEqual((reused['width'], reused['height']), (64, 48))

    def test_run_again_queues_the_same_upscale(self):
        name = upload((400, 300))
        captured = []
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'JOBS_DIR', Path(folder)),                 patch.object(server, 'enqueue_locked', side_effect=lambda job, data: captured.append((job, data))):
            job_id = 'c' * 32
            (Path(folder) / job_id).mkdir()
            request = {**ImageUpscaleRequest(source_image=name, scale=4, seed=9).model_dump(), 'kind': 'image-upscale',
                       'execution': 'local', 'images': [name], 'prompt': ''}
            write_json(Path(folder) / job_id / 'status.json', {'id': job_id, 'kind': 'image-upscale', 'status': 'succeeded',
                                                               'seed': 9, 'created_at': time.time(), 'request': request})
            result = self.client.post(f'/api/jobs/{job_id}/retry', headers=self.headers)
            self.assertEqual(result.status_code, 202, result.text)
            job, data = captured[-1]
            self.assertEqual((job['kind'], job['width'], job['height'], data['seed'], data['scale']), ('image-upscale', 1600, 1200, 9, 4))

    def test_fast_gan_upscaler_is_offered_up_to_its_own_scale(self):
        self.assertEqual(required({'model_id': 'nomos2-hq-mosr'}), ['nomos2-hq-mosr'])
        p = {**ImageUpscaleRequest(source_image='a.png', model_id='nomos2-hq-mosr').model_dump()}
        self.assertEqual(output_dimensions(1024, 768, {**p, 'scale': 4}), (4096, 3072))
        with self.assertRaisesRegex(ValueError, 'up to 4 times'):
            output_dimensions(600, 400, {**p, 'size_mode': 'long_side', 'long_side': 3000})
        self.assertEqual(output_dimensions(600, 400, {**p, 'model_id': 'seedvr2-3b-fp16', 'size_mode': 'long_side', 'long_side': 3000}), (3000, 2000))

    def test_advanced_knobs_reach_the_seedvr2_command(self):
        from backend.image_upscale import seed_options
        from backend.models import entry
        request = {**ImageUpscaleRequest(source_image='a.png', input_noise_scale=0.2, latent_noise_scale=0.1,
                                         vae_tile_size=768, vae_tile_overlap=64).model_dump(), '_short': 2048}
        fast, safe = seed_options(request, entry('seedvr2-3b-fp16'), 29.9)
        joined = ' '.join(fast)
        for flag in ('--input_noise_scale 0.2', '--latent_noise_scale 0.1', '--vae_encode_tile_size 768', '--vae_decode_tile_overlap 64', '--vae_decode_tiled'):
            self.assertIn(flag, joined)
        self.assertNotIn('--blocks_to_swap', joined)
        self.assertIn('--blocks_to_swap', ' '.join(safe))
        saving, _ = seed_options({**request, 'memory_mode': 'low_memory'}, entry('seedvr2-3b-fp16'), 29.9)
        self.assertEqual(saving, safe)
        with self.assertRaises(ValueError):
            ImageUpscaleRequest(source_image='a.png', vae_tile_size=100)

    def test_progress_reports_stage_fraction_and_estimate(self):
        from backend.image_upscale import Progress, seed_estimates, tile_starts
        writes = []
        progress = Progress(lambda **values: writes.append(values), seed_estimates(4.2))
        self.assertIn('upscaling', writes[0]['stage_estimates'])
        progress.stage('loading_model')
        progress.seed_events(stage='upscaling', message='Upscaling the image · 1 of 1', step=1, total_steps=1)
        last = writes[-1]
        self.assertEqual((last['stage'], last['step'], last['total_steps'], last['message']), ('upscaling', 3, 5, 'Upscaling the image'))
        self.assertLess(last['progress'], last['progress_stage_end'])
        self.assertGreater(last['progress_estimate_seconds'], 0)
        self.assertEqual(tile_starts(100, 512, 32), [0])
        starts = tile_starts(1000, 512, 32)
        self.assertEqual((starts[0], starts[-1]), (0, 488))
        self.assertTrue(all(b - a <= 480 for a, b in zip(starts, starts[1:])))


if __name__ == '__main__':
    unittest.main()

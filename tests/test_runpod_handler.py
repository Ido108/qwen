"""The RunPod worker runs the same engines; check that every model family reaches them with the right files."""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QWEN_DATA_DIR', tempfile.mkdtemp(prefix='runpod-tests-'))
from PIL import Image

from backend.common import JOBS_DIR, UPLOADS_DIR, read_json, write_json
from backend.ltx_settings import media_inputs


def load_handler():
    sys.modules.setdefault('fcntl', types.SimpleNamespace(flock=lambda *a: None, LOCK_EX=2, LOCK_UN=8))
    sys.modules['runpod'] = types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None, start=lambda *a: None))
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location('runpod_handler', Path(root) / 'runpod' / 'handler.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    """Stands in for backend.worker: records the request and writes a finished status and output."""
    def __init__(self, directory, suffix):
        request = read_json(directory / 'request.json')
        output = Path(request['output_dir']) / f'out{suffix}'
        output.write_bytes(b'x')
        status = read_json(directory / 'status.json')
        write_json(directory / 'status.json', {**status, 'status': 'succeeded', 'stage': 'complete', 'output_path': str(output)})
        self.request = request

    def poll(self):
        return 0

    def wait(self):
        return 0


class RunpodHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = load_handler()
        self.volume = Path(tempfile.mkdtemp(prefix='volume-'))
        self.handler.VOLUME = self.volume
        self.needed = []
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    def run_job(self, kind, request, inputs):
        """inputs: (media, name) in the order the desktop app uploads them (backend/server.py run_cloud_job)."""
        job_id = os.urandom(16).hex()
        prefix = f'qwen-transfer/{job_id}/'
        keys = []
        for index, (media, name) in enumerate(inputs):
            source = self.volume / prefix / f'input-{index}{Path(name).suffix}'
            source.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (64, 64), 'red').save(source) if media == 'image' else source.write_bytes(b'media')
            keys.append(prefix + source.name)
        started = []

        def popen(command, **kwargs):
            started.append(command)
            return FakeProcess(Path(command[-1]), '.mp4')

        def image_start(directory, req):
            started.append(['image-worker'])
            return FakeProcess(directory, '.png')

        with patch.object(self.handler.models, 'ensure_installed', side_effect=lambda ids, **k: self.needed.extend(ids)), \
             patch.object(self.handler.models, 'is_installed', return_value=True), \
             patch.object(self.handler.subprocess, 'Popen', side_effect=popen), \
             patch.object(self.handler.IMAGE_WORKER, 'start', side_effect=image_start), \
             patch.object(self.handler.IMAGE_WORKER, 'wait', return_value=0), \
             patch.object(self.handler.IMAGE_WORKER, 'close'):
            result = self.handler.handler({'input': {'job_id': job_id, 'kind': kind, 'output_prefix': prefix, 'request': request, 'input_keys': keys}})
        saved = read_json(JOBS_DIR / job_id / 'request.json')
        return result, saved, started

    def test_krea_turbo_with_a_style_image(self):
        image = 'a' * 32 + '.png'
        request = {'model_id': 'krea2-turbo-fp8', 'prompt': 'a lighthouse', 'images': [image], 'seed': 1, 'enhance': False,
                   'text_encoder': 'auto', 'vae_id': 'auto', 'loras': [], 'tool_id': None, 'steps': 8, 'auto_resolution': 1024}
        result, saved, started = self.run_job('image', request, [('image', image)])
        self.assertEqual(started, [['image-worker']])
        self.assertTrue(result['output_key'].endswith('result.png'))
        for needed in ('krea2-turbo-fp8', 'krea2-encoder-bf16', 'krea2-vae', 'krea2-lora-style-reference'):
            self.assertIn(needed, self.needed)
        self.assertNotIn('encoder-bf16', self.needed)

    def test_zimage_turbo_redraws_an_input_image(self):
        image = 'd' * 32 + '.png'
        request = {'model_id': 'zimage-turbo-bf16', 'prompt': 'a lighthouse', 'images': [image], 'seed': 1, 'enhance': False, 'denoise': 0.7,
                   'text_encoder': 'auto', 'vae_id': 'auto', 'loras': [], 'tool_id': None, 'steps': 8, 'auto_resolution': 1024}
        result, saved, started = self.run_job('image', request, [('image', image)])
        self.assertEqual(started, [['image-worker']])
        self.assertEqual(saved['denoise'], 0.7)
        for needed in ('zimage-turbo-bf16', 'zimage-encoder-bf16', 'zimage-vae'):
            self.assertIn(needed, self.needed)
        self.assertNotIn('encoder-bf16', self.needed)

    def test_ltx_text_and_first_last_frame_in_create_video(self):
        base = {'model_id': 'ltx25-msr-q4_k_m', 'prompt': 'a kite', 'seed': 2, 'enhance': False, 'quality': 'standard', 'steps': 8,
                'audio': True, 'kind': 'video', 'loras': [], 'randomize_seed': False}
        result, saved, started = self.run_job('video', {**base, 'mode': 'text', 'width': 768, 'height': 448, 'frames': 73,
                                                      'first_frame': None, 'last_frame': None}, [])
        self.assertEqual(saved['kind'], 'video')
        self.assertIn('backend.worker', started[0])
        self.assertIn('ltx25-spatial-upscaler', self.needed)
        self.assertNotIn('ltx25-msr-v2', self.needed)
        self.needed.clear()
        first, last = 'b' * 32 + '.png', 'c' * 32 + '.png'
        result, saved, _ = self.run_job('video', {**base, 'mode': 'image', 'width': 768, 'height': 448, 'frames': 73,
                                                'first_frame': first, 'last_frame': last}, [('image', first), ('image', last)])
        self.assertTrue(result['output_key'].endswith('result.mp4'))
        self.assertTrue((UPLOADS_DIR / first).is_file() and (UPLOADS_DIR / last).is_file())
        self.assertNotIn('ltx25-msr-v2', self.needed)

    def test_ref2va_with_images_only_in_reference_images(self):
        refs = ['d' * 32 + '.png', 'e' * 32 + '.png']
        request = {'model_id': 'h3-ref-turbo-int8', 'prompt': 'Image 1 talks to Image 2', 'reference_images': refs, 'background': None,
                   'audio_references': [], 'width': 768, 'height': 448, 'frames': 73, 'fps': 24, 'seed': 3, 'enhance': False,
                   'refine': False, 'quality': 'turbo', 'steps': None, 'audio': True, 'kind': 'multi-reference', 'loras': [],
                   'reference_strength': 1.0, 'reference_frames': 25, 'lora_strength': 1.0}
        result, saved, started = self.run_job('multi-reference', request, media_inputs(request))
        self.assertEqual(saved['kind'], 'multi-reference')
        self.assertIn('h3-ref-turbo-int8', self.needed)
        self.assertFalse(any(item.startswith('ltx25') for item in self.needed))
        self.assertTrue(result['output_key'].endswith('result.mp4'))

    def test_ltx_reference_images_keep_the_msr_lora(self):
        refs = ['f' * 32 + '.png']
        request = {'model_id': 'ltx25-msr-q4_k_m', 'prompt': 'Image 1 waves', 'reference_images': refs, 'background': None,
                   'audio_references': [], 'width': 768, 'height': 448, 'frames': 121, 'fps': 24, 'seed': 4, 'enhance': False,
                   'refine': False, 'audio': True, 'kind': 'multi-reference', 'loras': [], 'reference_strength': 1.0,
                   'reference_frames': 25, 'lora_strength': 1.0}
        self.run_job('multi-reference', request, media_inputs(request))
        self.assertIn('ltx25-msr-v2', self.needed)


if __name__ == '__main__':
    unittest.main()

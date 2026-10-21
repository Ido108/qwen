import json
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from backend import int8_copies, models, resident_worker
from tests.test_int8_convert import tiny_qwen

Usage = namedtuple('Usage', 'total used free')
GB = 1024**3


class CopyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library, self.data_models = self.root / 'library', self.root / 'data' / 'models'
        self.library.mkdir()
        self.source = self.root / 'civitai' / 'finalcut_fp32.safetensors'
        self.source.parent.mkdir()
        save_file(tiny_qwen(dtype=torch.float32), str(self.source), metadata={'format': 'pt'})
        self.item = {'id': 'civitai-model-1', 'kind': 'image', 'family': 'qwen-image-2.1', 'format': 'FP32', 'layout': 'comfy',
                     'title': 'FinalCut · fp32', 'directory': 'civitai', 'files': [{'path': self.source.name, 'bytes': self.source.stat().st_size}]}
        self.patches = [patch.object(int8_copies, 'REGISTRY', self.root / 'data' / 'int8-copies.json'),
                        patch.object(int8_copies, 'LIBRARY_DIR', self.library), patch.object(int8_copies, 'MODELS_DIR', self.data_models),
                        patch.object(models, 'local_path', lambda item, file: self.source if file['path'] == self.source.name else Path(file['path'])),
                        patch.object(int8_copies, '_running', {}), patch.object(int8_copies, '_plans', {})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def make_copy(self):
        target = self.data_models / 'int8-local' / 'finalcut_fp32_int8_convrot.safetensors'
        from backend import int8_convert
        result = int8_convert.convert([self.source], target, device='cpu')
        int8_copies.save({'copies': {self.item['id']: {'path': str(target), 'bytes': result['bytes'],
                                                         'source': int8_copies.signature([self.source])}}, 'original': [], 'failed': {}})
        return target

    def test_float_qwen_models_are_candidates_and_quantized_ones_are_not(self):
        self.assertIsNotNone(int8_copies.candidate(self.item))
        for other in ({**self.item, 'family': 'krea-2'}, {**self.item, 'format': 'GGUF'}, {**self.item, 'kind': 'image-lora'}):
            self.assertIsNone(int8_copies.candidate(other))
        fp8 = self.root / 'fp8.safetensors'
        sd = tiny_qwen()
        sd['transformer_blocks.0.attn.to_q.weight'] = sd['transformer_blocks.0.attn.to_q.weight'].to(torch.float8_e4m3fn)
        save_file(sd, str(fp8))
        with patch.object(models, 'local_path', lambda item, file: fp8):
            self.assertIsNone(int8_copies.candidate({**self.item, 'id': 'fp8', 'format': 'FP8'}))

    def test_the_copy_is_used_automatically_until_the_user_chooses_the_original(self):
        self.assertIsNone(int8_copies.active(self.item))
        self.assertEqual(int8_copies.info(self.item)['status'], 'waiting')
        target = self.make_copy()
        self.assertEqual(int8_copies.active(self.item), target)
        state = int8_copies.info(self.item)
        self.assertEqual((state['status'], state['label']), ('ready', 'INT8 (made on this computer)'))
        self.assertEqual(int8_copies.title(self.item, self.item['title']), 'FinalCut · fp32 · runs as INT8 (made on this computer)')
        with patch.object(models, 'entry', lambda model_id: self.item):
            self.assertEqual(int8_copies.set_choice(self.item['id'], 'original')['status'], 'original')
            self.assertIsNone(int8_copies.active(self.item))
            self.assertTrue(self.source.is_file() and target.is_file())  # both files stay
            int8_copies.set_choice(self.item['id'], 'int8')
        self.assertEqual(int8_copies.active(self.item), target)

    def test_a_changed_source_file_invalidates_the_copy(self):
        self.make_copy()
        save_file(tiny_qwen(dtype=torch.bfloat16), str(self.source), metadata={'format': 'pt'})
        self.assertIsNone(int8_copies.active(self.item))

    def test_official_bf16_models_use_the_installed_official_int8_file(self):
        official = self.root / 'qwen_image_2.1_int8_convrot.safetensors'
        official.write_bytes(b'x' * 10)
        int8_item = {'id': 'int8-convrot', 'files': [{'path': str(official)}]}
        with patch.object(models, 'is_installed', lambda model_id: model_id == 'int8-convrot'), \
                patch.object(models, 'entry', lambda model_id: int8_item), patch.object(models, 'local_path', lambda item, file: Path(file['path'])):
            self.assertEqual(int8_copies.active({'id': 'official-bf16'}), official)
            self.assertEqual(int8_copies.info({'id': 'official-bf16'})['label'], 'INT8 (official file)')
        with patch.object(models, 'is_installed', lambda model_id: False):
            self.assertIsNone(int8_copies.active({'id': 'official-bf16'}))

    def test_copy_goes_to_the_library_drive_only_with_3_gb_left(self):
        size = 7 * GB
        with patch.object(int8_copies.shutil, 'disk_usage', lambda path: Usage(0, 0, size + 4 * GB)):
            self.assertEqual(int8_copies.target_path(self.item, [self.source], size)[0].parent, self.library / 'diffusion_models' / 'int8-local')
        free = lambda path: Usage(0, 0, (size + 2 * GB) if str(path).startswith(str(self.library)) else 600 * GB)
        with patch.object(int8_copies.shutil, 'disk_usage', free):
            self.assertEqual(int8_copies.target_path(self.item, [self.source], size)[0].parent, self.data_models / 'int8-local')
        with patch.object(int8_copies.shutil, 'disk_usage', lambda path: Usage(0, 0, GB)):
            target, reason = int8_copies.target_path(self.item, [self.source], size)
            self.assertIsNone(target)
            self.assertIn('3 GB', reason)

    def test_conversion_starts_only_when_the_queue_is_idle(self):
        started = []
        with patch.object(int8_copies, 'pending', lambda: [(self.item, ([self.source], 1))]), \
                patch.object(int8_copies, '_start', lambda item, plan: started.append(item['id']) or True), \
                patch.object(int8_copies, '_scanned', [0.0]):
            int8_copies.tick(lambda: True)
            self.assertEqual(started, [])
            int8_copies.tick(lambda: False)
            self.assertEqual(started, ['civitai-model-1'])

    def test_a_running_conversion_moves_to_the_cpu_when_a_job_starts_and_is_recorded_when_done(self):
        target = self.data_models / 'int8-local' / 'x_int8_convrot.safetensors'
        target.parent.mkdir(parents=True)
        process = MagicMock()
        process.poll.return_value = None
        run = {'process': process, 'target': target, 'status': str(target) + '.status.json', 'flag': str(target) + '.use-cpu',
               'source': int8_copies.signature([self.source]), 'started': 0, 'bytes': 10}
        int8_copies._running[self.item['id']] = run
        int8_copies.tick(lambda: True)
        self.assertTrue(Path(run['flag']).exists())
        target.write_bytes(b'\x02\x00\x00\x00\x00\x00\x00\x00{}')
        process.poll.return_value = 0
        process.returncode = 0
        int8_copies.tick(lambda: True)
        self.assertEqual(int8_copies.read()['copies'][self.item['id']]['path'], str(target))
        self.assertFalse(Path(run['flag']).exists())

    def test_removing_the_model_deletes_only_the_apps_copy(self):
        target = self.make_copy()
        int8_copies.remove(self.item['id'])
        self.assertFalse(target.exists())
        self.assertTrue(self.source.is_file())
        self.assertEqual(int8_copies.read()['copies'], {})


class WorkerTests(unittest.TestCase):
    def test_resident_worker_reloads_when_the_int8_choice_changes(self):
        worker = resident_worker.ImageWorker()
        process = MagicMock()
        process.poll.return_value = None
        worker.process, worker.key, worker.weights = process, ('m', 'encoder-int8', 'vae'), None
        closed = []
        with patch.object(resident_worker, 'required_image', lambda request: ['m', 'encoder-int8', 'vae']), \
                patch.object(worker, 'close', lambda: closed.append(True) or setattr(worker, 'process', None)), \
                patch.object(worker, '_spawn', lambda log, key: setattr(worker, 'process', process)), \
                patch('backend.int8_copies.active_for_id', lambda model_id: 'E:/copy_int8_convrot.safetensors'):
            worker.start(Path(tempfile.gettempdir()), {'model_id': 'm'})
        self.assertEqual(closed, [True])
        worker.weights = 'E:/copy_int8_convrot.safetensors'
        with patch.object(resident_worker, 'required_image', lambda request: ['m', 'encoder-int8', 'vae']), \
                patch.object(worker, 'close', lambda: closed.append(True)), \
                patch('backend.int8_copies.active_for_id', lambda model_id: 'E:/copy_int8_convrot.safetensors'):
            worker.start(Path(tempfile.gettempdir()), {'model_id': 'm'})
        self.assertEqual(closed, [True])  # same file: the loaded model is reused


class StateTests(unittest.TestCase):
    def test_model_list_title_says_how_the_model_runs(self):
        item = {'id': 'civitai-model-1', 'kind': 'image', 'family': 'qwen-image-2.1', 'format': 'FP32', 'title': 'FinalCut · fp32',
                'revision': '1', 'directory': 'civitai', 'files': [], 'dependencies': [], 'repo': 'civitai'}
        ready = {'status': 'ready', 'official': False, 'label': 'INT8 (made on this computer)', 'bytes': 1}
        with patch.object(models, 'catalog', lambda: [item]), patch.object(models, 'installed_manifest', lambda: {item['id']: {'verified': True, 'revision': '1'}}), \
                patch.object(int8_copies, 'info', lambda entry: ready), patch('backend.orbit.profile', lambda item, lora: None):
            row = models.state()['models'][0]
        self.assertEqual(row['title'], 'FinalCut · fp32 · runs as INT8 (made on this computer)')
        self.assertEqual(row['int8'], ready)


if __name__ == '__main__':
    unittest.main()

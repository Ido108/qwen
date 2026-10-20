import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from backend.image_inputs import ImageInputs
from scripts.cleanup_image_inputs import candidates


class ImageInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'
        Image.new('RGBA', (16, 16), (100, 30, 20, 120)).save(self.source)

    def test_repeated_image_decodes_once_and_callers_cannot_change_cache(self):
        cache = ImageInputs()
        with patch('backend.image_inputs.Image.open', wraps=Image.open) as opened:
            first, hit = cache.load(self.source)
            self.assertFalse(hit)
            first.putpixel((0, 0), (0, 0, 0, 0))
            second, hit = cache.load(self.source)
            self.assertTrue(hit)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(second.getpixel((0, 0)), (100, 30, 20, 120))

    def test_changed_or_deleted_source_is_not_reused(self):
        cache = ImageInputs()
        cache.load(self.source)
        old = self.source.stat()
        Image.new('RGBA', (16, 16), 'blue').save(self.source)
        os.utime(self.source, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
        image, hit = cache.load(self.source)
        self.assertFalse(hit)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255, 255))
        self.assertEqual(len(cache.entries), 1)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            cache.load(self.source)

    def test_memory_and_entry_limits_evict_least_recent_image(self):
        cache = ImageInputs(max_bytes=2 * 16 * 16 * 4, max_entries=2)
        paths = []
        for i in range(3):
            path = self.root / f'{i}.png'
            Image.new('RGB', (16, 16), (i, 0, 0)).save(path)
            paths.append(path)
        cache.load(paths[0]); cache.load(paths[1]); cache.load(paths[0]); cache.load(paths[2])
        self.assertTrue(cache.load(paths[0])[1])
        self.assertFalse(cache.load(paths[1])[1])
        self.assertLessEqual(cache.bytes, cache.max_bytes)
        self.assertLessEqual(len(cache.entries), 2)
        tiny = ImageInputs(max_bytes=1)
        self.assertFalse(tiny.load(self.source)[1])
        self.assertFalse(tiny.load(self.source)[1])
        self.assertEqual(tiny.bytes, 0)

    def test_tool_cache_ignores_prompt_seed_but_tracks_size_options_and_pixels(self):
        from backend.image_tools import prepare
        cache = ImageInputs()
        image, _ = cache.load(self.source)
        tool = {'id': 'outpaint', 'prepare': 'outpaint', 'reference_resolution': 0}
        request = {'custom_size': True, 'width': 64, 'height': 64,
                   'tool_options': {'direction': 'left', 'amount': '50'}, 'prompt': 'first', 'seed': 1}
        with patch('backend.image_tools.prepare', wraps=prepare) as prepare_call:
            first, hit = cache.prepare(tool, request, [image])
            self.assertFalse(hit)
            original = first[0][0].tobytes()
            first[0][0].paste((0, 0, 0, 0), (0, 0, *first[0][0].size))
            first[2][2].putpixel((0, 0), (0, 0, 0, 0))
            second, hit = cache.prepare(tool, {**request, 'prompt': 'second', 'seed': 2}, [image])
            self.assertTrue(hit)
            self.assertEqual(second[0][0].tobytes(), original)
            self.assertEqual(prepare_call.call_count, 1)
            self.assertFalse(cache.prepare(tool, {**request, 'width': 96}, [image])[1])
            self.assertFalse(cache.prepare(tool, {**request, 'tool_options': {'direction': 'right', 'amount': '50'}}, [image])[1])
            self.assertFalse(cache.prepare(tool, request, [Image.new('RGBA', image.size, 'blue')])[1])
        self.assertLessEqual(cache.bytes, cache.max_bytes)

    def test_worker_reuses_inputs_without_job_png_copies_and_keeps_result_metadata(self):
        from backend import worker, models, image_settings
        output = self.root / 'outputs'; output.mkdir()
        request = {'model_id': 'test', 'images': ['source.png'], 'tool_id': None, 'enhance': False,
                   'prompt': 'A red cup', 'negative_prompt': '', 'output_dir': str(output),
                   'custom_size': True, 'width': 32, 'height': 32, 'seed': 1}
        def render(_request, _prompt, images, width, height, progress):
            progress(stage='generating', step=1, total_steps=1)
            return Image.new('RGBA', (width, height), 'red')
        with patch.object(worker, 'image_inputs', ImageInputs()), \
             patch.object(worker, 'resolve_image', return_value=self.source), \
             patch.object(worker, 'image_engine', return_value=render), \
             patch.object(models, 'entry', return_value={'title': 'CPU test', 'family': 'qwen-image-2.1', 'revision': 'test'}), \
             patch.object(models, 'image_tool', return_value={'title': 'Restore', 'prepare': 'restore', 'reference_resolution': 0}), \
             patch.object(image_settings, 'output_dimensions', return_value=(32, 32)), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('backend.image_inputs.Image.open', wraps=Image.open) as opened, \
             contextlib.redirect_stdout(io.StringIO()):
            for index, images in enumerate([['source.png'], ['source.png'], [], ['source.png'], ['source.png']]):
                folder = self.root / f'job-{index}'; folder.mkdir()
                (folder / 'request.json').write_text(json.dumps({**request, 'images': images, 'tool_id': 'restore' if index >= 3 else None}))
                (folder / 'status.json').write_text(json.dumps({'id': f'job-{index}', 'status': 'queued'}))
                self.assertEqual(worker.execute(folder), 0)
                saved = json.loads((folder / 'status.json').read_text())
                self.assertEqual(saved['input_cache_hits'], 1 if index in (1, 3, 4) else 0)
                if index >= 3:
                    self.assertEqual(saved['input_preparation_reused'], index == 4)
                self.assertEqual(saved['input_count'], len(images))
                self.assertEqual(saved['status'], 'succeeded')
                self.assertFalse(list(folder.glob('input-*.png')))
                self.assertFalse(list(folder.glob('prepared-*.png')))
                self.assertTrue(Path(saved['output_path']).is_file())
                self.assertTrue(Path(saved['output_path']).with_suffix('.json').is_file())
            self.assertEqual(opened.call_count, 1)
        self.assertTrue(self.source.is_file())

    def test_cleanup_only_selects_safe_finished_image_duplicates(self):
        uploads = self.root / 'uploads'; uploads.mkdir()
        jobs = self.root / 'jobs'; jobs.mkdir()
        Image.new('RGBA', (16, 16), 'red').save(uploads / 'a.png')
        for name, status, source, color, kind in [
            ('ok', 'succeeded', 'a.png', 'red', 'image'),
            ('busy', 'running', 'a.png', 'red', 'image'),
            ('queued', 'queued', 'a.png', 'red', 'image'),
            ('missing', 'succeeded', 'gone.png', 'red', 'image'),
            ('different', 'succeeded', 'a.png', 'blue', 'image'),
            ('video', 'succeeded', 'a.png', 'red', 'video'),
            ('outside', 'succeeded', '../source.png', 'red', 'image'),
        ]:
            folder = jobs / name; folder.mkdir()
            (folder / 'status.json').write_text(json.dumps({'status': status}))
            (folder / 'request.json').write_text(json.dumps({'images': [source], 'kind': kind}))
            Image.new('RGBA', (16, 16), color).save(folder / 'input-1.png')
            (folder / 'worker.log').write_text('keep')
        Image.new('RGBA', (32, 16), 'blue').save(jobs / 'ok' / 'prepared-1.png')
        (jobs / 'ok' / 'input-1.png.tmp').write_text('keep')
        selected = list(candidates(self.root, self.root / 'examples'))
        self.assertEqual(set(selected), {jobs / 'ok' / 'input-1.png', jobs / 'ok' / 'prepared-1.png'})
        self.assertTrue(all(p.exists() for p in selected))

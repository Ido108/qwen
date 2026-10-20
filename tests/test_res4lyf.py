import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import res4lyf_runtime as res
from backend.civitai import example_settings, find_sampler, image_defaults
from backend.image_sampling import validate


class SamplerTests(unittest.TestCase):
    def test_author_names_survive_import_without_an_alias(self):
        for name in res.SAMPLERS:
            with self.subTest(name=name):
                self.assertEqual(find_sampler('Use ' + name + ' with beta'), name)
                settings = example_settings({'images': [{'meta': {'steps': 50, 'cfgScale': 3, 'sampler': name, 'scheduler': 'beta'}}]})
                for family in ('qwen-image-2.1', 'krea-2'):
                    defaults = image_defaults(False, settings, 'example', family)
                    self.assertEqual((defaults['sampler'], defaults['scheduler']), (name, 'beta'))
                    validate({'sampler': name, 'scheduler': 'beta'}, {'family': family, 'generation_defaults': defaults})
        self.assertIsNone(find_sampler('res_2m_unknown'))

    def test_fixed_turbo_schedules_cannot_be_replaced(self):
        for family in ('qwen-image-2.1', 'krea-2', 'qwen-image-2.1-turbo', 'krea-2-turbo'):
            with self.subTest(family=family), self.assertRaises(ValueError):
                validate({'sampler': 'res_2s'}, {'family': family, 'generation_defaults': {'sigmas': [1, .5, 0]}})

    def test_missing_source_fails_before_algorithm_import(self):
        res.algorithm.cache_clear()
        with tempfile.TemporaryDirectory() as folder, patch.object(res, 'source_root', return_value=Path(folder)):
            with self.assertRaisesRegex(RuntimeError, 'missing or damaged'):
                res.ensure_sampler('res_2m')

    def test_builtin_sampler_does_not_import_extension(self):
        with patch.object(res, 'algorithm', side_effect=AssertionError('unexpected extension load')):
            res.ensure_sampler('euler')

    def test_source_validation_rejects_changes_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'test.py').write_bytes(b'original')
            entry = {'path': 'test.py', 'sha256': hashlib.sha256(b'original').hexdigest()}
            self.assertTrue(res.valid_file(root, entry))
            (root / 'test.py').write_bytes(b'changed')
            self.assertFalse(res.valid_file(root, entry))
            with self.assertRaises(ValueError):
                res.valid_file(root, dict(entry, path='../outside.py'))

    def test_bad_download_never_replaces_a_file(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(res, 'source_root', return_value=Path(folder)), \
             patch.object(res, 'source_files', return_value=[{'path': 'file.py', 'sha256': 'bad'}]), \
             patch.object(res.urllib.request, 'urlopen', return_value=BytesIO(b'wrong')):
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                res.install()
            self.assertFalse((Path(folder) / 'file.py').exists())

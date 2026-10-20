import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from backend.video_timing import gpu_load_stage


class VideoTimingTests(unittest.TestCase):
    def test_gpu_load_observer_preserves_arguments_result_and_restores_after_failure(self):
        seen=[];updates=[];result=object()
        def original(*args,**kwargs):seen.append((args,kwargs));return result
        memory=SimpleNamespace(load_models_gpu=original)
        sync=lambda:seen.append('sync')
        with gpu_load_stage(memory,lambda **v:updates.append(v),sync,'loading_model_gpu','Load','generating','Generate'):
            self.assertIs(memory.load_models_gpu(['model'],memory_required=123),result)
        self.assertIs(memory.load_models_gpu,original)
        self.assertEqual(seen,[((['model'],),{'memory_required':123}),'sync'])
        self.assertEqual([u['stage'] for u in updates],['loading_model_gpu','generating'])
        with self.assertRaisesRegex(RuntimeError,'failure'):
            with gpu_load_stage(memory,lambda **v:None,sync,'load','Load','generate','Generate'):
                raise RuntimeError('failure')
        self.assertIs(memory.load_models_gpu,original)

    def test_repeated_stages_in_separate_passes_keep_their_own_times(self):
        from backend import worker
        with tempfile.TemporaryDirectory() as folder, patch.object(worker,'directory',Path(folder)), \
             patch.object(worker,'job',{'stage':'generating','phase':'Replace','started_at':90,'stage_started_at':100}), \
             patch('backend.worker.time.time',side_effect=[105,112,120]),patch('builtins.print'):
            worker.update(stage='decoding')
            worker.update(stage='generating',phase='Camera',step=0)
            worker.update(stage='saving')
            records=worker.job['stage_history']
            self.assertEqual([(x['stage'],x['phase'],x['seconds']) for x in records],
                [('generating','Replace',5),('decoding','Replace',7),('generating','Camera',8)])
            self.assertEqual(worker.job['stage_seconds']['generating'],13)

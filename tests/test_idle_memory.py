"""Clock-controlled idle expiry, scheduler locking and server timer lifecycle.

No CUDA, model imports, network repair or model downloads are needed.
"""
from collections import deque
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from backend import server


class IdleMemoryTests(unittest.TestCase):
    def setUp(self):
        self.clock = Mock(wraps=time)
        self.clock.monotonic.return_value = 100.0
        self.loaded = True
        self.worker = Mock()
        self.worker.status.side_effect = lambda: {'loaded': self.loaded}
        self.worker.close.side_effect = self.close
        self.state = patch.multiple(
            server, time=self.clock, image_worker=self.worker,
            active_job=None, active_process=None, pending_jobs=[],
            shutting_down=False, queue_paused=False,
            image_warmup_suspended=False, image_idle_since=100.0)
        self.state.start()
        self.addCleanup(self.state.stop)

    def close(self):
        self.loaded = False

    def test_exact_ten_minute_boundary_and_release_once(self):
        self.assertEqual(server.IDLE_MODEL_RELEASE_SECONDS, 600)
        self.clock.monotonic.return_value = 699.999
        self.assertFalse(server.release_idle_image_worker())
        self.worker.close.assert_not_called()
        self.clock.monotonic.return_value = 700.0
        self.assertTrue(server.release_idle_image_worker())
        self.assertFalse(server.release_idle_image_worker())
        self.worker.close.assert_called_once()
        self.assertTrue(server.image_warmup_suspended)

    def test_active_pending_and_shutdown_guards(self):
        self.clock.monotonic.return_value = 7000
        for state in ({'active_job': 'running-image'}, {'active_job': 'running-video'},
                      {'active_job': 'training'}, {'active_job': 'captioning'},
                      {'active_process': Mock()}, {'pending_jobs': ['queued']},
                      {'pending_jobs': ['paused'], 'queue_paused': True},
                      {'shutting_down': True}):
            with self.subTest(state=state), patch.multiple(server, **state):
                self.assertFalse(server.release_idle_image_worker())
        self.worker.close.assert_not_called()
        self.assertFalse(server.image_warmup_suspended)

    def test_polling_empty_queue_controls_and_downloads_do_not_extend_timer(self):
        with patch.object(server, 'save_queue_locked'), \
                patch.object(server.models, 'transfer', {'status': 'running'}):
            for instant in (200, 400, 699):
                self.clock.monotonic.return_value = instant
                self.worker.status()
                with server.active_lock:
                    server.start_next_locked()
                self.assertFalse(server.release_idle_image_worker())
            self.assertEqual(server.image_idle_since, 100)
            self.clock.monotonic.return_value = 700
            self.assertTrue(server.release_idle_image_worker())
        # Downloads belong to the server, not the image worker.
        self.assertTrue(server.image_warmup_suspended)

    def test_empty_worker_stays_unloaded_after_expiry(self):
        self.loaded = False
        self.clock.monotonic.return_value = 700
        self.assertFalse(server.release_idle_image_worker())
        with server.active_lock:
            server.start_next_locked()
        self.worker.warm.assert_not_called()
        self.worker.close.assert_not_called()

    def test_warmup_cannot_restart_after_deadline_before_timer_tick(self):
        self.loaded = False
        self.clock.monotonic.return_value = 700
        with patch.object(server, 'all_jobs') as history:
            with server.active_lock:
                server.warm_image_worker_locked()
            history.assert_not_called()
            self.worker.warm.assert_not_called()

    def test_expiry_serializes_with_new_job_lock(self):
        self.clock.monotonic.return_value = 700
        entered, finish, submitted = threading.Event(), threading.Event(), threading.Event()

        def slow_close():
            entered.set()
            finish.wait(3)
            self.close()

        def submit():
            with server.active_lock:
                server.active_job = 'new-job'
                submitted.set()

        self.worker.close.side_effect = slow_close
        release = threading.Thread(target=server.release_idle_image_worker)
        submitter = threading.Thread(target=submit)
        release.start()
        try:
            self.assertTrue(entered.wait(2))
            submitter.start()
            self.assertFalse(submitted.wait(.05))
        finally:
            finish.set()
            release.join(3)
            if submitter.ident:
                submitter.join(3)
        self.assertFalse(release.is_alive())
        self.assertTrue(submitted.is_set())
        self.assertFalse(self.loaded)

    def test_lifespan_starts_and_stops_timer_without_startup_warmup(self):
        self.loaded = False
        with tempfile.TemporaryDirectory() as folder, \
                patch.multiple(server, DATA_DIR=Path(folder), JOBS_DIR=Path(folder)/'jobs', recent_jobs=deque(),
                               pending_jobs=[], active_lock=threading.Lock()), \
                patch.object(server, 'all_jobs') as history, \
                patch.dict(os.environ, {'QWEN_NO_WARMUP': ''}), \
                patch.object(server.cloud, 'settings', return_value={'mode': 'local'}), \
                patch.object(server.models, 'is_installed', return_value=True), \
                patch.object(server, 'recover_civitai_downloads'), \
                patch.object(server.civitai, 'repair_entries'), \
                patch.object(server.civitai, 'warm_indexes'), \
                patch.object(server.moise_remote, 'start', return_value=None):
            request = {'model_id': 'gguf-q8_0', 'text_encoder': 'auto'}
            old = {'id': 'a'*32, 'created_at': 1, 'kind': 'image',
                   'status': 'succeeded', 'request': request}
            directory = server.JOBS_DIR/old['id']
            directory.mkdir(parents=True)
            (directory/'request.json').write_text(json.dumps(request))
            history.return_value = [old]
            with TestClient(server.app):
                self.worker.warm.assert_not_called()
                self.assertTrue(server.image_warmup_suspended)
                timers = [t for t in threading.enumerate() if t.name == 'image-idle-release']
                self.assertEqual(len(timers), 1)
                # Simulate a completed image without running an AI model.
                self.loaded = True
                server.image_warmup_suspended = False
                self.clock.monotonic.return_value = 700
                deadline = time.monotonic() + 3
                while self.loaded and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertFalse(self.loaded)
                self.assertTrue(server.image_warmup_suspended)
            self.assertFalse(timers[0].is_alive())


if __name__ == '__main__':
    unittest.main()

"""Real lightweight child processes exercise IPC, reuse and crash recovery."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from backend.common import write_json
from backend.resident_worker import ImageWorker

SCRIPT = '''
import json, os, sys
from pathlib import Path
for line in sys.stdin:
    command=json.loads(line)
    if command.get('stop'): break
    folder=Path(command['directory'])
    request=json.loads((folder/'request.json').read_text())
    if request.get('crash'): sys.exit(7)
    (folder/'worker.done.json').write_text(json.dumps({'returncode': 0, 'pid': os.getpid()}))
'''


class ResidentWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = ImageWorker()
        self.count = 0
        self.processes = []
        real_popen = subprocess.Popen

        def launch(args, **kwargs):
            process = real_popen([sys.executable, '-u', '-c', SCRIPT], **kwargs)
            self.processes.append(process)
            return process
        self.patch = patch('backend.resident_worker.subprocess.Popen', side_effect=launch)
        self.patch.start()

    def tearDown(self):
        self.worker.close()
        self.patch.stop()
        self.temp.cleanup()

    def run_job(self, **changes):
        self.count += 1
        directory = self.root / str(self.count)
        directory.mkdir()
        request = {'model_id': 'gguf-q8_0', 'text_encoder': 'auto', **changes}
        write_json(directory / 'request.json', request)
        process = self.worker.start(directory, request)
        return process, self.worker.wait(directory, process)

    def test_same_model_reuses_process_and_switch_releases_previous(self):
        first, code = self.run_job()
        second, code = self.run_job(text_encoder='encoder-int8')
        self.assertIs(first, second)
        self.assertEqual(code, 0)
        self.assertTrue(self.worker.status()['loaded'])
        third, _ = self.run_job(model_id='gguf-q4_k_m')
        self.assertIsNot(first, third)
        self.assertEqual(first.poll(), 0)
        fourth, _ = self.run_job(model_id='gguf-q4_k_m', text_encoder='encoder-bf16')
        self.assertIsNot(third, fourth)
        self.assertEqual(third.poll(), 0)

    def test_crash_does_not_poison_next_job(self):
        failed, code = self.run_job(crash=True)
        self.assertEqual(code, 7)
        self.assertFalse(self.worker.status()['loaded'])
        success, code = self.run_job()
        self.assertEqual(code, 0)
        self.assertIsNot(failed, success)

    def test_completion_read_retries_windows_sharing_error(self):
        original = Path.read_text
        attempts = []
        def read(path, *args, **kwargs):
            if path.name == 'worker.done.json':
                attempts.append(path)
                if len(attempts) == 1: raise PermissionError('temporary file share')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            process, code = self.run_job()
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(attempts), 2)
        self.assertIsNone(process.poll())

    def test_release_and_parent_pipe_eof_exit_cleanly(self):
        process, _ = self.run_job()
        self.worker.close()
        self.assertEqual(process.poll(), 0)
        self.assertFalse(self.worker.status()['loaded'])
        process, _ = self.run_job()
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5), 0)

    def test_release_endpoint_stops_the_resident_process_and_next_job_can_load(self):
        from fastapi.testclient import TestClient
        from backend import server
        process, _ = self.run_job()
        with patch.multiple(server, image_worker=self.worker, active_job=None,
                            pending_jobs=[], image_warmup_suspended=False):
            # No lifespan: this check must not scan history, warm real models or start downloads.
            client = TestClient(server.app)
            response = client.post('/api/models/release', headers={'X-Studio-Token': server.SESSION_TOKEN})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(process.poll(), 0)
            self.assertFalse(response.json()['loaded'])
            self.assertTrue(server.image_warmup_suspended)
            self.assertTrue((self.root / '1' / 'request.json').exists())
            next_process, code = self.run_job()
            self.assertEqual(code, 0)
            self.assertIsNot(process, next_process)
            client.close()

    def test_idle_deadline_exits_real_worker_and_next_job_starts_fresh(self):
        from backend import server
        first, code = self.run_job()
        self.assertEqual(code, 0)
        clock = unittest.mock.Mock(wraps=time)
        clock.monotonic.return_value = 1599.99
        with patch.multiple(server, image_worker=self.worker, active_job=None,
                            active_process=None, pending_jobs=[], shutting_down=False,
                            image_idle_since=1000.0, image_warmup_suspended=False, time=clock):
            self.assertFalse(server.release_idle_image_worker())
            reused, code = self.run_job()
            self.assertIs(first, reused)
            self.assertEqual(code, 0)
            # The real scheduler resets this only after a job finishes.
            server.image_idle_since = clock.monotonic.return_value
            clock.monotonic.return_value += 600
            self.assertTrue(server.release_idle_image_worker())
            self.assertEqual(first.poll(), 0)
            self.assertFalse(self.worker.status()['loaded'])
            self.assertTrue(server.image_warmup_suspended)
            next_process, code = self.run_job()
            self.assertEqual(code, 0)
            self.assertIsNot(first, next_process)
            self.assertNotEqual(first.pid, next_process.pid)

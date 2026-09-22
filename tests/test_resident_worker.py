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

    def test_release_and_parent_pipe_eof_exit_cleanly(self):
        process, _ = self.run_job()
        self.worker.close()
        self.assertEqual(process.poll(), 0)
        self.assertFalse(self.worker.status()['loaded'])
        process, _ = self.run_job()
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5), 0)

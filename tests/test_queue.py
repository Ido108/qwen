"""Exercise the real scheduler/API with controlled worker processes, without CUDA."""
from collections import deque
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import server
from backend.common import write_json


def eventually(check):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('Scheduler did not reach the expected state')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.jobs = self.root / 'jobs'
        self.jobs.mkdir()
        self.started = []
        self.processes = {}
        self.inflight = 0
        self.peak = 0
        owner = self

        class Process:
            def __init__(self, args, **kwargs):
                self.directory = Path(args[-1])
                self.id = self.directory.name
                self.gate = threading.Event()
                self.code = 0
                self.done = False
                owner.inflight += 1
                owner.peak = max(owner.peak, owner.inflight)
                job = server.read_job(self.id)
                job.update(status='running', started_at=time.time())
                write_json(self.directory / 'status.json', job)
                owner.processes[self.id] = self
                owner.started.append(self.id)

            def wait(self):
                self.gate.wait(10)
                if self.code == 0:
                    job = server.read_job(self.id)
                    job.update(status='succeeded', finished_at=time.time())
                    write_json(self.directory / 'status.json', job)
                self.done = True
                owner.inflight -= 1
                return self.code

            def poll(self):
                return self.code if self.done else None

        self.patches = [
            patch.multiple(server, DATA_DIR=self.root, JOBS_DIR=self.jobs, active_job=None,
                           active_process=None, pending_jobs=[], recent_jobs=deque(maxlen=20),
                           cancelled_jobs=set(), queue_paused=False, shutting_down=False),
            patch.object(server.subprocess, 'Popen', Process),
            patch.object(server, 'stop_worker', lambda process: process.gate.set()),
            patch.object(server.models, 'is_installed', return_value=True),
        ]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app)
        self.client.__enter__()
        self.headers = {'X-Studio-Token': server.SESSION_TOKEN}

    def tearDown(self):
        with server.active_lock:
            server.shutting_down = True
        for process in self.processes.values():
            process.gate.set()
        eventually(lambda: server.active_job is None)
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def post(self, path, body=None):
        response = self.client.post(path, json=body, headers=self.headers)
        self.assertLess(response.status_code, 300, response.text)
        return response.json()

    def add(self, prompt, video=False, **kwargs):
        return self.post('/api/video/generate' if video else '/api/generate',
                         {'prompt': prompt, 'enhance': False, 'output_dir': str(self.root),
                          'seed': 1234, 'randomize_seed': False, **kwargs})

    def finish(self, job_id, code=0):
        eventually(lambda: job_id in self.processes)
        process = self.processes[job_id]
        process.code = code
        process.gate.set()

    def test_mixed_requests_are_accepted_while_running_and_execute_fifo(self):
        first = self.add('First image', steps=12)
        eventually(lambda: first['id'] in self.started)
        second = self.add('Second video', video=True)
        third = self.add('Third image', steps=25)
        queue = self.client.get('/api/queue').json()
        self.assertEqual([j['id'] for j in queue['pending']], [second['id'], third['id']])
        self.assertEqual(queue['active']['id'], first['id'])
        self.finish(first['id'])
        eventually(lambda: second['id'] in self.started)
        self.assertNotIn(third['id'], self.started)
        self.finish(second['id'])
        eventually(lambda: third['id'] in self.started)
        self.finish(third['id'])
        eventually(lambda: server.active_job is None)
        self.assertEqual(self.started, [first['id'], second['id'], third['id']])
        self.assertEqual(self.peak, 1)
        self.assertEqual(server.read_job(first['id'])['request']['steps'], 12)
        self.assertEqual(server.read_job(third['id'])['request']['steps'], 25)

    def test_pause_reorder_cancel_and_model_removal_protection(self):
        self.post('/api/queue/control', {'paused': True})
        first, second, third = [self.add(name) for name in ('First', 'Second', 'Third')]
        self.post(f'/api/jobs/{third["id"]}/move', {'position': 1})
        self.post(f'/api/jobs/{second["id"]}/cancel')
        self.assertEqual(self.started, [])
        self.assertEqual(server.read_job(second['id'])['status'], 'cancelled')
        self.assertEqual(self.client.delete('/api/models/gguf-q8_0', headers=self.headers).status_code, 409)
        self.post('/api/queue/control', {'paused': False})
        eventually(lambda: third['id'] in self.started)
        self.post('/api/queue/control', {'paused': True})
        self.finish(third['id'])
        eventually(lambda: server.active_job is None)
        self.assertEqual(self.started, [third['id']])
        self.post('/api/queue/control', {'paused': False})
        self.finish(first['id'])
        eventually(lambda: server.active_job is None)
        self.assertEqual(self.started, [third['id'], first['id']])

    def test_failure_and_active_cancel_do_not_stall_the_queue(self):
        first, second, third = [self.add(name) for name in ('Fail', 'Cancel', 'Continue')]
        self.finish(first['id'], 1)
        eventually(lambda: second['id'] in self.started)
        self.assertEqual(server.read_job(first['id'])['status'], 'failed')
        self.post(f'/api/jobs/{second["id"]}/cancel')
        eventually(lambda: third['id'] in self.started)
        self.assertEqual(server.read_job(second['id'])['status'], 'cancelled')
        self.finish(third['id'])
        eventually(lambda: server.active_job is None)
        self.assertEqual(self.peak, 1)

    def test_restart_restores_waiting_order_and_pause(self):
        self.post('/api/queue/control', {'paused': True})
        first, second = self.add('First'), self.add('Second')
        self.post(f'/api/jobs/{second["id"]}/move', {'position': 1})
        self.client.__exit__(None, None, None)
        self.client = TestClient(server.app)
        self.client.__enter__()
        queue = self.client.get('/api/queue').json()
        self.assertTrue(queue['paused'])
        self.assertEqual([j['id'] for j in queue['pending']], [second['id'], first['id']])
        self.assertEqual(self.started, [])

    def test_rerun_preserves_resolved_seed_steps_and_input_parameters(self):
        self.post('/api/queue/control', {'paused': True})
        original = self.add('Exact settings', seed=567, steps=17, auto_resolution=1024,
                            negative_prompt='No text', width=864, height=1184)
        rerun = self.post(f'/api/jobs/{original["id"]}/retry')
        self.assertNotEqual(rerun['id'], original['id'])
        self.assertEqual(rerun['request'], original['request'])

    def test_library_filters_searches_and_paginates_beyond_100_results(self):
        for i in range(125):
            job_id = f'{i:032x}'
            directory = self.jobs / job_id
            directory.mkdir()
            write_json(directory / 'status.json', {'id': job_id, 'status': 'succeeded',
                       'kind': 'video' if i % 2 else 'image', 'created_at': i,
                       'request': {'prompt': f'Garden {i}', 'model_id': 'test-model'}})
        first = self.client.get('/api/library?limit=100').json()
        second = self.client.get('/api/library?offset=100').json()
        self.assertEqual(first['total'], 125)
        self.assertEqual(len(first['items']), 100)
        self.assertEqual(len(second['items']), 25)
        self.assertFalse({j['id'] for j in first['items']} & {j['id'] for j in second['items']})
        filtered = self.client.get('/api/library?kind=video&search=Garden%20123').json()
        self.assertEqual(filtered['total'], 1)
        self.assertEqual(filtered['items'][0]['request']['prompt'], 'Garden 123')
        self.assertEqual(self.client.get('/api/library?offset=-1').status_code, 422)


if __name__ == '__main__':
    unittest.main()

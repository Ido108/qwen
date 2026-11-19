"""A repair of a long video is made only by the repair route (and a retry of that job), never by the form's route.
Real HTTP handlers; the queue step is replaced, so no model runs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import server
from backend.common import write_json

ORIGINAL = 'd34026bc' + '0' * 24


class RepairRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        jobs = patch.object(server, 'JOBS_DIR', self.root); jobs.start(); self.addCleanup(jobs.stop)
        self.queued = []
        queue = patch.object(server, 'queue_reference_video', lambda body: self.queued.append(body) or {'id': 'e' * 32})
        queue.start(); self.addCleanup(queue.stop)
        self.client = TestClient(server.app)
        self.headers = {'x-studio-token': self.client.get('/api/session').json()['token']}

    def job(self, job_id, request, **status):
        (self.root / job_id).mkdir()
        write_json(self.root / job_id / 'status.json', {'id': job_id, 'kind': 'reference-video', 'seed': 7, 'request': request, **status})

    def test_a_request_of_the_form_is_a_whole_video_also_when_it_carries_a_repair(self):
        # The request that failed on 2026-11-19: a new video of 362 frames, sent by a form restored from a repaired video
        body = {'workflow': 'faceswap', 'model_id': 'h3-ref-full-q8_0', 'source_video': '516515b392aa4114a78e61d2e9cf4fb0', 'frames': 362,
                'steps': 25, 'width': 576, 'height': 1024, 'prompt': 'the person', 'repair_of': ORIGINAL, 'repair_from_part': 17}
        answer = self.client.post('/api/reference-video/generate', json=body, headers=self.headers)
        self.assertEqual(answer.status_code, 202, answer.text)
        sent = self.queued[0]
        self.assertIsNone(sent.repair_of)
        self.assertIsNone(sent.repair_from_part)
        self.assertEqual((sent.source_video, sent.frames, sent.steps, sent.model_id), ('516515b392aa4114a78e61d2e9cf4fb0', 362, 25, 'h3-ref-full-q8_0'))
        # The same request with no repair in it goes out unchanged
        plain = {key: value for key, value in body.items() if not key.startswith('repair')}
        self.client.post('/api/reference-video/generate', json=plain, headers=self.headers)
        self.assertEqual(self.queued[1].model_dump(), sent.model_dump())

    def test_the_repair_route_and_a_retry_of_a_repair_keep_the_repair(self):
        video = self.root / 'original.mp4'; video.write_bytes(b'mp4')
        request = {'workflow': 'faceswap', 'model_id': 'h3-ref-full-q8_0', 'source_video': 'e77f93ddcfdb474ba09947d7f596d93e', 'frames': 1977, 'prompt': 'the person'}
        self.job(ORIGINAL, request, status='succeeded', total_parts=17, output_path=str(video))
        answer = self.client.post(f'/api/jobs/{ORIGINAL}/repair', json={'from_part': 17}, headers=self.headers)
        self.assertEqual(answer.status_code, 202, answer.text)
        self.assertEqual((self.queued[0].repair_of, self.queued[0].repair_from_part, self.queued[0].frames, self.queued[0].seed), (ORIGINAL, 17, 1977, 7))
        # A retry of the repair job is the same repair again
        repair = 'f' * 32
        self.job(repair, {**request, 'repair_of': ORIGINAL, 'repair_from_part': 17}, status='failed')
        self.client.post(f'/api/jobs/{repair}/retry', headers=self.headers)
        self.assertEqual((self.queued[1].repair_of, self.queued[1].repair_from_part), (ORIGINAL, 17))
        # A retry of a whole video has no repair
        self.client.post(f'/api/jobs/{ORIGINAL}/retry', headers=self.headers)
        self.assertIsNone(self.queued[2].repair_of)


if __name__ == '__main__':
    unittest.main()

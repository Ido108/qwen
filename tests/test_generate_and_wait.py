"""The one-call job for Moise's relay, via real HTTP handlers. No model startup or inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import server
from backend.common import write_json


class GenerateAndWaitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patch=patch.object(server,'JOBS_DIR',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.client=TestClient(server.app);self.token=self.client.get('/api/session').json()['token']
        self.id='c'*32;(self.root/self.id).mkdir()

    def finish(self,status,**extra):
        def fake_generate(body):
            write_json(self.root/self.id/'status.json',{'id':self.id,'kind':'image','status':status,**extra})
            return {'id':self.id}
        return patch.object(server,'generate',fake_generate)

    def post(self):
        return self.client.post('/api/generate-and-wait',json={'model_id':'m','prompt':'an apple'},headers={'x-studio-token':self.token})

    def test_the_finished_picture_comes_back_with_its_job(self):
        image=self.root/self.id/'result.png';image.write_bytes(b'\x89PNG fake')
        with self.finish('succeeded',output_path=str(image)):
            answer=self.post()
        self.assertEqual(answer.status_code,200)
        self.assertEqual(answer.content,b'\x89PNG fake')
        self.assertEqual(answer.headers['x-job-id'],self.id)

    def test_a_failed_job_answers_its_message(self):
        with self.finish('failed',message='Out of GPU memory'):
            answer=self.post()
        self.assertEqual(answer.status_code,500)
        self.assertEqual(answer.json()['detail'],'Out of GPU memory')


if __name__=='__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import server, video_reuse


class VideoReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.uploads = self.root / 'uploads'
        self.job_id = 'a' * 32
        self.folder = self.root / self.job_id
        self.folder.mkdir()
        self.source = self.root / 'result.mp4'
        self.source.write_bytes(b'video bytes remain unchanged')
        self.job = {'id':self.job_id,'kind':'video','status':'succeeded','output_path':str(self.source)}
        self.client = TestClient(server.app)
        self.headers = {'X-Studio-Token':server.SESSION_TOKEN}

    @staticmethod
    def probe(path, poster):
        poster.write_bytes(b'poster')
        return {'width':576,'height':1024,'fps':24,'duration_seconds':4.9,'has_audio':True}

    def send(self, part='final'):
        (self.folder / 'status.json').write_text(json.dumps(self.job),encoding='utf-8')
        with patch.object(server,'JOBS_DIR',self.root), patch.object(server,'VIDEO_UPLOADS_DIR',self.uploads):
            return self.client.post(f'/api/jobs/{self.job_id}/reuse-video?part={part}',headers=self.headers)

    def test_every_video_kind_becomes_an_input_with_actual_metadata(self):
        with patch.object(video_reuse,'inspect_video',side_effect=self.probe):
            for kind in server.VIDEO_KINDS:
                self.job.update(kind=kind,width=1344,height=768)
                response = self.send()
                self.assertEqual(response.status_code,200,response.text)
                info = response.json()
                self.assertEqual((info['width'],info['height']),(576,1024))
                self.assertEqual(info['duration_seconds'],4.9)
                self.assertTrue(info['has_audio'])
                self.assertEqual((self.uploads/info['id']).read_bytes(),self.source.read_bytes())
                self.assertTrue((self.uploads/(info['id']+'.json')).is_file())

    def test_repeated_and_concurrent_sends_reuse_one_input(self):
        from concurrent.futures import ThreadPoolExecutor
        with patch.object(video_reuse,'inspect_video',side_effect=self.probe) as probe:
            def send(_):return video_reuse.reuse_video_file(self.source,self.uploads,self.folder)
            with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(send,range(8)))
            self.assertEqual(len({r['id'] for r in results}),1)
            probe.assert_called_once()
            self.assertEqual(len(list(self.uploads.glob('*.mp4'))),1)

    def test_copy_fallback_keeps_output_and_rolls_back_failed_import(self):
        with patch.object(video_reuse.os,'link',side_effect=OSError('Different volumes')), \
             patch.object(video_reuse,'inspect_video',side_effect=ValueError('Invalid video')):
            response=self.send()
        self.assertEqual(response.status_code,422)
        self.assertEqual(list(self.uploads.iterdir()),[])
        self.assertTrue(self.source.is_file())
        with patch.object(video_reuse.os,'link',side_effect=OSError('Different volumes')), \
             patch.object(video_reuse,'inspect_video',side_effect=self.probe):
            self.assertEqual(self.send().status_code,200)

    def test_rejects_non_outputs_and_requires_current_session(self):
        for kind,status in [('image','succeeded'),('video','running'),('video','failed')]:
            self.job.update(kind=kind,status=status)
            self.assertEqual(self.send().status_code,404)
        self.job.update(kind='video',status='succeeded',output_path=str(self.root/'missing.mp4'))
        self.assertEqual(self.send().status_code,404)
        self.assertEqual(self.send('unknown').status_code,422)
        self.assertEqual(self.client.post(f'/api/jobs/{self.job_id}/reuse-video').status_code,403)

    def test_first_stage_can_be_sent_when_available_and_stays_separate(self):
        self.job.update(kind='reference-video',status='running',subject_replacement_ready=True)
        (self.folder/'subject-replacement.mp4').write_bytes(b'intermediate')
        with patch.object(video_reuse,'inspect_video',side_effect=self.probe):
            subject=self.send('subject')
            self.assertEqual(subject.status_code,200)
            self.job.update(status='succeeded')
            final=self.send()
        self.assertNotEqual(final.json()['id'],subject.json()['id'])
        self.assertEqual((self.uploads/subject.json()['id']).read_bytes(),b'intermediate')


if __name__=='__main__':unittest.main()

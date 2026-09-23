"""Cloud dispatch contracts without charging for a GPU job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend import cloud, server
from backend.common import GenerationRequest, VideoRequest


class CloudTests(unittest.TestCase):
    def test_image_and_video_requests_queue_for_runpod_without_local_models(self):
        with tempfile.TemporaryDirectory() as folder:
            captured = []
            with patch.object(cloud, 'settings', return_value={'mode': 'runpod'}), \
                 patch.object(cloud, 'ready', return_value=True), \
                 patch.object(server.models, 'is_installed', side_effect=AssertionError('local model checked')), \
                 patch.object(server, 'enqueue_locked', side_effect=lambda job, data: captured.append((job, data))), \
                 patch.object(server.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 2**40})()):
                image = server.generate(GenerationRequest(prompt='A room', enhance=False, output_dir=folder))
                video = server.generate_video(VideoRequest(prompt='Camera move', enhance=False, output_dir=folder))
            self.assertEqual(image['kind'], 'image')
            self.assertEqual(video['kind'], 'video')
            self.assertEqual([data['execution'] for _, data in captured], ['runpod', 'runpod'])
            self.assertEqual(captured[0][1]['steps'], server.models.entry('gguf-q8_0')['generation_defaults']['steps'])

    def test_remote_image_result_is_saved_locally_with_original_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_id = 'a' * 32
            job_folder = root / job_id
            job_folder.mkdir()
            request = GenerationRequest(prompt='Original brief', enhance=False, output_dir=folder).model_dump()
            (job_folder / 'status.json').write_text(json.dumps({'id': job_id, 'kind': 'image', 'status': 'queued',
                'request': request, 'created_at': 1, 'seed': 0}), encoding='utf-8')
            uploads = []
            def download(_key, destination):
                destination.write_bytes(b'PNG result')
            with patch.object(server, 'JOBS_DIR', root), \
                 patch.object(cloud, 'upload', side_effect=lambda *args: uploads.append(args)), \
                 patch.object(cloud, 'submit', return_value='remote-123'), \
                 patch.object(cloud, 'poll', return_value={'output_key': f'qwen-transfer/{job_id}/result.png', 'metadata': {'width': 1024}}), \
                 patch.object(cloud, 'download', side_effect=download):
                server.run_cloud_job(job_id, request, 'image')
            result = json.loads((job_folder / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(result['request']['prompt'], 'Original brief')
            self.assertEqual(Path(result['output_path']).read_bytes(), b'PNG result')
            self.assertEqual(uploads, [])


if __name__ == '__main__':
    unittest.main()

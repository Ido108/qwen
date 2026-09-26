"""Cloud dispatch contracts without charging for a GPU job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend import cloud, server
from backend.common import GenerationRequest, VideoRequest, ReferenceVideoRequest, UpscaleRequest


class CloudTests(unittest.TestCase):
    def test_progress_keeps_real_steps_and_drops_untrusted_status_fields(self):
        details={'stage':'generating','step':2,'total_steps':8,'stage_started_at':123,
                 'preview_step':2,'preview_key':'qwen-transfer/abc/preview.webp',
                 'status':'succeeded','output_path':'untrusted','request':{'prompt':'changed'}}
        for output in (details,[{'step':1},details]):
            safe=cloud.progress_details({'output':output})
            self.assertEqual(safe['step'],2)
            self.assertEqual(safe['stage'],'generating')
            self.assertNotIn('status',safe)
            self.assertNotIn('request',safe)
        self.assertEqual(cloud.progress_details({'output':'Loading model'}),{})

    def test_download_progress_survives_cloud_transport(self):
        progress={'stage':'downloading_models','stage_started_at':1,
                  'download_completed_bytes':100,'download_total_bytes':200}
        self.assertEqual(cloud.progress_details({'output':progress}),progress)

    def test_remote_preview_is_saved_while_generation_continues(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job_id='e'*32;directory=root/job_id;directory.mkdir()
            request=VideoRequest(prompt='Motion',enhance=False,output_dir=folder).model_dump()
            (directory/'status.json').write_text(json.dumps({'id':job_id,'kind':'video','status':'queued','request':request,'seed':0,'created_at':1}),encoding='utf-8')
            prefix=f'qwen-transfer/{job_id}/'
            def download(key,destination):
                if key.endswith('preview.webp'):Image.new('RGB',(8,8),'red').save(destination,format='WEBP')
                else:destination.write_bytes(b'video')
            def poll(kind,remote_id,update,cancelled):
                update('IN_PROGRESS',{'output':{'stage':'generating','step':3,'total_steps':8,'preview_key':prefix+'preview.webp','preview_updated_at':123}})
                current=json.loads((directory/'status.json').read_text())
                self.assertEqual(current['status'],'running')
                self.assertEqual(current['step'],3)
                self.assertEqual(current['preview_url'],f'/api/jobs/{job_id}/live-preview')
                self.assertTrue((directory/'live-preview.webp').is_file())
                return {'output_key':prefix+'result.mp4'}
            with patch.object(server,'JOBS_DIR',root),patch.object(cloud,'submit',return_value='remote'), \
                 patch.object(cloud,'download',side_effect=download),patch.object(cloud,'poll',side_effect=poll):
                server.run_cloud_job(job_id,request,'video')
            self.assertEqual(json.loads((directory/'status.json').read_text())['status'],'succeeded')

    def test_remote_catalog_offers_supported_video_workflows_only(self):
        with patch.object(cloud, 'settings', return_value={'mode': 'runpod'}), \
             patch.object(cloud, 'ready', return_value=True):
            state = server.catalog_state()
        by_id = {item['id']: item for item in state['models']}
        self.assertTrue(by_id['h3-ref-turbo-int8']['ready'])
        self.assertTrue(by_id['seedvr2-3b-q4']['ready'])
        self.assertFalse(by_id['ltx25-msr-v2']['ready'])

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

    def test_reference_and_upscale_requests_use_video_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / ('a' * 32 + '.mp4')
            source.write_bytes(b'video')
            reference = root / ('b' * 32 + '.png')
            reference.write_bytes(b'image')
            captured = []
            with patch.object(cloud, 'settings', return_value={'mode': 'runpod'}), \
                 patch.object(cloud, 'ready', return_value=True) as ready, \
                 patch.object(server, 'resolve_video', return_value=source), \
                 patch.object(server, 'probe_video_duration', return_value=12.0), \
                 patch.object(server, 'resolve_image', return_value=reference), \
                 patch.object(server.models, 'is_installed', side_effect=AssertionError('local model checked')), \
                 patch.object(server, 'enqueue_locked', side_effect=lambda job, data: captured.append((job, data))), \
                 patch.object(server.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 2**40})()):
                replacement = server.generate_reference_video(ReferenceVideoRequest(
                    workflow='crossview', replace_subjects_first=True,
                    source_video=source.name, reference_images=[reference.name],
                    source_start_seconds=2, source_end_seconds=9,
                    source_people=[{'id':'person-1','time':3,'box':(0.1,0.2,0.2,0.3)}],
                    person_assignments={'person-1':reference.name},
                    prompt='New camera angle', output_dir=folder, enhance=False,
                ))
                upscale = server.generate_upscale(UpscaleRequest(source_video=source.name, output_dir=folder))
            self.assertEqual([item['kind'] for item, _ in captured], ['reference-video', 'upscale'])
            self.assertEqual([data['execution'] for _, data in captured], ['runpod', 'runpod'])
            self.assertEqual(replacement['total_steps'], 16)
            self.assertEqual(captured[0][1]['source_start_seconds'], 2)
            self.assertEqual(captured[0][1]['source_end_seconds'], 9)
            self.assertEqual(captured[0][1]['frames'], 175)
            self.assertEqual(upscale['kind'], 'upscale')
            ready.assert_any_call('video')

    def test_remote_two_stage_result_transfers_the_intermediate_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / ('a' * 32 + '.mp4')
            source.write_bytes(b'source')
            reference = root / ('b' * 32 + '.png')
            reference.write_bytes(b'reference')
            job_id = 'c' * 32
            job_folder = root / job_id
            job_folder.mkdir()
            request = ReferenceVideoRequest(
                workflow='crossview', replace_subjects_first=True, source_video=source.name,
                reference_images=[reference.name], prompt='New camera angle', output_dir=folder,
                enhance=False, source_audio=False,
                source_people=[{'id':'person-1','time':0.5}],
                person_assignments={'person-1':reference.name},
            ).model_dump()
            request.update(kind='reference-video', execution='runpod', steps=8)
            (job_folder / 'status.json').write_text(json.dumps({
                'id': job_id, 'kind': 'reference-video', 'status': 'queued',
                'request': request, 'created_at': 1, 'seed': 0,
            }), encoding='utf-8')
            submitted = []
            uploads = []
            prefix = f'qwen-transfer/{job_id}/'
            result = {'output_key': prefix + 'result.mp4', 'poster_key': prefix + 'poster.png',
                      'subject_video_key': prefix + 'subject-replacement.mp4',
                      'metadata': {'subject_replacement_ready': True, 'width': 256}}
            def download(key, destination):
                destination.write_bytes(key.encode())
            with patch.object(server, 'JOBS_DIR', root), \
                 patch.object(server, 'resolve_video', return_value=source), \
                 patch.object(server, 'resolve_image', return_value=reference), \
                 patch.object(cloud, 'upload', side_effect=lambda *args: uploads.append(args)), \
                 patch.object(cloud, 'submit', side_effect=lambda kind, payload: submitted.append((kind, payload)) or 'remote-123'), \
                 patch.object(cloud, 'poll', return_value=result), \
                 patch.object(cloud, 'download', side_effect=download):
                server.run_cloud_job(job_id, request, 'video')
            saved = json.loads((job_folder / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'succeeded')
            self.assertTrue(saved['subject_replacement_ready'])
            self.assertEqual(len(uploads), 2)
            self.assertEqual(submitted[0][0], 'video')
            self.assertEqual(submitted[0][1]['kind'], 'reference-video')
            self.assertEqual(submitted[0][1]['request']['reference_images'], ['cloud-ref-0.png'])
            self.assertEqual(submitted[0][1]['request']['person_assignments'], {'person-1':'cloud-ref-0.png'})
            self.assertTrue((job_folder / 'subject-replacement.mp4').is_file())
            self.assertTrue(Path(saved['output_path']).is_file())


if __name__ == '__main__':
    unittest.main()

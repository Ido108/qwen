"""No inference: dependency, request, queue, cloud, and schedule contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from backend import models, server, cloud
from backend.ltx_settings import MultiReferenceRequest, required_ltx, media_inputs, SIGMAS, REFINE_SIGMAS


class LtxTests(unittest.TestCase):
    def test_all_ltx_variants_are_pinned_with_exact_eight_step_schedule(self):
        entries = [item for item in models.catalog() if item['kind'] == 'multi-reference']
        self.assertEqual(len(entries), 8)
        for item in entries:
            with self.subTest(model=item['id']):
                request = MultiReferenceRequest(model_id=item['id'], prompt='A person waves', reference_images=['a.png'])
                needed = required_ltx(request)
                self.assertEqual(needed, [item['id'], 'ltx25-gemma-int8', 'ltx25-video-vae', 'ltx25-audio-vae', 'ltx25-msr-v2'])
                self.assertEqual(item['generation_defaults']['steps'], len(SIGMAS)-1)
                self.assertEqual(item['generation_defaults']['sigmas'], SIGMAS)
                for name in needed:
                    companion = models.entry(name)
                    self.assertEqual(len(companion['revision']), 40)
                    for file in companion['files']:
                        self.assertEqual(len(file['sha256']), 64)
                        self.assertGreater(file['bytes'], 0)
                request.refine = True
                self.assertIn('ltx25-spatial-upscaler', required_ltx(request))
                self.assertEqual(len(REFINE_SIGMAS)-1, 3)

    def test_frame_grid_resolution_and_audio_pairs_are_validated(self):
        base = dict(prompt='Scene', reference_images=['a.png'])
        for data in [dict(frames=124), dict(width=1025), dict(reference_frames=17), dict(reference_images=[])]:
            with self.assertRaises(ValueError):
                MultiReferenceRequest(**(base | data))
        for data in [dict(width=1536,height=1536), dict(audio_references=[None,'voice.wav']), dict(model_id='h3-ref-turbo-int8')]:
            with self.assertRaises(ValueError):
                required_ltx(MultiReferenceRequest(**(base | data)))

    def test_media_order_preserves_sparse_voice_slots(self):
        request = MultiReferenceRequest(prompt='Scene', reference_images=['a.png','b.png'], background='bg.png',audio_references=[None,'voice.wav'])
        self.assertEqual(media_inputs(request.model_dump()), [('image','a.png'),('image','b.png'),('image','bg.png'),('audio','voice.wav')])

    def test_blank_output_folder_uses_video_library(self):
        from backend.common import VIDEOS_DIR
        request=MultiReferenceRequest(prompt='Scene',reference_images=['a.png'],output_dir='  ')
        self.assertEqual(request.output_dir,str(VIDEOS_DIR))

    def test_api_queues_uninstalled_ltx_without_starting_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            captured=[]
            with patch.object(server,'resolve_image',return_value=Path(folder)/'ref.png'), \
                 patch.object(cloud,'settings',return_value={'mode':'local'}), \
                 patch.object(server,'enqueue_locked',side_effect=lambda job,data:captured.append(data)), \
                 patch.object(models,'is_installed',side_effect=AssertionError('Submission must not require installation')):
                for refine,steps in [(False,8),(True,11)]:
                    result=server.generate_multi_reference(MultiReferenceRequest(prompt='Scene',reference_images=['ref.png'],output_dir=folder,refine=refine))
                    self.assertEqual(result['total_steps'],steps)
                    self.assertEqual(result['kind'],'multi-reference')
                    self.assertEqual(captured[-1]['execution'],'local')

    def test_cloud_preserves_images_background_and_voice_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job_id='d'*32;directory=root/job_id;directory.mkdir()
            request=MultiReferenceRequest(prompt='Two people speak', reference_images=['a.png','b.png'],
                background='bg.png',audio_references=[None,'voice.wav'],output_dir=folder).model_dump()
            request.update(kind='multi-reference',execution='runpod')
            (directory/'status.json').write_text(json.dumps({'id':job_id,'kind':'multi-reference','status':'queued','request':request,'seed':0,'created_at':1}))
            submitted=[]
            prefix=f'qwen-transfer/{job_id}/'
            with patch.object(server,'JOBS_DIR',root), \
                 patch.object(server,'resolve_image',side_effect=lambda name:root/name), \
                 patch.object(server,'resolve_audio',side_effect=lambda name:root/name), \
                 patch.object(cloud,'upload') as upload, \
                 patch.object(cloud,'submit',side_effect=lambda kind,payload:submitted.append((kind,payload)) or 'remote'), \
                 patch.object(cloud,'poll',return_value={'output_key':prefix+'result.mp4'}), \
                 patch.object(cloud,'download',side_effect=lambda key,path:path.write_bytes(b'mocked-video')):
                server.run_cloud_job(job_id,request,'video')
            self.assertEqual(upload.call_count,4)
            self.assertEqual(submitted[0][0],'video')
            self.assertEqual(submitted[0][1]['kind'],'multi-reference')
            remote=submitted[0][1]['request']
            self.assertEqual(media_inputs(remote),media_inputs(request))
            self.assertEqual(remote['audio_references'],[None,'voice.wav'])
            self.assertEqual(json.loads((directory/'status.json').read_text())['status'],'succeeded')

    def test_access_failure_prevents_any_large_download(self):
        import httpx
        from huggingface_hub.errors import GatedRepoError
        from backend.models import install
        with patch.object(models,'is_installed',return_value=False), \
             patch.dict(models.transfer, {'status':'idle'}, clear=True), \
             patch('huggingface_hub.get_hf_file_metadata',side_effect=GatedRepoError('Access required',response=httpx.Response(403,request=httpx.Request('HEAD','https://huggingface.co')))), \
             patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('Do not download before access check')):
            models.cancel_event.clear()
            install(required_ltx(MultiReferenceRequest(prompt='Scene',reference_images=['a.png'])),include_dependencies=False)
            self.assertEqual(models.transfer['status'],'failed')
            self.assertIn('accept the model terms',models.transfer['message'])

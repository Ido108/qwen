"""No inference: dependency, request, queue, cloud, and schedule contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from backend import models, server, cloud
from backend.ltx_settings import MultiReferenceRequest, required_ltx, media_inputs, SIGMAS, REFINE_SIGMAS


class LtxTests(unittest.TestCase):
    def test_imported_ltx_files_use_reference_adapter_without_changing_keyframe_dependencies(self):
        from backend.common import VideoRequest
        original=models.entry
        for variant in ('distilled','dev'):
            imported={**original('ltx25-msr-q4_k_m'),'id':'future-ltx-'+variant,'kind':'video','family':'ltx25',
                      'dependencies':['ltx25-gemma-int8','ltx25-video-vae','ltx25-audio-vae']+(['ltx25-distill-lora'] if variant=='dev' else [])}
            with patch.object(models,'entry',side_effect=lambda id:imported if id==imported['id'] else original(id)):
                refs=required_ltx(MultiReferenceRequest(model_id=imported['id'],prompt='@Image1 walks.',reference_images=['a.png']))
                frames=models.required_video(VideoRequest(model_id=imported['id'],prompt='A scene.',mode='image',first_frame='a.png',frames=121,width=768,height=448))
            self.assertEqual(refs.count('ltx25-msr-v2'),1)
            self.assertNotIn('ltx25-msr-v2',frames)
            self.assertEqual('ltx25-distill-lora' in refs,variant=='dev')
            self.assertEqual('ltx25-distill-lora' in frames,variant=='dev')

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
        for data in [dict(frames=120), dict(width=1025), dict(reference_frames=17), dict(reference_images=[])]:
            with self.assertRaises(ValueError):
                MultiReferenceRequest(**(base | data))
        # 124 frames is a MiniMax H3 length (17n+5), not an LTX length (8n+1).
        for data in [dict(width=1536,height=1536), dict(audio_references=[None,'voice.wav']), dict(frames=124),
                     dict(model_id='h3-ref-turbo-int8'), dict(model_id='h3-ref-turbo-int8',frames=124,refine=True),
                     dict(model_id='h3-ref-turbo-int8',frames=124,fps=25), dict(model_id='fasth3-q4_k_m')]:
            with self.assertRaises(ValueError):
                required_ltx(MultiReferenceRequest(**(base | data)))

    def test_ref2va_models_accept_reference_images_without_a_source_video(self):
        request = MultiReferenceRequest(model_id='h3-ref-base-int8', prompt='Image 1 waves in Image 2.',
            reference_images=['a.png'], background='bg.png', audio_references=['voice.wav'], frames=124, width=768, height=448)
        self.assertEqual(required_ltx(request), ['h3-ref-base-int8', 'h3-encoder', 'h3-video-vae', 'h3-audio-vae'])
        turbo = request.model_copy(update={'quality': 'turbo'})
        self.assertIn('h3-ref-turbo-lora', required_ltx(turbo))
        from backend.ltx_settings import ref2va_prompt
        prompt = ref2va_prompt(request.model_dump())
        self.assertTrue(prompt.startswith('<Picture 1> waves in <Picture 2>.'))
        self.assertIn('<Picture 2> as the background', prompt)
        self.assertIn('<Audio 1> is the voice of the subject in <Picture 1>.', prompt)
        with self.assertRaises(ValueError):
            ref2va_prompt({**request.model_dump(), 'prompt': 'Image 3 dances'})

    def test_api_queues_ref2va_reference_images_with_model_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(server,'resolve_image',return_value=Path(folder)/'ref.png'),                  patch.object(cloud,'settings',return_value={'mode':'local'}),                  patch.object(server,'enqueue_locked'):
                result=server.generate_multi_reference(MultiReferenceRequest(model_id='h3-ref-turbo-int8',prompt='Scene',
                    reference_images=['ref.png'],frames=124,output_dir=folder))
            self.assertEqual(result['total_steps'],8)
            self.assertEqual(result['request']['quality'],'turbo')

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

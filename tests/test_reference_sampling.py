import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from math import prod

from backend.models import catalog, entry, required_reference
from backend.common import ReferenceVideoRequest
from backend.reference_sampling import reference_sampling
from backend.reference_memory import account_for_reference_memory, reference_latent_elements


class ReferenceSamplingTests(unittest.TestCase):
    def test_api_click_only_request_uses_model_default_without_prompt_auth(self):
        from backend import server
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);captured=[]
            with patch.object(server.cloud,'settings',return_value={'mode':'local'}), \
                 patch.object(server.models,'is_installed',return_value=True), \
                 patch.object(server,'resolve_video',return_value=root/'source.mp4'), \
                 patch.object(server,'resolve_image',return_value=root/'ref.png'), \
                 patch.object(server,'probe_video_duration',return_value=5), \
                 patch.object(server,'enqueue_locked',side_effect=lambda job,data:captured.append(data)), \
                 patch.object(server.antigravity,'status',side_effect=AssertionError('Empty instructions must not require OAuth')):
                for model_id,steps in [('h3-ref-turbo-int8',8),('h3-ref-pruned-q4_k_m',20),('h3-ref-full-q8_0',25),('h3-ref-eros-q4',25)]:
                    result=server.generate_reference_video(ReferenceVideoRequest(model_id=model_id,
                        source_video='no-real-media.mp4',reference_images=['ref.png'],prompt='',enhance=True,output_dir=folder))
                    self.assertEqual(result['total_steps'],steps)
                    self.assertEqual(captured[-1]['steps'],steps)
                    self.assertFalse(captured[-1]['enhance'])
                    self.assertEqual(captured[-1]['steps_mode'],'model')

    def test_published_checkpoint_defaults_and_explicit_override(self):
        expected={'h3-ref-base-int8':20,'h3-ref-turbo-int8':8,
                  'h3-ref-eros-q4':25,'h3-ref-eros-q8':25,
                  'h3-ref-pruned-q4_k_m':20,'h3-ref-full-q8_0':25}
        for model_id, steps in expected.items():
            with self.subTest(model=model_id):
                item=entry(model_id)
                self.assertEqual(reference_sampling(item)['steps'],steps)
                self.assertEqual(reference_sampling(item,steps=12)['steps'],12)
                request=ReferenceVideoRequest(model_id=model_id,source_video='source.mp4',reference_images=['ref.png'])
                self.assertIn(model_id,required_reference(request))
        turbo=reference_sampling(entry('h3-ref-base-int8'),'turbo')
        self.assertEqual((turbo['steps'],turbo['sampler'],turbo['video_shift']),(8,'euler',6))
        self.assertEqual(turbo['lora_id'],'h3-ref-turbo-lora')
        self.assertNotIn('lora_id',reference_sampling(entry('h3-ref-turbo-int8')))

    def test_new_gguf_variants_are_pinned_with_shared_companions(self):
        added=[item for item in catalog() if item['id'].startswith(('h3-ref-pruned-','h3-ref-full-'))]
        self.assertEqual(len(added),17)
        for item in added:
            self.assertEqual(len(item['revision']),40)
            self.assertEqual(len(item['files'][0]['sha256']),64)
            self.assertEqual(item['bytes'],item['files'][0]['bytes'])
            self.assertEqual(item['dependencies'],['h3-encoder','h3-video-vae','h3-audio-vae'])
            self.assertEqual(reference_sampling(item)['guidance'],1)
            with self.assertRaises(ValueError):reference_sampling(item,'turbo')

    def test_reference_and_guide_tokens_add_to_memory_budget(self):
        video=SimpleNamespace(shape=(1,24,102,64,36))
        audio=SimpleNamespace(shape=(1,32,2,575))
        image=SimpleNamespace(shape=(1,24,1,64,36))
        conditioning=[[None,{'minimax_refs':[{'latent':video,'audio_latent':audio},{'latent':image}],
                             'minimax_keyframes':[{'latent':video}]}]]
        expected=sum(prod(v.shape[1:]) for v in (video,audio,image))
        self.assertEqual(reference_latent_elements(conditioning),expected)
        class Model:
            def memory_required(self, shape, cond_shapes=None):return shape[0]*prod(shape[2:])*2
        model=Model()
        account_for_reference_memory(model,conditioning)
        self.assertEqual(model.memory_required([1,1,100]),2*(100+expected))
        self.assertEqual(model.memory_required([2,1,100]),4*(100+expected))

    def test_unreferenced_video_keeps_existing_memory_calculation(self):
        model=SimpleNamespace(memory_required=lambda shape,cond=None:42)
        self.assertEqual(account_for_reference_memory(model,[[None,{}]]),0)
        self.assertEqual(model.memory_required([1,1,100]),42)

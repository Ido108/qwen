"""Count limits must not differ between image and video request paths."""
import unittest

from backend.common import GenerationRequest, VideoRequest, ReferenceVideoRequest
from backend.ltx_settings import MultiReferenceRequest


class LoraStackRequests(unittest.TestCase):
    def test_all_request_types_accept_large_ordered_stacks(self):
        choices = [{'id': f'style-{i}', 'strength': .75,
                    'prompt_addition': f'Words for {i}.'} for i in range(100)]
        for schema, fields in [
            (GenerationRequest, {'prompt': 'A cup.'}),
            (VideoRequest, {'prompt': 'A moving cup.'}),
            (ReferenceVideoRequest, {'source_video': 'clip'}),
            (MultiReferenceRequest, {'prompt': 'A moving cup.', 'reference_images': ['photo']}),
        ]:
            with self.subTest(schema=schema.__name__):
                request = schema(**fields, loras=choices)
                self.assertEqual(request.model_dump()['loras'], choices)
                self.assertNotIn('maxItems', schema.model_json_schema()['properties']['loras'])

    def test_removing_lora_cap_keeps_image_and_strength_limits(self):
        with self.assertRaises(ValueError):
            MultiReferenceRequest(prompt='Cup.', reference_images=['photo'] * 5)
        with self.assertRaises(ValueError):
            GenerationRequest(prompt='Cup.', loras=[{'id':'style', 'strength':3}])

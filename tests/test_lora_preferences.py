import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import lora_preferences as prefs
from backend.common import GenerationRequest
from backend.image_sampling import validate


class PreferencesTests(unittest.TestCase):
    def test_saved_text_survives_import_metadata_and_queue_snapshot(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(prefs, 'PATH', Path(folder)/'settings.json'), patch('backend.models.entry', return_value={'kind':'image-lora'}):
            prefs.save('ink', '  Paper texture.  ')
            request = {'loras':[{'id':'ink', 'strength':1}, {'id':'old','prompt_addition':'Saved old text.'}]}
            prefs.snapshot(request)
            prefs.save('ink', 'New texture.')
            prefs.snapshot(request)
            self.assertEqual(request['loras'][0]['prompt_addition'], 'Paper texture.')
            self.assertEqual(request['loras'][1]['prompt_addition'], 'Saved old text.')
            self.assertEqual(prefs.read()['ink']['prompt_addition'], 'New texture.')

    def test_empty_prompt_needs_image_and_sampling_rejects_unknown_choices(self):
        self.assertEqual(GenerationRequest(prompt=' ', images=['image']).prompt, '')
        with self.assertRaises(ValueError): GenerationRequest(prompt=' ')
        validate({'sampler':'res_multistep','scheduler':'beta'}, {'family':'qwen-image-2.1'})
        validate({'sampler':'res_2m','scheduler':'beta'}, {'family':'qwen-image-2.1'})
        validate({'sampler':'res_2s','scheduler':'beta'}, {'family':'krea-2'})
        with self.assertRaises(ValueError): validate({'sampler':'unknown_custom'}, {'family':'qwen-image-2.1'})
        # A schedule override replaces Turbo's trained sigmas (image_sampling.resolve); other model types have no controls.
        validate({'scheduler':'beta'}, {'family':'qwen-image-2.1-turbo'})
        with self.assertRaises(ValueError): validate({'scheduler':'beta'}, {'family':'minimax-h3'})

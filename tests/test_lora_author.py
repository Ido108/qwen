import unittest
from backend.lora_author import profile, strength


class AuthorAdviceTests(unittest.TestCase):
    def advice(self, page, version='', **extra):
        return profile({'id': 42, 'name': 'v1 Qwen2.1', 'baseModel': 'Qwen 2.1',
                        'description': version, **extra}, {'id': 7, 'description': page})

    def test_strength_before_word_and_best_not_range_maximum(self):
        self.assertEqual(strength('Use at 0.7 strength'), .7)
        self.assertEqual(strength('Strength: Up to 2.0. 1.5 is the best. 1.0 does little.'), 1.5)
        self.assertEqual(strength('Recommended strength: 0.5–1.0, 0.75 is the best in my test.'), .75)
        self.assertIsNone(strength('Training steps: 2000. Image scale 2.'))

    def test_architecture_sections_and_legacy_qwen_are_separate(self):
        p = self.advice('--Qwen2.1\nUse at 0.7 strength\n--Krea2\nStrength: 1.2\nTrigger: other\n--Qwen\nStrength: 1.5')
        self.assertEqual(p['strength'], .7)
        self.assertEqual(p['trained_words'], [])
        self.assertNotIn('1.2', p['description'])

    def test_missing_qwen_advice_stays_unknown(self):
        p = self.advice('--Qwen2.1\nNew dataset\n--Krea2\nRecommended weight 0.6–0.8\n--Anima\nStrength: 0.4')
        self.assertIsNone(p['strength'])

    def test_exact_version_wins_and_generic_footer_is_shared(self):
        p = self.advice('👇Qwen-Image-2.1\nA restoration adapter.\n👇Krea2edit_v1\nOther setup\n🚀Suggestions\nWeight: 0.3~0.9\ntigger words : high quality')
        self.assertEqual(p['trained_words'], ['high quality'])
        self.assertEqual(p['strength'], .3)
        self.assertEqual(self.advice('Strength: .3', 'Strength: 1')['strength'], 1)

    def test_shared_declared_trigger_needs_exact_version_evidence(self):
        page = '--Qwen2.1\nUse at 0.7 strength\n--Qwen\nRecommended Trigger:\nThe only trigger word you truly need for this model is “tapeStyle”.'
        self.assertEqual(self.advice(page)['trained_words'], [])
        images = [{'meta': {'prompt': 'tapeStyle, a tree'}}, {'meta': {'prompt': 'A cup in tapeStyle'}}]
        p = self.advice(page, images=images)
        self.assertEqual(p['trained_words'], ['tapeStyle'])
        self.assertEqual(p['trigger_source'], 'description-and-version-examples')
        self.assertEqual(self.advice('No trigger words.', images=images)['trained_words'], [])

    def test_new_import_uses_same_profile_as_page(self):
        from backend.civitai import trained_words
        version = {'baseModel':'Qwen 2.1', 'trainedWords':[], 'description':'Tigger words: paperStyle'}
        self.assertEqual(trained_words(version, {}), ['paperStyle'])


class TrainedOnTests(unittest.TestCase):
    QWEN = ['qwen-image-2.1', 'qwen-image-2.1-turbo']

    def advice(self, family, examples):
        return {'workflow': {'family': family, 'steps': 50, 'cfg': 3.0, 'supported_sampler': 'res_2m',
                             'supported_scheduler': 'beta', 'examples': examples}}

    def test_author_example_strength_matches_this_file_before_single_lora_graphs(self):
        from backend.lora_author import example_strength
        graphs = [{'loras': [{'file': 'a\\lenovo_qwen21.safetensors', 'strength': .7}, {'file': 'a/nicegirls_v2_qwen21.safetensors', 'strength': .6}]},
                  {'loras': [{'file': 'step-3000.safetensors', 'strength': 1}]}]
        self.assertEqual(example_strength({'workflow': {'examples': graphs}}, 'nicegirls_v2_qwen21.safetensors'), .6)
        self.assertEqual(example_strength({'workflow': {'examples': graphs}}, 'other.safetensors'), 1)
        self.assertIsNone(example_strength({'workflow': {'examples': graphs[:1]}}, 'other.safetensors'))

    def test_standard_trained_style_lora_keeps_author_settings_and_vae(self):
        from backend.lora_author import trained_on
        advice = self.advice('qwen-image-2.1', [{'vae': ['qwen_image_2.1_vae_bf16.safetensors']}])
        found = trained_on(advice, {'ss_base_model_version': 'qwen_image_2'}, ['transformer_blocks.0.attn.to_q.lora_A.weight'],
                           'GrainScape', 'grainscape_qwen21.safetensors', self.QWEN)
        self.assertEqual(found['family'], 'qwen-image-2.1')
        self.assertEqual(found['settings'], {'steps': 50, 'cfg': 3.0, 'sampler': 'res_2m', 'scheduler': 'beta', 'vae_id': 'vae'})

    def test_speed_lora_and_turbo_names(self):
        from backend.lora_author import accelerator, trained_on
        keys = ['transformer.time_text_embed.timestep_embedder.linear_1.lora_A.weight']
        self.assertEqual(accelerator(keys, 'Accelerator', 'qwen-viggle-turbo-4step-lora.safetensors'), {'steps': 4})
        self.assertIsNone(accelerator(['blocks.0.attn.to_q.lora_A.weight'], 'Style', 'style.safetensors'))
        turbo = trained_on(None, {}, ['a.lora_A.weight'], 'Turbo Character Swap LoRA', 'swap.safetensors', self.QWEN)
        self.assertEqual(turbo['family'], 'qwen-image-2.1-turbo')
        other = trained_on(None, {'ss_base_model_version': 'krea2'}, [], 'Realistic Snapshot (Z-Image-Turbo + Krea2)', 'k.safetensors', ['krea-2', 'krea-2-turbo'])
        self.assertEqual(other, {'family': None, 'label': 'Krea 2', 'evidence': 'file header', 'settings': None})

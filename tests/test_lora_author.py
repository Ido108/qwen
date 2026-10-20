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

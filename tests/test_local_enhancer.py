from unittest.mock import patch
import tempfile
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import antigravity, models, prompt_enhancer
from backend.common import GenerationRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
import pe_core


class LocalEnhancerTests(unittest.TestCase):
    def test_enhance_is_off_unless_checked(self):
        self.assertFalse(GenerationRequest(prompt='a cat').enhance)

    def test_local_is_the_default_and_video_prompts_stay_on_a_cloud_model(self):
        self.assertEqual(antigravity.MODELS[0]['id'], 'local')
        self.assertEqual(antigravity.cloud_model('local'), antigravity.MODELS[1]['id'])
        self.assertEqual(antigravity.cloud_model('gemini-3.8-flash'), 'gemini-3.8-flash')

    def test_each_architecture_uses_its_own_official_prompt(self):
        self.assertEqual(prompt_enhancer.local_model([]), 'pe-t2i-heretic-q8')
        self.assertEqual(prompt_enhancer.local_model([object()]), 'pe-i2i-heretic-q8')
        self.assertEqual(prompt_enhancer.spec_key('z-image-turbo', [object()], redraw=True), 'z-image')
        self.assertEqual(prompt_enhancer.spec_key('krea-2-turbo', []), 'krea-2')
        self.assertEqual(prompt_enhancer.spec_key('krea-2-turbo', [object()]), 'krea-2-identity')
        self.assertEqual(prompt_enhancer.spec_key('moss-image-turbo', []), 'moss-image')
        self.assertIsNone(prompt_enhancer.local_model([], 'krea-2-turbo'))
        for key, spec in prompt_enhancer.prompts().items():
            if isinstance(spec, dict):
                self.assertTrue(spec['text'] and spec['source'], key)
        for model_id in prompt_enhancer.LOCAL_MODELS.values():
            item = models.entry(model_id)
            self.assertEqual((item['kind'], item['format']), ('image-support', 'GGUF Q8_0'))
            self.assertTrue(all(len(file['sha256']) == 64 for file in item['files']))

    def test_the_rewriter_downloads_with_the_job_only_when_enhancement_is_on(self):
        request = {'model_id': 'uc-int8-convrot', 'enhance': True, 'enhancement_model': 'local', 'images': ['a.png']}
        rewriters = {'pe-t2i-heretic-q8', 'pe-i2i-heretic-q8'}
        self.assertIn('pe-i2i-heretic-q8', models.required_image(request))
        self.assertIn('pe-t2i-heretic-q8', models.required_image({**request, 'images': []}))
        self.assertFalse(rewriters & set(models.required_image({**request, 'enhance': False})))
        # The cloud is now a per-architecture choice (Settings > Prompts), not the request's model id.
        from backend import enhancer_settings
        with tempfile.TemporaryDirectory() as folder, patch.object(enhancer_settings, 'PATH', Path(folder) / 'e.json'):
            enhancer_settings.save('qwen-image-2.1', 'cloud')
            self.assertFalse(rewriters & set(models.required_image(request)))
        self.assertFalse(rewriters & set(models.required_image({**request, 'model_id': 'krea2-turbo-nvfp4', 'images': []})))

    def test_chat_follows_each_official_layout(self):
        data = prompt_enhancer.prompts()
        chat = prompt_enhancer.messages(data['qwen-image-2.1-t2i'], 'a cat')
        self.assertEqual([m['role'] for m in chat], ['system', 'user'])
        self.assertEqual(chat[1]['content'], 'a cat')
        self.assertTrue(prompt_enhancer.messages(data['krea-2'], 'a cat')[0]['content'].endswith("User's Input:\na cat"))
        self.assertIn('a cat', prompt_enhancer.messages(data['z-image'], 'a cat')[0]['content'])
        from PIL import Image
        content = prompt_enhancer.messages(data['qwen-image-2.1-edit'], 'make it night', [Image.new('RGB', (8, 8))])[1]['content']
        self.assertEqual([part['type'] for part in content], ['image_url', 'text'])

    def test_moss_caption_json_is_unwrapped(self):
        parsed = prompt_enhancer.parse(prompt_enhancer.prompts()['moss-image'], '{"caption": "A man holds a sign reading \\"שלום\\"."}', 't2i')
        self.assertEqual(parsed['positive_prompt'], 'A man holds a sign reading "שלום".')

    def test_the_official_answer_contract_is_parsed_after_the_thinking_block(self):
        output = 'Planning the scene.\n</think>\n' + json.dumps({'rewritten_prompt': 'A red fox in snow at dawn.', 'wh_ratio': '16:9'})
        thinking, answer = pe_core.split_thinking('<think>\n' + output)
        updates = {}
        parsed = prompt_enhancer._finish(pe_core.parse_answer(answer, pe_core.get_profile('t2i')), 'a fox', lambda **v: updates.update(v))
        self.assertEqual(thinking, 'Planning the scene.')
        self.assertEqual((parsed['positive_prompt'], parsed['wh_ratio']), ('A red fox in snow at dawn.', '16:9'))
        self.assertEqual(updates['rewritten_prompt'], 'A red fox in snow at dawn.')

    def test_an_unparsable_answer_stops_the_job(self):
        with self.assertRaises(RuntimeError):
            prompt_enhancer._finish(pe_core.parse_answer('no json here', pe_core.get_profile('t2i')), 'x', lambda **_: None)


if __name__ == '__main__':
    unittest.main()


class WorkerRouteTests(unittest.TestCase):
    def test_worker_passes_the_chosen_enhancer_to_the_local_rewriter(self):
        # The encoder choice once reused the enhancer variable, so every local Enhance failed with
        # "string indices must be integers" (live job 855a71f1, 2026-10-21).
        from backend import worker
        seen = {}
        def local(prompt, images, update, **options):
            seen.update(options)
            return {'positive_prompt': 'rewritten'}
        with patch.dict(worker.request, {'model_id': 'uc-int8-convrot', 'prompt': 'a sign that says "שלום"', 'seed': 1}, clear=True), \
                patch.object(worker, 'update', lambda **values: None), \
                patch('backend.prompt_enhancer.enhance_local', side_effect=local):
            self.assertEqual(worker.rewrite_prompt([]), {'positive_prompt': 'rewritten'})
        self.assertTrue(seen['spec'] and seen['runner'])

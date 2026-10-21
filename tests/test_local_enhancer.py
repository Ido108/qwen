from unittest.mock import patch
import tempfile
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import models, prompt_enhancer
from backend.common import GenerationRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
import pe_core


class LocalEnhancerTests(unittest.TestCase):
    def test_enhance_is_off_unless_checked(self):
        self.assertFalse(GenerationRequest(prompt='a cat').enhance)

    def test_each_architecture_uses_its_own_official_prompt(self):
        self.assertEqual(prompt_enhancer.local_model([]), 'pe-t2i-heretic-q8')
        self.assertEqual(prompt_enhancer.local_model([object()]), 'pe-i2i-heretic-q8')
        self.assertEqual(prompt_enhancer.spec_key('z-image-turbo', [object()], redraw=True), 'z-image')
        self.assertEqual(prompt_enhancer.spec_key('krea-2-turbo', []), 'krea-2')
        self.assertEqual(prompt_enhancer.spec_key('krea-2-turbo', [object()]), 'krea-2-identity')
        self.assertEqual(prompt_enhancer.spec_key('moss-image-turbo', []), 'moss-image')
        # Krea 2 defaults to the fast Qwen PE rewriter with its Krea template; its own encoder stays selectable.
        self.assertEqual(prompt_enhancer.local_model([], 'krea-2-turbo'), 'pe-t2i-heretic-q8')
        for key, spec in prompt_enhancer.prompts().items():
            if isinstance(spec, dict):
                self.assertTrue(spec['text'] and spec['source'], key)
        for model_id in prompt_enhancer.LOCAL_MODELS.values():
            item = models.entry(model_id)
            self.assertEqual((item['kind'], item['format']), ('image-support', 'GGUF Q8_0'))
            self.assertTrue(all(len(file['sha256']) == 64 for file in item['files']))

    def test_the_rewriter_downloads_with_the_job_only_when_enhancement_is_on(self):
        request = {'model_id': 'uc-int8-convrot', 'enhance': True, 'images': ['a.png']}
        rewriters = {'pe-t2i-heretic-q8', 'pe-i2i-heretic-q8'}
        self.assertIn('pe-i2i-heretic-q8', models.required_image(request))
        self.assertIn('pe-t2i-heretic-q8', models.required_image({**request, 'images': []}))
        self.assertFalse(rewriters & set(models.required_image({**request, 'enhance': False})))
        self.assertTrue(rewriters & set(models.required_image({**request, 'model_id': 'krea2-turbo-nvfp4', 'images': []})))

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

    def test_the_same_request_reuses_the_saved_rewrite(self):
        from backend import worker, enhance_cache
        from PIL import Image
        calls = []
        def local(prompt, images, update, **options):
            calls.append(prompt)
            return {'positive_prompt': f'rewritten {len(calls)}', 'parse_ok': True}
        with tempfile.TemporaryDirectory() as folder, patch.object(enhance_cache, 'PATH', Path(folder) / 'cache.json'), \
                patch.object(worker, 'update', lambda **values: None), \
                patch('backend.prompt_enhancer.enhance_local', side_effect=local):
            run = lambda prompt, images: (worker.request.clear(), worker.request.update({'model_id': 'uc-int8-convrot', 'prompt': prompt, 'seed': len(calls)}),
                                          worker.rewrite_prompt(images))[-1]
            red, blue = Image.new('RGB', (8, 8), 'red'), Image.new('RGB', (8, 8), 'blue')
            self.assertEqual(run('a cat', [])['positive_prompt'], 'rewritten 1')
            self.assertEqual(run('a cat', [])['positive_prompt'], 'rewritten 1')  # another seed, same request: saved
            self.assertEqual(run('a dog', [])['positive_prompt'], 'rewritten 2')
            self.assertEqual(run('a cat', [red])['positive_prompt'], 'rewritten 3')
            self.assertEqual(run('a cat', [red])['positive_prompt'], 'rewritten 3')
            self.assertEqual(run('a cat', [blue])['positive_prompt'], 'rewritten 4')
        worker.request.clear()

    def test_video_enhance_runs_locally_and_reuses_the_saved_answer(self):
        from backend import enhance_cache, llama_runtime, video_worker
        from PIL import Image
        answers, sent = iter(['first', 'second', 'third']), []
        class Server:
            def __init__(self, model, vision=None): sent.append({'vision': bool(vision)})
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def chat(self, body):
                sent[-1].update(body)
                return {'choices': [{'message': {'content': next(answers)}}], 'usage': {}}
        request = {'frames': 121, 'audio': True}
        with tempfile.TemporaryDirectory() as folder, patch.object(enhance_cache, 'PATH', Path(folder) / 'cache.json'), \
                patch.object(llama_runtime, 'Server', Server), patch.object(prompt_enhancer, '_free_comfy_gpu', lambda: None), \
                patch.object(models, 'local_path', lambda item, file: 'x.gguf'):
            frame = {'first_frame': Image.new('RGB', (8, 8), 'red')}
            run = lambda prompt, images: video_worker.enhance_ltx_video(prompt, images, request, lambda **_: None)
            self.assertEqual(run('a dog runs', frame), 'first')
            self.assertEqual(run('a dog runs', frame), 'first')
            self.assertEqual(run('a dog runs', {'first_frame': Image.new('RGB', (8, 8), 'blue')}), 'second')
            self.assertEqual(run('a dog runs', {}), 'third')
        self.assertEqual(len(sent), 3)
        self.assertTrue(sent[0]['vision'] and not sent[2]['vision'])
        self.assertFalse(sent[0]['chat_template_kwargs']['enable_thinking'])
        self.assertIn('REFERENCE IMAGE', sent[0]['messages'][0]['content'])  # the official LTX image to video prompt
        self.assertEqual(sent[2]['messages'][1]['content'][-1]['text'], 'user prompt: a dog runs')

    def test_fast_mode_skips_thinking_and_opens_the_qwen_answer(self):
        from backend import llama_runtime
        sent = {}
        class Server:
            process = type('Process', (), {'poll': lambda self: None})()
            def __init__(self, *_): pass
            def close(self): sent['closed'] = True
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def chat(self, body):
                sent.update(body)
                return {'choices': [{'message': {'content': 'a red sign that says "שלום"", "wh_ratio": "1:1"}'}}], 'usage': {}}
        with patch.object(llama_runtime, 'Server', Server), patch.object(prompt_enhancer, '_free_comfy_gpu', lambda: None),                 patch.object(models, 'local_path', lambda item, file: 'x.gguf'):
            result = prompt_enhancer.enhance_local('שלט שכתוב עליו "שלום"', [], spec='qwen-image-2.1-t2i', runner='llama', fast=True)
        self.assertFalse(sent['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(sent['closed'])  # no image model in the GPU here, so the rewriter does not stay loaded
        self.assertEqual(sent['messages'][-1], {'role': 'assistant', 'content': prompt_enhancer.FAST_OPENING})
        self.assertEqual(result['positive_prompt'], 'a red sign that says "שלום"')

    def test_the_rewriter_stays_loaded_only_when_the_gpu_has_room(self):
        from backend import llama_runtime
        started, closed = [], []
        class Server:
            process = type('Process', (), {'poll': lambda self: None})()
            def __init__(self, *files): started.append(files)
            def close(self): closed.append(True)
            def chat(self, body):
                return {'choices': [{'message': {'content': '{"rewritten_prompt": "a cat", "wh_ratio": "1:1"}'}}], 'usage': {}}
        run = lambda: prompt_enhancer.enhance_local('חתול', [], spec='qwen-image-2.1-t2i', runner='llama', fast=True)
        with patch.object(llama_runtime, 'Server', Server), patch.object(prompt_enhancer, '_free_comfy_gpu', lambda: None), \
                patch.object(models, 'local_path', lambda item, file: 'x.gguf'), \
                patch.object(prompt_enhancer, '_image_model_loaded', lambda: True):
            with patch.object(prompt_enhancer, '_free_vram', lambda: 20 * 1024**3):
                run(); run()
            self.assertEqual((len(started), closed), (1, []))  # the second job reused the loaded rewriter
            with patch.object(prompt_enhancer, '_free_vram', lambda: 3 * 1024**3):
                run()
            self.assertEqual((len(started), closed), (1, [True]))  # too little room left: it closes after writing
        llama_runtime.release()

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend import prompt_registry, sent_prompts, enhancer_settings

APP = Path(__file__).resolve().parents[1]


class FakeHF:
    def decode(self, ids):
        return ''.join({1: '<|im_start|>system\nDescribe the image.<|im_end|>\n', 2: '<|im_start|>user\n', 3: 'A cup', 4: '<|image_pad|>'}.get(i, '?') for i in ids)


class FakeTokenizer:
    def __init__(self):
        self.qwen = type('Inner', (), {'tokenizer': FakeHF()})()


class FakeCLIP:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def tokenize(self, text, return_word_ids=False, **kwargs):
        return {'qwen': [[(1, 1.0), (2, 1.0)] + [(4, 1.0)] * 20 + [(3, 1.0)]]}


class SentPromptTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        for module, name in ((prompt_registry, 'OVERRIDES'), (enhancer_settings, 'PATH')):
            stub = patch.object(module, name, Path(folder.name) / f'{name}.json'); stub.start(); self.addCleanup(stub.stop)
        sent_prompts.clear(); sent_prompts._memory.clear()

    def test_registry_lists_every_internal_prompt_and_keeps_edits_apart(self):
        keys = {item['key'] for item in prompt_registry.listing()}
        for key in ('enhancer:qwen-image-2.1-t2i', 'enhancer:krea-2', 'enhancer:moss-image', 'flux3:t2i', 'tool:restore', 'orbit',
                    'video:enhance', 'video:reference', 'video:ltx-enhance', 'krea:grounding', 'swap'):
            self.assertIn(key, keys)
        official = prompt_registry.get('tool:restore')
        prompt_registry.save('tool:restore', 'My restore text')
        self.assertEqual(prompt_registry.get('tool:restore'), 'My restore text')
        from backend import models
        self.assertEqual(next(t for t in models.image_tools() if t['id'] == 'restore')['prompt'], 'My restore text')
        self.assertNotEqual(official, 'My restore text')  # the official file is unchanged
        prompt_registry.restore('tool:restore')
        self.assertEqual(prompt_registry.get('tool:restore'), official)
        prompt_registry.save('flux3:t2i', 'Edited BFL')
        self.assertEqual(prompt_registry.apply_flux3({'t2i': 'x', 'i2i': 'y'}), {'t2i': 'Edited BFL', 'i2i': 'y'})
        with self.assertRaises(KeyError):
            prompt_registry.save('krea:grounding', 'no')

    def test_caption_prompt_edits_reach_the_captioner(self):
        from backend import training_captions
        before = training_captions.prompt({'name': 'Set', 'trigger_word': 'tw', 'subject': 'character'})
        prompt_registry.save('caption:default:outro', 'Closing for {name}.')
        after = training_captions.prompt({'name': 'Set', 'trigger_word': 'tw', 'subject': 'character'})
        self.assertNotEqual(before, after)
        self.assertTrue(after.endswith('Closing for "Set".'))

    def test_enhancer_edits_reach_the_rewriter_prompts(self):
        from backend.prompt_enhancer import prompts
        prompt_registry.save('enhancer:krea-2', 'Rewrite for Krea: ')
        self.assertEqual(prompts()['krea-2']['text'], 'Rewrite for Krea: ')
        self.assertIn('Assume clothing covers genitals', next(i for i in prompt_registry.listing() if i['key'] == 'enhancer:krea-2')['note'])

    def test_tokenizer_capture_decodes_templates_and_folds_image_tokens(self):
        holder = type('SD', (), {'CLIP': FakeCLIP})
        sent_prompts.install(holder)
        holder.CLIP().tokenize('A cup')
        records = []
        sent_prompts.flush(lambda **values: records.append(values['sent_prompt']), key=('m', 'A cup'))
        self.assertEqual(len(records), 1)
        self.assertIn('Describe the image.', records[0]['text'])
        self.assertIn('<|image_pad|> x 20', records[0]['text'])
        self.assertTrue(records[0]['text'].endswith('A cup'))
        reused = []
        sent_prompts.flush(lambda **values: reused.append(values['sent_prompt']), key=('m', 'A cup'))
        self.assertIn('reused', reused[0]['label'])

    def test_enhancer_routing_defaults_to_the_official_one_and_follows_settings(self):
        route = enhancer_settings.route
        self.assertEqual((route('qwen-image-2.1', [], False)['spec'], route('qwen-image-2.1', ['p'], False)['spec']),
                         ('qwen-image-2.1-t2i', 'qwen-image-2.1-edit'))
        self.assertEqual((route('krea-2-turbo')['runner'], route('moss-image')['spec'], route('z-image-turbo', [], True)['spec']),
                         ('llama', 'moss-image', 'z-image'))
        self.assertTrue(route('qwen-image-2.1')['fast'] and route('krea-2-turbo')['fast'])
        self.assertEqual(route('flux3-klein')['mode'], 'none')
        self.assertTrue(all(route(f)['official'] for f in ('qwen-image-2.1', 'z-image', 'moss-image')))
        enhancer_settings.save('krea-2', 'qwen-pe')
        self.assertEqual((route('krea-2')['runner'], route('krea-2')['official']), ('llama', False))
        enhancer_settings.save('moss-image', 'cloud')
        self.assertEqual(route('moss-image')['mode'], 'cloud')
        with self.assertRaises(ValueError):
            enhancer_settings.save('qwen-image-2.1', 'moss-encoder')

    def run_image(self, request, tool=None):
        from PIL import Image
        from backend import worker
        def render(request, prompt, images, width, height, progress):
            progress(sent_prompt={'label': 'Engine record', 'text': 'engine ' + prompt})
            return Image.new('RGBA', (32, 32), 'white')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); job = root / 'job'; job.mkdir()
            base = {'model_id': 'model', 'prompt': 'A cup', 'images': [], 'enhance': False, 'negative_prompt': '', 'output_dir': folder,
                    'custom_size': True, 'width': 32, 'height': 32, 'seed': 1, 'loras': []}
            (job / 'request.json').write_text(json.dumps({**base, **request}))
            (job / 'status.json').write_text(json.dumps({'id': 'sent-test', 'status': 'queued'}))
            item = {'title': 'CPU stand-in', 'family': request.pop('_family', 'qwen-image-2.1'), 'revision': 'test', 'dependencies': []}
            with patch.object(worker, 'image_engine', return_value=render), patch('backend.models.entry', return_value=item), \
                    patch('backend.models.image_tool', return_value=tool or {}), patch('torch.cuda.is_available', return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.execute(job), 0)
            return json.loads((job / 'status.json').read_text())['sent_prompts']

    def test_image_routes_record_what_they_sent(self):
        labels = lambda records: [r['label'] for r in records]
        plain = self.run_image({})
        self.assertIn('Prompt sent to the image model (after enhancement, LoRA words and tool text)', labels(plain))
        self.assertIn({'label': 'Engine record', 'text': 'engine A cup'}, plain)
        swap = self.run_image({'swap_prompt_template': 'Put {person} into {scene}'})
        self.assertEqual(swap[0], {'label': 'Swap template', 'text': 'Put {person} into {scene}'})
        moss = self.run_image({'_family': 'moss-image'})
        self.assertIn('Engine record', labels(moss))

    def test_every_engine_and_worker_route_records_its_prompts(self):
        for name in ('comfy_engine', 'krea_engine', 'zimage_engine', 'flux3_engine', 'moss_engine', 'video_engine', 'ltx_engine'):
            self.assertIn('sent_prompts.install(comfy.sd)', (APP / 'backend' / f'{name}.py').read_text(encoding='utf-8'), name)
        flux3 = (APP / 'backend/flux3_engine.py').read_text(encoding='utf-8')
        self.assertIn('FLUX.3 prompt model (BFL upsampler): full input', flux3)
        for name in ('video_worker', 'ltx_worker'):
            source = (APP / 'backend' / f'{name}.py').read_text(encoding='utf-8')
            self.assertIn("sent_prompt={'label': 'Prompt sent to the video model", source)
            self.assertIn('sent_prompts.flush(update)', source)
        enhancer = (APP / 'backend/prompt_enhancer.py').read_text(encoding='utf-8')
        self.assertGreaterEqual(enhancer.count('_record(update'), 4)
        self.assertNotIn('sent_prompt', (APP / 'backend/upscale_worker.py').read_text(encoding='utf-8'))  # upscalers send no prompt

    def test_rewriter_messages_are_recorded_in_full(self):
        from backend.prompt_enhancer import _record, messages
        records = []
        spec = {'role': 'system', 'text': 'Official system prompt'}
        _record(lambda **v: records.append(v['sent_prompt']), 'Qwen PE', messages(spec, 'a cup'))
        self.assertEqual([r['text'] for r in records], ['Official system prompt', 'a cup'])
        self.assertTrue(records[0]['label'].startswith('Enhance with Qwen PE'))


if __name__ == '__main__':
    unittest.main()

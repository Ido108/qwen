import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from backend.lora_prompts import internal_prompt, description_words


class LoraPromptTests(unittest.TestCase):
    def setUp(self):
        self.items = {
            'style': {'title': 'Ink style', 'trained_words': ['inkStyle', 'fine lines']},
            'other': {'title': 'Other', 'trained_words': ['inkStyle', 'softLight']},
            'none': {'title': 'No trigger', 'trained_words': []},
        }
        self.lookup = patch('backend.models.entry', side_effect=lambda id:self.items[id])
        self.lookup.start(); self.addCleanup(self.lookup.stop)

    def test_each_run_gets_words_without_mutating_user_text_or_request(self):
        request = {'prompt': 'A cup', 'loras': [{'id': 'style', 'strength': 0.6}]}
        for _ in range(2):
            prompt, records = internal_prompt(request['prompt'], request)
            self.assertEqual(prompt, 'A cup, inkStyle, fine lines')
            self.assertEqual(records[0]['added'], ['inkStyle', 'fine lines'])
        self.assertEqual(request['prompt'], 'A cup')
        self.assertEqual(internal_prompt(prompt, request)[0], prompt)

    def test_bypass_zero_and_none_do_not_add_words(self):
        request = {'loras': [{'id':'style', 'enabled':False}, {'id':'other', 'strength':0}, {'id':'none'}]}
        self.assertEqual(internal_prompt('A cup', request), ('A cup', []))
        request['loras'][0]['enabled'] = True
        self.assertIn('inkStyle', internal_prompt('A cup', request)[0])

    def test_custom_additions_stack_and_queued_empty_snapshot_wins(self):
        self.items['style']['prompt_addition'] = 'Rough paper texture.'
        self.items['other']['prompt_addition'] = 'Soft evening light.'
        request = {'loras': [{'id': 'style'}, {'id': 'other'}]}
        prompt, records = internal_prompt('A cup', request)
        self.assertIn('Rough paper texture.', prompt)
        self.assertIn('Soft evening light.', prompt)
        self.assertEqual(internal_prompt(prompt, request)[0], prompt)
        request['loras'] = [{'id':'style', 'prompt_addition':''}, {'id':'other', 'strength':0}]
        self.assertNotIn('texture', internal_prompt('', request)[0])
        request['loras'][0]['enabled'] = False
        self.assertEqual(internal_prompt('A cup', request), ('A cup', []))

    def test_custom_additions_expand_friendly_image_mentions(self):
        self.items['model'] = {'family': 'qwen-image-2.1'}
        request = {'model_id':'model', 'images':['a'], 'image_labels':[], 'loras':[{'id':'none', 'prompt_addition':'Use @Image1 in pencil.'}]}
        # No label: the plain picture reference; a label names what the mention selects.
        self.assertEqual(internal_prompt('', request)[0], 'Use <image1> in pencil.')
        request['image_labels'] = ['woman']
        self.assertEqual(internal_prompt('', request)[0], 'Use the woman from <image1> in pencil.')

    def test_deduplicate_whole_phrases_and_shared_words(self):
        request={'loras':[{'id':'style'}, {'id':'other'}]}
        prompt, records=internal_prompt('INKSTYLE. Fine   lines on a cup.', request)
        self.assertEqual(prompt, 'INKSTYLE. Fine   lines on a cup. softLight')
        self.assertEqual(records[0]['added'], [])
        self.items['style']['trained_words']=['art']
        self.assertEqual(internal_prompt('A cart', {'loras':[{'id':'style'}]})[0], 'A cart, art')

    def test_header_fallback_reads_only_explicit_trigger_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.safetensors'
            metadata={'trigger_word':'paperStyle','ss_tag_frequency':json.dumps({'private-caption':{'invented-tag':80}})}
            head=json.dumps({'__metadata__':metadata}).encode()
            path.write_bytes(struct.pack('<Q',len(head))+head+b'\0')
            self.items['none']['files']=[{'path':'weights.safetensors'}]
            with patch('backend.models.local_path',return_value=path):
                prompt,records=internal_prompt('A cup',{'loras':[{'id':'none'}]})
            self.assertEqual(prompt,'A cup, paperStyle')
            self.assertEqual(records[0]['words'],['paperStyle'])

    def test_author_labels_and_import_precedence(self):
        from backend.civitai import trained_words
        self.assertEqual(description_words('Trigger words: `paperStyle`, fine lines\nExample: a cat'),['paperStyle','fine lines'])
        self.assertEqual(description_words('Trigger word: Not required'),[])
        self.assertEqual(description_words('A beautiful style with many examples'),[])
        self.assertEqual(description_words('• Trigger word :- paperStyle'),['paperStyle'])
        self.assertEqual(trained_words({'trainedWords':['explicit'],'description':'Trigger word: older'},{}),['explicit'])
        self.assertEqual(trained_words({'description':'<p>Trigger word: paperStyle</p>'},{}),['paperStyle'])

    def test_prompt_changes_invalidate_conditioning_and_triggerless_lora_reuses_text(self):
        from backend.image_cache import ConditioningCache
        cache=ConditioningCache()
        base='A cup'
        enabled=internal_prompt(base,{'loras':[{'id':'style'}]})[0]
        disabled=internal_prompt(base,{'loras':[]})[0]
        self.assertNotEqual(cache.fingerprint(base,'',[]),cache.fingerprint(enabled,'',[]))
        self.assertEqual(cache.fingerprint(base,'',[]),cache.fingerprint(disabled,'',[]))

    def test_worker_sends_and_saves_internal_prompt_after_enhancement(self):
        import contextlib,io
        from PIL import Image
        from backend import worker
        self.items['model']={'title':'CPU stand-in','family':'qwen-image-2.1','revision':'test'}
        received=[]
        def render(request,prompt,images,width,height,progress):
            received.append(prompt)
            return Image.new('RGBA',(32,32),'white')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            request={'model_id':'model','prompt':'A cup','images':[],'enhance':True,'negative_prompt':'',
                'output_dir':folder,'custom_size':True,'width':32,'height':32,'seed':1,'loras':[{'id':'style','strength':.6}],
                'enhancement_result':{'positive_prompt':'A red cup on a table.'}}
            job=root/'job';job.mkdir()
            (job/'request.json').write_text(json.dumps(request))
            (job/'status.json').write_text(json.dumps({'id':'cpu-lora-test','status':'queued'}))
            with patch.object(worker,'image_engine',return_value=render),patch('torch.cuda.is_available',return_value=True),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.execute(job),0)
            saved=json.loads((job/'status.json').read_text())
            self.assertEqual(received,['A red cup on a table. inkStyle, fine lines'])
            self.assertEqual(saved['actual_prompt'],received[0])
            self.assertEqual(saved['lora_prompt_words'][0]['added'],['inkStyle','fine lines'])
            self.assertEqual(json.loads((job/'request.json').read_text())['prompt'],'A cup')
            with Image.open(saved['output_path']) as picture:
                self.assertEqual(picture.info['Prompt'],'A cup')
                self.assertEqual(picture.info['Actual prompt'],received[0])

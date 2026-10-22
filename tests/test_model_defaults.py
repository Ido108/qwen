import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import model_defaults, models, swap_prompts, image_people


class ModelDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'defaults.json'
        change = patch.object(model_defaults, 'PATH', self.path)
        change.start(); self.addCleanup(change.stop)

    def test_persistent_model_and_workflow_isolation(self):
        original = models.entry('turbo-bf16')['generation_defaults'].copy()
        model_defaults.save('turbo-bf16','image','Steps',{'steps':12})
        model_defaults.save('turbo-bf16','swap-head-swap','Effect strength',{'strength':1.4})
        model_defaults.save('official-bf16','image','Steps',{'steps':30})
        stored = json.loads(self.path.read_text())
        self.assertEqual(stored['turbo-bf16']['image']['Steps'], {'steps':12})
        self.assertEqual(stored['official-bf16']['image']['Steps'], {'steps':30})
        self.assertEqual(models.entry('turbo-bf16')['generation_defaults'], original)
        self.assertEqual(model_defaults.read(),stored)
        model_defaults.save('turbo-bf16','image','Steps',None)
        self.assertEqual(model_defaults.read()['turbo-bf16']['image'],{})
        self.assertIn('Effect strength',model_defaults.read()['turbo-bf16']['swap-head-swap'])

    def test_latest_save_wins_for_coupled_fields(self):
        model_defaults.save('turbo-bf16','image','Output size',{'width':1024,'height':1024,'custom_size':True})
        model_defaults.save('turbo-bf16','image','Width',{'width':768})
        model_defaults.save('turbo-bf16','image','Output size',{'width':896,'height':1152,'custom_size':True})
        merged={}
        for value in model_defaults.read()['turbo-bf16']['image'].values():merged.update(value)
        self.assertEqual(merged['width'],896)

    def test_reject_invalid_values_without_writing(self):
        for value in ({'images':['private']},{'seed':-1},{'cfg':float('nan')},{'steps':0},{'width':'bad'},{}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model_defaults.save('turbo-bf16','image','Steps',value)
        with self.assertRaises(ValueError):model_defaults.save('missing','image','Steps',{'steps':8})
        with self.assertRaises(ValueError):model_defaults.save('turbo-bf16','bad','Steps',{'steps':8})
        self.assertFalse(self.path.exists())

    def test_training_defaults_use_architecture(self):
        model_defaults.save('qwen-image-2.1-turbo','training','Rank',{'rank':32,'settings_mode':'custom'})
        self.assertIn('Rank',model_defaults.read()['qwen-image-2.1-turbo']['training'])

    def test_http_save_reload_and_remove(self):
        from fastapi.testclient import TestClient
        from backend import server
        client=TestClient(server.app)
        body={'scope':'swap-head-swap','field':'Effect strength','value':{'strength':1.35}}
        headers={'X-Studio-Token':server.SESSION_TOKEN}
        response=client.put('/api/model-defaults/turbo-bf16',json=body,headers=headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(client.get('/api/model-defaults').json()['turbo-bf16']['swap-head-swap']['Effect strength'],body['value'])
        response=client.put('/api/model-defaults/turbo-bf16',json={**body,'value':None},headers=headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json(),{})
        response=client.put('/api/model-defaults/turbo-bf16',json={**body,'value':{'images':['bad']}},headers=headers)
        self.assertEqual(response.status_code,400,response.text)

    def test_style_mode_keeps_complete_shared_prompt(self):
        faces={'revision':'a'*24,'faces':[{'id':'face-1','box':[.1,.1,.2,.2]}]}
        selection=dict(scene_image='s',reference_image='r',scene_revision='a'*24,reference_revision='a'*24,assignments={'face-1':'face-1'})
        for tool_id in ('head-swap','head-swap-expression','body-swap'):
            tool=models.image_tool(tool_id)
            for mode in ('realistic','non-realistic'):
                with patch.object(image_people,'analyze_image',return_value=faces):
                    prompt=image_people.build_swap_prompt(tool,selection,['s','r'],Path,style_mode=mode,template=swap_prompts.default_prompt(tool))
                base=swap_prompts.default_prompt(tool).replace('<pose_source>','<image1>').replace('<style_source>','<image1>')
                self.assertTrue(prompt.startswith(base))
                for phrase in ('exact head angle','contrast','blends into the image','occlusion','micro-expressions'):
                    self.assertIn(phrase,prompt)
                self.assertEqual('same original artwork' in prompt,mode=='non-realistic')


if __name__=='__main__':unittest.main()

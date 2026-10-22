"""Real HTTP validation/queue payloads, with scheduling disabled; no CUDA work."""
import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend import server, swap_prompts, image_people, models


class PromptApiTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(swap_prompts,'PATH',self.root/'prompts.json'))
        self.stack.enter_context(patch.object(server.models,'required',return_value=[]))
        self.stack.enter_context(patch.object(server.models,'required_video',return_value=[]))
        self.stack.enter_context(patch.object(server,'required_ltx',return_value=[]))
        self.stack.enter_context(patch.object(server.cloud,'settings',return_value={'mode':'local'}))
        self.stack.enter_context(patch.object(server,'resolve_image',side_effect=lambda id:self.root/id))
        self.scan={'revision':'a'*24,'faces':[{'id':'face-1','box':[.1,.1,.2,.2]}]}
        self.stack.enter_context(patch.object(image_people,'analyze_image',return_value=self.scan))
        self.queued=[]
        self.stack.enter_context(patch.object(server,'enqueue_locked',side_effect=lambda job,data:self.queued.append(copy.deepcopy(data))))
        self.client=TestClient(server.app)
        self.headers={'X-Studio-Token':server.SESSION_TOKEN}
        self.body=dict(prompt='Use @Image1.',model_id='turbo-gguf-q8_0',images=['scene','head'],enhance=False,seed=433848671,randomize_seed=False,output_dir=str(self.root))

    def post(self,path,body):return self.client.post(path,json=body,headers=self.headers)

    def test_swap_snapshot_and_seed_survive_later_prompt_edits(self):
        initial=self.client.get('/api/swap-prompts/head-swap').json()
        body={**self.body,'tool_id':'head-swap','swap_extra':'Keep the angle of <image 1>.','swap_selection':dict(scene_image='scene',reference_image='head',scene_revision='a'*24,reference_revision='a'*24,assignments={'face-1':'face-1'})}
        response=self.post('/api/generate',body)
        self.assertEqual(response.status_code,202,response.text)
        job=self.queued[0]
        self.assertEqual(job['seed'],433848671)
        self.assertFalse(job['randomize_seed'])
        self.assertEqual(job['swap_prompt_template'],initial['text'])
        self.assertEqual(job['swap_prompt_revision'],initial['current'])
        self.assertIn('angle of <image1>',job['prompt'])
        updated=self.client.put('/api/swap-prompts/head-swap',headers=self.headers,json=dict(expected=initial['current'],text=initial['text']+' Keep the grain.'))
        self.assertEqual(updated.status_code,200,updated.text)
        self.assertNotIn('Keep the grain.',job['prompt'])
        self.assertEqual(self.post('/api/generate',{**body,'swap_prompt_template':job['swap_prompt_template'],'swap_prompt_revision':job['swap_prompt_revision']}).status_code,202)
        self.assertEqual(self.queued[1]['prompt'],job['prompt'])
        self.assertEqual(self.client.put('/api/swap-prompts/head-swap',headers=self.headers,json=dict(expected=initial['current'],text=initial['text'])).status_code,409)
        preview=self.post('/api/swap-prompts/head-swap/preview',{**body,'swap_prompt_template':initial['text']})
        self.assertEqual(preview.status_code,200,preview.text)
        self.assertEqual(preview.json()['prompt'],job['prompt'])

    def test_job_plan_matches_the_queued_swap(self):
        """The page shows /api/job-plan; the queued job must carry the same steps, CFG, sampler and LoRA choice."""
        swap=dict(scene_image='scene',reference_image='head',scene_revision='a'*24,reference_revision='a'*24,assignments={'face-1':'2'})
        for model_id,speed,expected in (('uc-int8-convrot',None,(8,1.0,'deis_2m','simple',True)),
                                        ('uc-int8-convrot',False,(40,None,None,None,False)),
                                        ('turbo-int8-convrot',None,(8,None,None,None,False)),
                                        ('krea2-turbo-fp8',None,(8,1.0,'euler','simple',False))):
            tool='krea-head-swap' if model_id.startswith('krea') else 'head-swap'
            body={**self.body,'model_id':model_id,'tool_id':tool,'speed_lora':speed,'swap_selection':None if tool.startswith('krea') else swap}
            if tool.startswith('krea'):body['prompt']='head_swap: replace the head with the reference head.'
            plan=self.post('/api/job-plan',body).json()
            self.queued.clear()
            self.assertEqual(self.post('/api/generate',body).status_code,202)
            job=self.queued[0]
            self.assertEqual((job['steps'],job['cfg'],job['sampler'],job['scheduler'],plan['speed_lora'] is not None and plan['speed_lora']['on']),expected,model_id)
            self.assertEqual(job['steps'],plan['steps'],model_id)
            if job['cfg'] is not None:self.assertEqual(job['cfg'],plan['cfg'])
            if job['sampler'] is not None:self.assertEqual(job['sampler'],plan['sampler'])

    def test_all_image_mentions_resolve_before_the_queue(self):
        response=self.post('/api/generate',{**self.body,'prompt':'Use @Image2 in <image 1>.','negative_prompt':'Do not copy @Image2 background.'})
        self.assertEqual(response.status_code,202,response.text)
        self.assertEqual(self.queued[0]['prompt'],'Use the person from <image2> in <image1>.')
        self.assertEqual(self.queued[0]['negative_prompt'],'Do not copy the person from <image2> background.')
        self.assertEqual(self.post('/api/generate',{**self.body,'prompt':'Use @Image3.'}).status_code,400)
        self.assertEqual(len(self.queued),1)

    def test_non_realistic_swap_applies_source_style_and_editable_negative(self):
        selection=dict(scene_image='scene',reference_image='head',scene_revision='a'*24,reference_revision='a'*24,assignments={'face-1':'face-1'})
        from backend import models, swap_prompts
        # The app's scene template (the author's default prompt has no style placeholder).
        body={**self.body,'tool_id':'head-swap','swap_selection':selection,'swap_style_source':2,
              'swap_style_mode':'non-realistic','negative_prompt':'photographic skin, texture from @Image2',
              'swap_prompt_template':swap_prompts.default_prompt(models.image_tool('head-swap'))}
        preview=self.post('/api/swap-prompts/head-swap/preview',body)
        response=self.post('/api/generate',body)
        self.assertEqual(response.status_code,202,response.text)
        job=self.queued[-1]
        self.assertEqual(preview.json()['prompt'],job['prompt'])
        self.assertIn('exact visual style of <image1>',job['prompt'])
        self.assertIn('line weight, edge softness',job['prompt'])
        self.assertIn('same original artwork',job['prompt'])
        self.assertEqual(job['negative_prompt'],'photographic skin, texture from the person from <image2>')
        self.assertEqual(job['source_negative_prompt'],body['negative_prompt'])
        self.assertEqual(job['swap_style_mode'],'non-realistic')
        self.assertEqual(job['seed'],self.body['seed'])
        self.post('/api/generate',{**body,'negative_prompt':''})
        self.assertEqual(self.queued[-1]['negative_prompt'],'')
        self.post('/api/generate',{**body,'swap_style_mode':'realistic'})
        self.assertNotIn('same original artwork',self.queued[-1]['prompt'])
        self.assertIn('exact visual style of <image2>',self.queued[-1]['prompt'])
        self.assertEqual(self.post('/api/generate',{**body,'swap_style_mode':'invalid'}).status_code,422)

    def test_video_and_multi_reference_keep_their_own_notation(self):
        response=self.post('/api/video/generate',dict(prompt='Start at @FirstFrame.',model_id='h3-int8',mode='image',first_frame='scene',output_dir=str(self.root)))
        self.assertEqual(response.status_code,202,response.text)
        self.assertEqual(self.queued[-1]['prompt'],'Start at the first frame.')
        for model,frames in [('ltx25-msr-q4_k_m',121),('h3-ref-base-int8',124)]:
            response=self.post('/api/multi-reference/generate',dict(prompt='@Image1 walks through @Image2.',model_id=model,reference_images=['scene'],background='head',frames=frames,output_dir=str(self.root)))
            self.assertEqual(response.status_code,202,response.text)
            self.assertEqual(self.queued[-1]['prompt'],'the person from Image 1 walks through the background from Image 2.')
        from backend.ltx_settings import ref2va_prompt
        self.assertIn('the person from <Picture 1> walks through the background from <Picture 2>.',ref2va_prompt(self.queued[-1]))

    def test_subject_labels_and_original_prompt_survive_queue_snapshot(self):
        text='@Image1 holds @Image2. Keep the person from @Image1 beside <image 2>.'
        response=self.post('/api/generate',{**self.body,'prompt':text,'image_labels':['person','red_car']})
        self.assertEqual(response.status_code,202,response.text)
        job=self.queued[-1]
        self.assertEqual(job['prompt'],'the person from <image1> holds the red car from <image2>. Keep the person from <image1> beside <image2>.')
        self.assertEqual(job['source_prompt'],text)
        self.assertEqual(job['image_labels'],['person','red car'])
        self.assertEqual(self.post('/api/generate',{**self.body,'image_labels':['person']*11}).status_code,422)

    def test_swap_extra_expands_subject_tags_but_keeps_template_image_references(self):
        selection=dict(scene_image='scene',reference_image='head',scene_revision='a'*24,reference_revision='a'*24,assignments={'face-1':'face-1'})
        body={**self.body,'tool_id':'head-swap','swap_selection':selection,'swap_extra':'Keep @Image1 beside @Image2.','image_labels':['person','statue']}
        preview=self.post('/api/swap-prompts/head-swap/preview',body)
        queued=self.post('/api/generate',body)
        self.assertEqual(preview.status_code,200,preview.text)
        self.assertEqual(queued.status_code,202,queued.text)
        self.assertEqual(preview.json()['prompt'],self.queued[-1]['prompt'])
        self.assertIn('head_swap: start with <image1> as the base image, keeping its lighting',self.queued[-1]['prompt'])
        self.assertIn('Keep the person from <image1> beside the statue from <image2>.',self.queued[-1]['prompt'])

    def test_custom_labels_reach_both_reference_video_families(self):
        for model,frames in [('ltx25-msr-q4_k_m',121),('h3-ref-base-int8',124)]:
            response=self.post('/api/multi-reference/generate',dict(prompt='@Image1 moves across @Image2.',model_id=model,reference_images=['scene'],background='head',image_labels=['dog','background'],frames=frames,output_dir=str(self.root)))
            self.assertEqual(response.status_code,202,response.text)
            self.assertEqual(self.queued[-1]['prompt'],'the dog from Image 1 moves across the background from Image 2.')
            self.assertEqual(self.queued[-1]['source_prompt'],'@Image1 moves across @Image2.')


if __name__=='__main__':unittest.main()

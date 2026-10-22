import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import models, swap_prompts, image_people
from backend.prompt_mentions import image_mentions, frame_mentions


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'swap-prompts.json'
        self.patch = patch.object(swap_prompts, 'PATH', self.path)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.tool = models.image_tool('head-swap')

    def test_history_survives_reload_and_restore_does_not_erase_versions(self):
        first = swap_prompts.get(self.tool)
        text = first['text'] + ' Keep the original image grain.'
        saved = swap_prompts.save(self.tool, text, first['current'])
        self.assertEqual(swap_prompts.get(self.tool)['text'], text)
        restored = swap_prompts.save(self.tool, '', saved['current'], first['current'])
        self.assertEqual(restored['text'], first['text'])
        self.assertEqual(len(restored['versions']),len(first['versions'])+2)
        self.assertTrue(any(v['text']==text for v in restored['versions']))
        self.assertEqual(restored['versions'][0]['restored_from'],first['current'])
        self.assertEqual(swap_prompts.get(models.image_tool('body-swap'))['current'],'author-v1')

    def test_stale_screen_cannot_overwrite_a_new_version(self):
        first=swap_prompts.get(self.tool)
        swap_prompts.save(self.tool,first['text']+' Keep the earrings.',first['current'])
        with self.assertRaises(swap_prompts.PromptConflict):
            swap_prompts.save(self.tool,first['text']+' Keep the hat.',first['current'])
        self.assertTrue(swap_prompts.get(self.tool)['text'].endswith('Keep the earrings.'))

    def test_custom_prompt_reaches_compiler_with_pose_and_style_controls(self):
        text='head_swap: Use @Image1 as the scene and @Image2 for identity. Keep the texture of @StyleSource and the angle of @PoseSource.'
        saved=swap_prompts.save(self.tool,text,swap_prompts.get(self.tool)['current'])
        faces={'revision':'a'*24,'faces':[{'id':'face-1','box':[.1,.1,.2,.2]}]}
        selection=dict(scene_image='s',reference_image='r',scene_revision='a'*24,reference_revision='a'*24,assignments={'face-1':'face-1'})
        with patch.object(image_people,'analyze_image',return_value=faces):
            prompt=image_people.build_swap_prompt(self.tool,selection,['s','r'],Path,'Keep the ear from @Image1.',2,1,saved['text'])
        self.assertIn('texture of <image1> and the angle of <image2>',prompt)
        self.assertIn('ear from <image1>',prompt)
        self.assertNotIn('@',prompt)
        with self.assertRaises(ValueError): swap_prompts.validate(self.tool,'Replace both people.')

    def test_positive_style_default_keeps_prior_prompt_and_user_edits(self):
        for tool_id in ('head-swap','head-swap-expression','body-swap'):
            tool=models.image_tool(tool_id)
            state=swap_prompts.get(tool)
            # The author's prompt is the default; the app's scene template stays in the history.
            self.assertEqual(state['current'],'author-v1')
            self.assertEqual(state['text'],tool['prompt'])
            scene=next(v['text'] for v in state['versions'] if v['id']=='scene-v4')
            self.assertEqual(scene,swap_prompts.validate(tool,scene))
            self.assertIn('exact visual style of <style_source>',scene)
            self.assertIn('contrast',scene)
            self.assertIn('exact head angle',scene)
            self.assertIn('blends into the image',scene)
            self.assertIn('occlusion',scene)
            self.assertTrue(any(v['id']=='style-v3' for v in state['versions']))
            self.assertNotRegex(scene.lower(),r"\b(?:not|no|without|avoid|don't|photorealistic|cartoon|anime|illustration)\b")
            old=next(v for v in state['versions'] if v['id']=='scene-v2')
            self.assertEqual(old['text'],swap_prompts._scene_prompt_v2(tool))
        custom='head_swap: Use <image1> for the scene and <image2> for identity. Keep the image grain.'
        saved=swap_prompts.save(self.tool,custom,'author-v1')
        self.assertEqual(swap_prompts.get(self.tool)['text'],custom)
        restored=swap_prompts.save(self.tool,'',saved['current'],'scene-v2')
        self.assertEqual(restored['text'],swap_prompts._scene_prompt_v2(self.tool))

    def test_image_notation_validation_and_normal_text(self):
        text='Keep @Image1; use <image 2> and <Picture 3>. name@example.com @unrelated'
        self.assertEqual(image_mentions(text,3),'Keep <image1>; use <image2> and <image3>. name@example.com @unrelated')
        self.assertEqual(image_mentions('@Image1 with @Image2',2,'plain'),'Image 1 with Image 2')
        for text in ('@Image0','@Image3','<image 10>'):
            with self.subTest(text=text),self.assertRaises(ValueError): image_mentions(text,2)
        self.assertEqual(frame_mentions('Start at @FirstFrame and end at @LastFrame.',True,True),'Start at the first frame and end at the last frame.')
        with self.assertRaises(ValueError): frame_mentions('@LastFrame',True,False)


if __name__=='__main__': unittest.main()

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import image_people, models


def scan(count):
    return {'revision': 'a'*24, 'faces': [
        {'id': f'face-{i+1}', 'box': [.05+i*.3, .2, .15, .25], 'preview': ''}
        for i in range(count)]}


class ImagePeopleTests(unittest.TestCase):
    def test_large_cropped_face_retries_at_detector_scale(self):
        import numpy as np
        from unittest.mock import Mock
        detector=Mock()
        detector.detect.side_effect=[(None,None),(None,np.array([[138,0,186,340]]))]
        with tempfile.TemporaryDirectory() as folder,patch('backend.reference_people.face_detector',return_value=detector):
            path=Path(folder)/'closeup.png';Image.new('RGB',(500,500)).save(path)
            found=image_people.analyze_image(path)
            self.assertEqual(len(found['faces']),1)
            self.assertEqual(found['faces'][0]['box'],[.43125,0,.56875,1])
            self.assertEqual(detector.setInputSize.call_args.args,((320,320),))

    def setUp(self):
        self.scene, self.reference = scan(3), scan(2)
        self.selection = {'scene_image': 'scene.png', 'reference_image': 'reference.png',
                          'scene_revision': 'a'*24, 'reference_revision': 'a'*24,
                          'assignments': {'face-1': 'face-2', 'face-2': None, 'face-3': 'face-1'}}

    def prompt(self, selection=None, tool='head-swap'):
        with patch.object(image_people, 'analyze_image', side_effect=[self.scene, self.reference]):
            return image_people.build_swap_prompt(models.image_tool(tool), selection or self.selection,
                                                  ['scene.png', 'reference.png'], Path, 'Keep the earrings.')

    def test_crossed_choices_preserve_the_unselected_person(self):
        text = self.prompt()
        self.assertTrue(text.startswith('head_swap:'))
        self.assertIn('person 1 in <image1> on the left', text)
        self.assertIn('person 2 in <image2> on the center', text)
        self.assertIn('Keep person 2 in <image1>', text)
        self.assertIn('person 3 in <image1> on the right', text)
        self.assertIn('person 1 in <image2> on the left', text)
        self.assertIn('x=65.0% to 80.0%', text)
        self.assertIn('take priority over conflicting instructions', text)
        self.assertIn('Keep the earrings.', text)
        body = self.prompt(tool='body-swap')
        self.assertTrue(body.startswith('body_swap:'))
        self.assertIn('whole person (face, clothing, body shape and proportions)', body)

    def test_invalid_or_stale_choices_never_fall_back_to_a_random_person(self):
        for update in ({'scene_image': 'different.png'}, {'reference_revision': 'b'*24},
                       {'assignments': {'face-1': None}}, {'assignments': {'face-9': 'face-1'}},
                       {'assignments': {'face-1': 'face-9'}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.prompt({**self.selection, **update})
        self.scene['faces'] = []
        with self.assertRaisesRegex(ValueError, 'No faces'):
            self.prompt()

    def test_pose_and_style_choices_reach_the_authoritative_prompt(self):
        with patch.object(image_people,'analyze_image',side_effect=[self.scene,self.reference]):
            prompt=image_people.build_swap_prompt(models.image_tool('head-swap-expression'),self.selection,
                ['scene.png','reference.png'],Path,'Keep the earrings.',pose_source=2,style_source=1)
        self.assertIn('expression of person 2 in <image2>',prompt)
        self.assertIn('exact visual style of <image1>',prompt)
        self.assertIn('background and text of <image1>',prompt)

    def test_single_face_prompt_preserves_style_without_forced_sharpening(self):
        self.scene, self.reference = scan(1), scan(1)
        prompt=self.prompt({**self.selection,'assignments':{'face-1':'face-1'}})
        self.assertNotIn('coordinates',prompt)
        self.assertNotIn('x=',prompt)
        self.assertNotIn('4k',prompt.lower())
        self.assertNotIn('sharp details',prompt.lower())
        for instruction in ('exact framing','camera perspective','visual style of <image1>','depth and occlusion','facial features','<image2>'):
            self.assertIn(instruction,prompt)

    def test_style_adaptation_follows_the_chosen_image_for_every_swap(self):
        for tool_id in ('head-swap','head-swap-expression','body-swap'):
            for style_source in (1,2):
                with self.subTest(tool=tool_id,style=style_source),patch.object(image_people,'analyze_image',side_effect=[scan(1),scan(1)]):
                    prompt=image_people.build_swap_prompt(models.image_tool(tool_id),
                        {**self.selection,'assignments':{'face-1':'face-1'}},['scene.png','reference.png'],Path,
                        pose_source=1,style_source=style_source)
                self.assertIn(f'exact visual style of <image{style_source}>',prompt)
                self.assertIn('head angle, head rotation, gaze, expression and micro-expressions from <image1>',prompt)
                self.assertNotIn('<style_source>',prompt)
                self.assertNotRegex(prompt.lower(),r'\b(?:without|photorealistic|cartoon|anime|illustration)\b')

    def test_detector_orders_faces_and_invalidates_changed_pictures(self):
        import numpy as np
        from unittest.mock import Mock
        detector = Mock()
        detector.detect.return_value = (None, np.array([[130, 30, 30, 40], [10, 20, 25, 35]]))
        with tempfile.TemporaryDirectory() as folder, patch('backend.reference_people.face_detector', return_value=detector):
            path = Path(folder)/'picture.png'
            Image.new('RGB', (200, 100)).save(path)
            result = copy.deepcopy(image_people.analyze_image(path))
            self.assertEqual(len(result['faces']), 2)
            self.assertEqual(result['faces'][0]['box'], [.05, .2, .125, .35])
            self.assertTrue(result['faces'][0]['preview'].startswith('data:image/jpeg;base64,'))
            self.assertEqual(image_people.analyze_image(path), result)
            self.assertEqual(detector.detect.call_count, 1)
            Image.new('RGB', (220, 100)).save(path)
            self.assertNotEqual(image_people.analyze_image(path)['revision'], result['revision'])
            self.assertEqual(detector.detect.call_count, 2)


if __name__ == '__main__':
    unittest.main()

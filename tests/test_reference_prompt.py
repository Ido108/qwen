"""Selections alone must produce full prompts in each execution path."""
import unittest

from backend.common import ReferenceVideoRequest
from backend.reference_prompt import build_reference_prompt, reference_instructions


class ReferencePromptTests(unittest.TestCase):
    def setUp(self):
        self.request = ReferenceVideoRequest(source_video='source.mp4', reference_images=['a.png','b.png'],
            source_start_seconds=2, source_end_seconds=8,
            source_people=[{'id':'person-1','time':3,'box':[0.1,0.1,0.2,0.3]},
                           {'id':'person-2','time':4,'box':[0.7,0.1,0.2,0.3]}],
            person_assignments={'person-1':'b.png','person-2':None}).model_dump()

    def test_clicks_alone_replace_selected_person_and_keep_others(self):
        result = build_reference_prompt(self.request)
        self.assertTrue(result.startswith('Faceswap.'))
        self.assertIn('at 1.0s on the left',result)
        self.assertIn('identity in <Picture 2>',result)
        self.assertIn('Keep the source person first seen at 2.0s on the right',result)
        self.assertNotIn('Replace the main person',result)
        self.assertEqual(result.count('Person assignments:'),1)

    def test_reassigning_updates_prompt_without_changing_text(self):
        self.request['person_assignments']={'person-1':None,'person-2':'a.png'}
        result=build_reference_prompt(self.request)
        self.assertIn('Replace only the source person first seen at 2.0s on the right',result)
        self.assertIn('identity in <Picture 1>',result)

    def test_extra_text_and_tags_do_not_remove_choices(self):
        self.request['prompt']='Keep the glasses on @Person1. Use @Ref2 in @SourceVideo.'
        result=build_reference_prompt(self.request)
        self.assertIn('Additional instructions: Keep the glasses',result)
        self.assertNotIn('@Person',result)
        enhanced=build_reference_prompt(self.request,'Preserve the glasses.')
        self.assertIn('identity in <Picture 2>',enhanced)
        self.assertIn('selected person assignments take priority',enhanced)

    def test_camera_and_replacement_stages_are_independent_with_both_fields_empty(self):
        self.request.update(workflow='crossview',replace_subjects_first=True)
        camera=build_reference_prompt(self.request)
        replacement=build_reference_prompt(self.request,replacement=True)
        self.assertTrue(camera.startswith('crossview.'))
        self.assertNotIn('Person assignments',camera)
        self.assertNotIn('<Picture',camera)
        self.assertIn('identity in <Picture 2>',replacement)

    def test_camera_instructions_cannot_reference_absent_replacement_images(self):
        self.request.update(workflow='crossview',prompt='Use @Ref1')
        with self.assertRaisesRegex(ValueError,'no uploaded'):
            build_reference_prompt(self.request)

    def test_reference_guided_has_useful_default(self):
        self.request['workflow']='reference'
        self.assertIn('Follow the movement',build_reference_prompt(self.request))
        self.assertEqual(reference_instructions(self.request),'')

    def test_face_replacement_prioritizes_original_camera_after_enhancement(self):
        self.request.update(camera_azimuth=90, camera_elevation=60, warp_video='stale-camera.mp4')
        result = build_reference_prompt(self.request, 'Zoom in and move the camera.')
        self.assertGreater(result.index('Scene preservation takes priority'), result.index('Zoom in'))
        self.assertIn('exact source camera angle', result)
        self.assertNotIn('supplied camera guide', result)
        self.assertNotIn('crossview', result.lower())

    def test_no_detected_people_uses_first_reference(self):
        self.request.update(source_people=[],person_assignments={})
        self.assertIn('Replace the main person with the identity in <Picture 1>',build_reference_prompt(self.request))

    def test_keep_all_and_stale_reference_fail_instead_of_swapping_somebody_else(self):
        self.request['person_assignments']={'person-1':None,'person-2':None}
        with self.assertRaisesRegex(ValueError,'at least one'):
            build_reference_prompt(self.request)
        self.request['person_assignments']={'person-1':'deleted.png'}
        with self.assertRaisesRegex(ValueError,'removed reference'):
            build_reference_prompt(self.request)

    def grouped(self):
        return ReferenceVideoRequest(source_video='source.mp4',
            reference_people=[{'id':'ref-person-a','images':['front.png','side.png']},
                              {'id':'ref-person-b','images':['other.png']}],
            source_people=self.request['source_people'],source_start_seconds=2,source_end_seconds=8,
            person_assignments={'person-1':'ref-person-a','person-2':None}).model_dump()

    def test_person_selection_uses_all_views_without_merging_identities(self):
        request=self.grouped()
        self.assertEqual(request['reference_images'],['front.png','side.png','other.png'])
        prompt=build_reference_prompt(request)
        self.assertIn('identity in <Picture 1> and <Picture 2>',prompt)
        self.assertIn('Person 2: <Picture 3>',prompt)
        self.assertIn('Keep the source person first seen at 2.0s',prompt)
        self.assertNotIn('identity in <Picture 1> and <Picture 2> and <Picture 3>',prompt)

    def test_person_and_source_tags_have_separate_meanings(self):
        request=self.grouped();request['prompt']='Use @Person1 for @Source2. Keep @Person2 separate.'
        prompt=reference_instructions(request)
        self.assertIn('person shown in <Picture 1> and <Picture 2>',prompt)
        self.assertIn('source person first seen at 2.0s',prompt)
        self.assertIn('person shown in <Picture 3>',prompt)
        self.assertNotIn('Reference people:',prompt)

    def test_fallback_uses_only_first_group_and_camera_uses_no_groups(self):
        request=self.grouped();request['person_assignments']={}
        prompt=build_reference_prompt(request)
        self.assertIn('Replace the main person with the identity in <Picture 1> and <Picture 2>',prompt)
        request.update(workflow='crossview',replace_subjects_first=True)
        self.assertNotIn('<Picture',build_reference_prompt(request))
        self.assertIn('<Picture 1> and <Picture 2>',build_reference_prompt(request,replacement=True))

    def test_group_image_limit_duplicates_and_flat_order_are_validated(self):
        for groups,images in [([{'id':'ref-person-a','images':[]}],[]),
            ([{'id':'ref-person-a','images':['same.png','same.png']}],[]),
            ([{'id':'ref-person-a','images':['a.png']},{'id':'ref-person-a','images':['b.png']}],[]),
            ([{'id':'ref-person-a','images':['a.png']}],['wrong.png']),
            ([{'id':'ref-person-a','images':[f'{i}.png' for i in range(9)]},{'id':'ref-person-b','images':['last.png']}],[])]:
            with self.subTest(groups=groups),self.assertRaises(ValueError):
                ReferenceVideoRequest(source_video='v.mp4',reference_people=groups,reference_images=images)

"""Keep independent subject replacement and camera workflows independent."""
import unittest

from backend.common import ReferenceVideoRequest, UpscaleRequest
from backend.models import required_reference, required_upscale


class ReferenceWorkflowTests(unittest.TestCase):
    def test_camera_view_does_not_require_subject_replacement(self):
        camera = ReferenceVideoRequest(
            workflow='crossview', source_video='source.mp4', prompt='New camera angle',
        )
        needed = required_reference(camera)
        self.assertIn('h3-ref-crossview', needed)
        self.assertIn('moge2-geometry', needed)
        self.assertNotIn('h3-ref-faceswap', needed)

    def test_subject_replacement_does_not_require_camera_view(self):
        replacement = ReferenceVideoRequest(
            workflow='faceswap', source_video='source.mp4', reference_images=['person.png'],
            prompt='Replace the person',
        )
        needed = required_reference(replacement)
        self.assertIn('h3-ref-faceswap', needed)
        self.assertNotIn('h3-ref-crossview', needed)
        self.assertNotIn('moge2-geometry', needed)

    def test_optional_two_stage_workflow_requires_both_model_adapters(self):
        combined = ReferenceVideoRequest(
            workflow='crossview', source_video='source.mp4', reference_images=['person.png'],
            replace_subjects_first=True, prompt='New camera angle',
        )
        needed = required_reference(combined)
        self.assertIn('h3-ref-faceswap', needed)
        self.assertIn('h3-ref-crossview', needed)
        self.assertIn('moge2-geometry', needed)
        combined.reference_images = []
        with self.assertRaisesRegex(ValueError, 'reference images'):
            required_reference(combined)

    def test_gguf_reference_model_uses_standard_steps(self):
        request = ReferenceVideoRequest(
            workflow='faceswap', source_video='source.mp4', reference_images=['person.png'],
            model_id='h3-ref-eros-q4', quality='standard', prompt='Replace the person',
        )
        self.assertIn('h3-encoder', required_reference(request))
        request.quality = 'turbo'
        with self.assertRaisesRegex(ValueError, 'not distilled'):
            required_reference(request)

    def test_upscale_defaults_avoid_reprocessing_nearly_every_frame(self):
        request = UpscaleRequest(source_video='source.mp4')
        self.assertEqual((request.batch_size, request.temporal_overlap), (5, 1))
        self.assertIn('seedvr2-vae', required_upscale(request))


if __name__ == '__main__':
    unittest.main()

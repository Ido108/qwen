"""No inference: which video models each video page accepts, and their per-family rules."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError
from backend import models, server, cloud
from backend.common import VideoRequest

LTX_FILES = ['ltx25-gemma-int8', 'ltx25-video-vae', 'ltx25-audio-vae']


class CreateVideoModelTests(unittest.TestCase):
    def test_every_ltx_transformer_runs_text_to_video_without_the_msr_lora(self):
        for item in [m for m in models.catalog() if m['kind'] == 'multi-reference']:
            with self.subTest(model=item['id']):
                needed = models.required_video(VideoRequest(model_id=item['id'], prompt='A beach', frames=121, width=1280, height=704))
                self.assertEqual(needed, [item['id'], *LTX_FILES, 'ltx25-spatial-upscaler'])
                self.assertNotIn('ltx25-msr-v2', needed)

    def test_ltx_image_modes_follow_the_official_workflows(self):
        first = VideoRequest(model_id='ltx25-msr-q4_k_m', prompt='Waves', mode='image', first_frame='a.png', frames=73, width=768, height=448)
        self.assertIn('ltx25-spatial-upscaler', models.required_video(first))
        both = first.model_copy(update={'last_frame': 'b.png', 'width': 736, 'height': 416})
        # First and last frame is one full-size pass: no upscaler and multiples of 32 are enough.
        self.assertNotIn('ltx25-spatial-upscaler', models.required_video(both))

    def test_ltx_frame_and_size_rules(self):
        base = dict(model_id='ltx25-msr-q4_k_m', prompt='A beach', width=768, height=448)
        for values in ({'frames': 124}, {'frames': 121, 'width': 736}, {'frames': 121, 'width': 1536, 'height': 1536},
                       {'frames': 121, 'mode': 'image', 'first_frame': 'a.png', 'last_frame': 'b.png', 'width': 1536, 'height': 1024}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                models.required_video(VideoRequest(**(base | values)))

    def test_minimax_rules_still_apply_to_minimax_models(self):
        for values in ({'frames': 121}, {'width': 1536, 'height': 512}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                models.required_video(VideoRequest(prompt='A beach', model_id='h3-gguf-q4', **values))
        with self.assertRaises(ValidationError):
            VideoRequest(prompt='A beach', frames=100)

    def test_reference_video_models_stay_out_of_create_video(self):
        with self.assertRaises(ValueError):
            models.required_video(VideoRequest(model_id='h3-ref-turbo-int8', prompt='A beach'))

    def test_video_loras_only_match_their_family(self):
        h3 = {'id': 'civitai-lora-h3', 'title': 'H3 LoRA', 'kind': 'video-lora', 'families': ['minimax-h3', 'fasth3', 'minimax-h3-ref2va']}
        ltx = {'id': 'civitai-lora-ltx', 'title': 'LTX LoRA', 'kind': 'video-lora', 'families': ['ltx25', 'ltx25-msr']}
        real = models.entry
        with patch.object(models, 'entry', side_effect=lambda name: {'civitai-lora-h3': h3, 'civitai-lora-ltx': ltx}.get(name) or real(name)):
            ltx_request = VideoRequest(model_id='ltx25-msr-q4_k_m', prompt='A beach', frames=121, width=768, height=448,
                                       loras=[{'id': 'civitai-lora-ltx'}])
            self.assertIn('civitai-lora-ltx', models.required_video(ltx_request))
            with self.assertRaises(ValueError):
                models.required_video(ltx_request.model_copy(update={'loras': [{'id': 'civitai-lora-h3', 'strength': 1}]}))
            with self.assertRaises(ValueError):
                models.required_video(VideoRequest(prompt='A beach', model_id='h3-gguf-q4', loras=[{'id': 'civitai-lora-ltx'}]))

    def test_refinement_pass_gets_conditioning_without_cleared_guides(self):
        # LTXVCropGuides sets these keys to None; the MSR guide of the 2x pass must see them absent.
        from backend.ltx_engine import _drop_cleared_guides
        cleared = [['t', {'keyframe_idxs': None, 'guide_attention_entries': None, 'frame_rate': 24}]]
        self.assertEqual(_drop_cleared_guides(cleared), [['t', {'frame_rate': 24}]])
        kept = [['t', {'guide_attention_entries': [1]}]]
        self.assertEqual(_drop_cleared_guides(kept), kept)

    def test_api_reports_ltx_step_count(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(cloud, 'settings', return_value={'mode': 'local'}), patch.object(server, 'enqueue_locked'), \
                 patch.object(server, 'resolve_image', return_value=Path(folder) / 'a.png'):
                two_pass = server.generate_video(VideoRequest(model_id='ltx25-msr-q4_k_m', prompt='A beach', frames=121,
                                                              width=768, height=448, output_dir=folder))
                keyframes = server.generate_video(VideoRequest(model_id='ltx25-msr-q4_k_m', prompt='A beach', frames=121, mode='image',
                    first_frame='a.png', last_frame='b.png', width=768, height=448, output_dir=folder))
        self.assertEqual(two_pass['total_steps'], 11)
        self.assertEqual(keyframes['total_steps'], 8)


if __name__ == '__main__':
    unittest.main()

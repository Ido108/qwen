"""Shot-by-shot face swap: cuts, padding to the H3 grid and joining in source order."""
import unittest
from unittest import mock

import torch

from backend import reference_shots
from backend.reference_shots import grid_length, shot_bounds, swap_by_shot


def clip(*shots):
    """Frames whose brightness jumps at each shot start; small motion inside each shot."""
    parts = []
    for length, level in shots:
        ramp = torch.linspace(0, 0.02, length).view(-1, 1, 1, 1)
        parts.append((torch.full((length, 64, 36, 3), level) + ramp).clamp(0, 1))
    return torch.cat(parts)


class ShotTests(unittest.TestCase):
    def test_cuts_are_found_and_motion_is_not_a_cut(self):
        self.assertEqual(shot_bounds(clip((30, 0.1), (20, 0.8), (12, 0.3))), [(0, 30), (30, 50), (50, 62)])
        moving = torch.rand(40, 64, 36, 3) * 0.05 + torch.linspace(0, 0.6, 40).view(-1, 1, 1, 1)
        self.assertEqual(shot_bounds(moving), [(0, 40)])

    def test_grid_length(self):
        self.assertEqual([grid_length(n) for n in (1, 5, 6, 22, 23, 117, 124)], [5, 5, 22, 22, 39, 124, 124])

    def test_shots_with_faces_are_swapped_alone_and_joined_in_order(self):
        source = clip((30, 0.1), (20, 0.8), (12, 0.3))
        audio = {'waveform': torch.arange(62 * 1000, dtype=torch.float32).view(1, 1, -1).repeat(1, 2, 1), 'sample_rate': 24000}
        calls, updates = [], []

        def generate(request, prompt, images, progress, preview_directory=None):
            calls.append((request['frames'], request['source_start_seconds'], images['source_video'].shape[0],
                          images['source_audio']['waveform'].shape[-1]))
            progress(stage='generating', step=8, total_steps=8)
            return torch.full((request['frames'], 128, 72, 3), 0.5), {'waveform': torch.ones(1, 2, request['frames'] * 2000), 'sample_rate': 48000}

        # Output is larger than the source frames, as with a 480x848 source at 576x1024.
        request = {'steps': 8, 'frames': 62, 'source_start_seconds': 2.0, 'width': 72, 'height': 128}
        with mock.patch.object(reference_shots, 'has_face', side_effect=[True, False, True]):
            frames, sound = swap_by_shot(request, 'Faceswap', {'source_video': source, 'source_audio': audio},
                                         lambda **values: updates.append(values), generate)
        self.assertEqual([call[:3] for call in calls], [(39, 2.0, 39), (22, 2.0 + 50 / 24, 22)])
        self.assertEqual([call[3] for call in calls], [39 * 1000, 22 * 1000])
        self.assertEqual(tuple(frames.shape), (62, 128, 72, 3))
        self.assertTrue(torch.allclose(frames[30:50].mean(dim=(1, 2, 3)), source[30:50].mean(dim=(1, 2, 3)), atol=1e-3))
        self.assertTrue(bool((frames[:30] == 0.5).all()) and bool((frames[50:] == 0.5).all()))
        self.assertEqual(sound['sample_rate'], 48000)
        self.assertEqual(sound['waveform'].shape, (1, 2, 62 * 2000))
        steps = [u['step'] for u in updates if u.get('stage') == 'generating']
        self.assertEqual(steps, [8, 16])
        self.assertEqual({u['total_steps'] for u in updates if 'total_steps' in u}, {16})
        self.assertEqual([(u['shot'], u['total_shots']) for u in updates if 'shot' in u], [(1, 2), (2, 2)])

    def test_single_shot_or_no_face_runs_one_generate(self):
        calls = []
        def generate(request, prompt, images, progress, preview_directory=None):
            calls.append(request['frames'])
            return 'frames', 'audio'
        with mock.patch.object(reference_shots, 'has_face', return_value=True):
            self.assertEqual(swap_by_shot({'steps': 8, 'frames': 40}, 'Faceswap', {'source_video': clip((40, 0.2))}, lambda **v: None, generate), ('frames', 'audio'))
        with mock.patch.object(reference_shots, 'has_face', return_value=False):
            swap_by_shot({'steps': 8, 'frames': 50}, 'Faceswap', {'source_video': clip((30, 0.1), (20, 0.8))}, lambda **v: None, generate)
        self.assertEqual(calls, [40, 50])


class ReferenceCropTests(unittest.TestCase):
    def test_picture_without_a_face_is_unchanged(self):
        from PIL import Image
        from backend.reference_people import crop_to_face
        picture = Image.new('RGB', (640, 480), 'gray')
        self.assertIs(crop_to_face(picture), picture)


if __name__ == '__main__':
    unittest.main()

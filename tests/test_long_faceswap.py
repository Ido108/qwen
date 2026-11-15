"""A face swap of any length runs in parts and is written to one MP4 with its audio."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from backend.common import ReferenceVideoRequest
from backend.long_faceswap import PART_FRAMES, is_long_faceswap, part_plan
from backend.reference_timing import resolve_reference_clip


class PartPlanTests(unittest.TestCase):
    def test_every_part_is_one_124_frame_pass_with_its_overlap(self):
        from backend.long_faceswap import OVERLAP_FRAMES, MIN_LAST_PART
        from backend.reference_shots import grid_length
        for total in (22, 124, 125, 362, 497, 720, 1433, 1705, 86405):
            plan = part_plan(total)
            sizes = [end - start for start, end in plan]
            self.assertEqual(plan[0][0], 0)
            self.assertEqual(plan[-1][1], total)
            self.assertTrue(all(plan[i][1] == plan[i + 1][0] for i in range(len(plan) - 1)))
            reads = [grid_length(size + (OVERLAP_FRAMES if index else 0)) for index, size in enumerate(sizes)]
            # Every part but the last reads exactly one 124-frame pass; a short last part joins the one before it.
            self.assertTrue(all(read == PART_FRAMES for read in reads[:-1]), (total, reads))
            self.assertLessEqual(reads[-1], grid_length(PART_FRAMES + MIN_LAST_PART))
            if len(sizes) > 1:
                self.assertGreaterEqual(sizes[-1], MIN_LAST_PART, 'a short last part joins the part before it')

    def test_only_a_face_swap_longer_than_one_part_runs_in_parts(self):
        self.assertTrue(is_long_faceswap({'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 141}))
        self.assertFalse(is_long_faceswap({'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 124}))
        self.assertTrue(is_long_faceswap({'kind': 'reference-video', 'workflow': 'reference', 'frames': 141}))
        self.assertFalse(is_long_faceswap({'kind': 'reference-video', 'workflow': 'crossview', 'frames': 362}))


class LengthLimitTests(unittest.TestCase):
    def test_face_swap_selection_has_no_length_limit(self):
        for seconds, frames in ((15.0, 362), (60.0, 1433), (70.9, 1705)):
            request = SimpleNamespace(source_start_seconds=0.0, source_end_seconds=seconds, frames=124)
            resolve_reference_clip(request, 70.92, 362, unlimited=True)
            self.assertEqual(request.frames, frames)
            self.assertEqual(request.frames % 17, 5)

    def test_other_workflows_keep_one_pass(self):
        with self.assertRaisesRegex(ValueError, 'at most 15.1 seconds'):
            resolve_reference_clip(SimpleNamespace(source_start_seconds=0.0, source_end_seconds=31.0, frames=124), 70.92, 362)
        base = dict(source_video='x.mp4', reference_images=['a.png'], width=1024, height=576)
        for workflow in ('faceswap', 'reference'):
            self.assertEqual(ReferenceVideoRequest(**base, workflow=workflow, frames=1705).frames, 1705)
        for workflow in ('crossview',):
            with self.assertRaisesRegex(ValueError, 'at most 15.1 seconds'):
                ReferenceVideoRequest(**base, workflow=workflow, frames=1433)


class SwapLongTests(unittest.TestCase):
    def test_parts_are_read_one_at_a_time_and_written_to_one_mp4_with_audio(self):
        import av
        import torch
        from backend import long_faceswap

        reads, calls = [], []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            reads.append((count, round(start_seconds, 4), round(end_seconds, 4)))
            # Each frame carries its source time, to check the order of the written frames.
            frame = torch.zeros(count, 64, 64, 3)
            for index in range(count):
                frame[index] = (start_seconds * 24 + index) / 1000
            return frame

        def read_audio(path, seconds, start_seconds, end_seconds):
            return {'waveform': torch.ones(1, 2, round(seconds * 32000)) * 0.1, 'sample_rate': 32000}

        def generate(request, prompt, images, progress, preview_directory=None):
            guide = images.get('continuity_guide')
            calls.append((request['frames'], None if guide is None else round(float(guide[0, 0, 0, 0]) * 1000)))
            assert images['source_video'].shape[0] == request['frames']
            progress(stage='generating', step=1)
            return images['source_video'].clone(), {'waveform': torch.ones(1, 2, round(request['frames'] / 24 * 32000)) * 0.2, 'sample_rate': 32000}

        updates = []
        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 2.0,
                   'source_audio': True, 'output_audio': 'generated', 'width': 64, 'height': 64}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch('backend.reference_media.read_audio', read_audio), \
                mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]):
            destination = Path(folder) / 'out.mp4'
            count, first, has_audio, size, rate = long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'),
                                                                          lambda **values: updates.append(values), generate, destination)
            self.assertEqual((count, has_audio, size, rate), (400, True, (64, 64), 32000))
            self.assertEqual(len(reads), 3, '400 frames: parts of 124, 119 and 157 (the last 38 frames join the third part), each read on its own')
            self.assertEqual([r[0] for r in reads], [124, 124, 175], 'each part is read on the 17n+5 grid')
            # Parts after the first start 5 frames early, where the previous part's last 5 frames are anchored.
            self.assertEqual([r[1] for r in reads], [2.0, round(2.0 + 119 / 24, 4), round(2.0 + 238 / 24, 4)])
            self.assertEqual([c[0] for c in calls], [124, 124, 175])
            # The guide is the previous part's last 5 written frames: frames 119-123 and 238-242 of the selection.
            self.assertEqual([c[1] for c in calls], [None, 48 + 119, 48 + 238])
            self.assertTrue(any(u.get('part') == 3 and u.get('total_parts') == 3 for u in updates))
            self.assertFalse((Path(folder) / 'out.video.mp4').exists(), 'the video-only file is removed')
            # Each finished part is saved as its own MP4 with a poster, and stays next to the whole video.
            self.assertEqual([u['parts_saved'] for u in updates if 'parts_saved' in u], [1, 2, 3])
            self.assertEqual([(u['part_start_seconds'], u['part_end_seconds']) for u in updates if 'part_start_seconds' in u],
                             [(2.0, round(2 + 124 / 24, 3)), (round(2 + 124 / 24, 3), round(2 + 243 / 24, 3)), (round(2 + 243 / 24, 3), round(2 + 400 / 24, 3))])
            parts = Path(folder) / 'out parts'
            self.assertEqual(sorted(path.name for path in parts.iterdir()),
                             ['part-001.jpg', 'part-001.mp4', 'part-002.jpg', 'part-002.mp4', 'part-003.jpg', 'part-003.mp4'])
            self.assertEqual([part['index'] for part in [u for u in updates if 'parts' in u][-1]['parts']], [1, 2, 3])
            with av.open(str(parts / 'part-002.mp4')) as container:
                self.assertEqual(sum(1 for _ in container.decode(container.streams.video[0])), 119)
                self.assertEqual(len(container.streams.audio), 1)
            with av.open(str(destination)) as container:
                video = container.streams.video[0]
                frames = sum(1 for _ in container.decode(video))
                self.assertEqual(frames, 400)
                self.assertEqual(len(container.streams.audio), 1)
                self.assertAlmostEqual(float(container.duration) / 1e6, 400 / 24, delta=0.2)


class TwoPassTests(unittest.TestCase):
    def test_every_part_is_prepared_and_encoded_before_the_parts_are_generated_in_order(self):
        import torch
        from backend import long_faceswap
        events = []

        class Encoder:
            def prepare(self, request, prompt, images):
                events.append(('prepare', request['frames'], 'continuity_guide' in images))
                return {'frames': request['frames'], 'id': len([e for e in events if e[0] == 'prepare'])}

            def encode_all(self, units):
                events.append(('encode_all', len(units)))
                return [f'positive-{unit["id"]}' for unit in units]

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 32, 32, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            events.append(('generate', request['frames'], images.get('precomputed_positive'), 'continuity_guide' in images))
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 32, 'height': 32}
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('backend.reference_media.read_frames', read_frames),                 mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]):
            count, *_ = long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'), lambda **values: None,
                                                generate, Path(folder) / 'out.mp4', encoder=Encoder())
        self.assertEqual(count, 400)
        self.assertEqual(events, [('prepare', 124, False), ('prepare', 124, False), ('prepare', 175, False), ('encode_all', 3),
                                  ('generate', 124, 'positive-1', False), ('generate', 124, 'positive-2', True),
                                  ('generate', 175, 'positive-3', True)])


if __name__ == '__main__':
    unittest.main()

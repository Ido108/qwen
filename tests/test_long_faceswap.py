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
        self.assertTrue(is_long_faceswap({'kind': 'reference-video', 'workflow': 'crossview', 'frames': 362}))
        # Camera view with its own guide video, or with Replace subjects first (two passes), keeps one pass.
        self.assertFalse(is_long_faceswap({'kind': 'reference-video', 'workflow': 'crossview', 'frames': 362, 'warp_video': 'w.mp4'}))
        self.assertFalse(is_long_faceswap({'kind': 'reference-video', 'workflow': 'crossview', 'frames': 362, 'replace_subjects_first': True}))


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
        for workflow in ('faceswap', 'reference', 'crossview'):
            self.assertEqual(ReferenceVideoRequest(**base, workflow=workflow, frames=1705).frames, 1705)
        for extra in ({'warp_video': 'w.mp4'}, {'replace_subjects_first': True}):
            with self.assertRaisesRegex(ValueError, 'at most 15.1 seconds'):
                ReferenceVideoRequest(**base, workflow='crossview', frames=1433, **extra)


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
                   'source_audio': True, 'output_audio': 'generated', 'width': 64, 'height': 64, 'long_parts': 'chain'}
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
            # Parts after the first start 5 frames early, where the previous part's last 5 frames are anchored (the seam).
            self.assertEqual([r[1] for r in reads], [2.0, round(2.0 + 119 / 24, 4), round(2.0 + 238 / 24, 4)])
            self.assertEqual([c[0] for c in calls], [124, 124, 175])
            # The anchor is the previous part's last 5 written frames: frames 119-123 and 238-242 of the selection.
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
                   'source_audio': False, 'output_audio': 'mute', 'width': 32, 'height': 32, 'long_parts': 'chain'}
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('backend.reference_media.read_frames', read_frames),                 mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]):
            count, *_ = long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'), lambda **values: None,
                                                generate, Path(folder) / 'out.mp4', encoder=Encoder())
        self.assertEqual(count, 400)
        self.assertEqual(events, [('prepare', 124, False), ('prepare', 124, False), ('prepare', 175, False), ('encode_all', 3),
                                  ('generate', 124, 'positive-1', False), ('generate', 124, 'positive-2', True),
                                  ('generate', 175, 'positive-3', True)])


class FollowPartOneTests(unittest.TestCase):
    def test_later_parts_take_their_input_image_and_prompt_from_part_1(self):
        import torch
        from PIL import Image
        from backend import long_faceswap
        events, picked = [], Image.new('RGB', (8, 8), 'red')
        photo = Image.new('RGB', (8, 8), 'blue')

        class Encoder:
            def prepare(self, request, prompt, images):
                second = images['ref_image_2'].getpixel((0, 0)) if 'ref_image_2' in images else None
                events.append(('prepare', images['ref_image_1'].getpixel((0, 0)), second, len(request['reference_images']), prompt))
                return {'frames': request['frames'], 'id': len([e for e in events if e[0] == 'prepare'])}

            def encode_all(self, units):
                events.append(('encode_all', [unit['part'] for unit in units]))
                return [f'positive-{unit["id"]}' for unit in units]

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 32, 32, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            events.append(('generate', images['precomputed_positive'], 'ref_image_2' in images, prompt))
            # Following the frame of part 1, no part starts from the frames of the part before.
            self.assertNotIn('continuity_guide', images)
            return images['source_video'].clone(), None

        chosen = []

        def best(frames, reference):
            chosen.append((frames.shape[0], reference.getpixel((0, 0))))
            return picked, 40, 0.71

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 32, 'height': 32, 'reference_images': ['a.png'], 'long_parts': 'follow'}
        updates = []
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]), \
                mock.patch.object(long_faceswap, 'best_reference_frame', best):
            count, *_ = long_faceswap.swap_long(request, 'Faceswap', {'ref_image_1': photo}, Path('source.mp4'),
                                                lambda **values: updates.append(values), generate, Path(folder) / 'out.mp4',
                                                preview_directory=folder, encoder=Encoder())
            self.assertTrue((Path(folder) / 'reference-from-part-1.png').is_file())
        self.assertEqual(count, 400)
        # The frame is picked from part 1's 124 frames, compared with the input photo.
        self.assertEqual(chosen, [(124, (0, 0, 255))])
        later = 'Faceswap' + long_faceswap.FOLLOW_PROMPT
        # Part 1 is prepared alone and made first; parts 2 and 3 are prepared once, after it, with the input photo
        # first, the frame of part 1 as the second picture, and the added prompt.
        after = ('prepare', (0, 0, 255), (255, 0, 0), 2, later)
        self.assertEqual(events, [('prepare', (0, 0, 255), None, 1, 'Faceswap'), ('encode_all', [1]),
                                  ('generate', 'positive-1', False, 'Faceswap'),
                                  after, after, ('encode_all', [2, 3]),
                                  ('generate', 'positive-2', True, later), ('generate', 'positive-3', True, later)])
        self.assertNotIn('face', long_faceswap.FOLLOW_PROMPT.lower())
        self.assertNotIn('person', long_faceswap.FOLLOW_PROMPT.lower())
        self.assertTrue(any(u.get('reference_frame') == 40 for u in updates))

    def test_continue_starts_each_part_from_the_last_frames_of_the_part_before_and_keeps_both_pictures(self):
        import torch
        from PIL import Image
        from backend import long_faceswap
        made, prepared = [], []
        photo, picked = Image.new('RGB', (8, 8), 'blue'), Image.new('RGB', (8, 8), 'red')

        class Encoder:
            def prepare(self, request, prompt, images):
                prepared.append((prompt, 'ref_image_2' in images))
                return {'frames': request['frames'], 'id': len(prepared)}

            def encode_all(self, units):
                return [f'positive-{unit["id"]}' for unit in units]

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            guide = images.get('continuity_guide')
            made.append((images['precomputed_positive'], None if guide is None else guide.shape[0], 'ref_image_2' in images))
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16, 'reference_images': ['a.png'],
                   'seed': 100, 'long_parts': 'continue'}
        # The first try of part 3 loses the face; its second try is made on its own.
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]), \
                mock.patch.object(long_faceswap, 'best_reference_frame', lambda frames, reference: (picked, 40, 0.7)), \
                mock.patch.object(long_faceswap, 'face_lost', lambda *a, **k: len(made) == 3):
            count, *_ = long_faceswap.swap_long(request, 'Faceswap', {'ref_image_1': photo}, Path('source.mp4'),
                                                lambda **values: None, generate, Path(folder) / 'out.mp4', encoder=Encoder())
        self.assertEqual(count, 400)
        later = 'Faceswap' + long_faceswap.FOLLOW_PROMPT + long_faceswap.QUALITY_PROMPT
        self.assertEqual(prepared, [('Faceswap', False), (later, True), (later, True)])
        # Part 1 has nothing before it. Parts 2 and 3 start from the 5 last frames of the part before, with both
        # pictures. The second try of part 3 uses the same prepared conditioning without those frames.
        self.assertEqual(made, [('positive-1', None, False), ('positive-2', 5, True), ('positive-3', 5, True),
                                ('positive-3', None, True)])

    def test_by_default_three_parts_in_a_row_start_from_the_part_before_and_the_next_part_is_made_on_its_own(self):
        import torch
        from PIL import Image
        from backend import long_faceswap
        made = []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            made.append(('continuity_guide' in images, 'ref_image_2' in images, prompt.endswith(long_faceswap.QUALITY_PROMPT)))
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 838, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16, 'reference_images': ['a.png'], 'seed': 100}

        def run(**extra):
            made.clear()
            with tempfile.TemporaryDirectory() as folder,                     mock.patch('backend.reference_media.read_frames', read_frames),                     mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]),                     mock.patch.object(long_faceswap, 'best_reference_frame', lambda frames, reference: (Image.new('RGB', (8, 8), 'red'), 40, 0.7)),                     mock.patch.object(long_faceswap, 'face_lost', lambda *a, **k: False):
                long_faceswap.swap_long({**request, **extra}, 'Faceswap', {'ref_image_1': Image.new('RGB', (8, 8), 'blue')},
                                        Path('source.mp4'), lambda **values: None, generate, Path(folder) / 'out.mp4')
            return [part[0] for part in made]

        # 7 parts. Part 1 is made on its own; parts 2 and 3 start from the part before (3 in a row with part 1);
        # part 4 is made on its own again, and parts 5 and 6 start from the part before; part 7 on its own.
        self.assertEqual(run(), [False, True, True, False, True, True, False])
        self.assertTrue(all(second and quality for _, second, quality in made[1:]))
        self.assertEqual(run(continue_run=0), [False, True, True, True, True, True, True])
        self.assertEqual(run(long_parts='follow'), [False] * 7)

    def test_more_than_one_input_image_keeps_the_input_images(self):
        import torch
        from backend import long_faceswap
        events = []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            events.append(prompt)
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16, 'reference_images': ['a.png', 'b.png'],
                   'long_parts': 'follow'}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]), \
                mock.patch.object(long_faceswap, 'best_reference_frame', lambda *a: self.fail('no pick with two people')):
            long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'), lambda **values: None, generate,
                                    Path(folder) / 'out.mp4')
        self.assertEqual(events, ['Faceswap'] * 3)


class FaceGuardTests(unittest.TestCase):
    REQUEST = {'kind': 'reference-video', 'workflow': 'faceswap', 'steps': 8, 'source_start_seconds': 0,
               'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16, 'reference_images': ['a.png'],
               'seed': 100, 'long_parts': 'independent'}

    def swap(self, frames, generate, lost, encoder=None, updates=None):
        import torch
        from PIL import Image
        from backend import long_faceswap

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]), \
                mock.patch.object(long_faceswap, 'face_lost', lost):
            count, *_ = long_faceswap.swap_long({**self.REQUEST, 'frames': frames}, 'Faceswap',
                                                {'ref_image_1': Image.new('RGB', (8, 8), 'blue')}, Path('source.mp4'),
                                                lambda **values: (updates if updates is not None else []).append(values),
                                                generate, Path(folder) / 'out.mp4', encoder=encoder)
        return count

    def test_a_part_that_goes_back_to_the_source_face_is_made_once_more_from_the_frames_of_the_part_before(self):
        from backend import long_faceswap
        made, prepared, updates = [], [], []

        class Encoder:
            def prepare(self, request, prompt, images):
                prepared.append(prompt)
                return {'frames': request['frames'], 'id': len(prepared)}

            def encode_all(self, units):
                return [f'positive-{unit["id"]}' for unit in units]

        def generate(request, prompt, images, progress, preview_directory=None):
            made.append((request['frames'], request['seed'], images['precomputed_positive'], 'continuity_guide' in images))
            return images['source_video'].clone(), None

        # The first try of part 2 shows the source face; every other try keeps the swap.
        count = self.swap(400, generate, lambda frames, source, reference, count=12: len(made) == 2, Encoder(), updates)
        self.assertEqual(count, 400)
        better = 'Faceswap' + long_faceswap.QUALITY_PROMPT
        # Part 2 is prepared once more, with the added prompt, and made once more with the same seed, starting from
        # the last frames of part 1. Parts 1 and 3 run once.
        self.assertEqual(prepared, ['Faceswap', 'Faceswap', 'Faceswap', better])
        self.assertEqual(made, [(124, 100, 'positive-1', False), (124, 100, 'positive-2', False),
                                (124, 100, 'positive-4', True), (175, 100, 'positive-3', False)])
        self.assertEqual([u['part_retries'] for u in updates if 'part_retries' in u], [1])

    def test_a_part_that_is_lost_twice_is_not_made_a_third_time(self):
        made = []

        def generate(request, prompt, images, progress, preview_directory=None):
            made.append((request['frames'], request['seed'], 'continuity_guide' in images))
            return images['source_video'].clone(), None

        self.assertEqual(self.swap(243, generate, lambda *a, **k: True), 243)
        # Part 1 has no part before it: its second try is the next seed. Part 2: one more try, from part 1's frames.
        self.assertEqual(made, [(124, 100, False), (124, 101, False), (124, 100, False), (124, 100, True)])

    def test_a_part_whose_scene_leaves_part_1_is_made_once_more_from_the_frames_of_the_part_before(self):
        from backend import long_faceswap
        made = []

        def generate(request, prompt, images, progress, preview_directory=None):
            guided = 'continuity_guide' in images
            made.append((request['frames'], request['seed'], guided, prompt))
            frames = images['source_video'].clone()
            if len(made) > 1 and not guided:
                # Part 2 comes out with a dark garment until it starts from the frames of part 1.
                frames[:, 8:] = 0.05
            return frames, None

        self.assertEqual(self.swap(243, generate, lambda *a, **k: False), 243)
        self.assertEqual(made, [(124, 100, False, 'Faceswap'), (124, 100, False, 'Faceswap'),
                                (124, 100, True, 'Faceswap' + long_faceswap.QUALITY_PROMPT)])

    def test_the_face_is_lost_only_where_the_made_face_is_the_source_face(self):
        import numpy as np
        import torch
        from PIL import Image
        from backend import long_faceswap
        photo_face, swapped, original = (np.array(v, np.float32) for v in ([1, 0, 0], [0.8, 0, 0.6], [0, 1, 0]))

        def embedder():
            # The input image itself is nearly black. A dark image shows its person as the model draws him, a bright
            # one the source person, a gray one no face.
            return lambda bgr: photo_face if bgr.mean() < 15 else swapped if bgr.mean() < 60 else original if bgr.mean() > 190 else None

        photo = Image.new('RGB', (8, 8), (10, 10, 10))
        source = torch.full((24, 8, 8, 3), 0.9)
        with mock.patch.object(long_faceswap, '_face_embedder', embedder):
            kept = torch.full((24, 8, 8, 3), 0.1)
            self.assertFalse(long_faceswap.face_lost(kept, source, photo))
            covered = kept.clone()
            covered[6:] = 0.5  # no face to read in most of the part: objects in front of it, a turned head
            self.assertFalse(long_faceswap.face_lost(covered, source, photo))
            faded = kept.clone()
            faded[20:] = 0.9  # the source face is back at the end of the part
            self.assertTrue(long_faceswap.face_lost(faded, source, photo))
            shown = kept.clone()
            shown[20:] = 10 / 255  # the input image itself as the last shot of the part
            self.assertTrue(long_faceswap.face_lost(shown, source, photo))

    def test_a_face_check_that_fails_does_not_stop_the_video(self):
        import torch
        from PIL import Image
        from backend import long_faceswap

        def broken():
            raise RuntimeError('no face model')

        with mock.patch.object(long_faceswap, '_face_embedder', broken):
            self.assertFalse(long_faceswap.face_lost(torch.zeros(4, 8, 8, 3), torch.zeros(4, 8, 8, 3), Image.new('RGB', (8, 8))))


class IndependentPartsTests(unittest.TestCase):
    def test_independent_parts_use_the_input_image_and_no_anchor(self):
        import torch
        from PIL import Image
        from backend import long_faceswap
        seen = []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            seen.append((prompt, 'continuity_guide' in images, images['ref_image_1'].getpixel((0, 0))))
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16, 'reference_images': ['a.png'],
                   'long_parts': 'independent'}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]), \
                mock.patch.object(long_faceswap, 'best_reference_frame', lambda *a: self.fail('no frame pick')):
            long_faceswap.swap_long(request, 'Faceswap', {'ref_image_1': Image.new('RGB', (8, 8), 'blue')},
                                    Path('source.mp4'), lambda **values: None, generate, Path(folder) / 'out.mp4')
        self.assertEqual(seen, [('Faceswap', False, (0, 0, 255))] * 3)


class WorkingMemoryTests(unittest.TestCase):
    def test_later_parts_reserve_the_working_memory_measured_on_the_part_before(self):
        import torch
        from backend import long_faceswap
        reserved = []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            reserved.append(request.get('working_memory_gb'))
            progress(sampling_working_gb=5.0)
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16}
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('backend.reference_media.read_frames', read_frames),                 mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]):
            long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'), lambda **values: None, generate, Path(folder) / 'out.mp4')
        # Part 1 keeps ComfyUI's estimate; part 2 (124 frames) 5 + 2 GB; part 3 (175 frames) scaled to its length.
        self.assertEqual(reserved, [None, 7.0, round(5.0 * 175 / 124 + 2.0, 2)])


class CameraViewPartsTests(unittest.TestCase):
    def test_camera_view_parts_are_whole_parts_without_the_continuity_anchor(self):
        import torch
        from backend import long_faceswap
        calls = []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        def generate(request, prompt, images, progress, preview_directory=None):
            calls.append((request['frames'], 'continuity_guide' in images))
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'crossview', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 16, 'height': 16}
        cuts = mock.Mock(side_effect=AssertionError('Camera view parts are not split by shot'))
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('backend.reference_media.read_frames', read_frames),                 mock.patch('backend.reference_shots.shot_bounds', cuts):
            count, *_ = long_faceswap.swap_long(request, 'crossview', {}, Path('source.mp4'), lambda **values: None, generate, Path(folder) / 'out.mp4')
        self.assertEqual(count, 400)
        self.assertEqual(calls, [(124, False), (124, False), (175, False)])


class SeamTests(unittest.TestCase):
    def test_the_seam_goes_from_the_previous_part_to_the_next_over_the_overlap_frames(self):
        import av
        import numpy as np
        import torch
        from backend import long_faceswap
        made, reads = [], []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            reads.append(count)
            return torch.full((count, 64, 64, 3), 0.2 if len(reads) == 1 else 0.6)

        def generate(request, prompt, images, progress, preview_directory=None):
            made.append(request['frames'])
            return images['source_video'].clone(), None

        request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 243, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'mute', 'width': 64, 'height': 64}
        with tempfile.TemporaryDirectory() as folder,                 mock.patch('backend.reference_media.read_frames', read_frames),                 mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]):
            destination = Path(folder) / 'out.mp4'
            count, *_ = long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'), lambda **values: None, generate, destination)
            with av.open(str(destination)) as container:
                levels = [float(frame.to_ndarray(format='rgb24').mean()) / 255 for frame in container.decode(container.streams.video[0])]
        self.assertEqual((count, len(levels), made), (243, 243, [124, 124]))
        # Frames 0-118 are part 1, 119-123 the seam (part 1 going into part 2), 124-242 part 2.
        self.assertAlmostEqual(levels[100], 0.2, delta=0.02)
        self.assertAlmostEqual(levels[200], 0.6, delta=0.02)
        seam = levels[119:124]
        self.assertTrue(all(a < b for a, b in zip(seam, seam[1:])), seam)
        self.assertTrue(0.2 < seam[0] < 0.35 and 0.45 < seam[-1] < 0.6, seam)

    def test_the_seam_moves_a_shape_between_its_two_places_and_never_shows_it_twice(self):
        import numpy as np
        import torch
        from backend.long_faceswap import seam_morph

        def frames(left):
            # A soft bright blob on a textured dark ground, at column `left`.
            ys, xs = np.mgrid[0:96, 0:160].astype(np.float32)
            image = 0.15 + 0.03 * np.sin(xs / 5.0) * np.cos(ys / 7.0) + 0.7 * np.exp(-(((xs - left) / 12.0) ** 2 + ((ys - 48) / 12.0) ** 2))
            return torch.from_numpy(np.repeat(image[None, ..., None], 3, axis=3).astype(np.float32)).repeat(5, 1, 1, 1)

        held, ahead = frames(70.0), frames(82.0)
        seam = seam_morph(held, ahead)
        self.assertEqual(tuple(seam.shape), tuple(held.shape))
        row = seam[2, 48, :, 0].numpy()
        # Halfway through the seam the blob is one shape between its two places (about column 76) at close to its
        # full brightness; a plain mix would show two dimmer blobs at 70 and 82.
        self.assertTrue(73 <= int(row.argmax()) <= 79, int(row.argmax()))
        self.assertGreater(float(row.max()), 0.75)
        centers = [float((seam[i, 48, :, 0].numpy() * np.arange(160)).sum() / seam[i, 48, :, 0].numpy().sum()) for i in range(5)]
        self.assertEqual(centers, sorted(centers))


class SoundSeamTests(unittest.TestCase):
    def test_the_sound_of_one_part_fades_into_the_next_over_the_overlap(self):
        import torch
        from backend import long_faceswap
        finished = {}

        class Writer:
            def __init__(self, destination):
                self.frames, self.size = 0, (16, 16)

            def add(self, frames):
                self.frames += frames.shape[0]

            def finish(self, audio):
                finished['audio'] = audio

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.full((count, 16, 16, 3), 0.5)

        calls = []

        def generate(request, prompt, images, progress, preview_directory=None):
            calls.append(request['frames'])
            level = 0.2 if len(calls) == 1 else 0.6
            return images['source_video'].clone(), {'waveform': torch.full((1, 2, round(request['frames'] / 24 * 24000)), level),
                                                    'sample_rate': 24000}

        request = {'kind': 'reference-video', 'workflow': 'reference', 'frames': 243, 'steps': 8, 'source_start_seconds': 0,
                   'source_audio': False, 'output_audio': 'generated', 'width': 16, 'height': 16}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch('backend.reference_media.read_frames', read_frames), \
                mock.patch.object(long_faceswap, 'Mp4PartWriter', Writer), \
                mock.patch.object(long_faceswap, '_save_part', lambda *a: None):
            long_faceswap.swap_long(request, 'Prompt', {}, Path('source.mp4'), lambda **values: None, generate,
                                    Path(folder) / 'out.mp4')
        sound = finished['audio']['waveform'][0, 0]
        # 243 frames of sound at 24000 Hz; the 5 overlap frames (119 to 123) are 5000 samples.
        self.assertEqual(sound.shape[-1], 243000)
        self.assertAlmostEqual(float(sound[118000]), 0.2, places=4)
        self.assertAlmostEqual(float(sound[124500]), 0.6, places=4)
        seam = sound[119000:124000]
        self.assertTrue(bool((seam[1:] >= seam[:-1]).all()), 'the level only rises through the seam')
        self.assertTrue(0.2 < float(seam[0]) < 0.21 and 0.59 < float(seam[-1]) < 0.6)
        self.assertAlmostEqual(float(seam[2500]), 0.4, places=2)


class RepairTests(unittest.TestCase):
    def test_a_repair_keeps_the_parts_before_it_and_makes_the_others_again(self):
        import json
        import av
        import torch
        from backend import long_faceswap
        from backend.video_worker import save_mp4
        made = []

        def read_frames(path, count, start_seconds, end_seconds, progress=None):
            return torch.zeros(count, 64, 64, 3)

        def generate(request, prompt, images, progress, preview_directory=None):
            made.append(request['frames'])
            return torch.full((request['frames'], 64, 64, 3), 0.6), None

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            jobs = folder / 'jobs'; (jobs / ('a' * 32)).mkdir(parents=True)
            original_video = folder / 'original.mp4'
            sound = {'waveform': torch.full((1, 2, round(400 / 24 * 32000)), 0.1), 'sample_rate': 32000}
            save_mp4(original_video, torch.full((400, 64, 64, 3), 0.2), sound)
            parts = [{'index': 1, 'start_seconds': 0.0, 'end_seconds': 124 / 24}, {'index': 2, 'start_seconds': 124 / 24, 'end_seconds': 243 / 24},
                     {'index': 3, 'start_seconds': 243 / 24, 'end_seconds': 400 / 24}]
            (jobs / ('a' * 32) / 'status.json').write_text(json.dumps({'status': 'succeeded', 'output_path': str(original_video), 'has_audio': True,
                'total_parts': 3, 'parts': parts, 'request': {'frames': 400}}), encoding='utf-8')
            request = {'kind': 'reference-video', 'workflow': 'faceswap', 'frames': 400, 'steps': 8, 'source_start_seconds': 0,
                       'source_audio': False, 'output_audio': 'generated', 'width': 64, 'height': 64,
                       'repair_of': 'a' * 32, 'repair_from_part': 2}
            with mock.patch('backend.common.JOBS_DIR', jobs),                     mock.patch('backend.reference_media.read_frames', read_frames),                     mock.patch('backend.reference_shots.shot_bounds', lambda frames: [(0, frames.shape[0])]):
                destination = folder / 'out.mp4'
                count, *_ = long_faceswap.swap_long(request, 'Faceswap', {}, Path('source.mp4'), lambda **values: None, generate, destination)
            with av.open(str(destination)) as container:
                levels = [float(frame.to_ndarray(format='rgb24').mean()) / 255 for frame in container.decode(container.streams.video[0])]
                has_sound = len(container.streams.audio) == 1
        # Only parts 2 and 3 are made again (part 2 with its 5 seam frames, part 3 with the last frames joined to it).
        self.assertEqual((count, len(levels), made, has_sound), (400, 400, [124, 175], True))
        # The new parts came out brighter (0.6) from the same source; they are matched to the kept part 1 (0.2).
        self.assertAlmostEqual(levels[60], 0.2, delta=0.02)
        self.assertAlmostEqual(levels[300], 0.2, delta=0.02)
        self.assertLess(max(levels) - min(levels), 0.04)


class LookTests(unittest.TestCase):
    def test_a_part_darker_than_part_1_from_the_same_source_is_brought_back_to_part_1(self):
        import torch
        from backend.long_faceswap import LookMatcher
        torch.manual_seed(0)
        source = torch.rand(40, 72, 128, 3) * 0.6 + 0.2
        part_1 = (source * 1.05).clamp(0, 1)
        darker = (source * 0.8).clamp(0, 1)
        looks = LookMatcher()
        self.assertIs(looks.apply(part_1, source), part_1, 'part 1 sets the look and stays as it is')
        matched = looks.apply(darker, source)
        brightness = lambda frames: LookMatcher.look(frames)[0][0]
        self.assertLess(abs(brightness(matched) - brightness(part_1)), 1.0)
        self.assertGreater(abs(brightness(darker) - brightness(part_1)), 10.0)

    def test_a_real_change_of_light_in_the_source_carries_through(self):
        import torch
        from backend.long_faceswap import LookMatcher
        looks = LookMatcher()
        looks.apply(torch.full((12, 72, 128, 3), 0.4), torch.full((12, 72, 128, 3), 0.4))
        brighter = looks.apply(torch.full((12, 72, 128, 3), 0.6), torch.full((12, 72, 128, 3), 0.6))
        self.assertAlmostEqual(float(brighter.mean()), 0.6, delta=0.03)


if __name__ == '__main__':
    unittest.main()


class PartsSoFarTests(unittest.TestCase):
    def test_the_finished_parts_are_one_playable_file_that_grows_with_each_part(self):
        import av
        import torch
        from backend import long_faceswap, parts_video

        def frames(path):
            with av.open(str(path)) as container:
                return sum(1 for _ in container.decode(container.streams.video[0])), bool(container.streams.audio)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.assertIsNone(parts_video.joined_parts(folder, 3))
            for index, count in enumerate((12, 10, 7)):
                sound = torch.zeros(1, 2, round(count / 24 * 48000))
                long_faceswap._save_part(folder, index, torch.full((count, 32, 32, 3), 0.2 * (index + 1)), sound, 48000)
            first = parts_video.joined_parts(folder, 2)
            self.assertEqual(first.name, 'parts-001-to-002.mp4')
            self.assertEqual(frames(first), (22, True))
            self.assertEqual(parts_video.joined_parts(folder, 2), first)
            third = parts_video.joined_parts(folder, 3)
            self.assertEqual(frames(third), (29, True))
            # One file of the run so far: the shorter one is removed, and no working file stays.
            self.assertEqual(sorted(path.name for path in folder.glob('*') if not path.name.startswith('part-')), ['parts-001-to-003.mp4'])
            # A part that is not on disk ends the join there.
            (folder / 'part-002.mp4').unlink()
            self.assertEqual(frames(parts_video.joined_parts(folder, 3)), (12, True))

    def test_only_a_job_that_did_not_finish_and_has_parts_counts(self):
        from backend.parts_video import unfinished_with_parts
        self.assertTrue(unfinished_with_parts({'status': 'running', 'parts_saved': 5, 'parts_folder': 'x'}))
        self.assertTrue(unfinished_with_parts({'status': 'cancelled', 'parts_saved': 15, 'parts_folder': 'x'}))
        self.assertTrue(unfinished_with_parts({'status': 'failed', 'parts_saved': 1, 'parts_folder': 'x'}))
        self.assertFalse(unfinished_with_parts({'status': 'succeeded', 'parts_saved': 17, 'parts_folder': 'x'}))
        self.assertFalse(unfinished_with_parts({'status': 'running', 'parts_saved': 0, 'parts_folder': 'x'}))
        self.assertFalse(unfinished_with_parts({'status': 'queued'}))

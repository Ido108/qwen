"""Check that the selected source interval reaches the model as matching video and audio."""
from pathlib import Path
import tempfile
import unittest

from backend.common import ReferenceVideoRequest
from backend.models import required_reference
from backend.reference_prompt import render_reference_prompt
from backend.reference_timing import nearest_frames, resolve_reference_clip


class ReferenceTimingTests(unittest.TestCase):
    def test_model_resolution_grid_and_aspect_limits(self):
        portrait = ReferenceVideoRequest(source_video='sample.mp4',reference_images=['person.png'],
                                         width=576,height=1024)
        self.assertIn('h3-ref-turbo-int8',required_reference(portrait))
        extreme = portrait.model_copy(update={'width':256,'height':1344})
        with self.assertRaisesRegex(ValueError,'aspect ratios'):
            required_reference(extreme)
        oversized = portrait.model_copy(update={'width':1344,'height':1024})
        with self.assertRaisesRegex(ValueError,'must not exceed'):
            required_reference(oversized)

    def test_default_and_trim_follow_source_with_model_frame_grid(self):
        request = ReferenceVideoRequest(source_video='sample.mp4', frames=124,
                                        source_start_seconds=0, source_end_seconds=14)
        resolve_reference_clip(request, 14)
        self.assertEqual(request.frames, nearest_frames(14))
        self.assertEqual((request.source_start_seconds, request.source_end_seconds), (0, 14))
        request.source_start_seconds, request.source_end_seconds = 8, 14
        resolve_reference_clip(request, 14)
        self.assertEqual(request.frames, nearest_frames(6))
        with self.assertRaisesRegex(ValueError, 'at most'):
            resolve_reference_clip(ReferenceVideoRequest(source_video='sample.mp4', source_end_seconds=14),
                                   14, max_frames=124)

    def test_trimmed_video_and_audio_use_the_same_later_scene(self):
        import numpy as np
        import torch
        from backend.reference_media import read_audio, read_frames
        from backend.video_worker import save_mp4

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'two-scenes.mp4'
            frames = torch.zeros((72, 64, 64, 3))
            frames[:24,:,:,0] = 1
            frames[24:48,:,:,1] = 1
            frames[42:48,:,:,1] = 0
            frames[42:48,:,:,2] = 1
            frames[48:,:,:,2] = 1
            sample = np.arange(3 * 32000, dtype=np.float32) / 32000
            waveform = (np.sin(2*np.pi*np.where(sample<1,440,np.where(sample<2,660,880))*sample)*0.15).astype(np.float32)
            audio = {'waveform':torch.from_numpy(np.stack([waveform,waveform])).unsqueeze(0),
                     'sample_rate':32000}
            save_mp4(path, frames, audio)
            selected_frames = read_frames(path, 22, start_seconds=1.0, end_seconds=1.0+22/24)
            self.assertEqual(selected_frames.shape, (22,64,64,3))
            self.assertGreater(selected_frames[0,:,:,1].mean().item(), 0.8)
            self.assertLess(selected_frames[0,:,:,0].mean().item(), 0.1)
            full_segment = read_frames(path, 39, start_seconds=0, end_seconds=2.0)
            self.assertGreater(full_segment[-1,:,:,2].mean().item(), 0.8)
            selected_audio = read_audio(path, 22/24, start_seconds=1.0, end_seconds=1.0+22/24)
            self.assertEqual(selected_audio['waveform'].shape, (1,2,round(22/24*32000)))
            chunk = selected_audio['waveform'][0,0,1000:16000].numpy()
            spectrum = np.abs(np.fft.rfft(chunk))
            dominant = np.fft.rfftfreq(len(chunk),1/32000)[int(np.argmax(spectrum))]
            self.assertAlmostEqual(dominant,660,delta=10)

    def test_person_and_reference_tags_resolve_to_model_media_and_validate_mapping(self):
        request = {'reference_images':['a.png','b.png'],
                   'source_people':[{'id':'person-1','time':0.5},{'id':'person-2','time':2.5}],
                   'person_assignments':{'person-1':'b.png','person-2':None}}
        result = render_reference_prompt('Use @Ref2 for @Person1; keep @Person2 in @SourceVideo.',request)
        self.assertIn('<Picture 2>',result)
        self.assertIn('<Video 1>',result)
        self.assertIn('Keep the source person first seen at 2.5s',result)
        with self.assertRaisesRegex(ValueError,'removed reference'):
            render_reference_prompt('Test',{**request,'person_assignments':{'person-1':'missing.png'}})


if __name__ == '__main__':
    unittest.main()

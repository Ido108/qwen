"""Check the actual video container, frame clock, audio and invalid output handling."""
from pathlib import Path
import tempfile
import unittest


class VideoOutputTests(unittest.TestCase):
    def test_mp4_has_exact_frames_fps_and_optional_audio(self):
        import av
        import torch
        from backend.video_worker import save_mp4
        frames=torch.zeros((24,32,64,3))
        frames[:,:,:,0]=torch.linspace(0,1,24).view(24,1,1)
        tone=torch.sin(torch.arange(32000)*2*torch.pi*440/32000)*0.1
        audio={'waveform':tone.repeat(2,1).unsqueeze(0),'sample_rate':32000}
        with tempfile.TemporaryDirectory() as directory:
            for include_audio in (True,False):
                path=Path(directory)/f'video-{include_audio}.mp4'
                save_mp4(path,frames,audio if include_audio else None)
                with av.open(str(path)) as container:
                    stream=container.streams.video[0]
                    self.assertEqual(stream.average_rate,24)
                    self.assertEqual((stream.width,stream.height),(64,32))
                    self.assertEqual(len(container.streams.audio),int(include_audio))
                    decoded=list(container.decode(video=0))
                    self.assertEqual(len(decoded),24)
                    self.assertAlmostEqual(float(decoded[-1].time),23/24,places=4)
                    self.assertGreater(decoded[-1].to_ndarray(format='rgb24')[:,:,0].mean(),240)
                if include_audio:
                    with av.open(str(path)) as container:
                        stream=container.streams.audio[0]
                        self.assertEqual(stream.codec_context.sample_rate,32000)
                        self.assertEqual(stream.codec_context.channels,2)
                        self.assertAlmostEqual(float(stream.duration*stream.time_base),1,places=2)

    def test_invalid_pixels_are_not_saved_as_successful_video(self):
        import torch
        from backend.video_worker import save_mp4
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'invalid video pixels'):
                save_mp4(Path(directory)/'invalid.mp4',torch.full((1,32,32,3),float('nan')),None)

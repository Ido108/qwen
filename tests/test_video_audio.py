"""Codec-only fixtures, no models or CUDA inference."""
from pathlib import Path
import tempfile
import unittest
import subprocess
import json
from unittest.mock import patch
import numpy as np
import av
import imageio_ffmpeg
from backend.video_audio import add_source_audio
from backend.seedvr2_cli import H264Writer


class VideoAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.video=self.root/'video.mp4'
        writer=H264Writer(self.video,32,48,24)
        for i in range(24):writer.write(np.full((48,32,3),i*8,dtype=np.uint8))
        writer.release()
    def test_h264_output_plays_and_source_audio_is_bounded_for_short_and_long_tracks(self):
        for duration in (.2,2):
            audio=self.root/f'audio-{duration}.wav'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-f','lavfi','-i',
                f'sine=frequency=440:duration={duration}','-y',str(audio)],check=True,timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name=='nt' else 0)
            target=self.root/f'output-{duration}.mp4';events=[]
            add_source_audio(self.video,audio,target,lambda **v:events.append(v))
            with av.open(str(target)) as media:
                self.assertEqual(media.streams.video[0].codec_context.name,'h264')
                self.assertEqual(media.streams.audio[0].codec_context.name,'aac')
                self.assertLess(abs(media.duration/av.time_base-1),.06)
                self.assertEqual(sum(1 for _ in media.decode(video=0)),24)
            self.assertEqual(events[0]['stage'],'muxing_audio')
    def test_no_audio_does_not_create_silence_or_lose_frames(self):
        target=self.root/'silent.mp4'
        add_source_audio(self.video,self.video,target,lambda **v:None)
        self.assertEqual(target.read_bytes(),self.video.read_bytes())
    def test_legacy_mpeg4_without_audio_is_normalized_to_h264(self):
        legacy=self.root/'legacy.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-i',str(self.video),
            '-an','-c:v','mpeg4','-y',str(legacy)],check=True,timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name=='nt' else 0)
        target=self.root/'playable.mp4'
        add_source_audio(legacy,legacy,target,lambda **v:None)
        with av.open(str(target)) as media:
            self.assertEqual(media.streams.video[0].codec_context.name,'h264')
            self.assertFalse(media.streams.audio)
            self.assertEqual(sum(1 for _ in media.decode(video=0)),24)
    def test_timeout_stops_only_the_muxer(self):
        class Media:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            streams=type('Streams',(),{'audio':[1],'video':[type('Video',(),{'codec_context':type('Codec',(),{'name':'h264'})()})()]})()
        with patch('av.open',return_value=Media()),patch('backend.reference_timing.probe_video_duration',return_value=1), \
             patch('backend.video_audio.subprocess.Popen') as process,patch('backend.video_audio.time.monotonic',side_effect=[0,121]):
            process.return_value.communicate.side_effect=[subprocess.TimeoutExpired('ffmpeg',1),('','')]
            process.return_value.poll.return_value=None
            with self.assertRaisesRegex(RuntimeError,'took too long'):
                add_source_audio(self.video,self.video,self.root/'output.mp4',lambda **v:None)
            process.return_value.kill.assert_called_once()


if __name__=='__main__':unittest.main()

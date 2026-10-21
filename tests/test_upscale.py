"""Model contracts and tiny codec fixtures only. No weights, CUDA work or downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from backend.common import UpscaleRequest
from backend.models import catalog,entry,required_upscale
from backend.upscale_settings import output_dimensions,seed_memory_options,seed_vae_options
from backend.stream_upscale import frame_batches,swift
from backend.seedvr2_cli import H264Writer,publish_chunk


class UpscaleTests(unittest.TestCase):
    def test_every_model_has_one_step_and_complete_pinned_dependencies(self):
        models=[m for m in catalog() if m['kind']=='upscale']
        self.assertGreaterEqual(len(models),25)
        seen=set()
        for item in models:
            with self.subTest(model=item['id']):
                self.assertEqual(item['upscale_defaults']['steps'],1)
                self.assertEqual(len(item['revision']),40)
                needed=required_upscale(UpscaleRequest(source_video='a.mp4',model_id=item['id']))
                for model_id in needed:
                    for file in entry(model_id)['files']:
                        self.assertGreater(file['bytes'],0)
                        self.assertRegex(file['sha256'],r'^[a-f0-9]{64}$')
                key=(item['repo'],item['files'][0]['path'])
                self.assertNotIn(key,seen);seen.add(key)
                if item['family']=='seedvr2':self.assertIn('seedvr2-vae',needed)
                if item['family']=='swiftvr':self.assertIn('swiftvr-reae',needed)
                if item['family']=='flashvsr':self.assertIn('flashvsr-tcdecoder-fp16',needed)
        for group in ('SeedVR2 3B','SeedVR2 7B','SeedVR2 7B Sharp'):
            quants={m['quantization'] for m in models if m['model_group']==group}
            self.assertTrue({'Q3_K_M','Q4_K_M','Q5_K_M','Q6_K','Q8_0'}.issubset(quants))

    def test_geometry_preserves_source_and_limits_codec_rounding_to_two_pixels(self):
        p=UpscaleRequest(source_video='a.mp4').model_dump()
        self.assertEqual(output_dimensions(1080,1920,p),(1080,1920))
        self.assertEqual(output_dimensions(3840,2160,p),(1920,1080))
        self.assertEqual(output_dimensions(640,480,{**p,'size_mode':'scale','scale':4}),(2560,1920))
        for w,h in [(541,961),(1920,803),(1001,1001),(1928,1080)]:
            x,y=output_dimensions(w,h,p)
            self.assertLess(abs(x/y-w/h),.003)
            self.assertFalse(x%2 or y%2)
        with self.assertRaisesRegex(ValueError,'smaller'):output_dimensions(3840,2160,{**p,'size_mode':'scale','scale':4})

    def test_family_specific_validation_and_memory_policy(self):
        full=required_upscale(UpscaleRequest(source_video='a.mp4',model_id='flashvsr11-fp32',flash_decoder='full'))
        self.assertIn('flashvsr-wan-vae',full)
        self.assertNotIn('flashvsr-tcdecoder-fp16',full)
        with self.assertRaises(ValueError):UpscaleRequest(source_video='a.mp4',clip_len=7)
        with self.assertRaises(ValueError):required_upscale(UpscaleRequest(source_video='a.mp4',temporal_overlap=5))
        self.assertIn('swiftvr-reae',required_upscale(UpscaleRequest(source_video='a.mp4',model_id='swiftvr-bf16',temporal_overlap=9)))
        request=UpscaleRequest(source_video='a.mp4').model_dump()
        self.assertEqual(seed_memory_options(entry('seedvr2-3b-q4'),request,24),[])
        low=seed_memory_options(entry('seedvr2-7b-fp16'),request,12)
        self.assertIn('--blocks_to_swap',low)
        self.assertEqual(seed_memory_options(entry('seedvr2-7b-fp16'),{**request,'memory_mode':'gpu'},12),[])
        # 24 GB or more: 3B FP16 with 21 frames stays on the GPU at 1080p and 2160p; no forced BlockSwap at 4K.
        self.assertEqual(seed_memory_options(entry('seedvr2-3b-fp16'),{**request,'batch_size':21},28),[])
        self.assertEqual(seed_memory_options(entry('seedvr2-3b-fp16'),{**request,'resolution':2160},28),[])
        # The decoder stays tiled at 1080p: untiled it spilled 11 GB to shared memory.
        self.assertEqual(seed_vae_options(1080,28),['--vae_decode_tiled','--vae_decode_tile_size','768'])
        self.assertEqual(seed_vae_options(2160,28),['--vae_encode_tiled','--vae_decode_tiled'])
        self.assertEqual(seed_vae_options(1080,12),['--vae_encode_tiled','--vae_decode_tiled'])
        self.assertEqual(seed_vae_options(1080,28,conservative=True),['--vae_encode_tiled','--vae_decode_tiled'])
        self.assertEqual(seed_vae_options(720,28),[])
        self.assertEqual(UpscaleRequest(source_video='a.mp4',batch_size=121).batch_size,121)

    def test_stream_reader_overlap_and_preview_use_real_frames(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.mp4'
            writer=H264Writer(source,32,48,24)
            for i in range(37):writer.write(np.full((48,32,3),i*5,dtype=np.uint8))
            writer.release()
            chunks=list(frame_batches(source,17,8))
            self.assertEqual(sum(len(frames)-skip for frames,skip in chunks),37)
            self.assertTrue(all(len(frames)<=17 for frames,_ in chunks))
            details=publish_chunk(torch.from_numpy(chunks[0][0]).float()/255,root,1,24)
            self.assertEqual(details['preview_chunk'],1)
            with Image.open(root/'live-preview.webp') as im:
                self.assertGreater(im.n_frames,1)
                self.assertEqual(im.size,(32,48))

    def test_swift_preserves_tail_for_every_frame_remainder_without_real_inference(self):
        class Session:
            def __init__(self):self.left=[];self.first=True
            def step(self,frames):
                self.left.extend(list(frames));n=len(self.left)//4*4
                if not n:return None
                batch=self.left[:n];self.left=self.left[n:]
                output=torch.stack(batch).float()/255
                if self.first:output=output[3:];self.first=False
                return output.permute(0,3,1,2).unsqueeze(0)
            def flush(self):
                if not self.left:return None
                return self.step(torch.stack([self.left[-1]]*(4-len(self.left))))
        class Pipe:
            reae=type('ReAE',(),{'frames_to_trim':3})()
            def stream(self,**kw):return Session()
        class Output:
            dimensions=(32,48)
            def __init__(self):self.count=0
            def write(self,frames):self.count+=len(frames)
        for length in [1,2,3,4,5,8,9,12,14,17,24,25,28,37]:
            with self.subTest(frames=length):
                frames=np.zeros((length,48,32,3),np.uint8)
                batches=[(frames[i:i+8],0) for i in range(0,length,8)]
                output=Output()
                with patch('backend.stream_upscale.load_swift',return_value=Pipe()),patch('backend.stream_upscale.frame_batches',return_value=batches):
                    swift({'clip_len':8},{},None,output,lambda **kw:None)
                self.assertEqual(output.count,length)


if __name__=='__main__':unittest.main()

"""Exercise preview transport with small CPU tensors; never load a model or CUDA."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.video_preview import LiveVideoPreview
from backend.video_conditioning import ProgressVAE, SingleUseEncoder


class PreviewTests(unittest.TestCase):
    def test_actual_latent_samples_produce_changing_animated_preview(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            updates=[]
            # Three channels with known RGB mapping make output provenance testable.
            format=SimpleNamespace(latent_rgb_factors=[[1,0,0],[0,1,0],[0,0,1]],latent_rgb_factors_bias=[0,0,0])
            preview=LiveVideoPreview(Path(folder),format,lambda **v:updates.append(v),5)
            video=torch.zeros(1,3,20,4,6)
            video[:,0]=torch.linspace(-1,1,20).reshape(1,20,1,1)
            preview.publish(video,1)
            target=Path(folder)/'live-preview.webp'
            with Image.open(target) as image:
                self.assertGreater(image.n_frames,1)
                self.assertLessEqual(image.n_frames,12)
                self.assertEqual(image.size,(24,16))
                image.seek(0);first=image.convert('RGB').getpixel((0,0))[0]
                image.seek(image.n_frames-1);last=image.convert('RGB').getpixel((0,0))[0]
                self.assertGreater(last,first+150)
            self.assertEqual(updates[-1]['preview_step'],1)
            preview.publish(video+0.1,2)
            self.assertEqual(updates[-1]['preview_step'],2)
            self.assertEqual(updates[-1]['preview_url'],updates[0]['preview_url'])

    def test_bad_preview_never_breaks_generation(self):
        import torch
        updates=[]
        preview=LiveVideoPreview(Path('.'),SimpleNamespace(latent_rgb_factors=[[1,0,0]]),lambda **v:updates.append(v),5)
        with patch('backend.video_preview.logging.exception'):
            preview.publish(torch.zeros(1,3,4,4),1)
        self.assertFalse(preview.enabled)
        self.assertTrue(updates[-1]['preview_unavailable'])

    def test_stage_wrappers_forward_exact_inputs_and_outputs(self):
        import torch
        updates=[]
        value=torch.zeros(22,4,4,3)
        class VAE:
            def encode(self,pixels):
                self.seen=pixels
                return pixels
        vae=VAE()
        wrapped=ProgressVAE(vae,lambda **v:updates.append(v))
        self.assertIs(wrapped.encode(value),value)
        self.assertIs(vae.seen,value)
        self.assertEqual(updates[-1]['stage'],'encoding_source_video')
        wrapped.encode(value[:1])
        self.assertEqual(updates[-1]['stage'],'encoding_reference_image')
        ProgressVAE(vae,lambda **v:updates.append(v),audio=True).encode(value)
        self.assertEqual(updates[-1]['stage'],'encoding_source_audio')

    def test_optional_preview_endpoint_serves_only_existing_job_files(self):
        from backend import server
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as folder, patch.object(server,'JOBS_DIR',Path(folder)), \
             patch.object(server,'read_job',return_value={'kind':'reference-video'}):
            with self.assertRaises(HTTPException):server.live_video_preview('a'*32)
            path=Path(folder)/('a'*32);path.mkdir();(path/'live-preview.webp').write_bytes(b'preview')
            result=server.live_video_preview('a'*32)
            self.assertEqual(result.media_type,'image/webp')
            self.assertEqual(result.headers['cache-control'],'no-store')

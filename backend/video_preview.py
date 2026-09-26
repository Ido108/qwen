"""Small animated previews of denoising estimates, without a second VAE."""
from __future__ import annotations

import logging
from pathlib import Path
import time


class LiveVideoPreview:
    def __init__(self, directory: Path | None, latent_format, progress, duration: float):
        self.directory, self.progress, self.duration = directory, progress, duration
        self.factors = getattr(latent_format, 'latent_rgb_factors', None)
        self.bias = getattr(latent_format, 'latent_rgb_factors_bias', None)
        self.enabled = directory is not None and self.factors is not None

    def publish(self, estimate, step: int):
        if not self.enabled:
            return
        started = time.monotonic()
        try:
            import torch
            from PIL import Image
            # ComfyUI's sampler callback unpacks joint video/audio latents.
            video = estimate.unbind()[0] if estimate.is_nested else estimate
            if video.ndim != 5 or video.shape[1] != len(self.factors):
                raise ValueError('The sampler did not provide compatible video latents.')
            count = min(12, video.shape[2])
            indices = torch.linspace(0, video.shape[2]-1, count, device=video.device).round().long()
            # Copy only a few latent frames. Never decode the full clip or move
            # model weights off the GPU for a preview.
            frames = video.detach()[0].index_select(1, indices).movedim(0, -1).float().cpu()
            rgb = frames @ torch.tensor(self.factors, dtype=torch.float32)
            if self.bias is not None:
                rgb += torch.tensor(self.bias, dtype=torch.float32)
            pixels = ((rgb+1)/2).clamp(0, 1).mul(255).byte().numpy()
            pictures = []
            for array in pixels:
                picture = Image.fromarray(array)
                scale = min(4, 384/max(picture.size))
                picture = picture.resize(tuple(max(1,round(v*scale)) for v in picture.size), Image.Resampling.BILINEAR)
                pictures.append(picture)
            target = self.directory / 'live-preview.webp'
            temporary = self.directory / 'live-preview.tmp'
            pictures[0].save(temporary, format='WEBP', save_all=True, append_images=pictures[1:],
                duration=max(80,min(600,round(self.duration/count*1000))), loop=0, quality=65, method=0)
            temporary.replace(target)
            self.progress(preview_url=f'/api/jobs/{self.directory.name}/live-preview', preview_step=step,
                preview_updated_at=time.time(), preview_seconds=round(time.monotonic()-started,3))
        except Exception:
            # Preview errors must not discard an expensive running generation.
            logging.exception('Could not prepare the optional video preview')
            self.enabled = False
            self.progress(preview_unavailable=True)

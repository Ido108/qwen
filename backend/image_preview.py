"""Live image previews: each sampling step's estimate as a small picture, the way ComfyUI's latent2rgb preview does
it (no second VAE, no weights moved). The worker sets `directory` to the running job's folder."""
from __future__ import annotations

import logging
import time
from pathlib import Path

directory: Path | None = None


class LiveImagePreview:
    def __init__(self, latent_format, progress):
        self.progress = progress
        self.factors = getattr(latent_format, 'latent_rgb_factors', None)
        self.bias = getattr(latent_format, 'latent_rgb_factors_bias', None)
        self.reshape = getattr(latent_format, 'latent_rgb_factors_reshape', None)
        self.directory = directory
        self.enabled = self.directory is not None and self.factors is not None
        self.last = 0.0

    def publish(self, estimate, step: int, total: int):
        # At most about three pictures a second, and always the last step.
        if not self.enabled or estimate is None or (step + 1 < total and time.monotonic() - self.last < 0.3):
            return
        self.last = time.monotonic()
        try:
            import torch
            from PIL import Image
            latent = estimate.detach()[:1]
            if self.reshape is not None:
                latent = self.reshape(latent)
            if latent.ndim == 5:  # a single-frame video latent
                latent = latent[:, :, 0]
            if latent.shape[1] != len(self.factors):
                raise ValueError('The sampler latent does not match the preview colors.')
            rgb = torch.einsum('chw,cr->hwr', latent[0].float(), torch.tensor(self.factors, dtype=torch.float32, device=latent.device))
            if self.bias is not None:
                rgb += torch.tensor(self.bias, dtype=torch.float32, device=latent.device)
            picture = Image.fromarray(((rgb + 1) / 2).clamp(0, 1).mul(255).byte().cpu().numpy())
            scale = 768 / max(picture.size)
            picture = picture.resize(tuple(max(1, round(v * scale)) for v in picture.size), Image.Resampling.BILINEAR)
            temporary = self.directory / 'live-preview.tmp'
            picture.save(temporary, format='WEBP', quality=70, method=0)
            temporary.replace(self.directory / 'live-preview.webp')
            self.progress(preview_url=f'/api/jobs/{self.directory.name}/live-preview', preview_step=step + 1,
                          preview_updated_at=time.time())
        except Exception:
            # A preview must never stop the image being made.
            logging.exception('Could not prepare the optional image preview')
            self.enabled = False

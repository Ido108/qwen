"""Release the one-use H3 encoder before the upstream node starts VAE work."""
import hashlib
import json

CACHE_ENTRIES = 24


def conditioning_key(parts: dict, images: dict) -> str:
    """Encoder output depends on the encoder and VAE files, the prompt, the size and the keyframe pixels, never the seed."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode('utf-8'))
    for name in sorted(images):
        image = images[name]
        digest.update(f'{name}|{image.mode}|{image.size}'.encode('utf-8'))
        digest.update(image.tobytes())
    return digest.hexdigest()


def load_conditioning(directory, key):
    import torch
    path = directory / f'{key}.pt'
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
        path.touch()
        return value
    except Exception:
        return None  # Missing, older or damaged entry: encode again.


def save_conditioning(directory, key, value) -> None:
    import torch
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temp = directory / f'{key}.tmp'
        torch.save(value, temp)
        temp.replace(directory / f'{key}.pt')
        entries = sorted(directory.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in entries[CACHE_ENTRIES:]:
            stale.unlink(missing_ok=True)
    except Exception:
        pass  # The cache only saves time; a failed write never stops the job.


class SingleUseEncoder:
    def __init__(self, encoder, release, progress=lambda **_: None):
        self.encoder = encoder
        self.release = release
        self.progress = progress

    def tokenize(self, *args, **kwargs):
        self.progress(stage='preparing_conditioning', message='Preparing prompt and reference inputs…')
        return self.encoder.tokenize(*args, **kwargs)

    def encode_from_tokens_scheduled(self, *args, **kwargs):
        self.progress(stage='encoding_prompt', message='Encoding prompt and visual references on GPU…')
        try:
            # Imports stay lazy so small CPU contract tests need no Comfy runtime.
            if getattr(self.encoder,'qwen_timing',None):
                with self.encoder.qwen_timing():
                    return self.encoder.encode_from_tokens_scheduled(*args, **kwargs)
            return self.encoder.encode_from_tokens_scheduled(*args, **kwargs)
        finally:
            # Dropping the weights avoids ComfyUI copying partially loaded
            # quantized weights back to CPU, which can crash on Windows.
            self.encoder = None
            self.release()


class ProgressVAE:
    """Report actual upstream VAE calls without changing their tensors or device."""
    def __init__(self, vae, progress, *, audio=False):
        self.vae, self.progress, self.audio = vae, progress, audio

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, pixels, *args, **kwargs):
        if self.audio:
            stage, message = 'encoding_source_audio', 'Encoding source audio on GPU…'
        elif pixels.shape[0] > 1:
            stage, message = 'encoding_source_video', f'Encoding source video on GPU ({pixels.shape[0]} frames)…'
        else:
            stage, message = 'encoding_reference_image', 'Encoding reference image on GPU…'
        self.progress(stage=stage, message=message)
        if getattr(self.vae,'qwen_timing',None):
            with self.vae.qwen_timing(stage,message):
                return self.vae.encode(pixels,*args,**kwargs)
        return self.vae.encode(pixels, *args, **kwargs)

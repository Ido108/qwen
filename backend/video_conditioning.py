"""Release the one-use H3 encoder before the upstream node starts VAE work."""


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

"""Release the one-use H3 encoder before the upstream node starts VAE work."""


class SingleUseEncoder:
    def __init__(self, encoder, release):
        self.encoder = encoder
        self.release = release

    def tokenize(self, *args, **kwargs):
        return self.encoder.tokenize(*args, **kwargs)

    def encode_from_tokens_scheduled(self, *args, **kwargs):
        try:
            return self.encoder.encode_from_tokens_scheduled(*args, **kwargs)
        finally:
            # Dropping the weights avoids ComfyUI copying partially loaded
            # quantized weights back to CPU, which can crash on Windows.
            self.encoder = None
            self.release()

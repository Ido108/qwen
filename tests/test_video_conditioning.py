import gc
import unittest
import weakref

from backend.video_conditioning import SingleUseEncoder


class VideoConditioningTests(unittest.TestCase):
    def test_encoder_weights_are_released_before_keyframe_vae_encoding(self):
        class Encoder:
            def tokenize(self, prompt, images):
                return (prompt, images)

            def encode_from_tokens_scheduled(self, tokens):
                return {'conditioning': tokens}

        encoder = Encoder()
        weights_owner = weakref.ref(encoder)
        released = []

        def release():
            gc.collect()
            released.append(weights_owner() is None)

        one_shot = SingleUseEncoder(encoder, release)
        del encoder
        tokens = one_shot.tokenize('Dance', images=['first', 'last'])
        conditioning = one_shot.encode_from_tokens_scheduled(tokens)
        # This is the boundary at which the upstream node calls VAE.encode.
        self.assertIsNone(weights_owner())
        self.assertEqual(released, [True])
        self.assertEqual(conditioning, {'conditioning': ('Dance', ['first', 'last'])})

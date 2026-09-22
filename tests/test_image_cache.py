import unittest
from unittest.mock import Mock
from PIL import Image
from backend.image_cache import ConditioningCache, PositiveOnlyEncoder, retain_cpu_weights


class ImageCacheTests(unittest.TestCase):
    def test_conditioning_key_tracks_prompt_negative_pixels_order_and_shape(self):
        cache = ConditioningCache()
        red, blue = Image.new('RGB', (8, 8), 'red'), Image.new('RGB', (8, 8), 'blue')
        key = cache.fingerprint('prompt', '', [red, blue])
        cache.put(key, ('positive', 'negative'))
        self.assertEqual(cache.get(cache.fingerprint('prompt', '', [red.copy(), blue])), ('positive', 'negative'))
        for args in [('changed', '', [red, blue]), ('prompt', 'negative', [red, blue]),
                     ('prompt', '', [blue, red]), ('prompt', '', [red])]:
            self.assertIsNone(cache.get(cache.fingerprint(*args)))
        cache.put('new', ('next', 'next'))
        self.assertIsNone(cache.get(key))

    def test_cfg_one_skips_negative_but_cfg_two_encodes_both(self):
        for negative, expected in [(False, 1), (True, 2)]:
            clip = Mock()
            wrapper = PositiveOnlyEncoder(clip, negative)
            wrapper.encode_from_tokens_scheduled(wrapper.tokenize('positive'))
            wrapper.encode_from_tokens_scheduled(wrapper.tokenize('negative'))
            self.assertEqual(clip.tokenize.call_count, expected)
            self.assertEqual(clip.encode_from_tokens_scheduled.call_count, expected)

    def test_immutable_cpu_weights_survive_repeated_eviction(self):
        import torch
        model = torch.nn.Linear(3, 2)
        original = model.weight.detach().clone()
        patcher = Mock(model=model, patches={})
        patcher.loaded_size.return_value = 24
        patcher.unpatch_model.side_effect = lambda **kwargs: model.to('cpu')
        retain_cpu_weights(patcher)
        for _ in range(3):
            model.weight = torch.nn.Parameter(torch.ones_like(model.weight))
            with torch.inference_mode():
                self.assertEqual(patcher.partially_unload(torch.device('cpu'), 1), 24)
            self.assertTrue(torch.equal(model.weight, original))

    def test_ggml_self_detaching_views_do_not_mutate_the_saved_cpu_weights(self):
        import torch

        class SelfDetachingTensor(torch.Tensor):
            def __new__(cls, data, **kwargs):
                return torch.Tensor._make_subclass(cls, data, False)

            def __init__(self, data, tensor_type, tensor_shape, patches):
                self.tensor_type, self.tensor_shape, self.patches = tensor_type, tensor_shape, patches

            def detach(self):
                return self

        model = torch.nn.Module()
        original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        model.weight = torch.nn.Parameter(SelfDetachingTensor(original, tensor_type='test', tensor_shape=(2, 3), patches=[]), requires_grad=False)
        patcher = Mock(model=model, patches={})
        retain_cpu_weights(patcher)
        for _ in range(3):
            model.weight.data = torch.ones_like(original)
            with torch.inference_mode():
                patcher.unpatch_model(device_to='cpu')
            self.assertTrue(torch.equal(model.weight, original))

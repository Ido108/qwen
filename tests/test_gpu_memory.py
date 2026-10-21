import unittest
from backend.gpu_memory import GIB, reserve_for_dedicated_memory


class VideoMemoryTests(unittest.TestCase):
    def test_wddm_virtual_budget_does_not_hide_other_apps(self):
        reserved=reserve_for_dedicated_memory(22.5*GIB,17.2*GIB)
        self.assertAlmostEqual(reserved/GIB,7.3)
        self.assertAlmostEqual((22.5*GIB-reserved)/GIB,15.2)

    def test_matching_or_lagging_measurements_keep_safety_margin(self):
        self.assertEqual(reserve_for_dedicated_memory(17*GIB,17*GIB),2*GIB)
        self.assertEqual(reserve_for_dedicated_memory(17*GIB,18*GIB),2*GIB)

    def test_attention_choice_respects_opt_out_and_unavailable_kernel(self):
        import os, sys
        import torch  # before patch.dict(sys.modules) restores the module table
        from types import SimpleNamespace
        from unittest.mock import patch
        from backend.gpu_memory import configure_attention
        args=SimpleNamespace(use_ck_attention=False)
        with patch.dict(os.environ,{'QWEN_ATTENTION':'pytorch'}):
            self.assertEqual(configure_attention(args),'pytorch')
        self.assertFalse(args.use_ck_attention)
        for available,expected in [(False,'pytorch'),(True,'comfy_kitchen_int8')]:
            kitchen=SimpleNamespace(int8_attention_is_available=lambda device,available=available:available)
            with patch.dict(sys.modules,{'comfy_kitchen':kitchen}),patch.dict(os.environ,{'QWEN_ATTENTION':''}):
                self.assertEqual(configure_attention(args),expected)
            self.assertEqual(args.use_ck_attention,available)

    def test_plain_fp8_detection_skips_scaled_and_quantized_files(self):
        import torch
        from backend.comfy_engine import plain_fp8
        fp8={'a.weight':torch.zeros(2,2,dtype=torch.float8_e4m3fn),'a.bias':torch.zeros(2,dtype=torch.bfloat16)}
        self.assertTrue(plain_fp8(fp8,{}))
        self.assertFalse(plain_fp8({**fp8,'a.scale_weight':torch.ones(())},{}))
        self.assertFalse(plain_fp8({**fp8,'a.comfy_quant':torch.zeros(1,dtype=torch.uint8)},{}))
        self.assertFalse(plain_fp8(fp8,{'_quantization_metadata':'{}'}))
        self.assertFalse(plain_fp8({'a.weight':torch.zeros(2,2,dtype=torch.bfloat16)},None))

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from backend import int8_convert

OFFICIAL = Path('D:/models/diffusion_models/Comfy-Org-Qwen-Image-2.1')
DIFFUSERS = Path('D:/models/diffusers/Qwen-Image-2.1/transformer')


def tiny_qwen(blocks=2, dim=256, hidden=512, split=False, dtype=torch.bfloat16):
    """A small Qwen-Image-2.1 transformer with the real tensor names (dimensions divisible by the ConvRot group of 256)."""
    g = torch.Generator().manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, generator=g).to(dtype)
    sd = {'img_in.weight': rand(dim, 64), 'modulation.1.weight': rand(4 * dim, dim), 'norm_out.linear.weight': rand(dim, dim),
          'proj_out.weight': rand(64, dim), 'time_text_embed.timestep_embedder.linear_1.weight': rand(dim, 256),
          'time_text_embed.timestep_embedder.linear_2.weight': rand(dim, dim), 'txt_in.in_layer.weight': rand(dim, dim),
          'txt_in.out_layer.weight': rand(dim, dim), 'txt_in.text_norm.weight': rand(dim)}
    for b in range(blocks):
        p = f'transformer_blocks.{b}.'
        sd |= {p + 'attn.norm_q.weight': rand(128), p + 'attn.norm_k.weight': rand(128), p + 'attn.to_q.weight': rand(dim, dim),
               p + 'attn.to_k.weight': rand(dim, dim), p + 'attn.to_v.weight': rand(dim, dim), p + 'attn.to_out.0.weight': rand(dim, dim),
               p + 'img_mlp.out.weight': rand(dim, hidden)}
        if split:
            sd |= {p + 'img_mlp.gate_layer.weight': rand(hidden, dim), p + 'img_mlp.proj.weight': rand(hidden, dim)}
        else:
            sd[p + 'img_mlp.gate_up.weight'] = rand(2 * hidden, dim)
    return sd


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, sd, name='model.safetensors'):
        path = self.root / name
        save_file(sd, str(path), metadata={'format': 'pt'})
        return path

    def test_quantizes_exactly_the_six_block_linears_like_the_official_file(self):
        planned = int8_convert.plan(int8_convert.source_files([self.save(tiny_qwen())]))
        kinds = {name: (dtype, recipe[0]) for name, dtype, _, recipe in planned}
        quantized = sorted(name[:-len('.weight')] for name, (dtype, kind) in kinds.items() if kind == 'quantize')
        self.assertEqual(quantized, sorted(f'transformer_blocks.{b}.{layer}' for b in range(2) for layer in
                                           ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0', 'img_mlp.gate_up', 'img_mlp.out')))
        for layer in quantized:
            self.assertEqual(kinds[layer + '.weight'][0], 'I8')
            self.assertEqual(kinds[layer + '.weight_scale'][0], 'F32')
            self.assertEqual(kinds[layer + '.comfy_quant'][0], 'U8')
        # Every other tensor stays BF16, as in qwen_image_2.1_int8_convrot.safetensors.
        others = {name: dtype for name, (dtype, kind) in kinds.items() if kind == 'copy'}
        self.assertIn('transformer_blocks.0.attn.norm_q.weight', others)
        self.assertEqual(set(others.values()), {'BF16'})
        conf = next(recipe[1] for name, _, _, recipe in planned if name.endswith('.comfy_quant'))
        self.assertEqual(conf, b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}')
        self.assertEqual(len(conf), 72)
        self.assertFalse(any('comfy_attention' in name for name in kinds))

    def test_turbo_adds_int8_attention_to_every_block_but_the_first(self):
        planned = int8_convert.plan(int8_convert.source_files([self.save(tiny_qwen(blocks=3))]), turbo=True)
        attention = {name: recipe[1] for name, _, _, recipe in planned if 'comfy_attention' in name}
        self.assertEqual(sorted(attention), [f'transformer_blocks.{b}.attn.comfy_attention.config' for b in (1, 2)])
        self.assertEqual(set(attention.values()), {b'{"attention": "comfy_kitchen_int8"}'})

    def test_refuses_files_that_are_not_float_qwen_image_2_1(self):
        sd = tiny_qwen()
        sd['transformer_blocks.0.attn.to_q.weight'] = sd['transformer_blocks.0.attn.to_q.weight'].to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, 'only BF16, FP16 or FP32'):
            int8_convert.plan(int8_convert.source_files([self.save(sd, 'fp8.safetensors')]))
        sd = tiny_qwen()
        del sd['modulation.1.weight']
        with self.assertRaisesRegex(ValueError, 'Not a Qwen-Image-2.1'):
            int8_convert.plan(int8_convert.source_files([self.save(sd, 'other.safetensors')]))

    def test_strips_the_comfyui_checkpoint_prefix(self):
        sd = {'model.diffusion_model.' + k: v for k, v in tiny_qwen().items()}
        planned = int8_convert.plan(int8_convert.source_files([self.save(sd)]))
        self.assertIn('transformer_blocks.0.attn.to_q.weight', {name for name, *_ in planned})

    def test_layout_uses_safetensors_order_and_padding(self):
        planned = int8_convert.plan(int8_convert.source_files([self.save(tiny_qwen())]))
        header, offsets, size = int8_convert.layout(planned)
        length = struct.unpack('<Q', header[:8])[0]
        self.assertEqual(length % 8, 0)
        parsed = json.loads(header[8:])
        self.assertEqual(next(iter(parsed)), '__metadata__')
        self.assertEqual(parsed['__metadata__'], {'format': 'pt'})
        order = [parsed[name]['dtype'] for name in sorted(offsets, key=offsets.get)]
        groups = [d for i, d in enumerate(order) if i == 0 or order[i - 1] != d]
        self.assertEqual(groups, ['F32', 'BF16', 'I8', 'U8'])
        self.assertEqual(size, int8_convert.output_bytes(planned))

    def test_written_file_loads_and_dequantizes_to_the_source(self):
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
        sd = tiny_qwen(split=True, dtype=torch.float32)
        source = self.save(sd)
        target = self.root / 'out' / 'model_int8_convrot.safetensors'
        result = int8_convert.convert([source], target, device='cpu')
        self.assertEqual(target.stat().st_size, result['bytes'])
        self.assertEqual(result['layers'], 12)
        self.assertFalse(target.with_name(target.name + '.part').exists())
        with safe_open(str(target), 'pt') as f:
            self.assertEqual(f.metadata(), {'format': 'pt'})
            self.assertEqual(f.get_tensor('img_in.weight').dtype, torch.bfloat16)
            q, s = f.get_tensor('transformer_blocks.1.img_mlp.gate_up.weight'), f.get_tensor('transformer_blocks.1.img_mlp.gate_up.weight_scale')
            self.assertEqual((q.dtype, s.dtype, tuple(s.shape)), (torch.int8, torch.float32, (1024, 1)))
        # The Diffusers halves join as [gate_layer; proj], ComfyUI's fused gate_up.
        expected = torch.cat([sd['transformer_blocks.1.img_mlp.gate_layer.weight'], sd['transformer_blocks.1.img_mlp.proj.weight']])
        restored = dequantize_int8_convrot_weight(q, s, 256)
        self.assertLess(((restored - expected).abs().max() / expected.abs().max()).item(), 0.02)
        swapped = torch.cat([sd['transformer_blocks.1.img_mlp.proj.weight'], sd['transformer_blocks.1.img_mlp.gate_layer.weight']])
        self.assertGreater(((restored - swapped).abs().max() / swapped.abs().max()).item(), 0.5)

    def test_moss_image_follows_the_official_moss_int8_file(self):
        g = torch.Generator().manual_seed(1)
        rand = lambda *shape: torch.randn(*shape, generator=g).to(torch.bfloat16)
        sd = {'shared_block_modulation.weight': rand(1536, 256), 'video_embedder.weight': rand(256, 128), 'final_linear_video.weight': rand(128, 256),
              'rope.bands': torch.randn(16)}
        for b in range(3):
            sd |= {f'layers.{b}.proj.qkv.weight': rand(768, 256), f'layers.{b}.proj.o.weight': rand(256, 256),
                   f'layers.{b}.proj.ffn_up_gate.weight': rand(1024, 256), f'layers.{b}.proj.ffn_down.weight': rand(256, 512),
                   f'layers.{b}.q_norm.weight': rand(128)}
        planned = int8_convert.plan(int8_convert.source_files([self.save(sd, 'moss.safetensors')]))
        kinds = {name: (dtype, recipe[0]) for name, dtype, _, recipe in planned}
        self.assertEqual(planned.arch, 'moss-image')
        self.assertEqual(sorted(n[:-7] for n, (d, k) in kinds.items() if k == 'quantize'),
                         sorted(f'layers.{b}.proj.{p}' for b in range(3) for p in ('qkv', 'o', 'ffn_up_gate', 'ffn_down')))
        self.assertEqual(kinds['rope.bands'][0], 'F32')  # F32 in moss_image_sft_int8_convrot.safetensors too
        self.assertEqual(kinds['layers.0.q_norm.weight'][0], 'BF16')
        self.assertEqual(sorted(n for n in kinds if 'comfy_attention' in n), [f'layers.{b}.comfy_attention.config' for b in range(3)])
        header = int8_convert.layout(planned)[0]
        self.assertEqual(json.loads(header[8:])['__metadata__'], {})

    def test_matches_comfy_kitchen_quantize_layout(self):
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
        weight = torch.randn(512, 1024)
        q, scale = int8_convert.quantize(weight, 'cpu')
        expected, params = TensorWiseINT8Layout.quantize(weight, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
        self.assertTrue(torch.equal(q, expected))
        self.assertTrue(torch.equal(scale, params.scale.float()))


@unittest.skipUnless((OFFICIAL / 'qwen_image_2.1_int8_convrot.safetensors').is_file() and (DIFFUSERS / 'diffusion_pytorch_model.safetensors.index.json').is_file(),
                     'needs the official Qwen-Image-2.1 BF16 and INT8 files')
class OfficialFileTests(unittest.TestCase):
    def official_header(self, name):
        with open(OFFICIAL / name, 'rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            handle.seek(0)
            return handle.read(8 + length)

    def test_plan_of_the_official_bf16_is_the_official_int8_header_byte_for_byte(self):
        shards = sorted(DIFFUSERS.glob('diffusion_pytorch_model-*.safetensors'))
        planned = int8_convert.plan(int8_convert.source_files(shards))
        self.assertEqual(sum(1 for item in planned if item[3][0] == 'quantize'), 192)
        self.assertEqual(int8_convert.layout(planned)[0], self.official_header('qwen_image_2.1_int8_convrot.safetensors'))

    @unittest.skipUnless((OFFICIAL / 'qwen_image_2.1_turbo_bf16.safetensors').is_file(), 'needs the Turbo BF16 file')
    def test_turbo_plan_is_the_official_turbo_int8_header_byte_for_byte(self):
        planned = int8_convert.plan(int8_convert.source_files([OFFICIAL / 'qwen_image_2.1_turbo_bf16.safetensors']), turbo=True)
        self.assertEqual(int8_convert.layout(planned)[0], self.official_header('qwen_image_2.1_turbo_int8_convrot.safetensors'))


if __name__ == '__main__':
    unittest.main()

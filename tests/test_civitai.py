import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QWEN_DATA_DIR', tempfile.mkdtemp(prefix='civitai-tests-'))
from backend import civitai


def safetensors(path: Path, shapes: dict):
    header = {name: {'dtype': 'BF16', 'shape': shape, 'data_offsets': [0, 0]} for name, shape in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw)


class CivitaiTests(unittest.TestCase):
    def test_author_settings_prefer_the_version_and_the_author(self):
        version = "NVFP4 conversion. Author's settings for V4 Base: 25 steps, euler, simple, cfg 3 (cfg 1 runs about twice as fast)."
        model = 'The example images were made with: 80 steps, euler_ancestral_cfg_pp, beta, cfg 0.7. The author setting also works.'
        found = civitai.author_settings(version, model)
        self.assertEqual({k: found[k] for k in ('steps', 'cfg', 'sampler', 'scheduler')}, {'steps': 25, 'cfg': 3.0, 'sampler': 'euler', 'scheduler': 'simple'})
        self.assertEqual(civitai.author_settings('', model)['sampler'], 'euler_ancestral_cfg_pp')
        self.assertEqual(civitai.author_settings('No numbers here.'), {})

    def test_author_settings_read_lists_sampler_names_and_ranges(self):
        # Real page texts: a settings list on separate lines, "Euler A" (ancestral), an en dash range and a one-line list.
        moody = 'Skin Texture Demo (Euler Ancestral, bata, 8 Steps)\nRecommended Settings\nmy current recommended settings are:\n• Sampler: Euler A + BETA\n\n• Steps: 8'
        found = civitai.author_settings('', moody)
        self.assertEqual({k: found.get(k) for k in ('steps', 'sampler', 'scheduler')}, {'steps': 8, 'sampler': 'euler_ancestral', 'scheduler': 'beta'})
        realism = '- Fast: 8 steps · CFG 1 · drafts.\n- Best: 12 steps · CFG 1.5 · the richest version.\n- Extended range. CFG 1\u20133 and 8\u201320 steps.'
        self.assertEqual({k: civitai.author_settings(realism)[k] for k in ('steps', 'cfg')}, {'steps': 12, 'cfg': 1.5})
        line = civitai.author_settings('Steps: 10, Sampler: Euler a, CFG scale: 2, Seed: 3573086999')
        self.assertEqual({k: line.get(k) for k in ('steps', 'cfg', 'sampler')}, {'steps': 10, 'cfg': 2.0, 'sampler': 'euler_ancestral'})
        self.assertEqual(civitai.find_sampler('Euler Simple - 12 Steps'), 'euler')
        self.assertEqual(civitai.find_sampler('DPM++ 2M SDE Karras'), 'dpmpp_2m_sde')

    def test_status_is_connected_only_while_civitai_accepts_the_sign_in(self):
        from unittest import mock
        import httpx
        answer = lambda code: mock.Mock(status_code=code, json=lambda: {})
        try:
            civitai._save({'api_key': 'k' * 32, 'username': 'me'})
            civitai._checked['at'] = 0
            with mock.patch.object(civitai.httpx, 'get', side_effect=httpx.ConnectError('offline')):
                self.assertTrue(civitai.status()['connected'])  # unreachable is not a rejection
            civitai._checked['at'] = 0
            with mock.patch.object(civitai.httpx, 'get', return_value=answer(200)):
                self.assertTrue(civitai.status()['connected'])
            civitai._checked['at'] = 0
            with mock.patch.object(civitai.httpx, 'get', return_value=answer(401)):
                self.assertEqual(civitai.status(), {'connected': False, 'username': '', 'method': ''})
            civitai._save({'access_token': 'a', 'refresh_token': 'r', 'expires_at': 0, 'username': 'me'})
            civitai._checked['at'] = 0
            with mock.patch.object(civitai, '_auth_post', return_value=answer(400)):
                self.assertFalse(civitai.status()['connected'])
            self.assertFalse(civitai.TOKEN_FILE.exists())
        finally:
            civitai._forget()

    def test_turbo_detection_trusts_base_over_step_counts(self):
        self.assertIs(civitai.detect_turbo('V4.0 Base NVFP4', {'steps': 8}), False)
        self.assertIs(civitai.detect_turbo('turbo v1.3 int8', {}), True)
        self.assertIs(civitai.detect_turbo('v2', {'steps': 10}), True)
        self.assertIs(civitai.detect_turbo('v2', {'steps': 25}), False)
        self.assertIsNone(civitai.detect_turbo('v2', {}))
        self.assertIs(civitai.detect_turbo('krea2_raw_int8_convrot', {}), False)
        self.assertIs(civitai.detect_turbo('Krea 2 Raw FP8', {}), False)
        self.assertIsNone(civitai.detect_turbo('4. Raw Studio Shots', {}))  # a theme name, not the type
        self.assertIs(civitai.detect_turbo('v2', {'steps': 12, 'cfg': 1.0}), True)  # CFG 1 is a distilled model

    def test_page_settings_reach_generation_defaults(self):
        base = civitai.image_defaults(False, {'steps': 25, 'cfg': 3.0, 'sampler': 'euler', 'scheduler': 'simple', 'quote': 'q'}, 'u')
        self.assertEqual((base['steps'], base['cfg'], base['sampler'], base['scheduler']), (25, 3.0, 'euler', 'simple'))
        turbo = civitai.image_defaults(True, {'steps': 10, 'cfg': 2.0, 'quote': 'q'}, 'u')
        self.assertEqual((turbo['steps'], turbo['cfg'], len(turbo['sigmas'])), (10, 2.0, 9))
        self.assertNotIn('fixed_steps', turbo)
        self.assertEqual(civitai._known_vae('texture_fix_vae_for_qwen_image_2.1_bf16.safetensors'), 'vae-texture-fix')
        self.assertEqual(civitai._known_vae('qwen_image_2.1_vae_bf16.safetensors'), 'vae')

    def test_ltx_checkpoints_are_found_also_when_packed_to_4_bit(self):
        with tempfile.TemporaryDirectory() as folder:
            plain, packed = Path(folder) / 'plain.safetensors', Path(folder) / 'packed.safetensors'
            safetensors(plain, {'audio_adaln_single.linear.weight': [18432, 2048]})
            # W4A8 codebook files store half-width packed weights plus scales.
            safetensors(packed, {'audio_adaln_single.linear.weight': [18432, 1024], 'audio_adaln_single.linear.weight_s_channel': [18432],
                                 'av_ca_a2v_gate_adaln_single.linear.weight_codebook': [16]})
            self.assertEqual(civitai.inspect(plain)['arch'], 'ltx25')
            self.assertEqual(civitai.inspect(packed)['arch'], 'ltx25')

    def test_ltx_variant_reads_dev_before_distilled(self):
        self.assertEqual(civitai.ltx_variant('Joy-LTX 2.5 DEV merges - bring your own distilled LoRA', '070T30', {'steps': 25}, ''), 'dev')
        self.assertEqual(civitai.ltx_variant('LTX 2.5 INT8/INT4 ConvRot', 'LTX 2.5 DISTILLED INT8', {}, ''), 'distilled')
        self.assertEqual(civitai.ltx_variant('Remix', 'Beta1', {'steps': 8}, ''), 'distilled')
        self.assertIsNone(civitai.ltx_variant('Surfaces', 'v1.0', {}, 'a style merge'))

    def test_inspect_tells_qwen21_from_qwen1_and_models_from_loras(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = {
                'lora21.safetensors': ({'transformer_blocks.0.attn.to_q.lora_A.default.weight': [16, 4096],
                                        'transformer_blocks.0.attn.to_q.lora_B.default.weight': [4096, 16]}, ('lora', 'qwen-image-2.1')),
                'lora21-comfy.safetensors': ({'diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight': [32, 4096]}, ('lora', 'qwen-image-2.1')),
                'lora1.safetensors': ({'transformer_blocks.0.attn.to_q.lora_A.weight': [16, 3072]}, ('lora', 'qwen-image-1')),
                'model21.safetensors': ({'img_in.weight': [4096, 64], 'txt_in.weight': [4096, 4096]}, ('model', 'qwen-image-2.1')),
                'model1.safetensors': ({'model.diffusion_model.img_in.weight': [3072, 64]}, ('model', 'qwen-image-1')),
                'sdxl.safetensors': ({'lora_unet_down_blocks_0.lora_down.weight': [8, 320]}, ('lora', 'unknown')),
            }
            for name, (shapes, expected) in cases.items():
                path = Path(folder) / name
                safetensors(path, shapes)
                info = civitai.inspect(path)
                self.assertEqual((info['kind'], info['arch']), expected, name)

    def test_link_strength_and_text(self):
        self.assertEqual(civitai.resolve_link('https://civitai.red/models/1662740/lenovo?modelVersionId=3349565'), {'model_id': 1662740, 'version_id': 3349565})
        self.assertEqual(civitai.resolve_link('https://civitai.com/models/42'), {'model_id': 42, 'version_id': None})
        with self.assertRaises(ValueError):
            civitai.resolve_link('https://example.com/models/1')
        self.assertEqual(civitai.suggested_strength('Use it at strength 0.75 for best results'), 0.75)
        self.assertEqual(civitai.suggested_strength('LoRA weight: 1.2'), 1.2)
        self.assertIsNone(civitai.suggested_strength('No settings here, 2048 px'))
        self.assertEqual(civitai._text('<p>Hi &amp; <b>bye</b></p><ul><li>one</li></ul>'), 'Hi & bye\n• one')

    def test_user_lora_requirements_follow_model_family(self):
        from backend import models
        entry = {'id': 'civitai-lora-1', 'title': 'Test LoRA', 'kind': 'image-lora', 'format': 'LoRA', 'source': 'civitai', 'repo': 'civitai',
                 'revision': '1', 'directory': 'civitai/loras', 'files': [{'path': 'x.safetensors', 'bytes': 1, 'sha256': None}], 'bytes': 1,
                 'dependencies': [], 'families': ['qwen-image-2.1'], 'civitai': {'version_id': 1}}
        civitai._register(entry)
        try:
            self.assertEqual(models.required('gguf-q8_0', vae='vae', loras=['civitai-lora-1']), ['gguf-q8_0', 'encoder-int8', 'vae', 'civitai-lora-1'])
            with self.assertRaises(ValueError):
                models.required('turbo-gguf-q8_0', loras=['civitai-lora-1'])
            with self.assertRaises(ValueError):
                models.required('gguf-q8_0', loras=['vae'])
        finally:
            civitai.unregister('civitai-lora-1')
        self.assertFalse(any(item['id'] == 'civitai-lora-1' for item in models.catalog()))


if __name__ == '__main__':
    unittest.main()

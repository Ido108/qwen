"""Z-Image: catalog settings, file requirements, image to image, Civitai detection and installation."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QWEN_DATA_DIR', tempfile.mkdtemp(prefix='zimage-tests-'))
from backend import civitai, models, zimage_engine
from tests.test_krea import safetensors


class ZImageCatalogTests(unittest.TestCase):
    def test_official_settings(self):
        zimage = [item for item in models.catalog() if item['kind'] == 'image' and models.is_zimage(item)]
        self.assertEqual({item['id'] for item in zimage},
                         {'zimage-turbo-bf16', 'zimage-turbo-int8', 'zimage-turbo-nvfp4', 'zimage-base-bf16', 'zimage-base-int8'})
        for item in zimage:
            defaults = item['generation_defaults']
            # The official ComfyUI workflows: res_multistep, simple, ModelSamplingAuraFlow shift 3.
            self.assertEqual((defaults['sampler'], defaults['scheduler'], defaults['shift']), ('res_multistep', 'simple', 3.0), item['id'])
            self.assertEqual(item['dependencies'], ['zimage-encoder-bf16', 'zimage-vae'])
            self.assertEqual(item['layout'], 'comfy')
            self.assertRegex(item['files'][0]['sha256'], r'^[0-9a-f]{64}$')
            if item['family'] == 'z-image-turbo':
                self.assertEqual((defaults['steps'], defaults['cfg'], item['repo']), (8, 1.0, 'Comfy-Org/z_image_turbo'), item['id'])
            else:
                self.assertEqual((item['family'], defaults['steps'], defaults['cfg'], item['repo']), ('z-image', 30, 4.0, 'Comfy-Org/z_image'), item['id'])
        for support in ('zimage-encoder-bf16', 'zimage-vae'):
            self.assertEqual(models.entry(support)['families'], ['z-image', 'z-image-turbo'])

    def test_zimage_uses_its_own_encoder_and_vae(self):
        self.assertEqual(models.required('zimage-turbo-bf16'), ['zimage-turbo-bf16', 'zimage-encoder-bf16', 'zimage-vae'])
        # Qwen and Krea encoder and VAE choices do not apply to Z-Image.
        self.assertEqual(models.required('zimage-base-bf16', 'encoder-int8', vae='krea2-vae'), ['zimage-base-bf16', 'zimage-encoder-bf16', 'zimage-vae'])
        self.assertEqual(models.required_image({'model_id': 'zimage-turbo-int8', 'images': ['a']}), ['zimage-turbo-int8', 'zimage-encoder-bf16', 'zimage-vae'])
        with self.assertRaisesRegex(ValueError, 'only 1 image'):
            models.required('zimage-turbo-bf16', images=2)
        with self.assertRaisesRegex(ValueError, 'Qwen-Image-2.1 tool'):
            models.required('zimage-turbo-bf16', tool_id=models.image_tools()[0]['id'])
        # Neither Qwen nor Krea LoRAs fit Z-Image, and Z-Image encoders do not fit Qwen models.
        with self.assertRaisesRegex(ValueError, 'does not work'):
            models.required('zimage-turbo-bf16', loras=[models.KREA_STYLE_LORA])
        self.assertEqual(models.required('gguf-q8_0', 'zimage-encoder-bf16')[1], 'encoder-int8')

    def test_denoise_and_cfg(self):
        self.assertEqual(zimage_engine.zimage_denoise({'denoise': 0.4}, []), 1.0)  # text to image ignores it
        self.assertEqual(zimage_engine.zimage_denoise({}, ['image']), 0.7)
        self.assertEqual(zimage_engine.zimage_denoise({'denoise': 0.35}, ['image']), 0.35)
        self.assertEqual(zimage_engine.zimage_cfg({}, {'cfg': 4.0}), 4.0)
        self.assertEqual(zimage_engine.zimage_cfg({'cfg': 2}, {'cfg': 1.0}), 2.0)

    def test_worker_picks_the_engine_from_the_model(self):
        from backend import worker
        self.assertIs(worker.image_engine({'model_id': 'zimage-base-bf16'}), zimage_engine.generate)


class ZImageCivitaiTests(unittest.TestCase):
    def test_the_tag_names_the_type(self):
        self.assertEqual(civitai.BASE_MODELS['ZImageTurbo']['arch'], 'z-image')
        # Civitai's tag decides; a name or page settings cannot turn ZImageBase into Turbo or back.
        self.assertEqual(civitai._variant({'name': 'Turbo mix'}, {'name': 'v1 8 steps', 'baseModel': 'ZImageBase'}), 'standard')
        self.assertEqual(civitai._variant({'name': 'Base mix'}, {'name': 'base', 'baseModel': 'ZImageTurbo'}), 'turbo')
        self.assertIs(civitai.version_turbo({'name': 'x'}, {'name': 'v1', 'baseModel': 'ZImageBase', 'description': 'Use 8 steps CFG 1'}), False)

    def test_inspect_finds_zimage_models_and_loras(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = {
                'model.safetensors': ({'cap_embedder.1.weight': [3840, 2560], 'layers.0.attention.qkv.weight': [11520, 3840]}, ('model', 'z-image')),
                'aio.safetensors': ({'model.diffusion_model.cap_embedder.1.weight': [3840, 2560], 'vae.decoder.conv_in.weight': [512, 16, 3, 3]}, ('model', 'z-image')),
                'comfy-lora.safetensors': ({'diffusion_model.layers.0.attention.to_q.lora_A.default.weight': [32, 3840],
                                            'diffusion_model.layers.0.attention.to_q.lora_B.default.weight': [3840, 32]}, ('lora', 'z-image')),
                'qkv-lora.safetensors': ({'diffusion_model.layers.3.attention.qkv.lora_A.weight': [16, 3840]}, ('lora', 'z-image')),
                'kohya-lora.safetensors': ({'lora_unet_layers_0_attention_to_q.lora_down.weight': [16, 3840], 'lora_unet_layers_0_attention_to_q.alpha': []}, ('lora', 'z-image')),
                # Lumina 2 (2304 wide) uses the same names but is another model.
                'lumina-lora.safetensors': ({'diffusion_model.layers.0.attention.qkv.lora_A.weight': [16, 2304]}, ('lora', 'unknown')),
                'qwen21-lora.safetensors': ({'transformer_blocks.0.attn.to_q.lora_A.default.weight': [16, 4096]}, ('lora', 'qwen-image-2.1')),
                'krea-lora.safetensors': ({'diffusion_model.blocks.0.attn.wq.lora_A.weight': [64, 6144]}, ('lora', 'krea-2')),
            }
            for name, (shapes, expected) in cases.items():
                path = Path(folder) / name
                safetensors(path, shapes)
                info = civitai.inspect(path)
                self.assertEqual((info['kind'], info['arch']), expected, name)

    def test_page_with_settings_for_several_architectures(self):
        # civitai.red/models/1609320: three lists, two headings glued to the CFG line above them.
        page = ('IntoRealism\n⚡ DMD2 Usage\nSampling method: LCM / Exponential or Karras for V3.5\nSampling steps: 8 to 12\n'
                'Resolution: 896x1152 / 832x1216\nCFG: 1SDXL Usage\nSampling method: DPM++ 2M SDE Karras\nSampling steps: > 25\n'
                'Resolution: 896x1152 / 832x1216\nCFG: 4-5ZIT Usage\nSampling method: Euler / Beta\nSampling steps: 6/10\n'
                'Resolution: 896x1152 / 832x1216\nCFG: 1⚠️ Ignore the license tag above')
        found = civitai.author_settings('', page, arch='z-image')
        self.assertEqual({k: found[k] for k in ('sampler', 'scheduler', 'steps', 'cfg')}, {'sampler': 'euler', 'scheduler': 'beta', 'steps': 6, 'cfg': 1.0})
        self.assertTrue(found['quote'].startswith('ZIT Usage: '))
        # Without a matching heading the first list still wins, as before.
        self.assertEqual(civitai.author_settings('', page, arch='krea-2')['sampler'], 'lcm')
        # One list with a repeated name is two lists: the second does not overwrite the first.
        two = civitai.author_settings('Steps: 8\nCFG: 1\nSteps: 30\nCFG: 4', arch='z-image')
        self.assertEqual((two['steps'], two['cfg']), (8, 1.0))

    def test_defaults_take_page_settings(self):
        turbo = civitai.image_defaults(True, {}, 'u', 'z-image')
        self.assertEqual((turbo['steps'], turbo['cfg'], turbo['sampler'], turbo['shift'], turbo['source']), (8, 1.0, 'res_multistep', 3.0, 'u'))
        base = civitai.image_defaults(False, {'steps': 40, 'cfg': 5.0, 'sampler': 'euler', 'quote': 'q'}, 'u', 'z-image')
        self.assertEqual((base['steps'], base['cfg'], base['sampler'], base['scheduler']), (40, 5.0, 'euler', 'simple'))
        self.assertTrue(civitai._best_quality('Checkpoint', civitai.BASE_MODELS['ZImageBase']))

    def install(self, base_model, shapes, model_type='Checkpoint'):
        version = {'id': 501, 'modelId': 500, 'name': 'v2', 'baseModel': base_model, 'description': 'Recommended: 10 steps, CFG 1.5', 'trainedWords': ['zw'],
                   'files': [{'id': 1, 'name': 'file.safetensors', 'type': 'Model', 'sizeKB': 1, 'primary': True,
                              'metadata': {'format': 'SafeTensor', 'fp': 'fp16'}, 'hashes': {}, 'downloadUrl': 'https://example.invalid/1'}], 'images': []}
        model = {'id': 500, 'name': 'Some Z mix', 'type': model_type, 'description': '', 'modelVersions': [version]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.multiple(civitai, _get=lambda path, params=None: version if 'model-versions' in path else model,
                                _download=lambda url, target, *a: safetensors(target, shapes), _save_preview=lambda *a: None,
                                MODELS_DIR=root, USER_CATALOG=root / 'user.json'), patch.object(models, 'INSTALL_PATH', root / 'installed.json'):
                item = civitai.install(version['id'], None, False, lambda *a: None)
                required = models.required(item['id']) if item['kind'] == 'image' else None
            return item, required

    def test_install_maps_checkpoints_and_loras(self):
        item, required = self.install('ZImageTurbo', {'cap_embedder.1.weight': [3840, 2560]})
        self.assertEqual((item['kind'], item['family'], item['format'], item['layout']), ('image', 'z-image-turbo', 'FP16', 'comfy'))
        self.assertEqual(required, [item['id'], 'zimage-encoder-bf16', 'zimage-vae'])
        defaults = item['generation_defaults']
        self.assertEqual((defaults['steps'], defaults['cfg'], defaults['sampler']), (10, 1.5, 'res_multistep'))
        item, _ = self.install('ZImageBase', {'cap_embedder.1.weight': [3840, 2560]})
        self.assertEqual(item['family'], 'z-image')
        lora, _ = self.install('ZImageTurbo', {'diffusion_model.layers.0.attention.to_q.lora_A.weight': [32, 3840]}, 'LORA')
        self.assertEqual((lora['kind'], lora['families'], lora['trained_words']), ('image-lora', ['z-image', 'z-image-turbo'], ['zw']))
        # A Qwen-Image file uploaded under a Z-Image tag is refused.
        with self.assertRaisesRegex(ValueError, 'not a ZImageTurbo model'):
            self.install('ZImageTurbo', {'img_in.weight': [4096, 64]})


if __name__ == '__main__':
    unittest.main()

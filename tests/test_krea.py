"""Krea 2: catalog settings, file requirements, sampling schedule, Civitai detection and installation."""
import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QWEN_DATA_DIR', tempfile.mkdtemp(prefix='krea-tests-'))
from backend import civitai, krea_engine, models


def safetensors(path: Path, shapes: dict):
    header = {name: {'dtype': 'BF16', 'shape': shape, 'data_offsets': [0, 0]} for name, shape in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw)


def official_timesteps(seq_len, steps, mu=None, minres=256, maxres=1280, y1=0.5, y2=1.15):
    """krea-ai/krea-2 sampling.py timesteps() without torch."""
    x1, x2 = (minres // 16) ** 2, (maxres // 16) ** 2
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = [1 - i / steps for i in range(steps + 1)]
    return [math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)) if t > 0 else 0.0 for t in ts]


class KreaCatalogTests(unittest.TestCase):
    def test_official_settings(self):
        krea = [item for item in models.catalog() if item['kind'] == 'image' and models.is_krea(item)]
        self.assertEqual({item['family'] for item in krea}, {'krea-2', 'krea-2-turbo'})
        for item in krea:
            defaults = item['generation_defaults']
            self.assertEqual((defaults['sampler'], defaults['scheduler']), ('euler', 'simple'), item['id'])
            self.assertEqual(item['dependencies'], ['krea2-encoder-bf16', 'krea2-vae'])
            self.assertEqual((item['repo'], item['layout']), ('Comfy-Org/Krea-2', 'comfy'))
            self.assertRegex(item['files'][0]['sha256'], r'^[0-9a-f]{64}$')
            if item['family'] == 'krea-2-turbo':
                # README: --steps 8 --cfg 0.0 --mu 1.15 (no guidance is ComfyUI CFG 1).
                self.assertEqual((defaults['steps'], defaults['cfg'], defaults['mu']), (8, 1.0, 1.15), item['id'])
            else:
                # README: --steps 52 --cfg 3.5; Krea's cond + 3.5 (cond - uncond) is ComfyUI CFG 4.5.
                self.assertEqual((defaults['steps'], defaults['cfg']), (52, 4.5), item['id'])
                self.assertNotIn('mu', defaults)
        for support in ('krea2-encoder-bf16', 'krea2-vae', models.KREA_STYLE_LORA, models.KREA_IDENTITY_LORA):
            self.assertEqual(models.entry(support)['families'], ['krea-2', 'krea-2-turbo'])

    def test_krea_uses_its_own_encoder_and_vae(self):
        self.assertEqual(models.required('krea2-turbo-fp8'), ['krea2-turbo-fp8', 'krea2-encoder-bf16', 'krea2-vae'])
        # Qwen encoder and VAE choices made for Qwen models do not apply to Krea 2.
        self.assertEqual(models.required('krea2-raw-fp8', 'encoder-int8', vae='vae-texture-fix'), ['krea2-raw-fp8', 'krea2-encoder-bf16', 'krea2-vae'])
        # Input images keep the person by default (identity edit LoRA); style is the official style reference workflow.
        self.assertEqual(models.required_image({'model_id': 'krea2-turbo-nvfp4', 'vae_id': 'vae', 'images': ['a', 'b']}),
                         ['krea2-turbo-nvfp4', 'krea2-encoder-bf16', 'krea2-vae', models.KREA_IDENTITY_LORA])
        self.assertEqual(models.required_image({'model_id': 'krea2-turbo-nvfp4', 'images': ['a'], 'krea_reference_mode': 'style'})[-1], models.KREA_STYLE_LORA)
        with self.assertRaisesRegex(ValueError, 'identity or style'):
            models.required('krea2-turbo-nvfp4', images=1, krea_reference_mode='face')
        with self.assertRaisesRegex(ValueError, 'at most 3'):
            models.required('krea2-turbo-fp8', images=4)

    def test_qwen_tools_and_files_are_refused_for_krea(self):
        tool = models.image_tools()[0]
        with self.assertRaisesRegex(ValueError, 'Qwen-Image-2.1 tool'):
            models.required('krea2-turbo-fp8', tool_id=tool['id'])
        with self.assertRaises(ValueError):
            models.required('krea2-turbo-fp8', loras=['lora-restore'])

    def test_qwen_behavior_is_unchanged(self):
        self.assertEqual(models.required('gguf-q8_0'), ['gguf-q8_0', 'encoder-int8', 'vae-texture-fix'])
        self.assertEqual(models.required('turbo-bf16', 'encoder-int8', vae='vae'), ['turbo-bf16', 'encoder-int8', 'vae'])
        self.assertEqual(models.required('gguf-q8_0', 'krea2-encoder-bf16'), ['gguf-q8_0', 'encoder-int8', 'vae-texture-fix'])
        self.assertEqual(models.required_image({'model_id': 'gguf-q8_0', 'images': ['a', 'b', 'c', 'd']}), ['gguf-q8_0', 'encoder-int8', 'vae-texture-fix'])
        # The VAE list shows every VAE; a Krea 2 VAE chosen for a Qwen model changes to the automatic VAE.
        self.assertEqual(models.required('gguf-q8_0', vae='krea2-vae'), ['gguf-q8_0', 'encoder-int8', 'vae-texture-fix'])
        with self.assertRaises(ValueError):
            models.required('gguf-q8_0', vae='encoder-int8')
        with self.assertRaises(ValueError):
            models.required('gguf-q8_0', loras=[models.KREA_STYLE_LORA])
        with self.assertRaises(ValueError):
            models.required('gguf-q8_0', 'qwen3')


class KreaSamplingTests(unittest.TestCase):
    def test_time_shift_matches_the_official_sampler(self):
        turbo = models.entry('krea2-turbo-fp8')['generation_defaults']
        raw = models.entry('krea2-raw-fp8')['generation_defaults']
        self.assertEqual(krea_engine.krea_mu(turbo, 2048, 2048), 1.15)
        self.assertAlmostEqual(krea_engine.krea_mu(raw, 256, 256), 0.5)
        self.assertAlmostEqual(krea_engine.krea_mu(raw, 1280, 1280), 1.15)
        self.assertAlmostEqual(krea_engine.krea_mu(raw, 1024, 1024), 0.5 + 0.65 * (4096 - 256) / (6400 - 256))
        for steps, (width, height), defaults in ((8, (1024, 1024), turbo), (52, (1024, 1024), raw), (52, (832, 1216), raw), (8, (2048, 1152), turbo)):
            seq = (height // 16) * (width // 16)
            expected = official_timesteps(seq, steps, defaults.get('mu'))
            got = krea_engine.krea_sigmas(steps, krea_engine.krea_mu(defaults, width, height))
            self.assertEqual(len(got), steps + 1)
            for a, b in zip(got, expected):
                self.assertAlmostEqual(a, b, places=12)
            self.assertEqual((got[0], got[-1]), (1.0, 0.0))

    def test_cfg_follows_the_model_unless_typed(self):
        self.assertEqual(krea_engine.krea_cfg({'cfg': None}, {'cfg': 4.5}), 4.5)
        self.assertEqual(krea_engine.krea_cfg({}, {'cfg': 1.0}), 1.0)
        self.assertEqual(krea_engine.krea_cfg({'cfg': 3}, {'cfg': 1.0}), 3.0)

    def test_worker_picks_the_engine_from_the_model(self):
        from backend import worker, comfy_engine
        self.assertIs(worker.image_engine({'model_id': 'krea2-raw-bf16'}), krea_engine.generate)
        self.assertIs(worker.image_engine({'model_id': 'gguf-q8_0'}), comfy_engine.generate)


class KreaCivitaiTests(unittest.TestCase):
    def test_base_models_and_variants(self):
        self.assertEqual(civitai.BASE_MODELS['Krea 2']['arch'], 'krea-2')
        self.assertEqual({name for name, info in civitai.BASE_MODELS.items() if info['area'] == 'image'}, {'Qwen 2.1', 'Krea 2', 'Krea 2 Turbo', 'ZImageTurbo', 'ZImageBase', 'Flux.3 Klein', 'Flux.3 Klein-base', 'MOSS Image'})
        self.assertEqual(civitai._variant({'name': 'Muse'}, {'name': 'v3.0', 'baseModel': 'Krea 2 Turbo'}), 'turbo')
        self.assertEqual(civitai._variant({'name': 'Mix'}, {'name': 'v8', 'baseModel': 'Krea 2'}), 'unknown')  # nothing tells: the app does not guess
        self.assertEqual(civitai._variant({'name': 'Mix'}, {'name': 'v8', 'baseModel': 'Krea 2', 'description': 'Use 8 steps, CFG 1.'}), 'turbo')
        self.assertEqual(civitai._variant({'name': 'Krea 2 Turbo Official'}, {'name': 'krea2_raw_int8_convrot', 'baseModel': 'Krea 2'}), 'standard')

    def test_inspect_finds_krea_models_and_loras(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = {
                'model.safetensors': ({'blocks.0.attn.wq.weight': [6144, 6144], 'txtfusion.projector.weight': [1, 12], 'first.weight': [6144, 64]}, ('model', 'krea-2')),
                'aio.safetensors': ({'model.diffusion_model.txtfusion.projector.weight': [1, 12], 'vae.decoder.conv_in.weight': [384, 16, 3, 3, 3]}, ('model', 'krea-2')),
                'diffusers-lora.safetensors': ({'transformer.final_layer.linear.lora_A.weight': [32, 6144],
                                                'transformer.text_fusion.layerwise_blocks.0.attn.to_q.lora_A.weight': [32, 2560]}, ('lora', 'krea-2')),
                'blocks-lora.safetensors': ({'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': [32, 6144]}, ('lora', 'krea-2')),
                'comfy-lora.safetensors': ({'diffusion_model.blocks.0.attn.wq.lora_A.weight': [64, 6144], 'diffusion_model.blocks.0.attn.wq.lora_B.weight': [6144, 64]}, ('lora', 'krea-2')),
                'kohya-lora.safetensors': ({'lora_unet_blocks_0_attn_wq.lora_down.weight': [16, 6144], 'lora_unet_blocks_0_attn_wq.alpha': []}, ('lora', 'krea-2')),
                'qwen21-lora.safetensors': ({'transformer_blocks.0.attn.to_q.lora_A.default.weight': [16, 4096]}, ('lora', 'qwen-image-2.1')),
                'qwen21-model.safetensors': ({'img_in.weight': [4096, 64]}, ('model', 'qwen-image-2.1')),
            }
            for name, (shapes, expected) in cases.items():
                path = Path(folder) / name
                safetensors(path, shapes)
                info = civitai.inspect(path)
                self.assertEqual((info['kind'], info['arch']), expected, name)

    def test_example_images_tell_turbo_settings(self):
        version = {'images': [{'meta': {'steps': 8, 'cfgScale': 1, 'sampler': 'Euler', 'scheduler': 'normal'}}] * 5
                   + [{'meta': {'steps': 8, 'cfgScale': 1, 'sampler': 'Euler', 'scheduler': 'simple'}}] * 3 + [{'meta': None}, {}]}
        found = civitai.example_settings(version)
        self.assertEqual({k: found[k] for k in ('steps', 'cfg', 'sampler', 'scheduler')}, {'steps': 8, 'cfg': 1.0, 'sampler': 'euler', 'scheduler': 'normal'})
        self.assertIs(civitai.detect_turbo('X2_INT8_ConvRot (Fastest)', found), True)
        self.assertEqual(civitai.example_settings({'images': [{'meta': {'prompt': 'x'}}]}), {})
        self.assertEqual(civitai.example_settings({'images': [{'meta': {'steps': 30, 'sampler': 'DPM++ 2M', 'scheduler': 'Karras'}}]})['sampler'], 'dpmpp_2m')

    def test_krea_defaults_take_page_settings(self):
        turbo = civitai.image_defaults(True, {}, 'u', 'krea-2')
        self.assertEqual((turbo['steps'], turbo['cfg'], turbo['mu'], turbo['source']), (8, 1.0, 1.15, 'u'))
        raw = civitai.image_defaults(False, {'steps': 30, 'cfg': 4.0, 'scheduler': 'beta', 'quote': 'q'}, 'u', 'krea-2')
        self.assertEqual((raw['steps'], raw['cfg'], raw['sampler'], raw['scheduler']), (30, 4.0, 'euler', 'beta'))
        self.assertEqual(raw['mu_range'], [0.5, 1.15])

    def test_checkpoint_file_choice(self):
        def file(name, precision, primary=False, size=13, kind='Model', fmt='SafeTensor'):
            return {'id': name, 'name': name, 'type': kind, 'format': fmt, 'precision': precision, 'primary': primary, 'bytes': size * 1024**3, 'sha256': ''}
        files = [file('int4', 'int4', primary=True, size=6), file('int8', 'int8'), file('bf16', 'bf16', size=25), file('nvfp4', 'nvfp4', size=7),
                 file('fp8', 'fp8'), file('q8.gguf', None, size=13, fmt='GGUF'), file('wf.json', None, kind='Workflow', fmt='Other'),
                 file('cfg.yaml', None, kind='Config', fmt='Other'), file('data.zip', None, kind='Training Data', fmt='Other')]
        usable = [f for f in files if civitai._usable_file('Checkpoint', f)]
        self.assertEqual({f['id'] for f in usable}, {'int4', 'int8', 'bf16', 'nvfp4', 'fp8', 'q8.gguf'})
        best = lambda candidates: [f['id'] for f in sorted(candidates, key=lambda f: civitai._file_rank(f, True))]
        self.assertEqual(best(usable), ['fp8', 'nvfp4', 'bf16', 'int8', 'int4', 'q8.gguf'])
        # A mislabeled small "fp8" file is likely 4-bit: the full-size one wins.
        self.assertEqual(best([file('small', 'fp8', size=7), file('full', 'fp8', size=12)])[0], 'full')
        for base in ('Qwen 2.1', 'Krea 2', 'Krea 2 Turbo'):
            self.assertTrue(civitai._best_quality('Checkpoint', civitai.BASE_MODELS[base]))
        self.assertFalse(civitai._best_quality('LORA', civitai.BASE_MODELS['Krea 2']))
        self.assertFalse(civitai._best_quality('Checkpoint', civitai.BASE_MODELS['MiniMax H3']))
        # LoRAs keep the page's primary file.
        self.assertEqual(sorted(usable, key=civitai._file_rank)[0]['id'], 'int4')

    def test_install_maps_a_krea_checkpoint(self):
        version = {'id': 3339307, 'modelId': 2945475, 'name': 'X2_INT8_ConvRot (Fastest)', 'baseModel': 'Krea 2', 'description': '',
                   'files': [{'id': 1, 'name': 'model_int8.safetensors', 'type': 'Diffusion Model', 'sizeKB': 1, 'primary': True,
                              'metadata': {'format': 'SafeTensor', 'fp': 'int8'}, 'hashes': {}, 'downloadUrl': 'https://example.invalid/1'},
                             {'id': 2, 'name': 'model.json', 'type': 'Workflow', 'sizeKB': 1, 'metadata': {}, 'downloadUrl': 'x'}],
                   'images': [{'meta': {'steps': 8, 'cfgScale': 1, 'sampler': 'Euler', 'scheduler': 'simple'}}] * 3}
        model = {'id': 2945475, 'name': '[BSS] - DesireX - [Krea2]', 'type': 'Checkpoint', 'description': '', 'modelVersions': [version]}

        def get(path, params=None):
            return version if 'model-versions' in path else model

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def download(url, target, total, expected, progress):
                safetensors(target, {'txtfusion.projector.weight': [1, 12], 'blocks.0.attn.wq.weight': [6144, 6144]})
            with patch.multiple(civitai, _get=get, _download=download, _save_preview=lambda *a: None, MODELS_DIR=root, USER_CATALOG=root / 'user.json'), \
                    patch.object(models, 'INSTALL_PATH', root / 'installed.json'):
                item = civitai.install(version['id'], None, False, lambda *a: None)
                self.assertEqual((item['kind'], item['family'], item['format'], item['layout']), ('image', 'krea-2-turbo', 'INT8', 'comfy'))
                self.assertEqual(item['dependencies'], ['krea2-encoder-bf16', 'krea2-vae'])
                self.assertEqual(item['recommended_vae'], 'krea2-vae')
                defaults = item['generation_defaults']
                self.assertEqual((defaults['steps'], defaults['cfg'], defaults['sampler'], defaults['mu']), (8, 1.0, 'euler', 1.15))
                self.assertTrue(defaults['note'].startswith('From the Civitai page: 3 of 3 example images'))
                self.assertEqual(models.required(item['id']), [item['id'], 'krea2-encoder-bf16', 'krea2-vae'])


if __name__ == '__main__':
    unittest.main()


class KreaIdentityTests(unittest.TestCase):
    def test_card_settings_and_catalog_entry(self):
        from backend import krea_edit
        # Identity Edit V2 card: one photo ref_boost 2 beside; scene and person: person 2.5 beside, scene 1 aligned.
        self.assertEqual(krea_edit.reference_settings(1), {'ref_boost': 2.0, 'ref_boost_a': 1.0, 'ref_position': krea_edit.BESIDE, 'ref_position_a': krea_edit.ALIGNED})
        self.assertEqual(krea_edit.reference_settings(2)['ref_boost'], 2.5)
        self.assertEqual(krea_edit.GROUNDING_PX, 768)
        # The card: generate at 2 MP or less; the 4 MP auto size of a portrait becomes 1168 x 1760.
        self.assertEqual(krea_edit.identity_size(1664, 2496), (1168, 1760))
        self.assertEqual(krea_edit.identity_size(1024, 1024), (1024, 1024))
        item = models.entry(models.KREA_IDENTITY_LORA)
        self.assertEqual((item['repo'], item['files'][0]['path'], item['files'][0]['bytes']),
                         ('conradlocke/krea2-identity-edit', 'krea2_identity_edit_v2_r128.safetensors', 914188768))
        self.assertTrue((krea_edit.PACK / '__init__.py').is_file() and (krea_edit.PACK / 'LICENSE').is_file())

    def test_pack_places_the_photo_beside_the_target_on_frame_one(self):
        import torch
        from backend import krea_edit
        sys_path = str(krea_edit.APP_DIR / 'vendor' / 'ComfyUI')
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        pack = krea_edit.pack()
        ids = pack._imgids(1, 1, 2, 3, 'cpu')
        placed = pack._place_refs([ids], krea_edit.BESIDE, 2, 3)[0]
        self.assertTrue(torch.all(placed[..., 0] == 1))      # frame 1: the photo, not the target (frame 0)
        self.assertEqual(float(placed[..., 2].min()), 3.0)    # to the right of the 3-token-wide target

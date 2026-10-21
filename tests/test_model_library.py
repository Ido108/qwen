import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import civitai, model_library, models


def safetensors(path, tensors):
    """A header-only safetensors file: names and shapes are all the scan reads."""
    header = json.dumps({name: {'dtype': 'BF16', 'shape': shape, 'data_offsets': [0, 0]} for name, shape in tensors.items()}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(header)) + header)
    return path


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.legacy, self.library = root / 'data' / 'models', root / 'library'
        self.legacy.mkdir(parents=True)
        self.library.mkdir()
        self.patches = [patch.object(model_library, 'LIBRARY_DIR', self.library), patch.object(model_library, 'MODELS_DIR', self.legacy),
                        patch.object(models, 'MODELS_DIR', self.legacy), patch.object(model_library, 'SCAN_PATH', root / 'data' / 'library-scan.json'),
                        patch.object(model_library, '_manifest', {'mtime': None, 'files': {}})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def write(self, path, size=10):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * size)
        return path

    def test_files_go_to_comfyui_type_folders(self):
        cases = [
            ({'kind': 'image', 'directory': 'Comfy-Org-z_image_turbo'}, 'split_files/diffusion_models/z_image_turbo_bf16.safetensors',
             'diffusion_models/Comfy-Org-z_image_turbo/z_image_turbo_bf16.safetensors'),
            ({'kind': 'vae', 'directory': 'Comfy-Org-z_image_turbo'}, 'split_files/vae/ae.safetensors', 'vae/Comfy-Org-z_image_turbo/ae.safetensors'),
            ({'kind': 'encoder', 'directory': 'abenzerps-GGUF'}, 'text_encoders/qwen3vl_8b_int8_convrot.safetensors',
             'text_encoders/abenzerps-GGUF/qwen3vl_8b_int8_convrot.safetensors'),
            ({'kind': 'image', 'format': 'GGUF', 'directory': 'abenzerps-GGUF'}, 'qwen-image-2.1-Q8_0.gguf', 'diffusion_models/abenzerps-GGUF/qwen-image-2.1-Q8_0.gguf'),
            ({'kind': 'reference-video', 'directory': 'video/Abiray-MiniMax-H3-GGUF'}, 'unet/MiniMax-H3-Ref2VA-Q8_0.gguf',
             'diffusion_models/Abiray-MiniMax-H3-GGUF/MiniMax-H3-Ref2VA-Q8_0.gguf'),
            ({'kind': 'image-lora', 'directory': 'civitai/loras'}, '1_style.safetensors', 'loras/civitai/1_style.safetensors'),
            ({'kind': 'image-lora', 'directory': 'loras/akhaliq'}, 'checkpoints_v2/angles.safetensors', 'loras/akhaliq/angles.safetensors'),
            ({'kind': 'video-support', 'format': 'LoRA', 'directory': 'video/Comfy-Org-MiniMax-H3'}, 'loras/turbo.safetensors', 'loras/Comfy-Org-MiniMax-H3/turbo.safetensors'),
            ({'kind': 'video-support', 'directory': 'video/Lightricks-LTX-2.5'}, 'latent_upscale_models/up.safetensors', 'latent_upscale_models/Lightricks-LTX-2.5/up.safetensors'),
            ({'kind': 'upscale-support', 'directory': 'upscale/swiftvr'}, 'transformer/config.json', 'upscale_models/swiftvr/transformer/config.json'),
            ({'kind': 'image', 'directory': 'Qwen-Image-2.1', 'files': [{'path': 'transformer/a.safetensors'}, {'path': 'transformer/b.safetensors'}]},
             'transformer/a.safetensors', 'diffusers/Qwen-Image-2.1/transformer/a.safetensors'),
        ]
        for item, path, expected in cases:
            self.assertEqual(model_library.library_path(item, {'path': path}), (self.library / expected).resolve(), path)

    def test_reads_the_complete_library_copy_first(self):
        item, file = {'kind': 'vae', 'directory': 'flux3'}, {'path': 'vae/flux2-vae.safetensors', 'bytes': 10}
        legacy = self.write(self.legacy / 'flux3/vae/flux2-vae.safetensors')
        self.assertEqual(models.local_path(item, file), legacy)
        copy = self.write(self.library / 'vae/flux3/flux2-vae.safetensors', 4)  # incomplete: still the data folder
        self.assertEqual(models.local_path(item, file), legacy)
        self.write(copy)
        self.assertEqual(models.local_path(item, file), copy.resolve())
        with patch.object(model_library, 'LIBRARY_DIR', None):
            self.assertEqual(models.local_path(item, file), legacy)

    def test_a_reused_file_is_read_in_place_and_never_deleted(self):
        item, file = {'id': 'krea2-vae', 'kind': 'vae', 'directory': 'Comfy-Org-Krea-2'}, {'path': 'vae/qwen_image_vae.safetensors', 'bytes': 10}
        own = self.write(self.library / 'vae/Qwen_Image-VAE.safetensors')
        model_library.record(item, file, own, reused=True)
        self.assertEqual(models.local_path(item, file), own)
        model_library.remove(item, file)
        self.assertTrue(own.is_file())
        self.assertNotIn(model_library.key(item, file), model_library.recorded())

    def test_removal_deletes_the_app_copy_on_both_drives(self):
        item, file = {'kind': 'image-lora', 'directory': 'civitai/loras'}, {'path': '1_a.safetensors', 'bytes': 10}
        app = self.write(model_library.library_path(item, file))
        legacy = self.write(self.legacy / 'civitai/loras/1_a.safetensors')
        with patch.object(models, 'entry', return_value={**item, 'id': 'x', 'files': [file]}), patch.object(models, 'installed_manifest', return_value={}), \
                patch.object(models, 'write_json'):
            models.remove('x')
        self.assertFalse(app.exists() or legacy.exists())

    def test_new_downloads_go_to_the_library_when_it_has_room(self):
        item, file = {'kind': 'image', 'directory': 'flux3'}, {'path': 'diffusion_models/k.safetensors', 'bytes': 10}
        free = lambda value: patch.object(model_library.shutil, 'disk_usage', return_value=type('U', (), {'free': value})())
        with free(model_library.RESERVE_BYTES + 100):
            self.assertEqual(models.download_target(item, file), model_library.library_path(item, file))
            self.write(self.legacy / 'flux3/diffusion_models/k.safetensors.part')  # an unfinished download continues where it is
            self.assertEqual(models.download_target(item, file), models.store_path(item, file))
        with free(model_library.RESERVE_BYTES):
            other = {'path': 'diffusion_models/n.safetensors', 'bytes': 10}
            self.assertEqual(models.download_target(item, other), models.store_path(item, other))

    def test_an_upscaler_folder_stays_together(self):
        vae = {'kind': 'upscale-support', 'directory': 'upscale/SeedVR2'}
        self.write(self.legacy / 'upscale/SeedVR2/ema_vae_fp16.safetensors')
        dit = {'path': 'seedvr2_ema_3b_fp16.safetensors', 'bytes': 10}
        with patch.object(model_library.shutil, 'disk_usage', return_value=type('U', (), {'free': 10**15})()):
            self.assertEqual(models.download_target({**vae, 'kind': 'upscale'}, dit), models.store_path(vae, dit))
            self.write(self.library / 'upscale_models/SeedVR2/ema_vae_fp16.safetensors')
            self.assertEqual(models.download_target({**vae, 'kind': 'upscale'}, dit), model_library.library_path({**vae, 'kind': 'upscale'}, dit))

    def test_a_finished_download_moves_from_staging(self):
        staged = self.write(model_library.staging_dir({'directory': 'flux3'}) / 'vae/v.safetensors')
        target = self.library / 'vae/flux3/v.safetensors'
        self.assertEqual(model_library.settle(staged, target), target)
        self.assertTrue(target.is_file() and not staged.exists())

    def test_scan_adds_matching_library_models_once(self):
        lora = safetensors(self.library / 'loras/krea2/candid.safetensors',
                           {'diffusion_model.blocks.0.attn.wq.lora_A.weight': [16, civitai.KREA2_HIDDEN], 'diffusion_model.blocks.0.attn.wq.lora_B.weight': [civitai.KREA2_HIDDEN, 16]})
        safetensors(self.library / 'loras/flux1_person.safetensors', {'transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight': [16, 3072]})
        safetensors(self.library / 'loras/krea2/krea2_identity_edit_v2.safetensors', {'diffusion_model.blocks.0.attn.wq.lora_A.weight': [64, civitai.KREA2_HIDDEN]})
        (self.library / 'loras/page.safetensors').write_bytes(b'<!doctype html>\n<html></html>')
        safetensors(self.library / 'vae/ae.safetensors', {'decoder.conv_in.weight': [512, 16, 3, 3]})
        # The app's own copy of a catalog LoRA (not recorded in the manifest) is not a user file.
        tool = models.entry('lora-restore')
        own = model_library.library_path(tool, tool['files'][0])
        safetensors(own, {'diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight': [16, 4096]})
        result = model_library.scan()
        self.assertEqual([entry['library_path'] for entry in result['entries']], [str(lora)])
        self.assertNotIn(own.relative_to(self.library).as_posix(), {item['path'] for item in result['skipped']})
        entry = result['entries'][0]
        self.assertEqual((entry['kind'], entry['families'], entry['source']), ('image-lora', ['krea-2', 'krea-2-turbo'], 'library'))
        reasons = {item['path']: item['reason'] for item in result['skipped']}
        self.assertIn('FLUX.1', reasons['loras/flux1_person.safetensors'])
        self.assertIn('own edit mode', reasons['loras/krea2/krea2_identity_edit_v2.safetensors'])
        self.assertIn('web page', reasons['loras/page.safetensors'])
        self.assertIn('exact file', reasons['vae/ae.safetensors'])
        registered = []
        with patch.object(civitai, 'user_catalog', side_effect=lambda: list(registered)), patch.object(civitai, '_register', side_effect=registered.append), \
                patch.object(models, 'installed_manifest', return_value={}), patch.object(models, 'write_json') as saved:
            self.assertEqual(model_library.merge_scan(), [entry['id']])
            self.assertEqual(model_library.merge_scan(), [])
            self.assertTrue(saved.call_args[0][1][entry['id']]['verified'])
        self.assertEqual(models.local_path(registered[0], registered[0]['files'][0]), lora)
        model_library.dismiss(entry['id'])
        with patch.object(civitai, 'user_catalog', return_value=[]), patch.object(civitai, '_register') as register:
            self.assertEqual(model_library.merge_scan(), [])
            register.assert_not_called()


class SettingTests(unittest.TestCase):
    def test_library_comes_from_the_launcher_environment_only(self):
        # Tests and tools run without QWEN_MODELS_LIBRARY, so they never write into the user's library.
        self.assertEqual(os.environ.get('QWEN_MODELS_LIBRARY') is None, model_library.LIBRARY_DIR is None)


if __name__ == '__main__':
    unittest.main()

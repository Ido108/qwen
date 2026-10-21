"""MOSS Image: catalog settings, file requirements, tensor detection, LoRA names and source mapping."""
import os
import tempfile
import unittest

os.environ.setdefault('QWEN_DATA_DIR', tempfile.mkdtemp(prefix='moss-tests-'))
from backend import civitai, hf_hub, models
from backend.lora_loading import moss_aliases

# Real tensor names and shapes, read by HTTP range from Comfy-Org/MOSS-Image moss_image_sft_bf16.safetensors (2026-10-21).
MODEL = {'final_linear_video.weight': [128, 3072], 'layers.0.proj.qkv.weight': [9216, 3072], 'layers.0.proj.ffn_up_gate.weight': [16384, 3072],
         'layers.0.proj.ffn_down.weight': [3072, 8192], 'layers.0.proj.o.weight': [3072, 3072], 'rope.bands': [16],
         'shared_block_modulation.weight': [18432, 3072], 'text_embedder.proj.weight': [3072, 4096], 'video_embedder.weight': [3072, 128]}
# realrebelai/MOSS-Image_GGUFs MOSS-Image-SFT-Q8_0.gguf: split q/k/v, GGUF (reversed) dimensions.
GGUF = {'layers.0.proj.q.weight': [3072, 3072], 'layers.0.proj.ffn_up_gate.weight': [3072, 16384], 'shared_block_modulation.weight': [3072, 18432]}


class MossCatalogTests(unittest.TestCase):
    def test_official_settings(self):
        moss = [item for item in models.catalog() if item['kind'] == 'image' and models.is_moss(item)]
        self.assertTrue({'moss-sft-int8', 'moss-sft-bf16', 'moss-base-int8', 'moss-base-bf16', 'moss-sft-turbo-int8', 'moss-sft-gguf-q8_0'} <= {item['id'] for item in moss})
        for item in moss:
            defaults = item['generation_defaults']
            self.assertEqual((defaults['sampler'], defaults['scheduler'], defaults['shift']), ('euler', 'simple', 5.0), item['id'])
            speed = defaults.get('speed_lora')
            self.assertEqual(item['dependencies'], ['moss-encoder-int8', 'moss-vae-small'] + ([speed['id']] if speed else []))
            self.assertRegex(item['files'][0]['sha256'], r'^[0-9a-f]{64}$')
            if item['family'] == 'moss-image-turbo':
                self.assertEqual((defaults['steps'], defaults['cfg']), (8, 1.0), item['id'])
            elif item['id'].startswith('moss-serenity-'):
                # Serenity README: CFG 4; Turbo is 8 steps with the DMD LoRA, Full is 50.
                self.assertEqual((defaults['steps'], defaults['cfg']), (8 if speed else 50, 4.0), item['id'])
            else:
                # OpenMOSS: 50 steps, CFG 3, flow-matching Euler, shift 5.
                self.assertEqual((defaults['steps'], defaults['cfg']), (50, 3.0), item['id'])
            self.assertEqual(item.get('layout') == 'comfy', item['format'] != 'GGUF', item['id'])
        for support in ('moss-encoder-int8', 'moss-encoder-w4a8', 'moss-encoder-bf16', 'moss-vae-small', 'moss-lora-4step'):
            self.assertEqual(models.entry(support)['families'], ['moss-image', 'moss-image-base', 'moss-image-turbo'])
        fast = models.entry('moss-lora-4step')['accelerator']
        # The author's table: 4 steps up to 1024 px, 6 near 1536, 8 at 2048.
        self.assertEqual((fast['steps'], fast['steps_by_size']), (4, [[1048576, 4], [2359296, 6], [4194304, 8]]))

    def test_moss_uses_its_own_encoder_and_decoder(self):
        self.assertEqual(models.required('moss-sft-int8'), ['moss-sft-int8', 'moss-encoder-int8', 'moss-vae-small'])
        self.assertEqual(models.required('moss-sft-int8', 'moss-encoder-w4a8'), ['moss-sft-int8', 'moss-encoder-w4a8', 'moss-vae-small'])
        # A Qwen encoder or a Krea VAE does not apply; the full FLUX.2 VAE (the Civitai mirror's file) does.
        self.assertEqual(models.required('moss-base-int8', 'encoder-int8', vae='krea2-vae'), ['moss-base-int8', 'moss-encoder-int8', 'moss-vae-small'])
        self.assertEqual(models.required('moss-sft-int8', vae='flux3-vae'), ['moss-sft-int8', 'moss-encoder-int8', 'flux3-vae'])
        # One input image is redrawn (VAE encode + partial denoise); MOSS has no reference input for more.
        self.assertEqual(models.required('moss-sft-int8', images=1), ['moss-sft-int8', 'moss-encoder-int8', 'moss-vae-small'])
        with self.assertRaisesRegex(ValueError, 'one input image'):
            models.required('moss-sft-int8', images=2)
        with self.assertRaisesRegex(ValueError, 'needs Qwen-Image'):
            models.required('moss-sft-int8', tool_id=models.image_tools()[0]['id'])
        with self.assertRaisesRegex(ValueError, 'does not work'):
            models.required('moss-sft-int8', loras=['krea2-lora-style-reference'])
        self.assertIn('moss-lora-4step', models.required('moss-sft-int8', loras=['moss-lora-4step']))

    def test_engine_imports_without_the_runtime(self):
        import backend.moss_engine as engine
        self.assertEqual(engine.DEFAULT_SHIFT, 5.0)
        from backend.worker import image_engine
        self.assertIs(image_engine({'model_id': 'moss-sft-int8'}), engine.generate)


class MossDetectionTests(unittest.TestCase):
    def test_model_files(self):
        self.assertEqual(civitai.inspect_shapes(MODEL)['arch'], 'moss-image')
        self.assertEqual(civitai.inspect_shapes({'model.diffusion_model.' + k: v for k, v in MODEL.items()})['arch'], 'moss-image')
        self.assertEqual(civitai.inspect_shapes(GGUF)['arch'], 'moss-image')

    def test_lora_names(self):
        for prefix in ('diffusion_model.', 'transformer.', ''):
            found = civitai.inspect_shapes({f'{prefix}layers.0.proj.q.lora_A.weight': [16, 3072], f'{prefix}layers.0.proj.q.lora_B.weight': [3072, 16]})
            self.assertEqual((found['kind'], found['arch']), ('lora', 'moss-image'), prefix)
        kohya = civitai.inspect_shapes({'diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight': [128, 3072], 'diffusion_model.layers.0.proj.ffn_up_gate.alpha': []})
        self.assertEqual(kohya['arch'], 'moss-image')
        # Z-Image (layers.N.attention) and Qwen-Image-2.1 LoRAs stay what they are.
        self.assertEqual(civitai.inspect_shapes({'diffusion_model.layers.0.attention.to_q.lora_A.weight': [16, 3840]})['arch'], 'z-image')
        self.assertEqual(civitai.inspect_shapes({'transformer_blocks.0.attn.to_q.lora_A.weight': [16, 4096]})['arch'], 'qwen-image-2.1')

    def test_lora_aliases_reach_the_fused_qkv_slices(self):
        target = ('diffusion_model.layers.0.proj.qkv.weight', (0, 3072, 3072))
        mapping = moss_aliases({'diffusion_model.layers.0.proj.k': target, 'lora_unet_layers_0_proj_k': target})
        self.assertEqual(mapping['layers.0.proj.k'], target)
        self.assertEqual(mapping['transformer.layers.0.proj.k'], target)

    def test_types_and_sources(self):
        self.assertEqual(civitai.BASE_MODELS['MOSS Image']['arch'], 'moss-image')
        self.assertEqual(civitai.moss_family(True, 'v1'), 'moss-image-turbo')
        self.assertEqual(civitai.moss_family(False, 'Base'), 'moss-image-base')
        self.assertEqual(civitai.moss_family(False, 'SFT', 'mossImage_sft_3301904.safetensors'), 'moss-image')
        self.assertEqual(civitai.moss_family(False, 'realrebelai/MOSS-Image_GGUFs', 'MOSS-Image-Base-Q8_0.gguf'), 'moss-image-base')
        self.assertEqual(civitai.image_defaults(False, {}, 'u', 'moss-image')['steps'], 50)
        self.assertEqual(civitai.image_defaults(True, {}, 'u', 'moss-image')['steps'], 8)
        for repo in ('OpenMOSS-Team/MOSS-Image-SFT', 'OpenMOSS-Team/MOSS-Image-Base', 'Comfy-Org/MOSS-Image'):
            self.assertEqual(hf_hub.BASE_REPOS[repo][0], 'MOSS Image')
        fields = hf_hub._entry_fields('moss-image', False, True, False, {'precision': 'int8'}, {'url': 'u', 'repo': 'toxicdog/MOSS-Image-SFT-Turbo-INT8-ConvRot'}, '', 'x.safetensors')
        self.assertEqual((fields['family'], fields['text_encoder'], fields['recommended_vae'], fields['layout']), ('moss-image-turbo', 'moss-encoder-int8', 'moss-vae-small', 'comfy'))
        self.assertEqual(hf_hub._entry_fields('moss-image', True, False, False, {'precision': None}, {}, '', 'x')['families'], ['moss-image', 'moss-image-base', 'moss-image-turbo'])


if __name__ == '__main__':
    unittest.main()


class SerenityTests(unittest.TestCase):
    def test_serenity_entries_follow_the_readme(self):
        full, turbo = models.entry('moss-serenity-v2-bf16'), models.entry('moss-serenity-v2-turbo-bf16')
        self.assertEqual((full['family'], full['generation_defaults']['steps'], full['generation_defaults']['cfg'], full['generation_defaults']['negative_prompt']),
                         ('moss-image-base', 50, 4.0, 'bad ugly blurry'))
        self.assertEqual((turbo['generation_defaults']['steps'], turbo['generation_defaults']['cfg'], turbo['generation_defaults']['speed_lora']['id']), (8, 4.0, 'moss-lora-dmd8'))
        self.assertIn('moss-lora-dmd8', models.required('moss-serenity-v2-turbo-bf16'))
        self.assertEqual(len(full['known_people']), 68)
        self.assertEqual(len(models.entry('moss-serenity-v1-int8')['known_people']), 28)
        self.assertEqual({p['class'] for p in full['known_people']}, {'man', 'woman'})
        dmd = models.entry('moss-lora-dmd8')
        self.assertEqual((dmd['accelerator']['steps'], dmd['families']), (8, ['moss-image-base']))


class MossDenoiseTests(unittest.TestCase):
    def test_denoise_like_zimage(self):
        from backend.moss_engine import moss_denoise
        self.assertEqual(moss_denoise({}, []), 1.0)
        self.assertEqual(moss_denoise({}, ['image']), 0.7)
        self.assertEqual(moss_denoise({'denoise': 0.5}, ['image']), 0.5)
        self.assertEqual(moss_denoise({'denoise': 0.5}, []), 1.0)

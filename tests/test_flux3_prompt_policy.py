"""FLUX.3 prompt model inputs follow the official upsampler policy; runtime patch and Krea/Z-Image worker reuse."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from backend import flux3_runtime, models
from backend.flux3_engine import canvas_line, prompt_reference, prompt_sampling
from backend.resident_worker import swaps_model_only


class PromptPolicyTests(unittest.TestCase):
    def test_references_shrink_to_the_official_budget_per_count(self):
        photo=Image.new('RGB',(4000,3000),'red')
        for count,cap in ((1,1024*1024),(2,724*724),(3,512*512),(10,512*512)):
            out=prompt_reference(photo,count)
            self.assertLessEqual(out.width*out.height,cap)
            self.assertEqual((out.width%32,out.height%32),(0,0))
            self.assertAlmostEqual(out.width/out.height,4/3,delta=0.08)
        small=prompt_reference(Image.new('RGB',(640,480)),1)
        self.assertEqual(small.size,(640,480))

    def test_transparency_is_flattened_over_white(self):
        out=prompt_reference(Image.new('RGBA',(64,64),(0,0,0,0)),1)
        self.assertEqual(out.mode,'RGB')
        self.assertGreater(min(out.getpixel((32,32))),245)

    def test_sampling_is_plain_greedy(self):
        # The sampled top_k 1 path measured slower than greedy on this runtime.
        self.assertEqual(prompt_sampling([],7),{'sampling_mode':'off'})
        self.assertEqual(prompt_sampling([object()],7),{'sampling_mode':'off'})

    def test_canvas_line_only_for_non_square(self):
        self.assertEqual(canvas_line(1024,1024),'')
        self.assertEqual(canvas_line(1344,768),'TARGET CANVAS: 7:4\n')
        self.assertEqual(canvas_line(832,1248),'TARGET CANVAS: 2:3\n')

    def test_catalog_offers_int8_prompt_model_and_nvfp4(self):
        self.assertEqual(models.entry('flux3-prompt-int8')['kind'],'image-support')
        for model_id,family in (('flux3-klein-nvfp4','flux3-klein'),('flux3-klein-base-nvfp4','flux3-klein-base')):
            item=models.entry(model_id)
            self.assertEqual((item['kind'],item['format'],item['family']),('image','NVFP4',family))
            self.assertEqual(models.required(model_id)[-1],'flux3-prompt')


class RuntimePatchTests(unittest.TestCase):
    def test_lora_bias_patch_is_applied_once_and_keeps_line_endings(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'comfy/lora.py';target.parent.mkdir()
            # The pinned upstream lines before Comfy-Org/ComfyUI#17239.
            target.write_bytes('\n    '.join(old for _,old,_ in flux3_runtime.PATCHES).encode('utf-8')+b'\n')
            flux3_runtime.patch(Path(folder));once=target.read_bytes()
            flux3_runtime.patch(Path(folder))
            self.assertEqual(target.read_bytes(),once)
            self.assertEqual(once.count(b'def bias_key(to):'),1)
            self.assertEqual(once.count(b'patch_dict[bias_key(to_load[x])]'),2)
            self.assertNotIn(b'\r\n',once)


class WorkerReuseTests(unittest.TestCase):
    def test_only_same_family_checkpoints_with_same_encoder_and_vae_reuse_the_worker(self):
        families={'a':'krea-2-turbo','b':'krea-2','c':'z-image-turbo','q':'qwen-image-2.1'}
        with patch('backend.models.entry',lambda model_id:{'id':model_id,'family':families[model_id]}):
            self.assertTrue(swaps_model_only(('a','enc','vae'),('b','enc','vae')))
            self.assertFalse(swaps_model_only(('a','enc','vae'),('b','enc2','vae')))
            self.assertFalse(swaps_model_only(('a','enc','vae'),('c','enc','vae')))
            self.assertFalse(swaps_model_only(('q','enc','vae'),('a','enc','vae')))
            self.assertFalse(swaps_model_only(None,('a','enc','vae')))


if __name__=='__main__':
    unittest.main()

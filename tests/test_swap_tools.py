import unittest

from PIL import Image

from backend import image_tools, models


class SwapToolTests(unittest.TestCase):
    def test_swap_tools_use_the_bfs_loras_on_qwen_image_21(self):
        tools = {tool['id']: tool for tool in models.image_tools() if tool.get('swap')}
        self.assertEqual(set(tools), {'head-swap', 'head-swap-expression', 'body-swap'})
        for tool in tools.values():
            lora = models.entry(tool['lora'])
            self.assertEqual(lora['repo'], 'Alissonerdx/BFS-Best-Face-Swap')
            self.assertTrue(tool['prompt'].startswith(('head_swap: start with <image1>', 'body_swap: start with <image1>')))
            self.assertEqual(tool['families'], ['qwen-image-2.1', 'qwen-image-2.1-turbo'])

    def test_head_swap_scales_both_pictures_to_the_budget_up_or_down(self):
        tool = models.image_tool('head-swap')
        small, large = Image.new('RGB', (1024, 1024)), Image.new('RGB', (3000, 2000))
        images, size, box = image_tools.prepare(tool, {'auto_resolution': 2048}, [small, large])
        self.assertEqual(images[0].size, (1408, 1408), 'a small scene is scaled up to about 2 MP, as in the author workflow')
        # Fitted inside the same 1408 px square box as the scene (BFS workflow at 2 MP), shape kept.
        self.assertEqual(images[1].size, (1408, 928))
        self.assertEqual(size, images[0].size)
        self.assertIsNone(box)

    def test_body_swap_fits_both_pictures_in_the_same_box(self):
        tool = models.image_tool('body-swap')
        images, size, _ = image_tools.prepare(tool, {'auto_resolution': 2048}, [Image.new('RGB', (1600, 1200)), Image.new('RGB', (2000, 2000))])
        self.assertEqual(images[0].size, (1408, 1056))
        self.assertEqual(images[1].size, (1408, 1408))
        self.assertEqual(size, images[0].size)

    def test_custom_size_changes_output_without_distorting_reference_inputs(self):
        tool=models.image_tool('head-swap')
        inputs,size,_=image_tools.prepare(tool,dict(custom_size=True,width=1280,height=768),
                                         [Image.new('RGB',(1000,1000)),Image.new('RGB',(1200,800))])
        self.assertEqual(size,(1280,768))
        self.assertEqual(inputs[0].width,inputs[0].height)
        self.assertAlmostEqual(inputs[1].width/inputs[1].height,1.5,delta=.04)


if __name__ == '__main__':
    unittest.main()

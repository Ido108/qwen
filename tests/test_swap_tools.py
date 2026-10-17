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
        self.assertAlmostEqual(images[1].width * images[1].height / 2_000_000, 1, delta=0.05)
        self.assertEqual(size, images[0].size)
        self.assertIsNone(box)

    def test_body_swap_gives_the_reference_person_about_059_mp(self):
        tool = models.image_tool('body-swap')
        images, size, _ = image_tools.prepare(tool, {'auto_resolution': 2048}, [Image.new('RGB', (1600, 1200)), Image.new('RGB', (2000, 2000))])
        self.assertAlmostEqual(images[1].width * images[1].height / 590_000, 1, delta=0.05)
        self.assertAlmostEqual(images[0].width * images[0].height / 2_000_000, 1, delta=0.05)
        self.assertEqual(size, images[0].size)


if __name__ == '__main__':
    unittest.main()

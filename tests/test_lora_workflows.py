import json
import unittest
from backend.lora_workflows import inspect_graph
from backend.lora_author import workflow


def graph():
    return {'prompt': {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2.1_int8.safetensors'}},
        '2': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'paper.safetensors', 'strength_model': .7}},
        '3': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['2', 0], 'lora_name': 'ink.safetensors', 'strength_model': .6}},
        '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'prompt': 'A paper cup', 'negative_prompt': '', 'resolution': 1536}},
        '5': {'class_type': 'KSampler', 'inputs': {'model': ['3', 0], 'positive': ['4', 0], 'negative': ['4', 1], 'latent_image': ['8', 0], 'steps': 50, 'cfg': 3, 'sampler_name': 'res_2m', 'scheduler': 'beta', 'denoise': 1}},
        '6': {'class_type': 'SaveImage', 'inputs': {'images': ['7', 0]}},
        '7': {'class_type': 'VAEDecode', 'inputs': {'samples': ['5', 0]}},
        '8': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 1024, 'height': 1024}},
    }}


class WorkflowTests(unittest.TestCase):
    def test_connected_order_and_res_sampler_are_exact(self):
        data = graph()
        data['prompt']['99'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {'lora_name': 'unused', 'strength_model': 5}}
        row, = inspect_graph(json.dumps(data))
        self.assertEqual([v['strength'] for v in row['loras']], [.7, .6])
        self.assertEqual(row['reference_images'], 0)
        self.assertEqual(row['positive_prompt'], 'A paper cup')
        self.assertEqual(row['negative_prompt'], '')
        version = {'images': [{'meta': {'comfy': json.dumps(data), 'steps': 8, 'cfgScale': 1}}]}
        advice = workflow(version, 'Use res_multistep as another option.')
        self.assertEqual((advice['steps'], advice['cfg'], advice['sampler']), (50, 3, 'res_2m'))
        self.assertEqual(advice['supported_sampler'], 'res_2m')
        self.assertIsNone(advice['author_sampler_alternative'])

    def test_unknown_sampler_stays_unsupported(self):
        data = graph()
        data['prompt']['5']['inputs']['sampler_name'] = 'unknown_custom'
        advice = workflow({'images': [{'meta': {'comfy': json.dumps(data)}}]}, 'Use res_multistep as another option.')
        self.assertIsNone(advice['supported_sampler'])
        self.assertEqual(advice['sampler'], 'unknown_custom')
        self.assertEqual(advice['author_sampler_alternative'], 'res_multistep')

    def test_disabled_entries_are_not_counted_and_cycles_end(self):
        data = graph(); data['prompt']['3'] = {'class_type': 'Power Lora Loader (rgthree)', 'inputs': {
            'model': ['1', 0], 'lora_1': {'on': True, 'lora': 'one', 'strength': 1},
            'lora_2': {'on': False, 'lora': 'off', 'strength': 9}}}
        self.assertEqual([v['file'] for v in inspect_graph(data)[0]['loras']], ['one'])
        data['prompt']['1']['inputs']['model'] = ['3', 0]
        self.assertEqual(len(inspect_graph(data)), 1)
        data['prompt'].pop('6')
        self.assertEqual(inspect_graph(data), [])

    def test_visual_loader_and_reference_input(self):
        data = graph(); data['prompt']['3'] = {'class_type': 'VisualLoRALoader', 'inputs': {
            'model': ['1', 0], 'lora_data': json.dumps([{'enabled': True, 'name': 'edit', 'strength_model': .8}, {'enabled': False, 'name': 'old', 'strength_model': 1}])}}
        data['prompt']['9'] = {'class_type': 'LoadImage', 'inputs': {'image': 'input.png'}}
        data['prompt']['4']['inputs']['images.image_1'] = ['9', 0]
        row, = inspect_graph(data)
        self.assertEqual(row['reference_images'], 1)
        self.assertEqual([v['file'] for v in row['loras']], ['edit'])


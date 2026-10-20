import unittest
from unittest.mock import patch, Mock

from backend.civitai_workflows import sampling_from_workflow, attach_workflows, _read_config, _cache


def graph():
    nodes = []
    links = []

    def node(id, kind, values=(), **inputs):
        sockets = []
        for name, source in inputs.items():
            link_id = len(links) + 1
            links.append([link_id, source, 0, id, len(sockets), '*'])
            sockets.append({'name': name, 'link': link_id})
        nodes.append(dict(id=id, type=kind, widgets_values=list(values), inputs=sockets))

    node(1, 'UnetLoaderGGUF', ['h3_ref2va_turbo.gguf'])
    node(2, 'BasicGuider', model=1)
    node(3, 'MiniMaxH3SigmaShift', [6, 3], model=1)
    node(4, 'BasicScheduler', ['simple', 4, 1], model=3)
    node(5, 'MiniMaxH3TurboSampler')
    node(6, 'SamplerCustomAdvanced', guider=2, sigmas=4, sampler=5)
    node(7, 'BasicScheduler', ['normal', 50, 1], model=1)
    return dict(nodes=nodes, links=links)


class CivitaiWorkflowTests(unittest.TestCase):
    def test_connected_settings_ignore_disconnected_nodes(self):
        d=sampling_from_workflow(graph())
        self.assertEqual([d[k] for k in ('steps','sampler','scheduler','video_shift','scheduler_video_shift')],
                         [4,'euler','simple',12,6])
        self.assertTrue(d['reference'])

    def test_multiple_passes_and_extra_loras_require_explicit_support(self):
        d=graph();d['nodes'].append(dict(id=8,type='SamplerCustomAdvanced'))
        self.assertIsNone(sampling_from_workflow(d))
        d=graph();d['nodes'][0]['type']='LoraLoaderModelOnly'
        self.assertIsNone(sampling_from_workflow(d))
        d=graph();d['nodes'][4]['type']='UnknownSampler'
        self.assertIsNone(sampling_from_workflow(d))

    def test_only_bounded_json_configs_are_fetched(self):
        files=[dict(id=1,type='Model',name='weights.safetensors',sizeKB=20000000),
               dict(id=2,type='Config',name='workflow.json',sizeKB=27),
               dict(id=3,type='Config',name='too-large.json',sizeKB=2000),
               dict(id=4,type='Config',name='script.py',sizeKB=1)]
        v=dict(id=123,baseModel='MiniMax H3',files=files)
        with patch('backend.civitai_workflows._read_config',return_value={'steps':4}) as read:
            attach_workflows([v])
            read.assert_called_once_with(123,files[1])
        self.assertEqual(v['_workflow_settings'],[{'steps':4}])

    def test_oversized_response_does_not_read_or_execute_content(self):
        _cache.clear()
        response=Mock(headers={'content-length':'999999999'})
        context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
        with patch('backend.civitai.access_token',return_value=None),patch('backend.civitai_workflows.httpx.stream',return_value=context):
            self.assertIsNone(_read_config(999,dict(id=998)))
        response.iter_bytes.assert_not_called()

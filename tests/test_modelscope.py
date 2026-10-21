import json
import unittest
from unittest.mock import patch, MagicMock
from backend import civitai, hf_hub, modelscope


def answer(data):
    return MagicMock(status_code=200, json=MagicMock(return_value={'Code': 200, 'Data': data}))


RAW = {'Path': 'owner', 'Name': 'zit-style', 'NickName': 'Owner', 'Downloads': 50, 'Stars': 3, 'AigcType': 'LoRA',
       'BaseModel': ['Tongyi-MAI/Z-Image-Turbo'], 'BaseModelRelation': 'adapter', 'SubVisionFoundation': 'Z_IMAGE_TURBO',
       'MuseInfo': "{'cover': 'https://resources.modelscope.cn/cover-images/abc.png'}", 'Revision': 'master',
       'Description': 'Trigger: zit. 1.0-1.5 strength', 'ReadMeContent': '', 'TriggerWords': 'zit, style'}


class ModelScopeTests(unittest.TestCase):
    def setUp(self):
        hf_hub._cache.clear()

    def test_listing_filters_by_foundation_tags_and_type(self):
        bodies = []

        def request(method, url, content, params, headers, timeout, follow_redirects):
            bodies.append(json.loads(content))
            return answer({'Model': {'Models': [RAW], 'TotalCount': 30}})
        with patch.object(modelscope.httpx, 'request', side_effect=request):
            items, cursor, total = modelscope.listing(['Tongyi-MAI/Z-Image', 'Tongyi-MAI/Z-Image-Turbo'], 'LORA', 'Newest', 'style', None)
        criteria = {c['category']: c['values'] for c in bodies[0]['Criterion']}
        self.assertEqual(set(criteria['sub_vision_foundation']), {'Z_IMAGE', 'Z_IMAGE_TURBO'})
        self.assertEqual(criteria['aigc_type'], ['LoRA'])
        self.assertEqual((bodies[0]['SortBy'], bodies[0]['Name']), ('GmtModified', 'style'))
        self.assertEqual((cursor, total), ('2', 30))
        self.assertEqual(items[0]['repo'], 'owner/zit-style')
        self.assertEqual(items[0]['preview'], 'https://resources.modelscope.cn/cover-images/abc.png')

    def test_untagged_architectures_keep_only_declared_bases(self):
        flux2 = {**RAW, 'Name': 'flux2', 'BaseModel': ['black-forest-labs/FLUX.2-klein-9B'], 'SubVisionFoundation': ''}
        flux3 = {**RAW, 'Name': 'flux3', 'BaseModel': ['black-forest-labs/flux-3-image-klein'], 'SubVisionFoundation': ''}
        with patch.object(modelscope.httpx, 'request', return_value=answer({'Model': {'Models': [flux2, flux3], 'TotalCount': 2}})):
            items, _, _ = modelscope.listing(['black-forest-labs/flux-3-image-klein'], 'LORA', 'Most Downloaded', '', None)
        self.assertEqual([i['repo'] for i in items], ['owner/flux3'])

    def test_card_reads_files_checksums_and_trigger_words(self):
        files = {'Files': [{'Path': 'zit.safetensors', 'Size': 1000, 'Sha256': 'AB' * 32, 'Type': 'blob'}, {'Path': 'sub', 'Type': 'tree'}]}

        def request(method, url, content, params, headers, timeout, follow_redirects):
            return answer(files if url.endswith('/repo/files') else RAW)
        with patch.object(modelscope.httpx, 'request', side_effect=request):
            result = modelscope.card('owner/zit-style')
        self.assertEqual(result['files'], [{'path': 'zit.safetensors', 'bytes': 1000, 'sha256': 'ab' * 32}])
        self.assertEqual(result['trained_words'], ['zit', 'style'])
        self.assertEqual(result['type'], 'LORA')
        self.assertEqual(modelscope.file_url('owner/zit-style', 'master', 'a b.safetensors'),
                         'https://www.modelscope.cn/api/v1/models/owner/zit-style/repo?Revision=master&FilePath=a%20b.safetensors')

    def test_details_through_the_shared_hub_code(self):
        made = {**modelscope._item(RAW), 'source': 'modelscope', 'url': 'https://www.modelscope.cn/models/owner/zit-style', 'revision': 'master', 'gated': False,
                'type': 'LORA', 'text': '', 'trained_words': ['zit'], 'images': [], 'files': [{'path': 'zit.safetensors', 'bytes': 10, 'sha256': ''}]}
        lora = {'layers.0.attention.to_q.lora_A.weight': [16, 3840]}
        with patch.object(modelscope.SOURCE, 'card', return_value=made), patch.object(hf_hub, 'remote_shapes', return_value=lora), \
                patch.object(civitai, 'user_catalog', return_value=[]):
            result = hf_hub.details('modelscope', 'owner/zit-style')
        version = result['versions'][0]
        self.assertEqual((version['base_model'], version['supported'], version['trained_words']), ('ZImageTurbo', True, ['zit']))

    def test_errors_are_reported_as_connection_errors(self):
        with patch.object(modelscope.httpx, 'request', return_value=MagicMock(status_code=500)):
            with self.assertRaises(ConnectionError):
                modelscope.card('owner/x')


if __name__ == '__main__':
    unittest.main()

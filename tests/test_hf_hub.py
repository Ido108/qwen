import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from backend import civitai, hf_hub


def header(shapes):
    body = json.dumps({'__metadata__': {}, **{name: {'dtype': 'BF16', 'shape': shape, 'data_offsets': [0, 0]} for name, shape in shapes.items()}}).encode()
    return struct.pack('<Q', len(body)) + body


KREA_LORA = {'diffusion_model.blocks.0.attn.wq.lora_A.weight': [16, 6144], 'diffusion_model.blocks.0.attn.wq.lora_B.weight': [6144, 16]}
QWEN21_LORA = {'transformer_blocks.0.attn.to_q.lora_A.weight': [16, 4096], 'transformer_blocks.0.attn.to_q.lora_B.weight': [4096, 16]}
QWEN1_LORA = {'transformer_blocks.0.attn.to_q.lora_A.weight': [16, 3072]}
QWEN21_MODEL = {'img_in.weight': [4096, 64], 'img_in.bias': [4096]}


def card(repo='me/lora', bases=('krea/Krea-2-Turbo',), files=(), text='', relation='adapter'):
    return {'repo': repo, 'name': repo.split('/')[1], 'creator': 'me', 'downloads': 5, 'likes': 2, 'bases': list(bases), 'preview': None,
            'relation': relation, 'source': 'huggingface', 'url': f'https://huggingface.co/{repo}', 'revision': 'a' * 40, 'gated': False,
            'text': text, 'trained_words': [], 'images': ['https://huggingface.co/x.png'],
            'files': [{'path': path, 'bytes': 1000, 'sha256': f'{i:064x}'} for i, path in enumerate(files)]}


class HubTests(unittest.TestCase):
    def setUp(self):
        hf_hub._cache.clear()
        patcher = patch.object(civitai, 'user_catalog', return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def details(self, made, shapes):
        """details() with a fake repo; shapes maps file path -> tensor shapes read from its remote header."""
        with patch.object(hf_hub.HF, 'card', return_value=made), \
                patch.object(hf_hub, 'remote_shapes', side_effect=lambda url, headers: shapes[url.rsplit('/', 1)[-1]]):
            return hf_hub.details('huggingface', made['repo'])

    def test_remote_header_is_read_with_two_ranges(self):
        data = header(KREA_LORA)
        calls = []

        def get(url, headers, timeout, follow_redirects):
            first, last = map(int, headers['Range'].split('=')[1].split('-'))
            calls.append((first, last))
            return MagicMock(status_code=206, content=data[first:last + 1])
        with patch.object(hf_hub.httpx, 'get', side_effect=get):
            shapes = hf_hub.remote_shapes('https://huggingface.co/me/lora/resolve/x/a.safetensors', {})
        self.assertEqual(shapes, KREA_LORA)
        self.assertEqual(calls, [(0, 7), (8, len(data) - 1)])
        self.assertEqual(civitai.inspect_shapes(shapes)['arch'], 'krea-2')

    def test_inspect_uses_the_same_rules_for_local_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.safetensors'
            path.write_bytes(header(QWEN21_LORA))
            self.assertEqual(civitai.inspect(path), civitai.inspect_shapes(QWEN21_LORA))
            self.assertEqual(civitai.inspect(path)['arch'], 'qwen-image-2.1')

    def test_near_match_files_are_hidden(self):
        made = card(bases=('Qwen/Qwen-Image-2.1',), files=('good.safetensors', 'old_qwen1.safetensors', 'README.md'))
        result = self.details(made, {'good.safetensors': QWEN21_LORA, 'old_qwen1.safetensors': QWEN1_LORA})
        version = result['versions'][0]
        self.assertEqual(result['type'], 'LORA')
        self.assertEqual([f['name'] for f in version['files']], ['good.safetensors'])
        self.assertTrue(version['supported'])
        self.assertEqual(version['base_model'], 'Qwen 2.1')

    def test_no_matching_file_gives_a_reason(self):
        result = self.details(card(bases=('Qwen/Qwen-Image-2.1',), files=('old.safetensors',)), {'old.safetensors': QWEN1_LORA})
        self.assertFalse(result['versions'][0]['supported'])
        self.assertIn('tensor shapes', result['versions'][0]['problem'])

    def test_undeclared_base_is_not_supported(self):
        result = self.details(card(bases=('black-forest-labs/FLUX.2-klein-9B',), files=('a.safetensors',)), {'a.safetensors': KREA_LORA})
        self.assertFalse(result['versions'][0]['supported'])
        self.assertIn('does not declare', result['versions'][0]['problem'])

    def test_identity_edit_adapters_are_not_installed_as_plain_loras(self):
        for repo, text in (('conradlocke/krea2-identity-edit', ''), ('other/edit', 'Install comfyui-krea2edit. Both nodes are required.')):
            hf_hub._cache.clear()
            result = self.details(card(repo=repo, files=('a.safetensors',), text=text), {'a.safetensors': KREA_LORA})
            version = result['versions'][0]
            self.assertFalse(version['supported'])
            self.assertEqual(version['problem'], hf_hub.SPECIAL_PROBLEM)
            self.assertFalse(version['files'][0]['supported'])

    def test_mixed_official_repo_keeps_each_file_type(self):
        made = card(repo='Comfy-Org/Qwen-Image-2.1', bases=('Qwen/Qwen-Image-2.1',), relation='quantized',
                    files=('diffusion_models/qwen_image_2.1_bf16.safetensors', 'diffusion_models/qwen_image_2.1_turbo_bf16.safetensors'),
                    text='Turbo: use 8 steps, CFG 1.')
        result = self.details(made, {'qwen_image_2.1_bf16.safetensors': QWEN21_MODEL, 'qwen_image_2.1_turbo_bf16.safetensors': QWEN21_MODEL})
        files = {f['name'].rsplit('/', 1)[-1]: f for f in result['versions'][0]['files']}
        self.assertEqual(result['type'], 'Checkpoint')
        standard, turbo = files['qwen_image_2.1_bf16.safetensors'], files['qwen_image_2.1_turbo_bf16.safetensors']
        self.assertEqual((standard['family'], turbo['family']), ('qwen-image-2.1', 'qwen-image-2.1-turbo'))
        self.assertGreater(standard['generation_defaults']['steps'], 10)  # the Turbo sentence does not apply to the Standard file
        self.assertEqual(turbo['generation_defaults']['steps'], 8)

    def test_video_checkpoints_are_not_offered_yet(self):
        H3_MODEL = {'audio_patch_proj.weight': [5376, 64]}
        result = self.details(card(bases=('MiniMaxAI/MiniMax-H3',), relation='finetune', files=('h3.safetensors',)), {'h3.safetensors': H3_MODEL})
        self.assertEqual(result['versions'][0]['files'][0]['problem'], hf_hub.VIDEO_MODEL_PROBLEM)

    def test_search_keeps_only_declared_bases_and_maps_labels(self):
        items = [{'repo': 'a/one', 'name': 'one', 'creator': 'a', 'downloads': 9, 'likes': 1, 'bases': ['krea/Krea-2-Turbo'], 'preview': None, 'relation': 'adapter'},
                 {'repo': 'b/two', 'name': 'two', 'creator': 'b', 'downloads': 3, 'likes': 1, 'bases': ['some/Other'], 'preview': None, 'relation': 'adapter'}]
        with patch.object(hf_hub.HF, 'listing', return_value=(items, None, None)) as listing:
            result = hf_hub.search('huggingface', '', 'LORA', 'Most Downloaded', None, 'image', 'all', 'krea-2')
        self.assertEqual([c['id'] for c in result['items']], ['a/one'])
        self.assertEqual(result['items'][0]['versions'][0]['base_model'], 'Krea 2 Turbo')
        self.assertEqual(set(listing.call_args[0][0]), {'krea/Krea-2-Raw', 'krea/Krea-2-Turbo', 'Comfy-Org/Krea-2'})

    def test_checkpoint_variant_selects_base_repos(self):
        self.assertEqual(hf_hub._wanted('image', 'z-image', 'turbo', 'Checkpoint'), ['Tongyi-MAI/Z-Image-Turbo', 'Comfy-Org/z_image_turbo'])
        self.assertNotIn('Comfy-Org/Krea-2', hf_hub._wanted('image', 'krea-2', 'standard', 'Checkpoint'))
        self.assertEqual(hf_hub._wanted('video', 'qwen-image-2.1', 'all', 'LORA'), [])

    def test_hf_listing_uses_one_stream_per_base_and_relation(self):
        raw = lambda repo, downloads: {'id': repo, 'author': repo.split('/')[0], 'downloads': downloads, 'likes': 0, 'sha': 'b' * 40,
                                       'tags': ['base_model:adapter:krea/Krea-2-Turbo'], 'cardData': {}, 'siblings': []}
        seen = []

        def get(path, params):
            stream = dict(params)['filter']
            seen.append(stream)
            response = MagicMock(status_code=200)
            response.json.return_value = [raw(f'{stream[-4:]}/x{i}', i) for i in range(dict(params)['limit'] and int(dict(params)['limit']))]
            response.links = {'next': {'url': f'https://huggingface.co/api/models?cursor={stream}'}}
            return response
        with patch.object(hf_hub, '_hf_get', side_effect=get):
            items, cursor, _ = hf_hub._hf_listing(['krea/Krea-2-Turbo', 'krea/Krea-2-Raw'], 'Checkpoint', 'Most Downloaded', '', None)
        self.assertEqual(len(seen), 6)
        self.assertEqual(set(json.loads(cursor)), set(seen))
        self.assertEqual(items[0]['bases'], ['krea/Krea-2-Turbo'])

    def test_resolve_link(self):
        self.assertEqual(hf_hub.resolve_link('https://huggingface.co/RudySen/Krea2-realism-V2/tree/main'), {'source': 'huggingface', 'repo': 'RudySen/Krea2-realism-V2'})
        self.assertEqual(hf_hub.resolve_link('https://www.modelscope.cn/models/owner/name/files'), {'source': 'modelscope', 'repo': 'owner/name'})
        with self.assertRaises(ValueError):
            hf_hub.resolve_link('https://example.com/x')

    def test_install_registers_hub_entry(self):
        made = card(files=('a.safetensors',))
        registered, saved = [], {}
        with tempfile.TemporaryDirectory() as folder, \
                patch('backend.models.download_target', side_effect=lambda item, f: Path(folder) / item['directory'] / f['path']), \
                patch.object(hf_hub.HF, 'card', return_value=made), \
                patch.object(hf_hub, 'remote_shapes', return_value=KREA_LORA), \
                patch.object(hf_hub, 'download', side_effect=lambda url, target, *a: target.write_bytes(header(KREA_LORA))) as download, \
                patch.object(hf_hub, '_save_preview'), \
                patch.object(civitai, '_register', side_effect=registered.append), \
                patch('backend.models.installed_manifest', return_value=saved), \
                patch.object(hf_hub, 'write_json'):
            entry = hf_hub.install('huggingface', 'me/lora', 'a.safetensors', False, lambda *a: None)
            self.assertTrue((Path(folder) / entry['directory'] / 'a.safetensors').is_file())
        self.assertEqual(download.call_args[0][0], f"https://huggingface.co/me/lora/resolve/{'a' * 40}/a.safetensors")
        self.assertEqual(entry['source'], 'huggingface')
        self.assertEqual(entry['repo'], 'me/lora')
        self.assertEqual(entry['hub']['path'], 'a.safetensors')
        self.assertEqual(entry['families'], civitai.KREA2_FAMILIES)
        self.assertEqual(entry['kind'], 'image-lora')
        self.assertEqual(registered, [entry])
        self.assertIn(entry['id'], saved)

    def test_install_reuses_the_same_file_from_another_source(self):
        made = card(files=('a.safetensors',))
        existing = {'id': 'civitai-lora-1', 'directory': 'civitai/loras', 'files': [{'path': 'x.safetensors', 'sha256': f'{0:064x}'}]}
        with patch.object(hf_hub.HF, 'card', return_value=made), patch.object(hf_hub, 'remote_shapes', return_value=KREA_LORA), \
                patch.object(civitai, 'user_catalog', return_value=[existing]), \
                patch('backend.models.local_path', return_value=MagicMock(is_file=lambda: True)), \
                patch.object(hf_hub, 'download') as download:
            self.assertEqual(hf_hub.details('huggingface', 'me/lora')['versions'][0]['files'][0]['installed_as'], 'civitai-lora-1')
            self.assertIs(hf_hub.install('huggingface', 'me/lora', 'a.safetensors', False, lambda *a: None), existing)
        download.assert_not_called()

    def test_download_does_not_send_the_civitai_token(self):
        probe = MagicMock(status_code=206, url='https://cdn.example/a', headers={'content-range': 'bytes 0-0/10'})
        stream = MagicMock()
        stream.__enter__.return_value = probe
        with tempfile.TemporaryDirectory() as folder, patch.object(hf_hub.httpx, 'stream', return_value=stream) as opened, \
                patch.object(civitai, '_single', side_effect=lambda url, partial, total, progress: partial.write_bytes(b'0123456789')), \
                patch.object(civitai, 'access_token', return_value='civitai-secret'):
            target = Path(folder) / 'a.safetensors'
            hf_hub.download('https://huggingface.co/x', target, 10, '', {'User-Agent': 'x'}, lambda *a: None, 'Hugging Face')
            self.assertEqual(target.read_bytes(), b'0123456789')
        self.assertNotIn('Authorization', opened.call_args.kwargs['headers'])

    def test_checksum_mismatch_removes_the_partial_file(self):
        probe = MagicMock(status_code=200, url='https://cdn.example/a', headers={})
        stream = MagicMock()
        stream.__enter__.return_value = probe
        with tempfile.TemporaryDirectory() as folder, patch.object(hf_hub.httpx, 'stream', return_value=stream), \
                patch.object(civitai, '_single', side_effect=lambda url, partial, total, progress: partial.write_bytes(b'abc')):
            target = Path(folder) / 'a.safetensors'
            with self.assertRaises(ValueError):
                hf_hub.download('https://huggingface.co/x', target, 3, '0' * 64, {}, lambda *a: None, 'Hugging Face')
            self.assertFalse(target.exists() or target.with_name(target.name + '.part').exists())


class HubRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from backend.server import app
        cls.client = TestClient(app)
        cls.headers = {'X-Studio-Token': cls.client.get('/api/config').json()['token']}

    def test_search_and_details_routes(self):
        with patch.object(hf_hub, 'search', return_value={'items': [], 'total': 0, 'next': None, 'indexing': False}) as search:
            response = self.client.get('/api/hub/modelscope/search?kind=LORA&arch=krea-2', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(search.call_args[0][:2], ('modelscope', ''))
        self.assertEqual(self.client.get('/api/hub/elsewhere/search', headers=self.headers).status_code, 422)
        with patch.object(hf_hub, 'details', side_effect=ValueError('No file matches.')):
            response = self.client.get('/api/hub/huggingface/model?repo=a/b', headers=self.headers)
        self.assertEqual((response.status_code, response.json()['detail']), (400, 'No file matches.'))

    def test_install_joins_the_download_list_and_cancels(self):
        from backend import server
        started = threading.Event()

        def install(source, repo, path, turbo, progress):
            started.set()
            for _ in range(200):
                progress(1, 10, 'downloading')
                time.sleep(.01)
        selected = {'file': {'precision': 'BF16', 'format': 'SafeTensor', 'bytes': 10}}
        with patch.object(hf_hub, 'selected_file', return_value=selected), patch.object(hf_hub, 'install', side_effect=install):
            response = self.client.post('/api/hub/huggingface/install', json={'repo': 'a/b', 'path': 'x/y.safetensors', 'title': 'B'}, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            key = response.json()['download_id']
            self.assertEqual(key, 'huggingface:a/b:x/y.safetensors')
            started.wait(2)
            self.assertEqual(self.client.post('/api/hub/huggingface/install', json={'repo': 'a/b', 'path': 'x/y.safetensors'}, headers=self.headers).status_code, 409)
            record = server.civitai_downloads[key]
            # The download indicator cancels through the Civitai route with the record's version id.
            self.client.post(f"/api/civitai/install/{record['version_id']}/cancel", headers=self.headers)
            for _ in range(200):
                if record['status'] != 'running':
                    break
                time.sleep(.01)
        self.assertEqual(record['status'], 'cancelled')
        server.civitai_downloads.pop(key, None)


if __name__ == '__main__':
    unittest.main()

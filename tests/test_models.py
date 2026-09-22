"""Model selection, checksum verification, and deletion boundaries in temporary folders."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from backend import models

class ModelTests(unittest.TestCase):
    def test_image_defaults_match_both_official_comfy_workflows(self):
        import json
        def samplers(value):
            if isinstance(value,dict):
                if value.get('type')=='KSampler':
                    yield value['widgets_values'][2]
                for child in value.values():
                    yield from samplers(child)
            elif isinstance(value,list):
                for child in value:
                    yield from samplers(child)
        for filename in ('comfy_t2i.json','comfy_edit.json'):
            workflow=json.loads((models.APP_DIR/'reference'/filename).read_text(encoding='utf-8'))
            self.assertEqual(set(samplers(workflow)),{25})
        for item in models.catalog():
            if item['kind']=='image':
                self.assertEqual(item['generation_defaults']['steps'],25,item['id'])
                self.assertIn('workflow_templates',item['generation_defaults']['source'])

    def test_video_defaults_match_the_documented_model_families(self):
        for item in models.catalog():
            if item['kind']=='video':
                defaults=item['generation_defaults']
                self.assertEqual(defaults['steps'],8 if item['family']=='fasth3' else 20,item['id'])
                if item['family']=='minimax-h3':
                    self.assertEqual(defaults['turbo_steps'],8)

    def test_official_download_omits_duplicate_encoder_and_vae(self):
        item=models.entry('official-bf16')
        self.assertLess(item['bytes'],15_000_000_000)
        self.assertTrue(all(f['path'].startswith('transformer/') or f['path'] in ('LICENSE','README.md') for f in item['files']))
        self.assertTrue(any(f['path'].startswith('text_encoder/') for f in item['removal_files']))

    def test_removal_cleans_allowlisted_legacy_files_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            item={'id':'legacy-test','directory':'model','files':[{'path':'current.bin'}],
                  'removal_files':[{'path':'current.bin'},{'path':'old-encoder/weights.bin'}]}
            current=root/'model/current.bin'; legacy=root/'model/old-encoder/weights.bin'
            legacy.parent.mkdir(parents=True); current.write_bytes(b'current'); legacy.write_bytes(b'old')
            unrelated=root/'model/keep.txt'; unrelated.write_text('keep')
            with patch.multiple(models,MODELS_DIR=root,INSTALL_PATH=root/'installed.json'),patch.object(models,'entry',return_value=item),patch.object(models,'transfer',{'status':'idle'}):
                models.remove('legacy-test')
            self.assertFalse(current.exists()); self.assertFalse(legacy.exists()); self.assertTrue(unrelated.exists())

    def test_companions_follow_image_format(self):
        self.assertEqual(models.required('gguf-q8_0'),['gguf-q8_0','encoder-int8','vae'])
        self.assertEqual(models.required('gguf-q4_k_m'),['gguf-q4_k_m','encoder-int8','vae'])
        self.assertEqual(models.required('official-bf16'),['official-bf16','encoder-bf16','vae'])
        self.assertEqual(models.required('gguf-q4_k_m','encoder-bf16')[1],'encoder-bf16')
        with self.assertRaises(ValueError): models.required('gguf-q8_0','unknown')
        with self.assertRaises(ValueError): models.required('vae')

    def test_download_checksum_and_remove_preserve_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            content=b'valid test weights'
            item={'id':'test','title':'Test','directory':'test-model','repo':'test/model','revision':'fixed',
                  'bytes':len(content),'dependencies':[],
                  'files':[{'path':'weights.bin','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}]}
            def download(*args,**kwargs):
                path=root/'models'/'test-model'/'weights.bin'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(content)
                return str(path)
            with patch.multiple(models,DATA_DIR=root,MODELS_DIR=root/'models',INSTALL_PATH=root/'installed.json'), \
                 patch.object(models,'catalog',return_value=[item]), \
                 patch('huggingface_hub.hf_hub_download',side_effect=download):
                models.cancel_event.clear()
                models.transfer.update(status='running')
                models.install('test')
                self.assertEqual(models.transfer['status'],'succeeded')
                self.assertTrue(models.is_installed('test'))
                unrelated=root/'models'/'test-model'/'saved-image.png'
                unrelated.write_bytes(b'keep this')
                models.remove('test')
                self.assertFalse(models.is_installed('test'))
                self.assertEqual(unrelated.read_bytes(),b'keep this')
                item['files'][0]['sha256']='0'*64
                models.transfer.update(status='running')
                models.install('test')
                self.assertEqual(models.transfer['status'],'failed')
                self.assertFalse(models.is_installed('test'))

    def test_cancel_does_not_mark_model_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            item={'id':'test','title':'Test','directory':'model','repo':'test/model','revision':'fixed','bytes':1,'dependencies':[],
                  'files':[{'path':'weights.bin','bytes':1}]}
            with patch.multiple(models,DATA_DIR=root,MODELS_DIR=root/'models',INSTALL_PATH=root/'installed.json'),patch.object(models,'catalog',return_value=[item]):
                models.cancel_event.set()
                models.install('test')
                self.assertEqual(models.transfer['status'],'cancelled')
                self.assertFalse(models.is_installed('test'))
                models.cancel_event.clear()

    def test_escaping_model_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            models.local_path({'directory':'../../outside'},{'path':'unrelated.bin'})

if __name__=='__main__': unittest.main()

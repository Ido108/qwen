"""Model selection, checksum verification, and deletion boundaries in temporary folders."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from backend import models

class ModelTests(unittest.TestCase):
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

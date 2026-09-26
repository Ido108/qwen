"""Model-download coordination using tiny fake files; no network or inference."""
import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from backend import models


class AutomaticDownloadTests(unittest.TestCase):
    def test_exact_companions_are_verified_cached_and_not_downloaded_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);content=b'verified tiny fixture';calls=[]
            def item(name,deps=()):
                return {'id':name,'title':name,'kind':'image','directory':name,'repo':name,'revision':'pinned',
                        'bytes':len(content),'dependencies':list(deps),'files':[{'path':'weights.bin','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}]}
            entries=[item('image',['default-encoder','vae']),item('chosen-encoder'),item('default-encoder'),item('vae')]
            def download(repo,path,**kwargs):
                calls.append(repo);target=Path(kwargs['local_dir'])/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content);return target
            with patch.multiple(models,DATA_DIR=root,MODELS_DIR=root/'models',INSTALL_PATH=root/'installed.json',transfer={'status':'idle'},active_download=None), \
                 patch.object(models,'catalog',return_value=entries),patch('huggingface_hub.hf_hub_download',side_effect=download):
                needed=['image','chosen-encoder','vae']
                models.ensure_installed(needed,owner='job1')
                models.ensure_installed(needed,owner='job2')
                self.assertEqual(calls,needed)
                self.assertTrue(all(models.is_installed(name) for name in needed))
                self.assertFalse(models.is_installed('default-encoder'))

    def test_concurrent_manual_transfer_is_joined(self):
        installed=set();calls=[];entered=threading.Event();release=threading.Event()
        def install(ids,**kwargs):
            calls.extend(ids);entered.set();release.wait(3);installed.update(ids)
            with models.lock:models.transfer.update(status='succeeded')
        with patch.multiple(models,transfer={'status':'idle'},active_download=None), \
             patch.object(models,'entry',return_value={'kind':'image'}),patch.object(models,'is_installed',side_effect=lambda name:name in installed), \
             patch.object(models,'install',side_effect=install):
            models.start_download('image');self.assertTrue(entered.wait(1))
            result=[]
            thread=threading.Thread(target=lambda:(models.ensure_installed(['image'],owner='job'),result.append(True)))
            thread.start();release.set();thread.join(3)
            self.assertEqual(result,[True]);self.assertEqual(calls,['image'])

    def test_cancel_only_stops_download_owned_by_job(self):
        for manual in (False,True):
            entered=threading.Event();release=threading.Event();cancel=threading.Event();errors=[]
            def install(ids,**kwargs):
                entered.set();release.wait(3)
                with models.lock:models.transfer.update(status='cancelled')
            with patch.multiple(models,transfer={'status':'idle'},active_download=None,cancel_event=threading.Event()), \
                 patch.object(models,'entry',return_value={'kind':'image'}),patch.object(models,'is_installed',return_value=False), \
                 patch.object(models,'install',side_effect=install):
                if manual:models.start_download('other')
                def ensure():
                    try:models.ensure_installed(['image'],owner='job',cancelled=cancel.is_set)
                    except Exception as exc:errors.append(exc)
                thread=threading.Thread(target=ensure);thread.start();self.assertTrue(entered.wait(1));cancel.set();thread.join(2)
                self.assertEqual(models.cancel_event.is_set(),not manual)
                self.assertIsInstance(errors[0],models.DownloadCancelled)
                release.set();self.assertTrue(models.active_download['done'].wait(2))


if __name__=='__main__':unittest.main()

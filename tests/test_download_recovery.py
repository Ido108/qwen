import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from backend import server
from backend.common import read_json


class DownloadRecoveryTests(unittest.TestCase):
    def test_interruption_replays_exact_file_and_cancel_removes_recovery_request(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'CIVITAI_PENDING_PATH',Path(folder)/'pending.json'),patch.object(server,'civitai_downloads',{}),patch.object(server,'civitai_cancels',{}),patch.object(server.civitai,'download_file',return_value={'id':456,'name':'test.safetensors','precision':'fp8','format':'SafeTensor','bytes':100}):
            tasks=[]
            def thread(*args,**kwargs):
                tasks.append(kwargs['target']);return Mock()
            body=server.CivitaiInstall(version_id=123,file_id=456,turbo=True,title='Example model')
            with patch.object(server.threading,'Thread',side_effect=thread):
                server.civitai_install(body)
                self.assertEqual(read_json(server.CIVITAI_PENDING_PATH)['123:456']['file_id'],456)
                server.civitai_downloads.clear()
                server.recover_civitai_downloads()
                self.assertEqual(len(tasks),2)
            def cancelled(version,file,turbo,progress):
                progress(0,100,'downloading')
            server.civitai_install_cancel(123)
            with patch.object(server.civitai,'install',side_effect=cancelled):tasks[-1]()
            self.assertEqual(server.pending_civitai_downloads(),{})
            self.assertEqual(server.civitai_downloads['123:456']['status'],'cancelled')

    def test_two_files_of_one_version_have_separate_progress_cancel_and_recovery(self):
        from fastapi import HTTPException
        from backend.common import write_json
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'CIVITAI_PENDING_PATH',Path(folder)/'pending.json'),patch.object(server,'civitai_downloads',{}),patch.object(server,'civitai_cancels',{}),patch.object(server.civitai,'download_file',side_effect=lambda v,f:{'id':f or 456,'name':f'{f}.safetensors','precision':'fp8','format':'SafeTensor','bytes':100}):
            tasks=[]
            def thread(**kwargs):
                tasks.append(kwargs['target']);return Mock()
            write_json(server.CIVITAI_PENDING_PATH,{'123':{'version_id':123,'file_id':456,'title':'Legacy'}})
            with patch.object(server.threading,'Thread',side_effect=thread):
                server.recover_civitai_downloads()
                self.assertEqual(list(server.pending_civitai_downloads()),['123:456'])
                answer=server.civitai_install(server.CivitaiInstall(version_id=123,file_id=789,title='Second'))
                self.assertEqual(answer['download']['file_id'],789)
                self.assertEqual(answer['download_id'],'civitai:123:789')
                with self.assertRaises(HTTPException) as raised:
                    server.civitai_install(server.CivitaiInstall(version_id=123,file_id=456))
                self.assertEqual(raised.exception.status_code,409)
            self.assertEqual(len(server.pending_civitai_downloads()),2)
            with self.assertRaises(HTTPException):
                server.civitai_install_cancel(123)
            server.civitai_install_cancel(123,456)
            self.assertTrue(server.civitai_cancels['123:456'].is_set())
            self.assertFalse(server.civitai_cancels['123:789'].is_set())
            def install(version,file,turbo,progress):
                progress(55,100,'downloading')
                self.assertEqual(server.civitai_downloads[f'123:{file}']['completed_bytes'],55)
                return {'id':f'entry-{file}','title':str(file)}
            with patch.object(server.civitai,'install',side_effect=install):
                tasks[0]()
                self.assertEqual(list(server.pending_civitai_downloads()),['123:789'])
                tasks[1]()
            self.assertEqual(server.civitai_downloads['123:456']['status'],'cancelled')
            self.assertEqual(server.civitai_downloads['123:789']['status'],'succeeded')
            self.assertEqual(server.pending_civitai_downloads(),{})

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from backend import server
from backend.common import read_json


class DownloadRecoveryTests(unittest.TestCase):
    def test_interruption_replays_exact_file_and_cancel_removes_recovery_request(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'CIVITAI_PENDING_PATH',Path(folder)/'pending.json'),patch.object(server,'civitai_downloads',{}),patch.object(server,'civitai_cancels',{}):
            tasks=[]
            def thread(*args,**kwargs):
                tasks.append(kwargs['target']);return Mock()
            body=server.CivitaiInstall(version_id=123,file_id=456,turbo=True,title='Example model')
            with patch.object(server.threading,'Thread',side_effect=thread):
                server.civitai_install(body)
                self.assertEqual(read_json(server.CIVITAI_PENDING_PATH)['123']['file_id'],456)
                server.civitai_downloads.clear()
                server.recover_civitai_downloads()
                self.assertEqual(len(tasks),2)
            def cancelled(version,file,turbo,progress):
                progress(0,100,'downloading')
            server.civitai_install_cancel(123)
            with patch.object(server.civitai,'install',side_effect=cancelled):tasks[-1]()
            self.assertEqual(server.pending_civitai_downloads(),{})
            self.assertEqual(server.civitai_downloads['123']['status'],'cancelled')

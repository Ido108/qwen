"""The Create settings that Moise starts its jobs from, via real HTTP handlers. No model startup or inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import server


class CreateSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.patch=patch.object(server,'CREATE_SETTINGS_PATH',Path(self.temp.name)/'create-settings.json');self.patch.start();self.addCleanup(self.patch.stop)
        self.client=TestClient(server.app);self.token=self.client.get('/api/session').json()['token']

    def put(self,body):
        return self.client.put('/api/create-settings',json=body,headers={'x-studio-token':self.token})

    def test_settings_are_kept_without_the_prompt_and_pictures(self):
        self.assertEqual(self.client.get('/api/create-settings').json(),{})
        self.assertEqual(self.put({'model_id':'krea2-turbo-fp8','steps':12,'loras':[{'id':'a','strength':0.8}],'prompt':'p','images':['x.png'],'image_labels':['person'],'seed':5,'source_prompt':'old job','output_dir':'E:\\AI\\qwen-image-studio\\images'}).status_code,200)
        self.assertEqual(self.client.get('/api/create-settings').json(),{'model_id':'krea2-turbo-fp8','steps':12,'loras':[{'id':'a','strength':0.8}]})

    def test_an_output_folder_saved_earlier_is_not_returned(self):
        from backend.common import write_json
        write_json(server.CREATE_SETTINGS_PATH,{'model_id':'m','output_dir':'E:\\AI\\qwen-image-studio\\images'})
        self.assertEqual(self.client.get('/api/create-settings').json(),{'model_id':'m'})

    def test_settings_need_a_model(self):
        self.assertEqual(self.put({'steps':12}).status_code,400)


if __name__=='__main__':
    unittest.main()

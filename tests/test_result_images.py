"""Viewing copies/cache policy via real HTTP handlers. No model startup or inference."""
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from backend import server
from backend.common import write_json


class ResultImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patch=patch.object(server,'JOBS_DIR',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.id='a'*32;self.folder=self.root/self.id;self.folder.mkdir()
        self.source=self.root/'original.png'
        image=Image.new('RGBA',(2400,1200),(20,80,160,90));image.save(self.source)
        self.original=self.source.read_bytes()
        self.job={'id':self.id,'kind':'image','status':'succeeded','output_path':str(self.source),'request':{},'created_at':1}
        self.save();self.client=TestClient(server.app)
        self.base=f'/api/jobs/{self.id}'

    def save(self):write_json(self.folder/'status.json',self.job)

    def test_viewing_copy_preserves_ratio_alpha_and_original(self):
        response=self.client.get(self.base+'/display')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['content-type'],'image/webp')
        self.assertEqual(response.headers['cache-control'],'private, max-age=604800, immutable')
        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual(image.size,(1600,800));self.assertEqual(image.mode,'RGBA')
            self.assertEqual(image.getpixel((600,400))[3],90)
        for suffix in ['/image','/image?download=true']:
            original=self.client.get(self.base+suffix)
            self.assertEqual(original.content,self.original)
            self.assertIn('private',original.headers['cache-control'])
        self.assertIn('attachment',original.headers['content-disposition'])
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_small_image_is_not_upscaled_and_copy_is_reused(self):
        Image.new('RGB',(320,640),(50,60,70)).save(self.source)
        first=self.client.get(self.base+'/display');cached=self.folder/'display-v1.webp'
        before=cached.stat().st_mtime_ns
        with patch.object(server.Image,'open',side_effect=AssertionError('Copy must be reused')):
            second=self.client.get(self.base+'/display')
        self.assertEqual(first.content,second.content);self.assertEqual(before,cached.stat().st_mtime_ns)
        with Image.open(io.BytesIO(first.content)) as image:self.assertEqual(image.size,(320,640))

    def test_grayscale_alpha_is_preserved(self):
        Image.new('LA',(40,80),(90,70)).save(self.source)
        response=self.client.get(self.base+'/display')
        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual(image.mode,'RGBA');self.assertEqual(image.getpixel((20,40))[3],70)

    def test_parallel_first_views_share_one_valid_copy(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses=list(executor.map(lambda _:self.client.get(self.base+'/display'),range(4)))
        self.assertTrue(all(r.status_code==200 for r in responses))
        self.assertTrue(all(r.content==responses[0].content for r in responses))
        self.assertEqual(list(self.folder.glob('*.tmp')),[])

    def test_parallel_first_thumbnails_share_one_copy(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses=list(executor.map(lambda _:self.client.get(self.base+'/thumbnail'),range(8)))
        self.assertTrue(all(r.status_code==200 for r in responses))
        self.assertTrue(all(r.content==responses[0].content for r in responses))
        self.assertEqual(list(self.folder.glob('*.tmp')),[])

    def test_thumbnails_cache_but_api_and_missing_outputs_do_not(self):
        thumb=self.client.get(self.base+'/thumbnail');self.assertEqual(thumb.status_code,200)
        self.assertIn('private',thumb.headers['cache-control']);self.assertIn('immutable',thumb.headers['cache-control'])
        for path in ['/api/session','/api/history','/api/health']:
            self.assertEqual(self.client.get(path).headers['cache-control'],'no-store')
        self.source.unlink()
        for suffix in ['/image','/display']:
            response=self.client.get(self.base+suffix)
            self.assertEqual(response.status_code,404);self.assertEqual(response.headers['cache-control'],'no-store')

    def test_unfinished_video_and_invalid_jobs_never_get_image_copy(self):
        for change in [{'status':'running'},{'status':'succeeded','kind':'video'}]:
            self.job.update(change);self.save()
            response=self.client.get(self.base+'/display')
            self.assertEqual(response.status_code,404);self.assertEqual(response.headers['cache-control'],'no-store')
        self.assertEqual(self.client.get('/api/jobs/invalid/display').status_code,404)

import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import civitai, models
from backend.civitai_files import entry_location, saved_file, resume_path
from backend.civitai_settings import version_settings


def file(ident, precision='fp8', name=None):
    return {'id':ident,'name':name or f'example_{precision}.safetensors','sizeKB':100,
            'type':'Model','metadata':{'fp':precision,'format':'SafeTensor'},
            'hashes':{},'downloadUrl':f'https://example.com/{ident}'}


def release(base='Qwen 2.1', description='Recommended: 25 steps, CFG 3, euler/simple.'):
    v={'id':12,'modelId':10,'name':'v1.0','baseModel':base,'description':'','images':[],
       'files':[file(1,'fp8'),file(2,'int8')]}
    m={'id':10,'name':'Example','type':'Checkpoint','description':description,'modelVersions':[v]}
    return m,v


class CivitaiFileTests(unittest.TestCase):
    def test_precision_alone_keeps_shared_author_steps(self):
        m,v=release()
        for precision in ('nvfp4','fp8','int8','fp32'):
            f=file(1,precision)
            settings=version_settings(m,v,f)
            self.assertEqual((settings['steps'],settings['cfg']),(25,3))
            self.assertFalse(civitai.version_turbo(m,v,settings,f))

    def test_precision_specific_sections_override_common_settings(self):
        m,v=release(description='Common settings:\nRecommended: 25 steps, CFG 3.\nFP8:\nRecommended: 20 steps, CFG 2.\nINT8:\nRecommended: 30 steps, CFG 4.')
        for f,steps,cfg in [(v['files'][0],20,2),(v['files'][1],30,4)]:
            result=version_settings(m,v,f)
            self.assertEqual((result['steps'],result['cfg']),(steps,cfg))

    def test_explicit_base_and_turbo_files_for_each_image_architecture(self):
        for arch in ('Qwen 2.1','Krea 2','ZImageTurbo','Flux.3 Klein'):
            with self.subTest(arch=arch):
                m,v=release(arch,'Base:\nRecommended: 28 steps, CFG 3.\nTurbo (Distilled):\nRecommended: 8 steps, CFG 1.')
                v['files']=[file(1,name='example_base_fp8.safetensors'),file(2,name='example_distilled_fp8.safetensors')]
                for f,turbo,steps in [(v['files'][0],False,28),(v['files'][1],True,8)]:
                    settings=version_settings(m,v,f)
                    self.assertEqual(settings['steps'],steps)
                    self.assertEqual(civitai.version_turbo(m,v,settings,f),turbo)

    def test_mixed_files_do_not_inherit_unmatched_examples(self):
        m,v=release(description='')
        v['files']=[file(1,name='base.safetensors'),file(2,name='turbo.safetensors')]
        v['images']=[{'meta':{'steps':8,'cfgScale':1}}]
        self.assertEqual(version_settings(m,v,v['files'][0]),{})

    def test_named_file_step_count_overrides_shared_page(self):
        m,v=release()
        settings=version_settings(m,v,file(1,name='turbo_4step_fp8.safetensors'))
        self.assertEqual(settings['steps'],4)

    def test_pasted_example_data_and_training_notes_are_not_author_settings(self):
        # REDQW21 page: an old SDXL example's A1111 data and a training paragraph; the version images show 20 steps, CFG 1.
        m,v=release(description='Negative prompt: NSFW\nSteps: 10, Sampler: Euler a, CFG scale: 2, Seed: 3573086999, Size: 640x1280, Model hash: db0525a2bc\n'
                    'For one training image we noisify both to σ = 0.9567, the highest learnable step of the 8-step Turbo schedule.')
        v['images']=[{'meta':{'steps':20,'cfgScale':1,'sampler':'Euler'}}]*3
        settings=version_settings(m,v)
        self.assertEqual((settings['steps'],settings['cfg'],settings['sampler'],settings['settings_origin']),(20,1.0,'euler','examples'))

    def test_new_files_with_identical_names_get_distinct_ids_and_paths(self):
        first=entry_location([],12,file(1,name='same.safetensors'),False)
        second=entry_location([],12,file(2,name='same.safetensors'),False)
        self.assertNotEqual(first[0],second[0])
        self.assertNotEqual(first[2],second[2])
        self.assertNotEqual(first[0],entry_location([],12,file(1),True)[0])

    def test_legacy_installed_file_keeps_id_and_path(self):
        old={'id':'civitai-model-12','directory':'civitai/models','files':[{'path':'12_same.safetensors','sha256':'aa'}],
             'civitai':{'version_id':12,'file_id':1}}
        self.assertEqual(entry_location([old],12,file(1),False),(old['id'],old['directory'],old['files'][0]['path']))
        self.assertNotEqual(entry_location([old],12,file(2),False)[0],old['id'])

    def test_old_partial_download_resumes_only_for_an_unambiguous_file(self):
        f=file(1,name='same.safetensors')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);partial=root/'12_same.safetensors.part0'
            partial.write_bytes(b'partial')
            self.assertEqual(resume_path(root,'12_1_same.safetensors',12,f,[f]),'12_same.safetensors')
            self.assertEqual(resume_path(root,'12_1_same.safetensors',12,f,[f,file(2,name=f['name'])]),'12_1_same.safetensors')
            self.assertEqual(partial.read_bytes(),b'partial')

    def test_saved_identity_prefers_id_then_hash_and_rejects_ambiguous_names(self):
        files=[file(1,name='same.safetensors'),file(2,name='same.safetensors')]
        old={'files':[{'path':'12_same.safetensors','sha256':None}],'civitai':{'file_id':2}}
        self.assertEqual(saved_file(old,files)['id'],2)
        old['civitai'].clear()
        self.assertIsNone(saved_file(old,files))
        old['files'][0]['sha256']='AA';files[1]['hashes']={'SHA256':'aa'}
        self.assertEqual(saved_file(old,files)['id'],2)

    def test_download_selection_is_exact_and_rejects_attachment(self):
        m,v=release();v['model']={'type':'Checkpoint'}
        v['files'].append(dict(file(3),type='VAE'))
        with patch.object(civitai,'_get',return_value=v):
            self.assertEqual(civitai.download_file(12,2)['id'],2)
            self.assertEqual(civitai.download_file(12,None)['id'],1)
            with self.assertRaises(ValueError):civitai.download_file(12,3)
            with self.assertRaises(ValueError):civitai.download_file(12,999)

    def test_details_expose_distinct_files_and_shared_settings(self):
        m,v=release()
        with patch.object(civitai,'_get',return_value=m),patch.object(civitai,'_installed_versions',return_value=set()),patch('backend.civitai_workflows.attach_workflows'):
            result=civitai.details(10)['versions'][0]
        self.assertEqual(len(result['files']),2)
        self.assertEqual([f['precision'].lower() for f in result['files']],['fp8','int8'])
        self.assertTrue(all(f['supported'] and f['generation_defaults']['steps']==25 for f in result['files']))

    def test_ltx_support_is_checked_for_the_selected_file(self):
        m,v=release('LTXV 2.5','Dev:\nRecommended: 25 steps.\nDistilled:\nRecommended: 4 steps.')
        # Dev accepts the official adapter workflow; a custom 4-step distilled file does not.
        v['files']=[file(1,name='ltx_dev.safetensors'),file(2,name='ltx_distilled_4step.safetensors')]
        with patch.object(civitai,'_get',return_value=m),patch.object(civitai,'_installed_versions',return_value=set()),patch('backend.civitai_workflows.attach_workflows'):
            result=civitai.details(10)['versions'][0]
        self.assertTrue(result['supported'])
        self.assertTrue(result['files'][0]['supported'])
        self.assertFalse(result['files'][1]['supported'])
        self.assertEqual(result['default_file'],1)

    def test_ltx_shared_release_keeps_dev_and_distilled_settings_separate(self):
        m,v=release('LTXV 2.5','Dev:\nRecommended: 25 steps.\nDistilled:\nRecommended: 8 steps.')
        for name,steps in [('ltx_dev.safetensors',25),('ltx_distilled.safetensors',8)]:
            self.assertEqual(version_settings(m,v,file(1,name=name))['steps'],steps)

    def test_tiny_real_installs_keep_both_files_catalog_and_manifest(self):
        m,v=release();v['files']=[file(1,name='same.safetensors'),file(2,'int8',name='same.safetensors')]
        def get(path):
            return copy.deepcopy(v if 'model-versions' in path else m)
        def download(url,target,total,expected,progress):
            header={'img_in.weight':{'dtype':'BF16','shape':[4096,64],'data_offsets':[0,0]}}
            raw=json.dumps(header).encode()
            target.write_bytes(struct.pack('<Q',len(raw))+raw)
            progress(target.stat().st_size,target.stat().st_size,'verifying')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.multiple(civitai,USER_CATALOG=root/'catalog.json',MODELS_DIR=root/'models',PREVIEWS=root/'previews'),patch.object(models,'INSTALL_PATH',root/'installed.json'),patch.object(civitai,'_get',side_effect=get),patch.object(civitai,'_download',side_effect=download),patch.object(civitai,'_save_preview'),patch('backend.civitai_workflows.attach_workflows'):
                one=civitai.install(12,1,False,lambda *args:None)
                two=civitai.install(12,2,False,lambda *args:None)
                repeat=civitai.install(12,1,False,lambda *args:None)
                self.assertEqual(one['id'],repeat['id'])
                self.assertNotEqual(one['id'],two['id'])
                self.assertEqual(len(civitai.user_catalog()),2)
                self.assertEqual(set(models.installed_manifest()),{one['id'],two['id']})
                for item in (one,two):
                    self.assertTrue((root/'models'/item['directory']/item['files'][0]['path']).exists())
                    self.assertEqual(item['generation_defaults']['steps'],25)

    def test_repair_preserves_existing_ids_and_uses_exact_file_settings(self):
        m,v=release(description='Base:\nRecommended: 28 steps, CFG 3.\nTurbo:\nRecommended: 4 steps, CFG 1.')
        v['files']=[file(1,name='base.safetensors'),file(2,name='turbo.safetensors')]
        old={'id':'old-id','title':'Existing','kind':'image','family':'qwen-image-2.1',
             'format':'FP8','directory':'civitai/models','files':[{'path':'old.safetensors','bytes':100}],
             'civitai':{'model_id':10,'version_id':12,'file_id':2,'url':'https://example.com'}}
        with tempfile.TemporaryDirectory() as folder,patch.object(civitai,'USER_CATALOG',Path(folder)/'catalog.json'),patch.object(civitai,'_get',side_effect=lambda url:copy.deepcopy(v if 'model-versions' in url else m)),patch('backend.civitai_workflows.attach_workflows'):
            civitai._register(old)
            self.assertEqual(civitai.repair_entries(),['old-id'])
            updated=civitai.user_catalog()[0]
            self.assertEqual(updated['id'],'old-id')
            self.assertEqual(updated['files'],old['files'])
            self.assertEqual(updated['family'],'qwen-image-2.1-turbo')
            self.assertEqual(updated['generation_defaults']['steps'],4)
            self.assertEqual(updated['civitai']['file_id'],2)


if __name__=='__main__':
    unittest.main()

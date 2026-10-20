import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from backend import models,training,civitai
from backend.flux3_engine import PromptImages
from tests.test_krea import safetensors
from tests.test_training import REQUEST


class Flux3Tests(unittest.TestCase):
    def test_inference_has_its_exact_components_and_ten_reference_limit(self):
        self.assertEqual(models.required('flux3-klein-int8',images=10),
                         ['flux3-klein-int8','flux3-encoder-int8','flux3-vae','flux3-prompt'])
        with self.assertRaises(ValueError):models.required('flux3-klein-int8',images=11)
        with self.assertRaises(ValueError):models.required('flux3-base-int8')
        profile=models.entry('flux3-klein-int8')['generation_defaults']
        self.assertEqual((profile['steps'],profile['cfg'],profile['alpha']),(16,1,16))

    def test_training_explicitly_uses_base_and_reuses_matching_encoder_and_vae(self):
        process=training.build_config({**REQUEST,'arch':'flux3-klein-base'},Path('images'),Path('out'))['config']['process'][0]
        model=process['model'];kwargs=model['model_kwargs']
        self.assertEqual(model['arch'],'flux3')
        self.assertIn('flux3-image-klein-base-int8-convrot.safetensors',kwargs['model_path'])
        self.assertIn('qwen3vl_4b_int8_convrot.safetensors',kwargs['text_encoder_path'])
        self.assertIn('flux2-vae.safetensors',kwargs['ae_path'])
        self.assertEqual((model['qtype'],model['qtype_te']),('convrot8','convrot8'))
        self.assertTrue(kwargs['match_target_res'])
        self.assertEqual(Path(model['name_or_path']),Path(kwargs['model_path']).parent)
        self.assertTrue(Path(kwargs['config_path']).is_file())
        self.assertEqual((process['sample']['sample_steps'],process['sample']['guidance_scale']),(28,3.5))

    def test_tensor_structure_distinguishes_flux3_base_distilled_and_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.safetensors'
            for variant,keys in [('base',{'content_mode_blocks.image.0.attn_out.weight':[3072,3072]}),
                                 ('distilled',{'content_mode_blocks.image.0.attn_out.weight':[3072,3072],'ref_time_in.in_layer.weight':[3072,256]}),
                                 (None,{'diffusion_model.content_mode_blocks.image.0.attn_out.lora_A.weight':[16,3072]})]:
                safetensors(p,keys)
                found=civitai.inspect(p)
                self.assertEqual(found['arch'],'flux3-klein')
                self.assertEqual(found['variant'],variant)
                self.assertEqual(found['kind'],'lora' if variant is None else 'model')

    def test_prompt_model_receives_all_images_without_stretching_or_batching(self):
        images=[object(),object()];seen={}
        def tokenize(text,**kwargs):seen.update(text=text,**kwargs);return 'tokens'
        wrapper=PromptImages(SimpleNamespace(tokenize=tokenize,generate='generate'),images)
        self.assertEqual(wrapper.tokenize('edit',image=None,thinking=False),'tokens')
        self.assertIs(seen['images'],images)
        self.assertNotIn('image',seen)
        self.assertEqual(wrapper.generate,'generate')

    def test_runtime_accepts_the_loaded_namespace_on_the_next_job(self):
        from backend import flux3_runtime
        namespace=SimpleNamespace(__file__=None,__path__=[str(flux3_runtime.COMFY/'comfy')])
        with patch.object(flux3_runtime,'ensure'),patch.dict('sys.modules',{'comfy':namespace}):
            flux3_runtime.activate(lambda **_:None)
        namespace.__path__=[str(flux3_runtime.ROOT/'other/comfy')]
        with patch.object(flux3_runtime,'ensure'),patch.dict('sys.modules',{'comfy':namespace}):
            with self.assertRaises(RuntimeError):flux3_runtime.activate(lambda **_:None)

    def test_catalog_repair_preserves_flux3_architecture(self):
        item={'id':'future-flux','kind':'image','family':'flux3-klein','files':[],
              'civitai':{'version_id':2,'model_id':1,'url':'url'}}
        version={'id':2,'name':'Distilled','baseModel':'Flux.3 Klein','files':[]}
        with patch.object(civitai,'user_catalog',return_value=[item]),patch.object(civitai,'_get',side_effect=[version,{'id':1}]),patch.object(civitai,'_register'):
            self.assertEqual(civitai.repair_entries(),['future-flux'])
        self.assertEqual(item['family'],'flux3-klein')
        self.assertEqual(item['generation_defaults']['steps'],16)

    def test_training_tied_head_keeps_the_embedding_quantization(self):
        from backend.flux3_training import complete_tied_head_quantization
        weight=object();marker=object();scale=object();prefix='model.language_model.embed_tokens'
        state={prefix+'.weight':weight,prefix+'.comfy_quant':marker,prefix+'.weight_scale':scale,'lm_head.weight':weight}
        fixed=complete_tied_head_quantization(state)
        self.assertIs(fixed['lm_head.comfy_quant'],marker)
        self.assertIs(fixed['lm_head.weight_scale'],scale)
        separate={prefix+'.weight':weight,prefix+'.comfy_quant':marker,'lm_head.weight':object()}
        self.assertNotIn('lm_head.comfy_quant',complete_tied_head_quantization(separate))

    def test_prompt_identifiers_only_include_requested_lora_triggers(self):
        from backend.flux3_engine import prompt_identifiers,prompt_without_identifiers
        with patch('backend.models.entry',return_value={'title':'Style','trained_words':['myStyle','anotherStyle']}):
            words=prompt_identifiers({'loras':[{'id':'new-lora'}]},'mystyle teapot')
        self.assertEqual(words,['myStyle'])
        self.assertEqual(prompt_without_identifiers('mystyle, a teapot',words),'a teapot')
        self.assertEqual(prompt_without_identifiers('a sign saying "myStyle"',words),'a sign saying "myStyle"')

    def test_new_civitai_checkpoint_uses_its_tensor_variant_and_matching_files(self):
        # Model IDs are arbitrary; misleading page tags must not select the
        # wrong architecture profile after inspecting the actual file.
        for distilled in (True,False):
            version={'id':990011,'modelId':990010,'name':'New model','baseModel':'Flux.3 Klein-base' if distilled else 'Flux.3 Klein',
                     'files':[{'id':990012,'name':'future.safetensors','type':'Model','downloadUrl':'url','sizeKB':1,'metadata':{'format':'SafeTensor'}}]}
            model={'id':990010,'name':'Future model','type':'Checkpoint'}
            with tempfile.TemporaryDirectory() as folder:
                keys={'content_mode_blocks.image.0.attn_out.weight':[3072,3072]}
                if distilled:keys['ref_time_in.in_layer.weight']=[3072,256]
                with patch.object(civitai,'MODELS_DIR',Path(folder)),patch.object(civitai,'_get',side_effect=[version,model]), \
                     patch.object(civitai,'user_catalog',return_value=[]),patch.object(civitai,'_download',side_effect=lambda url,target,*args:safetensors(target,keys)), \
                     patch.object(civitai,'_save_preview'),patch.object(civitai,'_register'),patch.object(models,'installed_manifest',return_value={}), \
                     patch.object(models,'INSTALL_PATH',Path(folder)/'installed.json'):
                    found=civitai.install(990011,None,not distilled,lambda *args:None)
            self.assertEqual(found['family'],'flux3-klein' if distilled else 'flux3-klein-base')
            self.assertEqual(found['generation_defaults']['steps'],16 if distilled else 50)
            self.assertEqual(found['dependencies'],['flux3-encoder-int8','flux3-vae','flux3-prompt'])

    def test_civitai_browsing_accepts_locally_trained_loras_in_the_same_library(self):
        with patch.object(civitai,'user_catalog',return_value=[{'source':'trained'}, {'civitai':{'version_id':123}}]):
            self.assertEqual(civitai._installed_versions(),{123})

    def test_live_api_contract_accepts_flux3_search(self):
        from fastapi.testclient import TestClient
        from backend.server import app
        with patch.object(civitai,'search',return_value={'items':[]}) as search:
            response=TestClient(app).get('/api/civitai/search?arch=flux3-klein')
        self.assertEqual(response.status_code,200)
        self.assertEqual(search.call_args.args[-1],'flux3-klein')

    def test_author_base_naming_and_six_bit_filename_override_generic_tags(self):
        model={'name':'New exports','description':'Klein Distilled (flux3_image_klein_sd_*) and Klein Base (flux3_image_klein_*)'}
        version={'name':'Flux 3 Klein INT6','baseModel':'Flux.3 Klein'}
        self.assertFalse(civitai.version_turbo(model,version))
        self.assertTrue(civitai.version_turbo({'name':'Another page'},version))
        self.assertTrue(civitai.version_turbo(model,{**version,'name':'Flux 3 Klein SD INT6'}))
        self.assertEqual(civitai._file_precision({'name':'model_flux3KleinINT6.safetensors','metadata':{'fp':'int8'}}),'INT6')

    def test_civitai_base_tags_and_settings_are_separate(self):
        self.assertTrue(civitai.BASE_MODELS['Flux.3 Klein']['turbo'])
        self.assertFalse(civitai.BASE_MODELS['Flux.3 Klein-base']['turbo'])
        a=civitai.image_defaults(True,{},'url','flux3-klein')
        b=civitai.image_defaults(False,{},'url','flux3-klein')
        self.assertEqual((a['steps'],a['cfg'],b['steps'],b['cfg']),(16,1,50,3.5))

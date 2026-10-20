import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import civitai,models,server
from backend.civitai_settings import h3_defaults,version_settings,video_variant,h3_capabilities,version_section
from backend.common import VideoRequest
from backend.ltx_settings import MultiReferenceRequest
from backend.video_sampling import video_sampling


PAGE='Recommended Settings\nNon-Distilled\nres_mulistep/simple or euler/simple\nShift Video 10-12\nShift Audio 3-5\n20-25 Steps\nTurbo (Distilled)\neuler/simple or LCM/simple\nShift Video 6-12\nShift Audio 3-5\n4 or 8 Steps'


class VideoSettingsTests(unittest.TestCase):
    def test_ltx_exact_file_type_controls_adapter_and_reports_workflow(self):
        from backend.civitai_settings import ltx_defaults
        model={'name':'LTX 2.5 dev and distilled','description':''}
        version={'name':'Both variants','description':'','images':[]}
        for name,variant in [('ltx25_distilled_int8.safetensors','distilled'),('ltx25_dev_q4.gguf','dev')]:
            profile=ltx_defaults(model,version,'https://example.com',{'name':name})
            self.assertEqual(profile['variant'],variant)
            self.assertEqual((profile['steps'],profile['cfg']),(8,1))
            self.assertEqual('distilled LoRA' in profile['note'],variant=='dev')
        self.assertIsNone(civitai.ltx_variant('Unknown','v1',{'steps':4},''))
        self.assertEqual(civitai.ltx_variant('Pack','v1',{},'',{'name':'ltx_non-distilled.safetensors'}),'dev')
        version['description']='Use distilled LoRA strength: 0.7.'
        self.assertEqual(ltx_defaults(model,version,'u',{'name':'ltx25_dev.safetensors'})['distill_lora_strength'],0.7)
        problem=civitai.ltx_problem({'name':'Custom','type':'Checkpoint'},{'name':'distilled','baseModel':'LTXV 2.5'},{'steps':4},'')
        self.assertIn('different sampling schedule',problem)

    def version(self,name,description=''):
        return dict(id=1,name=name,description=description,baseModel='MiniMax H3',images=[])

    def test_quantization_and_training_counts_are_not_sampling_steps(self):
        v=self.version('Ref2VA v1','Calibration vs the official file: 100% within \u00b11 step.\nTraining steps: 120.\nStep 1: Download the model.')
        self.assertNotIn('steps',version_settings({},v))
        v['description']+='\nRecommended: 6 steps, er_sde/simple, no shift.'
        d=version_settings({},v)
        self.assertEqual((d['steps'],d['sampler'],d['video_shift']),(6,'er_sde',1))

    def test_benchmark_is_not_an_author_recommendation(self):
        v=self.version('Ref2VA v1')
        m={'description':'Same scene, same seed - 544x960, 243 frames, 10 steps.\nI tested 20 steps at 704x1280.'}
        self.assertNotIn('steps',version_settings(m,v))
        m['description']+='\nRecommended settings: 24 steps, euler/simple.'
        self.assertEqual(version_settings(m,v)['steps'],24)

    def test_partition_page_and_file_scopes(self):
        v=self.version('Ref2VA v0.1')
        m={'name':'MiniMax H3 Turbo GGUF','description':'Place minimax_h3_fl2v_turbo_4step.gguf in models.\nRecommended: 4 steps.\nVideo shift 6; audio shift 3.\nSupported input modes (FL2VA)'}
        d=h3_defaults(m,v,models.entry('h3-ref-base-int8'),'https://example.com',{'name':'h3_ref2va.gguf'})
        self.assertEqual(d['video_shift'],12)
        self.assertEqual(d['field_sources']['steps'],'workflow')
        self.assertIsNone(d['author_settings'])
        m['description']='FL2VA settings:\n4 steps, euler/simple. Video shift 6.\nRef2VA settings:\n8 steps, euler/simple. Video shift 12.'
        v=self.version('FL2VA / Ref2VA')
        for file,count,shift in [('h3_fl2va.gguf',4,6),('h3_ref2va.gguf',8,12)]:
            d=version_settings(m,v,{'name':file})
            self.assertEqual((d['steps'],d['video_shift']),(count,shift))

    def test_reference_hybrid_ancestry_does_not_discard_own_settings(self):
        v=self.version('beta6','Built on fl2va. Reference deltas added. Target 8-12 steps, 10/3 shift.')
        d=version_settings({'description':'beta 6:\nA reference model. Sampling uses FastH3 templates.'},v)
        self.assertEqual((d['steps'],d['video_shift'],d['sampler']),(8,10,'res_multistep'))

    def test_release_numbers_match_exactly(self):
        text='beta 1:\n4 steps\nbeta 10:\n12 steps'
        self.assertEqual(version_section(text,self.version('beta10')).strip(),'12 steps')

    def test_imported_base_keeps_own_steps_separate_from_optional_lora(self):
        v=self.version('Ref2VA Standard','Recommended: 30 steps, euler/simple.')
        d=h3_defaults({},v,models.entry('h3-ref-base-int8'),'https://example.com')
        self.assertEqual(d['profiles']['standard']['steps'],30)
        self.assertEqual(d['profiles']['turbo']['steps'],8)
        self.assertEqual(d['profiles']['turbo']['video_shift'],12)
        self.assertEqual(d['profiles']['turbo']['lora_id'],'h3-ref-turbo-lora')

    def test_examples_of_a_mixed_partition_version_are_not_file_defaults(self):
        v=self.version('FL2VA / Ref2VA')
        v.update(files=[{'name':'h3_fl2va.gguf'},{'name':'h3_ref2va.gguf'}],images=[{'meta':{'steps':4}}])
        self.assertNotIn('steps',version_settings({},v,{'name':'h3_ref2va.gguf'}))

    def test_attached_workflow_overrides_page_and_keeps_distinct_schedule(self):
        v=self.version('Ref2VA Turbo')
        v['_workflow_settings']=[dict(reference=True,checkpoint='ref2va_turbo.gguf',steps=4,sampler='euler',scheduler='simple',video_shift=12,audio_shift=3,scheduler_video_shift=6,scheduler_audio_shift=3,source='https://example.com/config.json')]
        d=h3_defaults({'description':'FL2VA settings: 8 steps. Video shift 6.'},v,models.entry('h3-ref-base-int8'),'https://example.com')
        self.assertEqual((d['steps'],d['video_shift'],d['scheduler_video_shift']),(4,12,6))
        self.assertEqual(d['field_sources']['steps'],'attached_workflow')
        self.assertEqual(d['profiles']['turbo']['scheduler_video_shift'],6)
        from backend.video_sampling import scheduler_sigmas
        from unittest.mock import Mock
        model=Mock();shift=Mock(return_value=[model]);calculate=Mock(return_value=[1,.5,0])
        self.assertEqual(scheduler_sigmas(d,model,shift,calculate),[1,.5,0])
        shift.assert_called_once_with(model,6,3)
        calculate.assert_called_once_with(model.get_model_object.return_value,'simple',4)
        shift.reset_mock();calculate.reset_mock()
        self.assertIsNone(scheduler_sigmas({**d,'scheduler_video_shift':12},model,shift,calculate))
        shift.assert_not_called();calculate.assert_not_called()

    def test_first_last_frame_models_keep_attached_schedule_settings(self):
        item=dict(family='minimax-h3',generation_defaults=dict(steps=4,sampler='euler',scheduler='simple',
                  video_shift=12,audio_shift=3,scheduler_video_shift=6,scheduler_audio_shift=3,
                  embedded_turbo=True,workflow_source='https://example.com/workflow.json'))
        profile=video_sampling(item)
        self.assertEqual((profile['steps'],profile['video_shift'],profile['scheduler_video_shift']),(4,12,6))
        self.assertEqual(profile['scheduler_audio_shift'],3)
        self.assertEqual(video_sampling(item,steps=7)['scheduler_video_shift'],6)

    def test_reference_image_queue_keeps_each_model_contract_and_override(self):
        from copy import deepcopy
        original=models.entry
        with tempfile.TemporaryDirectory() as folder,patch.object(server,'required_ltx'),patch.object(server,'resolve_image'),patch.object(server,'enqueue_locked'),patch.object(server.cloud,'settings',return_value={'mode':'local'}):
            for count,sampler,shift in [(4,'euler',12),(6,'er_sde',1),(8,'res_multistep',10),(20,'res_multistep',12),(25,'euler',12)]:
                item=deepcopy(original('h3-ref-turbo-int8'))
                item['id']='future-reference-model'
                profile=item['generation_defaults']['profiles']['turbo']
                profile.update(steps=count,sampler=sampler,video_shift=shift,scheduler_video_shift=6)
                with patch.object(models,'entry',side_effect=lambda key:item if key==item['id'] else original(key)):
                    request=MultiReferenceRequest(model_id=item['id'],prompt='A paper boat on water',reference_images=['neutral.png'],frames=124,output_dir=folder)
                    job=server.generate_multi_reference(request)
                    self.assertEqual(job['total_steps'],count)
                    self.assertEqual(job['request']['sampling_profile']['sampler'],sampler)
                    self.assertEqual(job['request']['sampling_profile']['scheduler_video_shift'],6)
                    self.assertEqual(job['request']['steps_mode'],'model')
                    custom=server.generate_multi_reference(request.model_copy(update={'steps':11}))
                    self.assertEqual(custom['total_steps'],11)
                    self.assertEqual(custom['request']['sampling_profile']['default_steps'],count)
                    self.assertEqual(custom['request']['steps_mode'],'custom')

    def test_version_turbo_does_not_borrow_base_or_sibling_settings(self):
        standard=self.version('Hybrid v4');turbo=self.version('Hybrid v4 Turbo','4step Turbo')
        model=dict(description=PAGE,modelVersions=[standard,turbo])
        settings=h3_defaults(model,turbo,models.entry('h3-int8'),'https://example.com')
        self.assertEqual([settings[k] for k in ('steps','sampler','scheduler','video_shift','audio_shift')],[4,'euler','simple',6,3])
        self.assertTrue(settings['embedded_turbo'])
        self.assertEqual(version_settings(model,standard)['steps'],20)
        self.assertFalse(video_variant(model,standard))
        self.assertIsNone(video_variant({},self.version('Version 1','CFG 1')))
        second=self.version('Hybrid Turbo v2','Turbo (4 till 8-step)')
        self.assertEqual(version_settings(model,second)['steps_range'],[4,8])

    def test_beta_names_and_previous_releases_never_select_a_scheduler(self):
        version=self.version('beta6','Merged 8+4 step turbos; target 8-12 steps, 10/3 shift. 4 step possible with added turbo.')
        model=dict(description='beta 6:\nUses FastH3 settings with 8-12 steps. It is different from beta 5 and normal hybrids.\nbeta 5:\ner_sde/beta 4-6 steps')
        d=h3_defaults(model,version,models.entry('h3-int8'),'https://example.com')
        self.assertEqual([d[k] for k in ('steps','sampler','scheduler','video_shift','audio_shift')],[8,'res_multistep','simple',10,3])
        self.assertEqual(d['steps_range'],[8,12])

    def test_file_turbo_and_standard_are_separate(self):
        v=self.version('beta5');m=dict(description=PAGE)
        self.assertTrue(video_variant(m,v,{'name':'weights_TURBO_int8.safetensors'}))
        self.assertFalse(video_variant(m,v,{'name':'weights_non-distilled_int8.safetensors'}))

    def test_partition_is_per_file_and_reference_does_not_imply_keyframes(self):
        v=self.version('FL2VA and Ref2VA')
        m=dict(name='MiniMax H3',description='FL2VA first/last frames. Ref2VA reference images.')
        self.assertEqual(h3_capabilities(m,v,dict(name='H3_ref2v_Q4.gguf')),(True,[]))
        self.assertEqual(h3_capabilities(m,v,dict(name='H3_fl2va_Q4.gguf')),(False,['text','image']))
        self.assertEqual(h3_capabilities(dict(description='Reference model with T2V ability'),self.version('Eros beta6')),(True,['text']))
        self.assertEqual(h3_capabilities(dict(name='SparseRef15',description='FL2VA-based hybrid; not a native Ref2VA model.'),v),(False,['text','image']))

    def test_unicode_ranges_title_steps_and_fast_workflow(self):
        v=self.version('beta6','Target 8\u201112 steps. Sampling uses FastH3 templates. 10/3 shift.')
        d=h3_defaults({},v,models.entry('h3-int8'),'https://example.com')
        self.assertEqual((d['steps'],d['steps_range'],d['sampler']),(8,[8,12],'res_multistep'))
        m=dict(name='MiniMax Turbo Merge 8-Step')
        v=self.version('comfy_fl2va')
        self.assertTrue(video_variant(m,v))
        self.assertEqual(version_settings(m,v)['steps'],8)
        self.assertEqual(version_settings(m,v,dict(name='fl2va_turbo_4step.safetensors'))['steps'],4)

    def test_queued_request_uses_defaults_and_never_adds_turbo_twice(self):
        item={**models.entry('h3-int8'),'id':'test-h3','generation_defaults':dict(steps=4,cfg=1,sampler='euler',scheduler='simple',video_shift=6,audio_shift=3,embedded_turbo=True)}
        real=models.entry
        with tempfile.TemporaryDirectory() as folder, patch.object(models,'entry',side_effect=lambda key:item if key=='test-h3' else real(key)),patch.object(server,'enqueue_locked'),patch.object(server.cloud,'settings',return_value={'mode':'local'}):
            request=VideoRequest(model_id='test-h3',prompt='A landscape',output_dir=folder)
            self.assertNotIn('h3-turbo',models.required_video(request.model_copy(update={'quality':'turbo'})))
            result=server.generate_video(request)
            self.assertEqual(result['total_steps'],4)
            self.assertEqual(result['request']['sampling']['video_shift'],6)
            self.assertEqual(result['request']['steps_mode'],'model')
            self.assertEqual(server.generate_video(request.model_copy(update={'steps':7}))['total_steps'],7)
        ltx=video_sampling(models.entry('ltx25-msr-q4_k_m'),steps=20,last_frame=True)
        self.assertEqual((ltx['steps'],ltx['total_steps'],ltx['sampler'],ltx['eta']),(8,8,'euler_ancestral',0))

    def test_unet_files_are_usable(self):
        self.assertTrue(civitai._usable_file('Checkpoint',dict(type='UNet',name='weights.safetensors',format='SafeTensor')))


class ProgressTimingTests(unittest.TestCase):
    def test_loading_time_is_excluded_and_new_pass_resets_timing(self):
        from backend import worker
        job=dict(stage='loading_model',started_at=0,step=0)
        with patch.object(worker,'job',job),patch.object(worker,'directory',Path('unused')),patch.object(worker,'write_json'),patch('builtins.print'),patch.object(worker.time,'time',side_effect=[10,80,87,94,100,130,137]):
            worker.update(stage='generating',step=0)
            worker.update(stage='generating',step=1)
            self.assertIsNone(job['sampling_step_seconds'])
            worker.update(stage='generating',step=2)
            worker.update(stage='generating',step=3)
            self.assertEqual(job['sampling_step_seconds'],7)
            worker.update(stage='generating',step=0)
            worker.update(stage='generating',step=1)
            self.assertIsNone(job['sampling_step_seconds'])
            worker.update(stage='generating',step=2)
            self.assertEqual(job['sampling_step_seconds'],7)

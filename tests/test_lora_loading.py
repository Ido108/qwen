"""Real ComfyUI patch math on tiny CPU tensors; no model download or inference."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def setUpModule():
    global guards,torch,comfy
    import torch
    guards=contextlib.ExitStack()
    guards.enter_context(patch('torch.cuda.is_available',return_value=False))
    guards.enter_context(patch('torch.cuda._lazy_init',side_effect=AssertionError('GPU use forbidden')))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'/'ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args
    args.cpu=True
    import comfy.model_base,comfy.model_patcher,comfy.lora


def tearDownModule():
    guards.close()


class LoraLoadingTests(unittest.TestCase):
    def model(self,fused):
        class TinyQwen(comfy.model_base.QwenImage):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.model_config=SimpleNamespace(unet_config={})
                mlp=torch.nn.Module()
                for name in (['gate_up'] if fused else ['gate_layer','proj']):
                    setattr(mlp,name,torch.nn.Linear(3,4 if fused else 2,bias=False))
                block=torch.nn.Module();block.img_mlp=mlp
                self.diffusion_model=torch.nn.Module()
                self.diffusion_model.transformer_blocks=torch.nn.ModuleList([block])
                for parameter in self.parameters():parameter.data.zero_()
        model=TinyQwen()
        return comfy.model_patcher.ModelPatcher(model,torch.device('cpu'),torch.device('cpu'))

    def weights(self,prefix='diffusion_model.transformer_blocks.0.img_mlp.gate_up'):
        return {prefix+'.lora_A.weight':torch.tensor([[1.,2.,3.]]),
                prefix+'.lora_B.weight':torch.tensor([[1.],[2.],[3.],[4.]])}

    def test_native_fused_and_split_formats_apply_every_half_at_real_strength(self):
        from backend.comfy_engine import split_fused_mlp
        from backend.lora_loading import load_model_lora
        from backend.image_cache import retain_cpu_weights,retain_lora_clone
        expected=torch.tensor([[1.,2.,3.],[2.,4.,6.],[3.,6.,9.],[4.,8.,12.]])
        for fused in [False,True]:
            base=self.model(fused)
            retain_cpu_weights(base)
            # Changes and repeated runs must not accumulate weight patches.
            for strength in [1.,0.,.5,1.5,0.,1.]:
                clone,check=load_model_lora(base,split_fused_mlp(self.weights()),strength,'Ink')
                retain_lora_clone(clone,base)
                self.assertEqual(check['patches'],2)
                self.assertEqual(base.patches,{})
                for key in clone.patches:clone.patch_weight_to_device(key,device_to=torch.device('cpu'))
                mlp=clone.model.diffusion_model.transformer_blocks[0].img_mlp
                actual=mlp.gate_up.weight if fused else torch.cat([mlp.gate_layer.weight,mlp.proj.weight])
                self.assertTrue(torch.allclose(actual,expected*strength), (fused,strength,actual))
                x=torch.tensor([[1.,1.,1.]])
                self.assertTrue(torch.allclose(torch.nn.functional.linear(x,actual),torch.nn.functional.linear(x,expected)*strength))
                clone.unpatch_model(device_to=torch.device('cpu'))
                self.assertTrue(all(torch.count_nonzero(p)==0 for p in base.model.parameters()))

    def test_comfy_drops_native_fused_halves_but_checked_loader_restores_them(self):
        from backend.comfy_engine import split_fused_mlp
        from backend.lora_loading import load_model_lora
        base=self.model(True);weights=split_fused_mlp(self.weights())
        mapping=comfy.lora.model_lora_keys_unet(base.model,{})
        self.assertEqual(comfy.lora.load_lora(weights,mapping,log_missing=False),{})
        self.assertEqual(load_model_lora(base,weights,1,'Ink')[1]['patches'],2)

    def test_kohya_aliases_and_multiple_loras_keep_separate_strengths(self):
        from backend.comfy_engine import split_fused_mlp
        from backend.lora_loading import load_model_lora
        base=self.model(True)
        first,check=load_model_lora(base,split_fused_mlp(self.weights('lora_unet_transformer_blocks_0_img_mlp_gate_up')),.25,'Ink')
        second,check=load_model_lora(first,split_fused_mlp(self.weights()),.5,'Paper')
        self.assertEqual(len(base.patches),0)
        # Each LoRA contributes two slices to the one fused weight.
        self.assertTrue(all(len(patches)==2 for patches in first.patches.values()))
        self.assertTrue(all(len(patches)==4 for patches in second.patches.values()))
        for key in second.patches:second.patch_weight_to_device(key,device_to=torch.device('cpu'))
        expected=torch.tensor([[1.,2.,3.],[2.,4.,6.],[3.,6.,9.],[4.,8.,12.]])*.75
        self.assertTrue(torch.allclose(second.model.diffusion_model.transformer_blocks[0].img_mlp.gate_up.weight,expected))
        second.unpatch_model(device_to=torch.device('cpu'))

    def test_no_match_partial_match_and_wrong_shapes_stop_before_sampling(self):
        from backend.comfy_engine import split_fused_mlp
        from backend.lora_loading import load_model_lora
        base=self.model(True)
        with self.assertRaisesRegex(ValueError,'did not attach any'):
            load_model_lora(base,self.weights('unknown'),1,'Ink')
        valid=split_fused_mlp(self.weights())
        with self.assertRaisesRegex(ValueError,'could not attach'):
            load_model_lora(base,{**valid,**self.weights('unknown')},1,'Ink')
        valid['diffusion_model.transformer_blocks.0.img_mlp.gate_layer.lora_A.weight']=torch.ones(1,5)
        with self.assertRaisesRegex(ValueError,'weight size'):
            load_model_lora(base,valid,1,'Ink')

    def test_gguf_uses_logical_shapes_and_applies_quantized_patches(self):
        import importlib.util
        import numpy as np
        import gguf
        from backend.comfy_engine import split_fused_mlp
        from backend.lora_loading import load_model_lora
        from backend.image_cache import retain_cpu_weights,retain_lora_clone
        directory=Path(__file__).resolve().parents[1]/'vendor'/'ComfyUI-GGUF'
        if 'studio_gguf' not in sys.modules:
            spec=importlib.util.spec_from_file_location('studio_gguf',directory/'__init__.py',submodule_search_locations=[str(directory)])
            package=importlib.util.module_from_spec(spec);sys.modules['studio_gguf']=package;spec.loader.exec_module(package)
        from studio_gguf.ops import GGMLOps,GGMLTensor
        from studio_gguf.nodes import GGUFModelPatcher
        for fused in [False,True]:
            base=self.model(fused)
            mlp=base.model.diffusion_model.transformer_blocks[0].img_mlp
            rows=4 if fused else 2
            for name in (['gate_up'] if fused else ['gate_layer','proj']):
                layer=GGMLOps.Linear(32,rows,bias=False)
                raw=gguf.quants.quantize(np.zeros((rows,32),dtype=np.float32),gguf.GGMLQuantizationType.Q8_0)
                layer.weight=torch.nn.Parameter(GGMLTensor(torch.from_numpy(raw),tensor_type=gguf.GGMLQuantizationType.Q8_0,tensor_shape=(rows,32)),requires_grad=False)
                setattr(mlp,name,layer)
                self.assertEqual(tuple(layer.state_dict()['weight'].shape),(rows,34))
            base=GGUFModelPatcher.clone(base);retain_cpu_weights(base)
            weights=self.weights();weights['diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight']=torch.ones(1,32)
            expected=torch.arange(1.,5.).unsqueeze(1).expand(4,32)
            for strength in [1.,0.,.5,1.]:
                clone,check=load_model_lora(base,split_fused_mlp(weights),strength,'Ink')
                retain_lora_clone(clone,base)
                self.assertEqual(check['patches'],2)
                for key in clone.patches:clone.patch_weight_to_device(key,device_to=torch.device('cpu'))
                layers=[mlp.gate_up] if fused else [mlp.gate_layer,mlp.proj]
                actual=torch.cat([layer.get_weight(layer.weight,torch.float32) for layer in layers])
                self.assertTrue(torch.allclose(actual,expected*strength),(fused,strength))
                clone.unpatch_model(device_to=torch.device('cpu'))
                self.assertTrue(all(not layer.weight.patches for layer in layers))
            from comfy.patcher_extension import WrapperExecutor,WrappersMP
            from backend.lora_runtime import KEY
            runtime,check=load_model_lora(base,split_fused_mlp(weights),.5,'Ink',runtime=True)
            self.assertEqual(check['application'],'runtime')
            def apply(x):return torch.cat([layer(x) for layer in layers],dim=-1)
            runner=WrapperExecutor(apply,base.model.diffusion_model,runtime.get_wrappers(WrappersMP.DIFFUSION_MODEL,KEY),0)
            actual=runner.execute(torch.ones(1,32))
            self.assertTrue(torch.allclose(actual,torch.tensor([[16.,32.,48.,64.]])))
            self.assertTrue(all(not layer.weight.patches and not layer._forward_hooks for layer in layers))

    def test_qwen_prefix_cache_resets_at_each_sampling_run(self):
        from unittest.mock import Mock
        holder=SimpleNamespace(diffusion_model=SimpleNamespace(reset_prefix_cache=Mock()))
        setter=comfy.model_base.QwenImage21.current_patcher.fset
        first=SimpleNamespace(hook_patches={})
        second=SimpleNamespace(hook_patches={})
        for value in [first,None,second,None]:setter(holder,value)
        self.assertEqual([call.args[0] for call in holder.diffusion_model.reset_prefix_cache.call_args_list],[True,False,True,False])

    def test_runtime_stack_matches_full_precision_math_including_fused_output(self):
        from backend.comfy_engine import split_fused_mlp
        from backend.lora_loading import load_model_lora
        from backend.lora_runtime import KEY
        from comfy.ldm.qwen_image21.model import SwiGLUFeedForward
        from comfy.patcher_extension import WrapperExecutor, WrappersMP
        import comfy.ops
        for fused in [False,True]:
            base=self.model(fused)
            block=base.model.diffusion_model.transformer_blocks[0]
            block.img_mlp=SwiGLUFeedForward(3,2,fused=fused,operations=comfy.ops.manual_cast,dtype=torch.float32)
            for p in block.parameters():p.data.fill_(.2)
            original={k:v.clone() for k,v in base.model.state_dict().items()}
            weights=split_fused_mlp(self.weights())
            weights.update({'diffusion_model.transformer_blocks.0.img_mlp.out.lora_A.weight':torch.tensor([[.1,.2]]),
                            'diffusion_model.transformer_blocks.0.img_mlp.out.lora_B.weight':torch.tensor([[.1],[.2],[.3]])})
            first,check=load_model_lora(base,weights,.25,'Ink',runtime=True)
            second,_=load_model_lora(first,weights,.5,'Paper',runtime=True)
            self.assertEqual(check['application'],'runtime')
            self.assertFalse(base.patches)
            self.assertFalse(second.patches)
            self.assertEqual(len(second.get_additional_models()),1)
            bank=second.get_attachment(KEY)
            self.assertGreater(second.get_additional_models()[0].model_size(),0)
            x=torch.ones(1,3)
            wrappers=second.get_wrappers(WrappersMP.DIFFUSION_MODEL,KEY)
            run=WrapperExecutor(block.img_mlp,base.model.diffusion_model,wrappers,0)
            actual=run.execute(x)
            from backend.lora_runtime import execution_report
            reports=execution_report(second)
            self.assertEqual({r['title'] for r in reports},{'Ink','Paper'})
            self.assertTrue(all(r['expected']==r['executed'] and r['model_calls']==1 for r in reports))
            merged,_=load_model_lora(base,weights,.75,'Expected')
            for key in merged.patches:merged.patch_weight_to_device(key,device_to=torch.device('cpu'))
            expected=block.img_mlp(x)
            self.assertTrue(torch.allclose(actual,expected,rtol=1e-5,atol=1e-5),(fused,actual,expected))
            merged.unpatch_model(device_to=torch.device('cpu'))
            self.assertTrue(all(torch.equal(v,original[k]) for k,v in base.model.state_dict().items()))
            self.assertTrue(all(not m._forward_hooks for m in block.modules()))
            def fail(x):
                block.img_mlp(x)
                raise RuntimeError('test interruption')
            with self.assertRaisesRegex(RuntimeError,'test interruption'):
                WrapperExecutor(fail,base.model.diffusion_model,wrappers,0).execute(x)
            self.assertTrue(all(not m._forward_hooks for m in block.modules()))
            # The first clone keeps its own stack when another LoRA is added.
            self.assertEqual(len(bank.records),len(first.get_attachment(KEY).records)*2)

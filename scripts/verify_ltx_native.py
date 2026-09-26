"""Exercise real MSR/native Comfy tensor plumbing with tiny CPU fixtures only.

No checkpoints, encoders, diffusion models, CUDA, or model sampling are loaded.
"""
import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/ComfyUI')]
import comfy.options
comfy.options.enable_args_parsing(False)
from comfy.cli_args import args
args.cpu = True
import torch
from backend.ltx_engine import load_extension
load_extension('qwen_ltx_msr', ROOT/'vendor/LTX25-MSR')
from qwen_ltx_msr.nodes import ComfyUILTX25MSRMultiReferenceGuide as Guide
from comfy_extras.nodes_lt import LTXVCropGuides, LTXVConcatAVLatent, LTXVSeparateAVLatent
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode, LTXVAudioVAEDecode
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler


class FakeVAE:
    downscale_index_formula = (8,32,32)
    def encode(self, pixels):
        return torch.zeros(1,128,(len(pixels)-1)//8+1,pixels.shape[1]//32,pixels.shape[2]//32)
    def encode_tiled(self,pixels,**kwargs):
        return self.encode(pixels)


slots={'frequencies':torch.ones(16),'net.0.weight':torch.zeros(256,33),'net.0.bias':torch.zeros(256),
       'net.2.weight':torch.zeros(128,256),'net.2.bias':torch.ones(128)*.1}
parameters={'slot_state':slots,'metadata':{},'reference_downscale_factor':1,'reference_temporal_scale_factor':1}
logging.getLogger().setLevel(logging.ERROR)
latent={'samples':torch.ones(1,128,5,2,2)}
conditioning=[[torch.zeros(1,1,1),{}]]
image=torch.zeros(1,64,64,3)
for reference_frames in (25,33):
    pos,neg,guided=Guide.execute(conditioning,conditioning,FakeVAE(),latent,pic1=image,pic2=image,pic3=image,pic4=image,background=image,
        strength=1,reference_frames=str(reference_frames),use_tiled_encode=True,tile_size=256,tile_overlap=64,msr_parameters=parameters)
    count=5*((reference_frames-1)//8+1)
    assert guided['samples'].shape==(1,128,5+count,2,2)
    assert torch.equal(guided['samples'][:,:,:5],latent['samples'])
    assert torch.count_nonzero(guided['noise_mask'][:,:,5:])==0
    joint=LTXVConcatAVLatent.execute(guided,{'samples':torch.zeros(1,8,20,16)})[0]
    video,audio=LTXVSeparateAVLatent.execute(joint)
    clean_pos,clean_neg,clean=LTXVCropGuides.execute(pos,neg,video)
    assert torch.equal(clean['samples'],latent['samples'])
    assert audio['samples'].shape==(1,8,20,16)
    assert clean_pos[0][1]['keyframe_idxs'] is None
    assert len(pos[0][1]['guide_attention_entries'])==5
print('PASS: 5 MSR image slots, learned embeddings, 25/33-frame references, AV masks, crop and native imports on CPU. No inference.')

# Voice 2 can be supplied without voice 1; its absolute slot must remain 2.
parameters.update(audio_slot_state=slots, lora_name='fixture', reference_audio_max_tokens=125,
    reference_audio_slot_duration_seconds=5.0, reference_audio_end_margin_seconds=.04)
pos,neg,guided=Guide.execute(conditioning,conditioning,FakeVAE(),latent,pic1=image,pic2=image,
    strength=1,reference_frames='33',use_tiled_encode=True,tile_size=256,tile_overlap=64,
    msr_parameters=parameters,audio_ref2={'samples':torch.zeros(1,8,10,16)})
assert pos[0][1]['ref_audio']['slot_ids']==(2,)
assert pos[0][1]['ref_audio']['slot_lengths']==(10,)
assert pos[0][1]['ref_audio']['tokens'].shape==(1,10,128)
assert pos[0][1]['ref_audio']['image_slot_count']==2
print('PASS: sparse voice pairing and 128-channel audio slot embeddings on CPU.')

"""Refresh explicitly pinned video file metadata; never download model weights."""
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
PINNED = {
    'Comfy-Org/MiniMax-H3': '7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1',
    'molbal/MiniMax-H3-GGUF': 'b45874371c61c49bf04096602f5527aac71b360b',
    'vantagewithai/FastVideo-FastH3-8-Step-V2-ComfyUI-GGUF': '448ffec4210efe431a6b4a945672d88aea5f25f4',
    'FastVideo/FastVideo-FastH3-Comfy': 'ec1e3aa374a91c57b0b94a1623b7e657c0498cf2',
}

def main():
    path = ROOT / 'reference/model-catalog.json'
    catalog = [m for m in json.loads(path.read_text('utf-8')) if not m['id'].startswith(('h3-', 'fasth3-'))]
    metadata = {}
    def add(model_id, title, repo, filename, kind, format, **extra):
        revision = PINNED[repo]
        if repo not in metadata:
            with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
                metadata[repo] = json.load(response)
        source = next(f for f in metadata[repo]['siblings'] if f['rfilename'] == filename)
        file = {'path': filename, 'bytes': source['size'], 'sha256': source.get('lfs', {}).get('sha256')}
        catalog.append(dict(id=model_id, title=title, repo=repo, revision=revision,
            directory='video/' + repo.split('/')[0] + '-' + repo.split('/')[1],
            files=[file], bytes=file['bytes'], kind=kind, format=format, dependencies=[], **extra))
    comfy = 'Comfy-Org/MiniMax-H3'
    add('h3-encoder', 'MiniMax H3 text encoder · NVFP4', comfy, 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'video-support', 'NVFP4')
    add('h3-video-vae', 'MiniMax H3 video VAE', comfy, 'vae/minimax_h3_video_vae_fp16.safetensors', 'video-support', 'FP16')
    add('h3-audio-vae', 'MiniMax H3 audio VAE', comfy, 'vae/minimax_h3_audio_vae_fp32.safetensors', 'video-support', 'FP32')
    add('h3-turbo', 'MiniMax H3 Turbo · 8 steps', comfy, 'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', 'video-support', 'LoRA')
    add('h3-gguf-q4', 'MiniMax H3 · Q4_0', 'molbal/MiniMax-H3-GGUF', 'minimax_h3_fl2va_pruned_fp8_Q4_0.gguf', 'video', 'GGUF', family='minimax-h3', modes=['text', 'image'])
    add('h3-gguf-q8', 'MiniMax H3 · Q8_0', 'molbal/MiniMax-H3-GGUF', 'minimax_h3_fl2va_pruned_fp8_Q8_0.gguf', 'video', 'GGUF', family='minimax-h3', modes=['text', 'image'])
    add('h3-int8', 'MiniMax H3 · INT8', comfy, 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors', 'video', 'INT8', family='minimax-h3', modes=['text', 'image'])
    for quant in ['Q4_K_M', 'Q5_K_M', 'Q6_K']:
        add('fasth3-' + quant.lower(), 'FastH3 8-Step V2 · ' + quant,
            'vantagewithai/FastVideo-FastH3-8-Step-V2-ComfyUI-GGUF',
            'fastvideo_fasth3_8step_v2_pruned-' + quant + '.gguf', 'video', 'GGUF',
            family='fasth3', modes=['text'], note='8-step text-to-video with audio. Use MiniMax H3 for image inputs.')
    add('fasth3-int8', 'FastH3 8-Step V2 · INT8', 'FastVideo/FastVideo-FastH3-Comfy',
        'diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors', 'video', 'INT8',
        family='fasth3', modes=['text'])
    for model in catalog:
        if model['kind'] == 'video':
            model['dependencies'] = ['h3-encoder', 'h3-video-vae', 'h3-audio-vae']
            fast=model['family']=='fasth3'
            model['generation_defaults']={'steps':8 if fast else 20,
                'source':'https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2' if fast else 'https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native',
                'note':'Trained eight-forward schedule.' if fast else 'Official ComfyUI FL2VA workflow.'}
            if not fast:
                model['generation_defaults']['turbo_steps']=8
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', 'utf-8')

if __name__ == '__main__':
    main()

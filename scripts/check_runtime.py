"""Verify the installed runtime without loading model weights."""
import fastapi
import httpx
import PIL
import torch
import av
import comfy_kitchen

if not torch.cuda.is_available():
    raise SystemExit('NVIDIA CUDA is unavailable. Update the graphics driver and restart Windows.')
print('Ready:', torch.cuda.get_device_name(0), '- CUDA', torch.version.cuda)
av.Codec('libx264', 'w')
av.Codec('aac', 'w')
if not comfy_kitchen.sol_attn_is_available(torch.device('cuda')):
    raise SystemExit('The FastH3 CUDA sparse attention kernel is unavailable. Check the NVIDIA driver and run Start Qwen.bat again.')
print('Ready: H.264 video, AAC audio and FastH3 sparse attention')

"""Verify the installed runtime without loading model weights."""
import fastapi
import httpx
import PIL
import torch

if not torch.cuda.is_available():
    raise SystemExit('NVIDIA CUDA is unavailable. Update the graphics driver and restart Windows.')
print('Ready:', torch.cuda.get_device_name(0), '- CUDA', torch.version.cuda)

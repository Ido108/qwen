"""Account for dedicated VRAM already used by other Windows applications."""
import functools
import os
import subprocess

GIB = 1024 ** 3
# Sizes measured on the 32 GB RTX 5090 apply from this much VRAM; a 24 GB RTX 4090 keeps the smaller ones.
LARGE_VRAM_GB = 30


def dedicated_memory_snapshot():
    """Report physical VRAM, rather than CUDA's virtual WDDM allocation limit."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        name, total, free = (part.strip() for part in result.stdout.splitlines()[0].rsplit(',', 2))
        return {'available': True, 'name': name, 'total_gb': round(int(total)/1024, 2),
                'free_gb': round(int(free)/1024, 2)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {'available': False}


@functools.lru_cache(maxsize=1)
def gpu_total_gb():
    """Physical VRAM of this computer's GPU, or 0 without an NVIDIA GPU."""
    snapshot = dedicated_memory_snapshot()
    return snapshot['total_gb'] if snapshot['available'] else 0


def large_vram():
    return gpu_total_gb() >= LARGE_VRAM_GB


@functools.lru_cache(maxsize=1)
def native_fp4():
    """RTX 50 (compute capability 12) runs NVFP4 and MXFP8 natively; older GPUs dequantize them. True when unknown."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True, timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return float(result.stdout.splitlines()[0]) >= 12
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return True


def reserve_for_dedicated_memory(cuda_free, dedicated_free, safety=2 * GIB):
    # WDDM CUDA mem_get_info can expose a virtual budget rather than free
    # dedicated VRAM. Reserve that difference before ComfyUI places weights.
    return safety + max(0, cuda_free - dedicated_free)


def nvidia_free_bytes(torch):
    """Free dedicated VRAM of this process's GPU as nvidia-smi reports it, or None."""
    gpu_uuid = str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'uuid', ''))
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        for line in result.stdout.splitlines():
            uuid, free_mib = (part.strip() for part in line.split(','))
            if uuid.removeprefix('GPU-') == gpu_uuid.removeprefix('GPU-'):
                return int(free_mib) * 1024 ** 2
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def dedicated_free_bytes(torch):
    """Physical VRAM this process can still use: nvidia-smi's free memory plus PyTorch's cached, unused blocks.
    On Windows the CUDA figure can exceed physical VRAM, and the driver then spills to system RAM in silence."""
    free = nvidia_free_bytes(torch)
    return None if free is None else free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


def configure_attention(args):
    """Comfy Kitchen INT8 attention (the kernel FLUX.3 and MiniMax H3 checkpoints select themselves) for every model,
    when this GPU supports it. Set QWEN_ATTENTION=pytorch to keep PyTorch attention. Must run before comfy.ldm imports."""
    if os.environ.get('QWEN_ATTENTION', '').lower() == 'pytorch':
        return 'pytorch'
    try:
        import torch
        import comfy_kitchen
        if comfy_kitchen.int8_attention_is_available(torch.device('cuda')):
            args.use_ck_attention = True
            return 'comfy_kitchen_int8'
    except Exception:
        pass
    return 'pytorch'


def configure_video_memory(torch, args, progress):
    cuda_free, total = torch.cuda.mem_get_info()
    dedicated_free = cuda_free
    source = 'CUDA'
    if os.name == 'nt':
        free = nvidia_free_bytes(torch)
        if free is not None:
            dedicated_free = min(cuda_free, free)
            source = 'NVIDIA dedicated VRAM'
    args.reserve_vram = reserve_for_dedicated_memory(cuda_free, dedicated_free) / GIB
    progress(gpu_total_gb=round(total/GIB, 2), cuda_free_gb=round(cuda_free/GIB, 2),
             dedicated_free_gb=round(dedicated_free/GIB, 2),
             reserved_vram_gb=round(args.reserve_vram, 2), memory_budget_source=source)

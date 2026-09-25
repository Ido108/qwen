"""Account for dedicated VRAM already used by other Windows applications."""
import os
import subprocess

GIB = 1024 ** 3


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


def reserve_for_dedicated_memory(cuda_free, dedicated_free, safety=2 * GIB):
    # WDDM CUDA mem_get_info can expose a virtual budget rather than free
    # dedicated VRAM. Reserve that difference before ComfyUI places weights.
    return safety + max(0, cuda_free - dedicated_free)


def configure_video_memory(torch, args, progress):
    cuda_free, total = torch.cuda.mem_get_info()
    dedicated_free = cuda_free
    source = 'CUDA'
    if os.name == 'nt':
        gpu_uuid = str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'uuid', ''))
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in result.stdout.splitlines():
                uuid, free_mib = (part.strip() for part in line.split(','))
                if uuid.removeprefix('GPU-') == gpu_uuid.removeprefix('GPU-'):
                    dedicated_free = min(cuda_free, int(free_mib) * 1024 ** 2)
                    source = 'NVIDIA dedicated VRAM'
                    break
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    args.reserve_vram = reserve_for_dedicated_memory(cuda_free, dedicated_free) / GIB
    progress(gpu_total_gb=round(total/GIB, 2), cuda_free_gb=round(cuda_free/GIB, 2),
             dedicated_free_gb=round(dedicated_free/GIB, 2),
             reserved_vram_gb=round(args.reserve_vram, 2), memory_budget_source=source)

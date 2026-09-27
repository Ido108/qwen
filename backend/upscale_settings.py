"""Shared output geometry and memory policy; no model imports or allocations."""
from __future__ import annotations
import math


def output_dimensions(width, height, request):
    if width <= 0 or height <= 0:
        raise ValueError('The source video has no valid dimensions.')
    factor = request.get('scale', 2) if request.get('size_mode') == 'scale' else request['resolution'] / min(width, height)
    # H.264 yuv420 needs even dimensions, not the model's 32/128-pixel padding.
    # Round only at the output edge; internal model padding is removed.
    result = tuple(max(2, int(math.floor(side * factor / 2 + .5)) * 2) for side in (width, height))
    if min(result) > 4320 or max(result) > 8192:
        raise ValueError('Choose a smaller output size (up to 4320px on the short side and 8192px on the long side).')
    return result


def seed_memory_options(selected, request, free_gb=None):
    size = selected.get('upscale_defaults', {}).get('parameters_billion', 7 if '7b' in selected['id'] else 3)
    mode = request.get('memory_mode', 'auto')
    if mode == 'gpu':
        return []
    # GGUF stays quantized; FP8 also keeps a working/dequantization buffer.
    weights_gb = selected['bytes'] / 1024**3
    short_side = request['resolution']
    reserve = 6 + 4 * (short_side / 1080)**2 * request.get('batch_size', 5) / 5
    pressure = free_gb is not None and weights_gb + reserve > free_gb
    swap = mode == 'low_memory' or pressure or short_side >= 2160
    if not swap:
        return []
    blocks = 36 if size == 7 else 32
    fraction = .65 if mode == 'low_memory' else max(.25, min(.8, (weights_gb + reserve - (free_gb or 20)) / max(weights_gb, 1)))
    return ['--dit_offload_device', 'cpu', '--blocks_to_swap', str(min(blocks-1, max(8, math.ceil(blocks * fraction)))),
            '--vae_offload_device', 'cpu']

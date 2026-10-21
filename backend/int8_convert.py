"""Qwen-Image-2.1 and MOSS Image BF16, FP16 or FP32 checkpoints as INT8 ConvRot, in the layout of Comfy-Org's official INT8 files.
MOSS Image follows the same method; its layer rule is in ARCHS (Comfy-Org/MOSS-Image moss_image_*_int8_convrot).

The official layout (Comfy-Org/Qwen-Image-2.1 diffusion_models/qwen_image_2.1_int8_convrot.safetensors and
qwen_image_2.1_turbo_int8_convrot.safetensors, headers read on 2026-10-22):
- In every transformer block the six Linear weights attn.to_q, attn.to_k, attn.to_v, attn.to_out.0, img_mlp.gate_up and
  img_mlp.out are INT8 ([out, in], I8) with a per-row F32 weight_scale ([out, 1]) and a U8 comfy_quant tensor that holds
  the JSON {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256} (192 layers in 32 blocks).
- Every other tensor (img_in, txt_in, time_text_embed, modulation, norm_q/norm_k, norm_out, proj_out) stays BF16.
- File metadata {"format": "pt"}; tensor data in safetensors order (F32, then BF16, then I8, then U8; by name).
- The Turbo file also has transformer_blocks.N.attn.comfy_attention.config = {"attention": "comfy_kitchen_int8"} in blocks 1 to
  31 (not block 0): ComfyAttention (comfy/ldm/modules/attention.py) then runs those blocks' attention in INT8. The standard
  file has none, so the standard conversion has none either.

The method is comfy_kitchen's own (0.2.35, backends/eager/quantization.py quantize_int8_convrot_weight): the weight in FP32,
rotated per 256 input columns by the regular Hadamard matrix (tensor/int8_utils.py _build_hadamard, _rotate_weight), then a
row-wise absmax/127 scale and round to nearest. Run on CUDA it reproduces the official files bit for bit (all 192 layers of
the Turbo and the standard file). ComfyUI writes the comfy_quant JSON the same way (vendor/ComfyUI/comfy/ops.py
_quantized_weight_state_dict) and loads it with TensorWiseINT8Layout (comfy/quant_ops.py, comfy/ops.py).
Diffusers files keep img_mlp.gate_layer and img_mlp.proj apart; ComfyUI's fused gate_up is [gate_layer; proj]
(comfy/lora.py, comfy/ldm/qwen_image21/model.py SwiGLUFeedForward), so they are joined in that order before quantizing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

GROUPSIZE = 256
QUANT_CONF = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': GROUPSIZE}
METHOD = 'comfy_kitchen quantize_int8_convrot_weight (eager, FP32), groupsize 256'
VERSION = 1
PREFIXES = ('model.diffusion_model.', 'diffusion_model.')
ATTENTION_CONF = {'attention': 'comfy_kitchen_int8'}


class Arch:
    """What the official Comfy-Org INT8 ConvRot file of one architecture quantizes and keeps."""
    def __init__(self, name, keys, quantized, metadata, attention, attention_from, split=None, keep_f32=()):
        self.name, self.keys, self.quantized, self.metadata = name, keys, re.compile(quantized), metadata
        self.attention, self.attention_from, self.split, self.keep_f32 = attention, attention_from, split and re.compile(split), set(keep_f32)


ARCHS = (
    # Comfy-Org/Qwen-Image-2.1 qwen_image_2.1_int8_convrot (standard: no INT8 attention) and qwen_image_2.1_turbo_int8_convrot
    # (INT8 attention from block 1). Keys from ComfyUI comfy/model_detection.py qwen_image21_keys.
    Arch('qwen-image-2.1', ('txt_in.text_norm.weight', 'modulation.1.weight', 'transformer_blocks.0.attn.norm_q.weight', 'img_in.weight', 'proj_out.weight'),
         r'^transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v|to_out\.0)|img_mlp\.(?:gate_up|out))$', {'format': 'pt'},
         'transformer_blocks.{}.attn.comfy_attention.config', {'turbo': 1}, split=r'^(transformer_blocks\.\d+\.img_mlp)\.(gate_layer|proj)\.weight$'),
    # Comfy-Org/MOSS-Image moss_image_sft_int8_convrot and moss_image_base_int8_convrot: the four Linear layers of every block,
    # INT8 attention in every block, rope.bands kept F32, empty metadata (headers read on 2026-10-22).
    Arch('moss-image', ('shared_block_modulation.weight', 'video_embedder.weight', 'layers.0.proj.qkv.weight', 'final_linear_video.weight'),
         r'^layers\.\d+\.proj\.(?:qkv|o|ffn_up_gate|ffn_down)$', {}, 'layers.{}.comfy_attention.config', {'turbo': 0, 'standard': 0},
         keep_f32=('rope.bands',)),
)
FLOATS = {'BF16', 'F16', 'F32'}
# safetensors writes tensor data sorted by dtype (its Dtype enum, highest first), then name.
DTYPE_RANK = {'U8': 4, 'I8': 5, 'F16': 11, 'BF16': 12, 'F32': 15}
DTYPE_BYTES = {'U8': 1, 'I8': 1, 'F16': 2, 'BF16': 2, 'F32': 4}


def read_header(path):
    with open(path, 'rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        return json.loads(handle.read(length))


def source_files(paths):
    """{tensor name without prefix: (file, name in file, dtype, shape)} for one file or a list of Diffusers shards."""
    tensors = {}
    for path in paths:
        for name, info in read_header(path).items():
            if name == '__metadata__':
                continue
            plain = next((name[len(p):] for p in PREFIXES if name.startswith(p)), name)
            tensors[plain] = (str(path), name, info['dtype'], info['shape'])
    return tensors


class Plan(list):
    """The planned output tensors, with the architecture and the file metadata of its official INT8 file."""
    arch = None
    metadata = None


def detect(tensors):
    for arch in ARCHS:
        if all(key in tensors for key in arch.keys):
            return arch
    raise ValueError('Not a Qwen-Image-2.1 or MOSS Image transformer.')


def plan(tensors, turbo=False):
    """The output tensors: [(name, dtype, shape, recipe)], where recipe is ('copy', source) or ('quantize', [sources]).
    turbo: a Turbo model, which gets the INT8 attention configs of the official Turbo file.
    Raises ValueError when the file is not a float transformer of a known architecture."""
    arch = detect(tensors)
    layers, copies = {}, {}
    for name in tensors:
        split = arch.split.match(name) if arch.split else None
        if split:
            layers.setdefault(split[1] + '.gate_up', {})[split[2]] = name
        elif name.endswith('.weight') and arch.quantized.match(name[:-len('.weight')]):
            layers.setdefault(name[:-len('.weight')], {})['weight'] = name
        else:
            copies[name] = name
    if not layers:
        raise ValueError('No transformer block layers found.')
    out = Plan()
    out.arch, out.metadata = arch.name, arch.metadata
    for layer, parts in sorted(layers.items()):
        sources = [parts['gate_layer'], parts['proj']] if 'gate_layer' in parts or 'proj' in parts else [parts.get('weight')]
        if None in sources or len(parts) != len(sources):
            raise ValueError(f'Incomplete layer {layer}.')
        dtypes = {tensors[s][2] for s in sources}
        if not dtypes <= FLOATS:
            raise ValueError(f'{layer} is already {"/".join(sorted(dtypes))}; only BF16, FP16 or FP32 files are converted.')
        shapes = [tensors[s][3] for s in sources]
        if any(len(shape) != 2 for shape in shapes) or len({shape[1] for shape in shapes}) != 1 or shapes[0][1] % GROUPSIZE:
            raise ValueError(f'{layer} has an unexpected shape {shapes}.')
        rows, cols = sum(shape[0] for shape in shapes), shapes[0][1]
        conf = json.dumps(QUANT_CONF).encode('utf-8')
        out += [(layer + '.weight', 'I8', [rows, cols], ('quantize', sources)),
                (layer + '.weight_scale', 'F32', [rows, 1], ('scale', sources)),
                (layer + '.comfy_quant', 'U8', [len(conf)], ('conf', conf))]
    for name in sorted(copies):
        dtype = tensors[name][2]
        kept = 'F32' if name in arch.keep_f32 and dtype == 'F32' else 'BF16'
        out.append((name, kept if dtype in FLOATS else dtype, tensors[name][3], ('copy', name)))
    first = arch.attention_from.get('turbo' if turbo else 'standard')
    if first is not None:
        conf = json.dumps(ATTENTION_CONF).encode('utf-8')
        blocks = sorted({int(name.split('.')[1]) for name in layers})
        out += [(arch.attention.format(block), 'U8', [len(conf)], ('conf', conf)) for block in blocks[first:]]
    return out


def layout(planned, metadata=None):
    """(header bytes, {name: absolute file offset}) exactly as safetensors writes them."""
    order = sorted(planned, key=lambda item: (-DTYPE_RANK.get(item[1], 0), item[0]))
    if metadata is None:
        metadata = planned.metadata if getattr(planned, 'metadata', None) is not None else {'format': 'pt'}
    header, offsets, position = {'__metadata__': metadata}, {}, 0
    for name, dtype, shape, _ in order:
        size = DTYPE_BYTES[dtype]
        for dim in shape:
            size *= dim
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [position, position + size]}
        offsets[name] = position
        position += size
    raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    start = 8 + len(raw)
    return struct.pack('<Q', len(raw)) + raw, {name: start + offset for name, offset in offsets.items()}, start + position


def output_bytes(planned):
    return layout(planned)[2]


class Reader:
    """Open safetensors files once each."""
    def __init__(self):
        self.handles = {}

    def get(self, tensors, name):
        from safetensors import safe_open
        path, key = tensors[name][0], tensors[name][1]
        if path not in self.handles:
            self.handles[path] = safe_open(path, framework='pt')
        return self.handles[path].get_tensor(key)


def quantize(weight, device):
    """comfy_kitchen's ConvRot INT8 weight quantization, eager path in FP32 (the official files' exact numbers)."""
    import torch
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    with torch.inference_mode():
        q, scale = quantize_int8_convrot_weight(weight.to(device=device, dtype=torch.float32), GROUPSIZE)
    return q.cpu(), scale.float().cpu()


def _count(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def produce(planned, tensors, device_for, progress=lambda done, total, name: None, placeholders=False):
    """Yield (name, tensor) for every planned output tensor; each quantized layer once (weight, then scale).
    placeholders: also yield (scale name, None) where the scale stands in the planned order."""
    import torch
    reader = Reader()
    total = len(planned)
    done = 0
    for name, dtype, shape, recipe in planned:
        kind = recipe[0]
        if kind == 'scale':
            if placeholders:
                yield name, None
            continue  # produced together with its weight
        if kind == 'quantize':
            parts = [reader.get(tensors, source) for source in recipe[1]]
            weight = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
            del parts
            q, scale = quantize(weight, device_for())
            del weight
            yield name, q
            yield name[:-len('.weight')] + '.weight_scale', scale
            done += 2
        elif kind == 'conf':
            yield name, torch.tensor(list(recipe[1]), dtype=torch.uint8)
            done += 1
        else:
            value = reader.get(tensors, recipe[1])
            yield name, value.to(torch.bfloat16) if dtype == 'BF16' else value
            done += 1
        progress(done, total, name)


def tensor_bytes(value):
    import torch
    return value.contiguous().view(torch.uint8).numpy().tobytes() if value.dtype != torch.uint8 else value.contiguous().numpy().tobytes()


def default_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def convert(sources, target, *, turbo=False, device=None, cpu_flag=None, progress=lambda done, total, name: None):
    """Write the INT8 ConvRot file to target (through target.part, renamed at the end). Returns a summary dict.
    cpu_flag: a path; while it exists the remaining layers use the CPU (the app sets it when a GPU job starts)."""
    started = time.time()
    tensors = source_files([Path(p) for p in sources])
    planned = plan(tensors, turbo)
    header, offsets, size = layout(planned)
    target = Path(target)
    part = target.with_name(target.name + '.part')
    target.parent.mkdir(parents=True, exist_ok=True)
    chosen = device or default_device()
    used = set()

    def device_for():
        current = 'cpu' if cpu_flag and Path(cpu_flag).exists() else chosen
        used.add(current)
        return current
    # Write in file order: NTFS fills a gap before a far write with zeros first, which made out-of-order writes 20x slower.
    # The scales come first in the file but only exist after their weight, so their few MB are zeros until the end.
    ordered = sorted(planned, key=lambda item: offsets[item[0]])
    sizes = {name: DTYPE_BYTES[dtype] * _count(shape) for name, dtype, shape, _ in planned}
    scales = {}
    with open(part, 'wb') as handle:
        handle.write(header)
        for name, value in produce(ordered, tensors, device_for, progress, placeholders=True):
            if name.endswith('.weight_scale') and value is not None:
                scales[name] = value
                continue
            if handle.tell() != offsets[name]:
                raise RuntimeError(f'Write order error at {name}.')
            handle.write(bytes(sizes[name]) if value is None else tensor_bytes(value))
        for name, value in scales.items():
            handle.seek(offsets[name])
            handle.write(tensor_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, target)
    layers = sum(1 for item in planned if item[3][0] == 'quantize')
    return {'path': str(target), 'bytes': size, 'layers': layers, 'turbo': turbo, 'seconds': round(time.time() - started, 1),
            'devices': sorted(used), 'method': METHOD, 'version': VERSION}


def compare(sources, reference, *, turbo=False, device=None):
    """Convert in memory and compare every tensor and the header with a reference INT8 file. Nothing is written."""
    import torch
    from safetensors import safe_open
    tensors = source_files([Path(p) for p in sources])
    planned = plan(tensors, turbo)
    header = layout(planned)[0]
    with open(reference, 'rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        handle.seek(0)
        reference_header = handle.read(8 + length)
    result = {'header_identical': header == reference_header, 'tensors': 0, 'identical': 0, 'int8_mismatches': 0, 'max_int8_diff': 0,
              'scale_mismatches': 0, 'other_mismatches': 0, 'missing': []}
    chosen = device or default_device()
    with safe_open(reference, framework='pt') as ref:
        names = set(ref.keys())
        for name, value in produce(planned, tensors, lambda: chosen):
            result['tensors'] += 1
            if name not in names:
                result['missing'].append(name)
                continue
            expected = ref.get_tensor(name)
            if expected.dtype == value.dtype and expected.shape == value.shape and torch.equal(expected, value):
                result['identical'] += 1
            elif name.endswith('.weight') and value.dtype == torch.int8:
                diff = (expected.int() - value.int()).abs()
                result['int8_mismatches'] += int((diff > 0).sum())
                result['max_int8_diff'] = max(result['max_int8_diff'], int(diff.max()))
            elif name.endswith('.weight_scale'):
                result['scale_mismatches'] += int((expected != value).sum())
            else:
                result['other_mismatches'] += 1
    result['reference_only'] = sorted(names - {item[0] for item in planned})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert a Qwen-Image-2.1 BF16/FP16/FP32 transformer to INT8 ConvRot (official Comfy-Org layout).')
    parser.add_argument('sources', nargs='+', help='One .safetensors file, or every Diffusers shard of the transformer.')
    parser.add_argument('--target', help='Output .safetensors file.')
    parser.add_argument('--compare', help='Compare with this INT8 file in memory instead of writing.')
    parser.add_argument('--turbo', action='store_true', help='A Turbo (8 step) model: INT8 attention as in the official Turbo file.')
    parser.add_argument('--device', choices=('cuda', 'cpu'))
    parser.add_argument('--cpu-flag', help='While this file exists, quantize on the CPU.')
    parser.add_argument('--status', help='Write progress JSON to this file.')
    args = parser.parse_args(argv)
    if args.compare:
        print(json.dumps(compare(args.sources, args.compare, turbo=args.turbo, device=args.device), indent=1))
        return 0
    if not args.target:
        parser.error('--target or --compare is needed.')
    status = Path(args.status) if args.status else None
    last = [0.0]

    def progress(done, total, name):
        if status and (time.time() - last[0] > 1 or done == total):
            last[0] = time.time()
            status.write_text(json.dumps({'done': done, 'total': total, 'name': name}), encoding='utf-8')
    print(json.dumps(convert(args.sources, args.target, turbo=args.turbo, device=args.device, cpu_flag=args.cpu_flag, progress=progress)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

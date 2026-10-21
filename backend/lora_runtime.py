"""Qwen linear LoRAs as separate branches, without requantizing base weights.

The Qwen Viggle author's runtime method also covers fused SwiGLU output kernels:
https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/comfyui/viggle_turbo.py
This implementation uses validated Comfy target names, stacks all adapters in one
wrapper, and registers adapter memory with Comfy's normal placement manager.
"""
from collections import defaultdict

KEY = 'studio_qwen_lora'


def attach(model, patches, strength, title='LoRA'):
    import torch
    import comfy.model_patcher
    import comfy.patcher_extension as pe
    records = []
    for target, adapter in patches.items():
        branch = linear_branch(adapter)
        if branch is None:
            return None
        kind, first, second, factor = branch
        key = target if isinstance(target, str) else target[0]
        offset = None if isinstance(target, str) else target[1]
        if not key.endswith('.weight') or offset is not None and offset[0] != 0:
            return None
        path = key.removeprefix('diffusion_model.').removesuffix('.weight')
        module = model.model.diffusion_model.get_submodule(path)
        if not hasattr(module, 'in_features'):
            return None
        rows = offset[2] if offset is not None else module.out_features
        if kind == 'lokr' and (first.shape[0] * second.shape[0] != rows or first.shape[1] * second.shape[1] != module.in_features):
            return None
        records.append((path, offset, first, second, strength * factor, kind, title))
    previous = model.get_attachment(KEY)
    records = (previous.records if previous else []) + records

    class Branches(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.records = records
            self.calls = 0
            self.executed = set()
            self.effect_samples = {}
            self.layers = torch.nn.ModuleList()
            self.targets = defaultdict(list)
            for path, offset, first, second, scale, kind, label in records:
                layer = torch.nn.Module()
                layer.kind = kind
                # LoRA: first=up, second=down. LoKr: first=w1, second=w2 of the Kronecker product.
                layer.register_parameter('first', torch.nn.Parameter(first, requires_grad=False))
                layer.register_parameter('second', torch.nn.Parameter(second, requires_grad=False))
                self.targets[path].append((len(self.layers), offset, scale))
                self.layers.append(layer)

        def get_dtype(self):
            return next(self.parameters()).dtype

        def memory_required(self, *args, **kwargs):
            return 0

        def add(self, path, x, out):
            result = out
            for index, offset, scale in self.targets[path]:
                if scale == 0:
                    continue
                self.executed.add(index)
                layer = self.layers[index]
                first = layer.first.to(device=x.device, dtype=x.dtype)
                second = layer.second.to(device=x.device, dtype=x.dtype)
                delta = branch_delta(layer.kind, first, second, x) * scale
                # One small sample per adapter/operation. Keep reductions on the
                # device until reporting; never synchronize the GPU per layer.
                title = self.records[index][-1]
                group = 'mlp-out' if path.endswith('.img_mlp.out') else 'mlp-in' if '.img_mlp.' in path else 'attention' if '.attn.' in path else 'other'
                sample_key = (title, group)
                if sample_key not in self.effect_samples:
                    original = result if offset is None else result[..., offset[1]:offset[1]+offset[2]]
                    # Select before flattening: a fused output slice can be
                    # noncontiguous and flattening it can copy the full video.
                    def first_values(tensor):
                        while tensor.ndim > 1:
                            tensor = tensor[0]
                        return tensor[:1024].detach()
                    base_sample = first_values(original)
                    branch_sample = first_values(delta).to(base_sample.dtype)
                    applied = (base_sample + branch_sample).float() - base_sample.float()
                    self.effect_samples[sample_key] = (path, torch.stack([
                        base_sample.float().square().mean().sqrt(),
                        branch_sample.float().square().mean().sqrt(),
                        applied.square().mean().sqrt()]))
                if offset is None:
                    result = result + delta.to(result.dtype)
                else:
                    _, start, length = offset
                    result = result.clone()
                    result[..., start:start+length] += delta.to(result.dtype)
            return result

        def run(self, executor, *args, **kwargs):
            self.calls += 1
            dm = executor.class_obj
            hooks, fused_outputs, captured, forced = [], {}, {}, []
            try:
                # One hook per target sums all LoRAs. A fused MLP's output
                # projection can bypass module hooks, so capture its final
                # gate/up activation and add that output branch on the parent.
                for path in self.targets:
                    parent, _, leaf = path.rpartition('.')
                    mlp = dm.get_submodule(parent) if parent else dm
                    if leaf == 'out' and getattr(mlp, 'fused', False):
                        fused_outputs[parent] = path
                        continue
                    module = dm.get_submodule(path)
                    # comfy.ops.linear_input_act (MOSS ffn_down, FLUX.3) fuses the input activation into the
                    # quantized matmul and skips module hooks; forced, it calls the module with the activated input.
                    if not getattr(module, 'comfy_force_forward', False):
                        module.comfy_force_forward = True
                        forced.append(module)
                    hooks.append(module.register_forward_hook(
                        lambda mod, inp, out, path=path: self.add(path, inp[0], out)))
                for parent, path in fused_outputs.items():
                    mlp = dm.get_submodule(parent)
                    hooks.append(mlp.gate_up.register_forward_hook(
                        lambda mod, inp, out, parent=parent: captured.__setitem__(parent, out)))
                    def output_hook(mod, inp, out, parent=parent, path=path):
                        gate, up = captured.pop(parent).chunk(2, -1)
                        return self.add(path, torch.nn.functional.silu(gate) * up, out)
                    hooks.append(mlp.register_forward_hook(output_hook))
                return executor(*args, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
                for module in forced:
                    module.comfy_force_forward = False

    bank = Branches()
    bank_patcher = comfy.model_patcher.ModelPatcher(bank, model.load_device, model.offload_device)
    clone = model.clone()
    clone.set_attachments(KEY, bank)
    clone.set_additional_models(KEY, [bank_patcher])
    clone.remove_wrappers_with_key(pe.WrappersMP.DIFFUSION_MODEL, KEY)
    clone.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, KEY, bank.run)
    clone.remove_callbacks_with_key(pe.CallbacksMP.ON_CLEANUP, KEY)
    def cleanup(_):
        import comfy.model_management
        comfy.model_management.unload_model_and_clones(bank_patcher)
    clone.add_callback_with_key(pe.CallbacksMP.ON_CLEANUP, KEY, cleanup)
    return clone


def linear_branch(adapter):
    """(kind, first, second, alpha factor) for a plain linear LoRA or LoKr, else None (merged by Comfy instead).
    The factor follows Comfy: LoRA alpha/rank; LoKr alpha/dim only for factorized matrices, else 1."""
    import torch
    values = getattr(adapter, 'weights', ())
    name = getattr(adapter, 'name', '')
    if name == 'lora' and len(values) == 6:
        up, down, alpha, mid, dora, reshape = values
        if up.ndim != 2 or down.ndim != 2 or any(x is not None for x in (mid, dora, reshape)):
            return None
        return 'lora', up, down, float(alpha) / down.shape[0] if alpha is not None else 1.
    if name == 'lokr' and len(values) >= 9:
        w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2, dora = values[:9]
        if t2 is not None or dora is not None:
            return None
        dim = None
        if w1 is None:
            dim = w1_b.shape[0]
            w1 = torch.mm(w1_a.float(), w1_b.float())
        if w2 is None:
            dim = w2_b.shape[0]
            w2 = torch.mm(w2_a.float(), w2_b.float())
        if w1.ndim != 2 or w2.ndim != 2:
            return None
        return 'lokr', w1, w2, float(alpha) / dim if alpha is not None and dim is not None else 1.
    return None


def branch_delta(kind, first, second, x):
    """The adapter's output change for input x, without building the full weight difference.
    kron(w1, w2) @ x equals w1 @ X @ w2.T with x reshaped to X (w1 columns by w2 columns)."""
    import torch
    if kind == 'lora':
        return torch.nn.functional.linear(torch.nn.functional.linear(x, second), first)
    grid = x.reshape(*x.shape[:-1], first.shape[1], second.shape[1])
    return torch.einsum('ij,...jl,kl->...ik', first, grid, second).reshape(*x.shape[:-1], first.shape[0] * second.shape[0])


def execution_report(model):
    """Check actual branch execution after sampling, not just accepted patches."""
    bank = model.get_attachment(KEY)
    if bank is None:
        return []
    reports = {}
    for index, record in enumerate(bank.records):
        title = record[-1]
        report = reports.setdefault(title, {'title': title, 'expected': 0, 'executed': 0, 'model_calls': bank.calls})
        if record[4] != 0:
            report['expected'] += 1
            report['executed'] += int(index in bank.executed)
    for report in reports.values():
        if report['expected'] != report['executed']:
            raise RuntimeError(f'The LoRA "{report["title"]}" did not run all its layers. Generation stopped.')
        report['layer_effects'] = [
            {'layer': path, 'base_rms': values[0], 'branch_rms': values[1], 'applied_rms': values[2]}
            for (title, _), (path, sample) in bank.effect_samples.items()
            if title == report['title'] for values in [sample.float().cpu().tolist()]]
    return list(reports.values())

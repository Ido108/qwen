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
        values = getattr(adapter, 'weights', ())
        if getattr(adapter, 'name', '') != 'lora' or len(values) != 6:
            return None
        up, down, alpha, mid, dora, reshape = values
        if up.ndim != 2 or down.ndim != 2 or any(x is not None for x in (mid, dora, reshape)):
            return None
        key = target if isinstance(target, str) else target[0]
        offset = None if isinstance(target, str) else target[1]
        if not key.endswith('.weight') or offset is not None and offset[0] != 0:
            return None
        path = key.removeprefix('diffusion_model.').removesuffix('.weight')
        module = model.model.diffusion_model.get_submodule(path)
        if not hasattr(module, 'in_features'):
            return None
        scale = strength * (float(alpha) / down.shape[0] if alpha is not None else 1.)
        records.append((path, offset, up, down, scale, title))
    previous = model.get_attachment(KEY)
    records = (previous.records if previous else []) + records

    class Branches(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.records = records
            self.calls = 0
            self.executed = set()
            self.layers = torch.nn.ModuleList()
            self.targets = defaultdict(list)
            for path, offset, up, down, scale, label in records:
                layer = torch.nn.Module()
                layer.register_parameter('up', torch.nn.Parameter(up, requires_grad=False))
                layer.register_parameter('down', torch.nn.Parameter(down, requires_grad=False))
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
                down = layer.down.to(device=x.device, dtype=x.dtype)
                up = layer.up.to(device=x.device, dtype=x.dtype)
                delta = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * scale
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
            hooks, fused_outputs, captured = [], {}, {}
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
    return list(reports.values())

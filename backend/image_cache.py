"""Bounded image caches and safe eviction of immutable, read-only CPU weights."""
import hashlib
from types import MethodType


class ConditioningCache:
    def __init__(self):
        self.key = None
        self.value = None

    @staticmethod
    def fingerprint(prompt, negative, images):
        digest = hashlib.sha256()
        for text in (prompt, negative):
            data = text.encode('utf-8')
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
        for image in images:
            digest.update(str((image.mode, image.size)).encode())
            digest.update(image.tobytes())
        return digest.hexdigest()

    def get(self, key):
        return self.value if key == self.key else None

    def put(self, key, value):
        # One entry only: reference latents and long multi-image prompts are large.
        self.key, self.value = key, value


class PositiveOnlyEncoder:
    """The official node encodes both sides; CFG=1 never uses the negative side."""
    def __init__(self, clip, use_negative):
        self.clip = clip
        self.use_negative = use_negative
        self.positive = None

    def tokenize(self, *args, **kwargs):
        if self.positive is not None and not self.use_negative:
            return None
        return self.clip.tokenize(*args, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        if self.positive is not None and not self.use_negative:
            return self.positive
        result = self.clip.encode_from_tokens_scheduled(tokens)
        self.positive = result
        return result


def retain_cpu_weights(patcher):
    """Keep CPU views, not copies; return to these when GPU memory is needed.

    Image weights are immutable (no LoRAs). Restoring the original CPU tensors
    avoids GPU-to-CPU copies into the read-only Windows mapping. Normal Comfy
    placement still decides when to evict/reload; idle models stay on the GPU.
    """
    import torch
    def cpu_view(value):
        # GGMLTensor.detach()/clone() deliberately return self. Construct a new
        # wrapper over the same CPU storage so Module.to() cannot mutate our
        # saved Parameter's .data into a CUDA tensor.
        if hasattr(value, 'tensor_type') and hasattr(value, 'tensor_shape'):
            return type(value)(value.as_subclass(torch.Tensor), tensor_type=value.tensor_type,
                               tensor_shape=value.tensor_shape, patches=[])
        return value.detach()
    if patcher.patches:
        raise ValueError('CPU weight retention requires unmodified image weights.')
    saved = []
    for module in patcher.model.modules():
        parameters, buffers = {}, {}
        for name, value in module._parameters.items():
            if value is not None:
                if value.device.type != 'cpu':
                    raise ValueError('Retain image weights before loading them on the GPU.')
                parameters[name] = cpu_view(value)
        for name, value in module._buffers.items():
            if value is not None and value.device.type == 'cpu':
                buffers[name] = value.detach()
        if parameters or buffers:
            saved.append((module, parameters, buffers))
    original_unpatch = patcher.unpatch_model

    def restore_cpu_weights():
        for module, parameters, buffers in saved:
            module._parameters.update({name: torch.nn.Parameter(cpu_view(value), requires_grad=False)
                                       for name, value in parameters.items()})
            module._buffers.update(buffers)
    patcher.studio_restore_cpu_weights = restore_cpu_weights

    def unpatch(self, device_to=None, unpatch_weights=True):
        # QuantizedTensor's view operations need ordinary version counters.
        with torch.inference_mode(False):
            if unpatch_weights and device_to is not None and torch.device(device_to).type == 'cpu':
                restore_cpu_weights()
            return original_unpatch(device_to=device_to, unpatch_weights=unpatch_weights)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Evict this component as a whole. It remains constructed in RAM, so
        # loading it again needs only transfer, not disk parsing/model creation.
        loaded = self.loaded_size()
        self.unpatch_model(device_to=device_to)
        return loaded

    patcher.unpatch_model = MethodType(unpatch, patcher)
    patcher.partially_unload = MethodType(partially_unload, patcher)


def retain_lora_clone(clone, base):
    """Give a LoRA clone of a retained model the same CPU-view eviction.

    The clone shares the base nn.Module. ComfyUI first removes the clone's own
    LoRA patches and weight backups; then the original read-only CPU views are
    put back instead of copying GPU tensors to RAM.
    """
    import torch
    restore = base.studio_restore_cpu_weights
    class_unpatch = type(clone).unpatch_model

    def unpatch(self, device_to=None, unpatch_weights=True):
        with torch.inference_mode(False):
            to_cpu = unpatch_weights and device_to is not None and torch.device(device_to).type == 'cpu'
            result = class_unpatch(self, device_to=None if to_cpu else device_to, unpatch_weights=unpatch_weights)
            if to_cpu:
                restore()
                self.model.device = torch.device(device_to)
            return result

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        loaded = self.loaded_size()
        self.unpatch_model(device_to=device_to)
        return loaded

    clone.unpatch_model = MethodType(unpatch, clone)
    clone.partially_unload = MethodType(partially_unload, clone)

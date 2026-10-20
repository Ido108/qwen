"""Checked model-only LoRA application using the installed ComfyUI adapters."""
import math
import re


def qwen_aliases(mapping):
    """Keep native/Kohya names for the two halves of fused Qwen 2.1 MLPs.

    Comfy maps those halves only under bare, transformer and LyCORIS names.
    The app also loads native diffusion_model and lora_unet names.
    """
    result = dict(mapping)
    for key, target in mapping.items():
        if key.startswith(('transformer_blocks.', 'time_text_embed.', 'modulation.',
                           'img_in.', 'txt_in.', 'proj_out.', 'norm_out.')):
            result.setdefault('diffusion_model.' + key, target)
            result.setdefault('lora_unet_' + key.replace('.', '_'), target)
    return result


def load_model_lora(model, weights, strength, title, *, runtime=False):
    import comfy.lora
    import comfy.lora_convert
    import comfy.model_base
    import comfy.utils
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    if isinstance(model.model, comfy.model_base.QwenImage):
        mapping = qwen_aliases(mapping)
    weights = comfy.lora_convert.convert_lora(weights)
    patches = comfy.lora.load_lora(weights, mapping, log_missing=False)
    if not patches:
        raise ValueError(f'The LoRA "{title}" did not attach any weights. Generation stopped.')
    used = set().union(*(getattr(adapter, 'loaded_keys', set()) for adapter in patches.values()))
    unmatched = [key for key in weights if key not in used and
                 re.search(r'(?:lora_(?:A|B|up|down|mid)(?:[._]|$)|lokr_|hada_|oft_|boft_|glora_)', key)]
    if unmatched:
        raise ValueError(f'The LoRA "{title}" has {len(unmatched)} weights that could not attach. Generation stopped.')
    # The upstream adapter logs shape errors and returns the unchanged weight.
    # Reject those before sampling, instead of reporting an ineffective LoRA.
    state = model.model.state_dict()
    for target, adapter in patches.items():
        key = target if isinstance(target, str) else target[0]
        if key not in state:
            raise ValueError(f'The LoRA "{title}" has a missing weight target. Generation stopped.')
        values = getattr(adapter, 'weights', ())
        if getattr(adapter, 'name', '').lower() == 'lora' and len(values) == 6 and values[3] is None and values[5] is None:
            up, down = values[:2]
            # GGUF state_dict contains packed byte shapes for memory estimates.
            # Read the live parameter's logical shape, without dequantizing it.
            weight = comfy.utils.get_attr(model.model, key)
            shape = list(getattr(weight, 'tensor_shape', weight.shape))
            if not isinstance(target, str):
                dimension, start, length = target[1]
                if start + length > shape[dimension]:
                    raise ValueError(f'The LoRA "{title}" has an invalid weight slice.')
                shape[dimension] = length
            if (math.prod(up.shape[1:]) != down.shape[0] or up.shape[0] != shape[0]
                    or math.prod(down.shape[1:]) != math.prod(shape[1:])):
                raise ValueError(f'The LoRA "{title}" has a weight size that does not match. Generation stopped.')
    if runtime and isinstance(model.model, (comfy.model_base.QwenImage, comfy.model_base.Krea2)):
        from .lora_runtime import attach
        clone = attach(model, patches, strength, title)
        if clone is not None:
            return clone, {'patches': len(patches), 'adapter_tensors': len(used), 'strength': strength, 'application': 'runtime'}
    clone = model.clone()
    accepted = clone.add_patches(patches, strength)
    if len(accepted) != len(patches):
        raise ValueError(f'The LoRA "{title}" did not attach all weights. Generation stopped.')
    return clone, {'patches': len(accepted), 'adapter_tensors': len(used), 'strength': strength, 'application': 'merged'}

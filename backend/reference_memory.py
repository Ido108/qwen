"""Include H3 reference tokens in ComfyUI's working-memory estimate."""
from math import prod


def reference_latent_elements(conditioning):
    # ComfyUI's base estimate sees the output's packed video/audio length only.
    # H3 also attends to every reference frame and guide frame at every step.
    maximum = 0
    for _, info in conditioning:
        seen = set()
        count = 0
        for key in ('minimax_refs', 'minimax_keyframes'):
            for item in info.get(key) or []:
                for name in ('latent', 'audio_latent'):
                    tensor = item.get(name)
                    if tensor is not None and id(tensor) not in seen:
                        seen.add(id(tensor))
                        count += prod(tensor.shape[1:])
        maximum = max(maximum, count)
    return maximum


def account_for_reference_memory(model, conditioning):
    elements = reference_latent_elements(conditioning)
    if not elements:
        return 0
    original = model.memory_required

    def memory_required(input_shape, cond_shapes=None):
        shapes = cond_shapes or {}
        # Use the same upstream formula and dtype, with references packed in the
        # same units as the output. This does not change or downsize any inputs.
        reference_shape = [input_shape[0], 1, elements]
        return original(input_shape, shapes) + original(reference_shape, {})

    model.memory_required = memory_required
    return elements

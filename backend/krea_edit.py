"""Krea 2 identity edit: conradlocke/krea2-identity-edit with its node pack (vendor/comfyui-krea2edit, unmodified).

The LoRA learned to copy a person from clean in-context photo tokens: [text | photo (RoPE frame 1) | target (frame 0)],
with the instruction encoded by Qwen3-VL while it sees the photo. Stock Krea 2 encoding gives the LoRA neither part.
Settings are the model card's (Identity Edit V2): Turbo, 8 steps, CFG 1, grounding 768 px; one photo: ref_boost 2,
placed beside the canvas (restaging); two photos (scene, then person): person 2.5 beside, scene 1 aligned.
"""
from __future__ import annotations
import importlib.util
import sys
from .common import APP_DIR

PACK = APP_DIR / 'vendor' / 'comfyui-krea2edit'
ALIGNED, BESIDE = 'aligned (trained)', 'beside (right of target)'
GROUNDING_PX = 768


def pack():
    """The node pack module, imported once from the vendored copy (its folder name is not a Python name)."""
    if 'qwen_krea2edit' not in sys.modules:
        spec = importlib.util.spec_from_file_location('qwen_krea2edit', PACK / '__init__.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules['qwen_krea2edit']


def reference_settings(count):
    """Card settings by photo count. The last photo is the person; with two, the first is the scene."""
    if count >= 2:
        return {'ref_boost': 2.5, 'ref_boost_a': 1.0, 'ref_position': BESIDE, 'ref_position_a': ALIGNED}
    return {'ref_boost': 2.0, 'ref_boost_a': 1.0, 'ref_position': BESIDE, 'ref_position_a': ALIGNED}


MAX_PIXELS = 2 * 1024 * 1024


def identity_size(width, height):
    """At most 2 MP (the card: generate at 2MP or less), same shape, multiples of 16."""
    if width * height <= MAX_PIXELS:
        return width, height
    scale = (MAX_PIXELS / (width * height)) ** 0.5
    return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)


def image_tensor(image):
    import numpy as np
    import torch
    return torch.from_numpy(np.array(image.convert('RGB')).astype(np.float32) / 255.0).unsqueeze(0)


def encode(clip, prompt, images):
    """Krea2EditGroundedEncode: the instruction with up to two photos in training order (scene, person)."""
    tensors = [image_tensor(image) for image in images[:2]]
    node = pack().Krea2EditGroundedEncode()
    return node.encode(clip, prompt, image=tensors[0], image_b=tensors[1] if len(tensors) > 1 else None, grounding_px=GROUNDING_PX)[0]


def patch(model, vae, images, latent):
    """Krea2EditModelPatch on the pixel path (fit, pre-encoded at the target size before sampling)."""
    import torch
    tensors = [image_tensor(image) for image in images[:2]]
    # The pixel path replaces source_latent; the node still scales it once, so a 1x1 placeholder is enough.
    placeholder = {'samples': torch.zeros(latent.shape[:-2] + (1, 1), dtype=latent.dtype)}
    patched = pack().Krea2EditModelPatch().patch(model, placeholder, vae=vae, source_image=tensors[0],
        source_image_b=tensors[1] if len(tensors) > 1 else None, fit_mode='fit', target_latent={'samples': latent},
        **reference_settings(len(tensors)))[0]
    # The node registers its forward in transformer_options, which ComfyUI places outside the patcher's wrappers
    # (sampler_helpers merges them first). Its forward never calls the next wrapper, so the runtime LoRA hooks
    # would never attach. Moved into the patcher list after the LoRAs, it runs inside them.
    import comfy.patcher_extension as pe
    wrappers = patched.model_options['transformer_options'].get('wrappers', {}).get(pe.WrappersMP.DIFFUSION_MODEL, {})
    for wrapper in wrappers.pop('krea2_edit', []):
        patched.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, 'krea2_edit', wrapper)
    return patched

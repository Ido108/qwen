"""Explicit sampling overrides for supported image paths; never ignore a choice."""
from .res4lyf_runtime import SAMPLERS as RES_SAMPLERS
SAMPLERS = ('euler', 'res_multistep', 'dpmpp_2m', 'heun', *RES_SAMPLERS)
SCHEDULERS = ('simple', 'normal', 'beta', 'karras', 'sgm_uniform')
FAMILIES = ('qwen-image-2.1', 'qwen-image-2.1-turbo', 'krea-2', 'krea-2-turbo',
            'z-image', 'z-image-turbo', 'flux3-klein', 'flux3-klein-base', 'moss-image', 'moss-image-base', 'moss-image-turbo')


def validate(request, model):
    sampler, scheduler = request.get('sampler'), request.get('scheduler')
    if not sampler and not scheduler:
        return
    if model.get('family') not in FAMILIES:
        raise ValueError('This model does not support image sampling controls.')
    if sampler and sampler not in SAMPLERS or scheduler and scheduler not in SCHEDULERS:
        raise ValueError('Choose a supported sampler and schedule.')


def resolve(request, defaults):
    """A sampler override keeps trained sigmas; only a schedule override replaces them."""
    sampler = request.get('sampler') or defaults.get('sampler', 'euler')
    trained = bool(defaults.get('sigmas')) and not request.get('scheduler')
    scheduler = 'turbo-sigmas' if trained else request.get('scheduler') or defaults.get('scheduler', 'simple')
    return sampler, scheduler


def sample_qwen(model, noise, request, defaults, cfg, positive, negative, latent, callback):
    import comfy.sample
    import comfy.samplers
    sampler, scheduler = resolve(request, defaults)
    if scheduler == 'turbo-sigmas':
        import numpy as np
        import torch
        sigmas = defaults['sigmas']
        if request['steps'] != len(sigmas) - 1:
            sigmas = np.interp(np.linspace(0, 1, request['steps'] + 1),
                               np.linspace(0, 1, len(sigmas)), sigmas).tolist()
        return comfy.sample.sample_custom(model, noise, cfg, comfy.samplers.sampler_object(sampler),
            torch.tensor(sigmas, dtype=torch.float32), positive, negative, latent,
            callback=callback, seed=request['seed'])
    return comfy.sample.sample(model, noise, request['steps'], cfg, sampler, scheduler,
        positive, negative, latent, callback=callback, seed=request['seed'])

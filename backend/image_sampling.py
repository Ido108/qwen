"""Explicit sampling overrides for supported image paths; never ignore a choice."""
from .res4lyf_runtime import SAMPLERS as RES_SAMPLERS
SAMPLERS = ('euler', 'res_multistep', 'dpmpp_2m', 'heun', *RES_SAMPLERS)
SCHEDULERS = ('simple', 'normal', 'beta', 'karras', 'sgm_uniform')


def validate(request, model):
    sampler, scheduler = request.get('sampler'), request.get('scheduler')
    if not sampler and not scheduler:
        return
    if model.get('family') not in ('qwen-image-2.1', 'krea-2') or model.get('generation_defaults', {}).get('sigmas'):
        raise ValueError('This model uses its own sampling schedule. Reset Sampler and Schedule to Model default.')
    if sampler and sampler not in SAMPLERS or scheduler and scheduler not in SCHEDULERS:
        raise ValueError('Choose a supported sampler and schedule.')

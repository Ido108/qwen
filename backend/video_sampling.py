"""Resolve the settings once, before queueing. Workers use the saved contract."""
from .reference_sampling import reference_sampling


def scheduler_sigmas(profile, model, shift_model, calculate_sigmas):
    """A publisher graph can shift its scheduler separately from its guider."""
    video = profile.get('scheduler_video_shift', profile['video_shift'])
    audio = profile.get('scheduler_audio_shift', profile['audio_shift'])
    if (video, audio) == (profile['video_shift'], profile['audio_shift']):
        return None
    schedule_model = shift_model(model, video, audio)[0]
    return calculate_sigmas(schedule_model.get_model_object('model_sampling'), profile['scheduler'], profile['steps'])


def video_sampling(item, quality=None, steps=None, last_frame=False):
    family = item.get('family')
    defaults = item.get('generation_defaults', {})
    if family in ('ltx25', 'ltx25-msr'):
        from .ltx_settings import SIGMAS, REFINE_SIGMAS
        return dict(steps=8, total_steps=8 if last_frame else 11, fixed_steps=True,
                    cfg=1.0, guidance=1.0, sampler='euler_ancestral', scheduler='manual', eta=0.0 if last_frame else 1.0,
                    sigmas=SIGMAS, refine_sigmas=[] if last_frame else REFINE_SIGMAS, quality='standard')
    if family == 'minimax-h3-ref2va':
        profile = reference_sampling(item, quality, steps)
    elif family == 'fasth3':
        profile = dict(steps=8, fixed_steps=True, sampler='res_multistep', scheduler='simple',
                       video_shift=10.0, audio_shift=3.0, quality='standard')
    elif family == 'minimax-h3':
        quality = quality or 'standard'
        profile = dict(steps=20, sampler='res_multistep', scheduler='simple', video_shift=12.0, audio_shift=3.0)
        profile.update({k:v for k,v in defaults.items() if k in (
            'steps','sampler','scheduler','video_shift','audio_shift','scheduler_video_shift','scheduler_audio_shift',
            'embedded_turbo','field_sources','steps_range','step_options','workflow_source')})
        if quality == 'turbo' and not defaults.get('embedded_turbo'):
            profile.update(steps=8, lora_id='h3-turbo')
        profile.update(quality=quality, default_steps=profile['steps'])
        if steps is not None:
            if isinstance(steps,bool) or not isinstance(steps,int) or not 1 <= steps <= 60:
                raise ValueError('Steps must be a whole number from 1 to 60.')
            profile['steps'] = steps
    else:
        raise ValueError('This model does not support video generation.')
    profile.update(cfg=1.0, guidance=1.0, total_steps=profile['steps'])
    return profile

"""Resolve a Ref2VA checkpoint's sampling contract from the model list."""


def reference_sampling(item, quality=None, steps=None):
    if item.get('family') != 'minimax-h3-ref2va':
        raise ValueError('Choose a MiniMax H3 Ref2VA model for this workflow.')
    defaults = item['generation_defaults']
    quality = quality or defaults['quality']
    profiles = defaults['profiles']
    if quality not in profiles:
        if item['format'] == 'GGUF':
            raise ValueError('This GGUF model is not distilled. Choose Standard quality.')
        raise ValueError('The selected transformer has Turbo merged. Choose Turbo or a standard model.')
    profile = dict(profiles[quality])
    profile.update(quality=quality, default_steps=profile['steps'])
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 60:
            raise ValueError('Steps must be a whole number from 1 to 60.')
        profile['steps'] = steps
    return profile

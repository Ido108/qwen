"""User defaults are separate from author metadata and immutable queued requests."""
import math
import re
import threading
from .common import DATA_DIR, read_json, write_json

PATH = DATA_DIR / 'model-defaults.json'
LOCK = threading.RLock()
SCOPES = {'image', 'swap-head-swap', 'swap-head-swap-expression', 'swap-body-swap',
          'video', 'references', 'video-to-video', 'orbit', 'upscale', 'training'}
FIELDS = set('steps cfg sampler scheduler custom_size auto_resolution width height seed randomize_seed '
             'negative_prompt vae_id vae_choice text_encoder denoise strength extra poseSource styleSource '
             'styleMode negativePrompt nonRealisticNegative quality frames fps refine reference_strength '
             'background_strength audio output_audio source_audio reference_image_size aspect_mode size_tier '
             'custom_aspect_width custom_aspect_height camera_azimuth camera_elevation camera_distance '
             'camera_height batch_size chunk_size temporal_overlap color_correction clip_len scale '
             'rank learning_rate resolutions save_every auto_caption settings_mode tool_options '
             'lora_strength output_dir reference_frames camera_fov camera_aim_lock camera_vertical_shift '
             'replace_subjects_first resolution size_mode preserve_audio memory_mode flash_chunk_size '
             'flash_sampling flash_sparse_ratio flash_local_range flash_conditioning flash_cache_format '
             'flash_decoder flash_color'.split())


def read():
    with LOCK:
        return read_json(PATH) if PATH.exists() else {}


def save(model_id, scope, field, value):
    from .models import entry
    if scope not in SCOPES or not re.fullmatch(r'[\w .()/?-]{1,80}', field):
        raise ValueError('Choose a valid setting.')
    if scope == 'training':
        from .training import ARCHS
        if model_id not in ARCHS:
            raise ValueError('Choose a training model.')
    else:
        entry(model_id)
    if value is not None:
        if not isinstance(value, dict) or not value or set(value) - FIELDS:
            raise ValueError('This value cannot be saved as a model default.')
        def valid(item, depth=0):
            if depth > 2: return False
            if item is None or isinstance(item, bool): return True
            if isinstance(item, (int, float)): return math.isfinite(item)
            if isinstance(item, str): return len(item) <= 8000
            if isinstance(item, list): return len(item) <= 16 and all(valid(v, depth+1) for v in item)
            if isinstance(item, dict): return len(item) <= 20 and all(isinstance(k,str) and len(k)<=80 and valid(v,depth+1) for k,v in item.items())
            return False
        if not valid(value):
            raise ValueError('Enter a valid setting value.')
        for key, low, high in [('steps',1,100000),('cfg',1,30),('width',64,16384),('height',64,16384),('seed',0,2147483647)]:
            number=value.get(key)
            if number is not None and (isinstance(number,bool) or not isinstance(number,(int,float)) or not low<=number<=high):
                raise ValueError(f'Enter a valid {key}.')
    with LOCK:
        data = read()
        settings = data.setdefault(model_id, {}).setdefault(scope, {})
        if value is None:
            settings.pop(field, None)
        else:
            settings.pop(field, None)
            settings[field] = value
        write_json(PATH, data)
        return settings

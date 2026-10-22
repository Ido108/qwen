"""What an image job will really run: steps, CFG, sampler, schedule and every LoRA with where it comes from.
The server fills jobs from it and the page shows it, so the screen and the job cannot disagree."""
from __future__ import annotations

from . import models


def _family(item):
    return item.get('family', 'qwen-image-2.1')


def tool_recipe(tool, item):
    """The tool author's sampling for this model family (BFS Qwen swaps on Standard), or {}."""
    recipe = (tool or {}).get('sampling') or {}
    families = recipe.get('families')
    return recipe if families is None or _family(item) in families else {}


def speed_lora(tool, item, request):
    """The author's few-step LoRA for this tool and family, unless the user switched it off."""
    if request.get('speed_lora') is False:
        return None
    return tool_recipe(tool, item).get('speed_lora')


def plan(request: dict) -> dict:
    item = models.entry(request['model_id'])
    defaults = item.get('generation_defaults') or {}
    family = _family(item)
    tool = models.image_tool(request['tool_id']) if request.get('tool_id') else None
    images = int(request.get('image_count', len(request.get('images') or [])))
    # The author's recipe counts only with its speed LoRA when it has one (Pruna's 8 steps need Pruna).
    recipe = dict(tool_recipe(tool, item))
    speed = speed_lora(tool, item, request)
    if recipe.get('speed_lora') and not speed:
        recipe = {}
    steps = request.get('steps') or (tool or {}).get('steps') or recipe.get('steps') or defaults.get('steps')
    sampler = request.get('sampler') or recipe.get('sampler') or defaults.get('sampler', 'euler')
    trained = bool(defaults.get('sigmas')) and not request.get('scheduler') and not recipe.get('scheduler')
    scheduler = 'trained' if trained else request.get('scheduler') or recipe.get('scheduler') or defaults.get('scheduler', 'simple')
    negative = (request.get('negative_prompt') or '').strip()
    if request.get('cfg'):
        cfg = float(request['cfg'])
    elif recipe.get('cfg'):
        cfg = float(recipe['cfg'])
    elif defaults.get('cfg'):
        cfg = float(defaults['cfg'])
    elif family.startswith('qwen-image'):
        # Qwen: Turbo 1; Standard 1, or 2 with a negative prompt (comfy_engine)
        cfg = 1.0 if defaults.get('sigmas') else 2.0 if negative else 1.0
    else:
        cfg = 1.0
    loras = []
    if tool:
        strength = request.get('lora_strength')
        loras.append({'id': tool['lora'], 'title': models.entry(tool['lora'])['title'], 'source': 'tool',
                      'strength': float(strength if strength is not None else tool['strength'])})
        if speed:
            loras.append({'id': speed, 'title': models.entry(speed)['title'], 'source': 'speed',
                          'strength': float(request.get('speed_lora_strength') or 1.0)})
    elif models.is_krea(item) and images:
        identity = (request.get('krea_reference_mode') or 'identity') == 'identity'
        lora = models.KREA_IDENTITY_LORA if identity else models.KREA_STYLE_LORA
        loras.append({'id': lora, 'title': models.entry(lora)['title'], 'source': 'model', 'strength': 1.0})
    if defaults.get('speed_lora'):
        loras.append({'id': defaults['speed_lora'], 'title': models.entry(defaults['speed_lora'])['title'], 'source': 'model', 'strength': 1.0})
    for choice in request.get('loras') or []:
        lora_id = choice['id'] if isinstance(choice, dict) else choice.id
        if (choice.get('enabled', True) if isinstance(choice, dict) else getattr(choice, 'enabled', True)) is False:
            continue
        loras.append({'id': lora_id, 'title': models.entry(lora_id)['title'], 'source': 'yours',
                      'strength': float(choice['strength'] if isinstance(choice, dict) else choice.strength)})
    return {'steps': int(steps or 1), 'cfg': cfg, 'sampler': sampler, 'scheduler': scheduler, 'loras': loras,
            'speed_lora': {'id': recipe.get('speed_lora') or tool_recipe(tool, item).get('speed_lora'), 'on': bool(speed)}
                          if tool_recipe(tool, item).get('speed_lora') else None,
            'recipe': bool(recipe)}

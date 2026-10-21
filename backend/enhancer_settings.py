"""Which prompt enhancer each image architecture uses. The default is the architecture's own official enhancer;
Settings can choose another installed local enhancer or the cloud (Antigravity). Stored in DATA_DIR/enhancer-settings.json."""
from __future__ import annotations
import json
from .common import DATA_DIR, write_json

PATH = DATA_DIR / 'enhancer-settings.json'
CLOUD = 'cloud'
FAST = 'fast'
# arch: (title, official note, [(choice, label, official, prompt key or None for the official one, runner)])
ARCHS = {
    # fast: the same official rewriter and prompt, answering directly without its thinking pass (measured on the 5090,
    # 2026-10-21: text 2.7 s instead of 9.5 s, edit with a photo 3.6 s instead of 30 s, sign text kept verbatim).
    'qwen-image-2.1': ('Qwen-Image-2.1', "Qwen's own rewriters Qwen-Image-2.1-PE-T2I and PE-I2I (edit), on this computer", [
        (FAST, 'Fast: Qwen PE T2I / I2I, no thinking (about 7 s)', True, None, 'llama'),
        ('official', 'Qwen PE T2I / I2I with thinking (30 to 60 s)', True, None, 'llama')]),
    'krea-2': ('Krea 2', "Krea 2's own text encoder with the official ComfyUI Krea template", [
        (FAST, 'Fast: Qwen PE T2I, Krea template, no thinking (about 7 s)', False, None, 'llama'),
        ('official', 'Official: Krea 2 text encoder (up to about 45 s)', True, None, 'encoder'),
        ('qwen-pe', 'Qwen PE T2I, Krea template, with thinking', False, None, 'llama')]),
    'z-image': ('Z-Image', "The official Z-Image prompt (Tongyi-MAI Space pe.py); its official LLM runs only in the cloud, so Qwen PE T2I runs it here", [
        ('official', 'Official: Z-Image prompt, Qwen PE T2I', True, None, 'llama'),
        ('qwen-prompt', 'Qwen PE T2I, Qwen prompt', False, 'qwen-image-2.1-t2i', 'llama')]),
    'moss-image': ('MOSS Image', 'The official MOSS-Image prompt enhancer skill (caption mode), run by Qwen PE T2I', [
        ('official', 'Official: MOSS skill, Qwen PE T2I', True, None, 'llama'),
        ('moss-encoder', 'MOSS text encoder, ComfyUI template', False, 'moss-image-comfy-template', 'encoder')]),
    'video': ('Video (MiniMax H3, LTX 2.5)', 'No official enhancer exists; the cloud rewrites with prompts written for this app', []),
}
FIXED = {'flux3-klein': ('FLUX.3 Klein', "BFL's prompt model always runs inside FLUX.3 (official); Enhance prompt is not used.")}


def arch_of(family):
    family = family or ''
    for prefix in ('krea-2', 'z-image', 'moss-image', 'flux3-klein'):
        if family.startswith(prefix):
            return prefix
    return 'qwen-image-2.1'


def saved():
    try:
        return json.loads(PATH.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return {}


def cloud_model():
    """The Antigravity model chosen in Settings (the first cloud model when the old global choice was 'local')."""
    from . import antigravity
    chosen = antigravity.selected_model()
    return chosen if chosen != 'local' else next(m['id'] for m in antigravity.MODELS if not m.get('local'))


def options(arch):
    title, note, choices = ARCHS[arch]
    from . import antigravity
    cloud = next((m['name'] for m in antigravity.MODELS if m['id'] == cloud_model()), cloud_model())
    return [dict(id=choice, label=label, official=official) for choice, label, official, _, _ in choices] + \
           [dict(id=CLOUD, label=f'Cloud: {cloud}', official=not choices)]


def choice(arch):
    value = saved().get(arch)
    ids = [option['id'] for option in options(arch)]
    return value if value in ids else ids[0]


def route(family, images=(), redraw=False):
    """{arch, choice, label, official, mode: local|cloud, spec: prompt key, runner: llama|encoder, model}."""
    from .prompt_enhancer import spec_key
    arch = arch_of(family)
    if arch in FIXED:
        return dict(arch=arch, choice='official', label=FIXED[arch][1], official=True, mode='none', spec=None, runner=None, model=None)
    picked = choice(arch)
    official_key = spec_key(family, images, redraw)
    if picked == CLOUD:
        return dict(arch=arch, choice=CLOUD, label=next(o['label'] for o in options(arch) if o['id'] == CLOUD), official=not ARCHS[arch][2],
                    mode='cloud', spec=official_key, runner=None, model=cloud_model())
    _, label, official, key, runner = next(entry for entry in ARCHS[arch][2] if entry[0] == picked)
    return dict(arch=arch, choice=picked, label=label, official=official, mode='local', spec=key or official_key, runner=runner, model=None,
                fast=picked == FAST)


def listing():
    rows = [dict(arch=arch, title=title, official_note=note, choice=choice(arch), options=options(arch), fixed=False)
            for arch, (title, note, _) in ARCHS.items()]
    return rows + [dict(arch=arch, title=title, official_note=note, choice='official', options=[], fixed=True) for arch, (title, note) in FIXED.items()]


def save(arch, value):
    if arch not in ARCHS or value not in [option['id'] for option in options(arch)]:
        raise ValueError('Choose one of the listed enhancers.')
    data = saved()
    data[arch] = value
    write_json(PATH, data)

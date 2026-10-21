"""Read exact-version LoRA advice without borrowing another architecture's settings."""
import re
from collections import Counter


def workflow(version, text):
    """Keep an exact example tuple together, including unsupported sampler names."""
    from .lora_workflows import examples
    graphs = examples(version)
    rows = []
    samples = [dict(steps=g['steps'], cfgScale=g['cfg'], sampler=g['sampler'], scheduler=g['scheduler'], Model=g['base_model']) for g in graphs]
    if not samples:
        samples = [image.get('meta') or {} for image in version.get('images', [])]
    for meta in samples:
        steps, cfg = meta.get('steps'), meta.get('cfgScale')
        if isinstance(steps, int) and 1 <= steps <= 80 and isinstance(cfg, (int, float)) and 1 <= cfg <= 15:
            rows.append((steps, cfg, str(meta.get('sampler') or '').lower().replace(' ', '_'),
                         str(meta.get('scheduler') or '').lower(), str(meta.get('Model') or meta.get('model') or '')))
    if not rows:
        return None
    row, count = Counter(rows).most_common(1)[0]
    steps, cfg, sampler, scheduler, base = row
    from .civitai import SAMPLERS, SCHEDULERS, SAMPLER_NAMES
    sampler = SAMPLER_NAMES.get(sampler.replace('_', ' '), sampler)
    alternative = next((s for s in SAMPLERS if re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', text, re.I)), None)
    family = None
    if re.search(r'qwen(?:[_ -]?image)?[_ -]?2[._]1', base, re.I):
        family = 'qwen-image-2.1' + ('-turbo' if re.search(r'turbo|viggle', base, re.I) else '')
    elif re.search(r'krea', base, re.I):
        if re.search(r'turbo', base, re.I): family = 'krea-2-turbo'
        elif re.search(r'raw', base, re.I): family = 'krea-2'
    return {'steps': steps, 'cfg': cfg, 'sampler': sampler, 'scheduler': scheduler,
            'base_model': base, 'family': family, 'matching_examples': count, 'total_examples': len(rows),
            'evidence': 'connected-workflow' if graphs else 'image-summary',
            'supported_sampler': sampler if sampler in SAMPLERS else None,
            'supported_scheduler': scheduler if scheduler in SCHEDULERS else None,
            'author_sampler_alternative': alternative if sampler not in SAMPLERS else None,
            'examples': [{k: value for k, value in g.items() if k not in ('positive_prompt', 'negative_prompt')}
                         for g in graphs]}


def architecture(text):
    text = re.sub(r'[\s_.-]', '', text).casefold()
    for pattern, name in ((r'qwen(?:image)?2[1Ⅰ]|qw21', 'qwen21'),
                          (r'qwen', 'qwen1'), (r'krea2', 'krea2'),
                          (r'zimage', 'zimage'), (r'flux3', 'flux3'),
                          (r'flux2', 'flux2'), (r'flux', 'flux1'),
                          (r'ltxv?25', 'ltx25'), (r'minimax|h3', 'h3'),
                          (r'anima', 'anima'), (r'ideogram', 'ideogram')):
        if re.search(pattern, text):
            return name
    return None


def scoped_text(text, base_model):
    """Select short architecture headings and common sections, not prose mentions."""
    wanted = architecture(base_model)
    if not wanted:
        return text
    selected, current = [], None
    for line in text.splitlines():
        heading = line.strip(' \t-*#•👇:')
        heading = re.sub(r'^[^\w<]+', '', heading)
        heading = re.sub(r'^(?:info for|new:|for)\s+', '', heading, flags=re.I)
        found = architecture(heading)
        if found and len(heading) < 85 and re.match(
                r'^(?:qwen|qw21|krea|z[ -]?image|flux|minimax|h3|ltx|anima|ideogram)', heading, re.I):
            # "Qwen works best ..." is a sentence, not a section boundary.
            if not re.search(r'\b(?:works|supports|uses|is|can|was|needs)\b', heading, re.I):
                current = found
        elif re.fullmatch(r'(?:suggestions|general (?:settings|usage|tips)|common (?:settings|usage|tips))', heading, re.I):
            current = None
        if current is None or current == wanted:
            selected.append(line)
    return '\n'.join(selected)


def strength(text):
    text = text.replace('**', '').replace('`', '')
    number = r'(\d+(?:\.\d+)?)'
    # An explicit best value wins over the maximum of a supported range.
    patterns = [number + r'\s+(?:is\s+)?(?:the\s+)?best\b',
                r'(?:start|best results|recommended)\s+(?:with|at)\s+(?:strength\s*)?' + number,
                number + r'\s+(?:LoRA\s+)?(?:strength|weight)\b',
                r'\b(?:strength|weight)\s*(?:is|of|at|:|=)?\s*' + number]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = float(match[1])
            if 0.1 <= value <= 2:
                return value
    return None


def profile(version, model):
    from .civitai import _text, SITE
    from .lora_prompts import description_words, contains_word
    from .civitai_settings import version_section
    base = version.get('baseModel', '')
    exact = scoped_text(_text(version.get('description')), base)
    page = _text(model.get('description'))
    selected = scoped_text(version_section(page, version), base)
    explicit = [w.strip() for w in version.get('trainedWords') or [] if isinstance(w, str) and w.strip()]
    words = explicit or description_words(exact) or description_words(selected)
    source = 'version' if explicit else 'description' if words else None
    if not words:
        # Some old shared descriptions declare a trigger under an older release.
        # Accept it only when the exact selected version also uses that declared
        # word in at least two example prompts. Never infer new words from captions.
        prompts = [(im.get('meta') or {}).get('prompt', '') for im in version.get('images', [])]
        words = [word for word in description_words(page)
                 if sum(contains_word(p, word) for p in prompts if isinstance(p, str)) >= 2]
        if words:
            source = 'description-and-version-examples'
    value = strength(exact)
    if value is None:
        value = strength(selected)
    return {'base_model': base, 'version_name': version.get('name', ''),
            'source': f"{SITE}/models/{model.get('id')}?modelVersionId={version.get('id')}",
            'strength': value, 'trigger_source': source, 'trained_words': words,
            'workflow': workflow(version, exact + '\n' + selected),
            'description': (exact + '\n' + selected).strip()}


FAMILY_LABELS = {'qwen-image-2.1': 'Qwen-Image-2.1 Standard', 'qwen-image-2.1-turbo': 'Qwen-Image-2.1 Turbo',
                 'krea-2': 'Krea 2 Raw', 'krea-2-turbo': 'Krea 2 Turbo', 'z-image': 'Z-Image Base', 'z-image-turbo': 'Z-Image Turbo',
                 'flux3-klein': 'FLUX.3 Klein', 'flux3-klein-base': 'FLUX.3 Klein Base',
                 'moss-image': 'MOSS Image SFT', 'moss-image-base': 'MOSS Image Base', 'moss-image-turbo': 'MOSS Image Turbo'}
HEADER_ARCH = {'qwen_image_2': 'Qwen-Image-2.1', 'krea2': 'Krea 2', 'zimage': 'Z-Image', 'flux3': 'FLUX.3 Klein',
               'minimax_h3': 'MiniMax H3', 'ltx2': 'LTX-2', 'ltx25': 'LTX 2.5'}


def _stem(name):
    name = re.split(r'[\\/]', name or '')[-1].casefold()
    return re.sub(r'^\d+_', '', re.sub(r'\.(safetensors|ckpt|pt)$', '', name)).replace(' ', '_')


def example_strength(advice, file_name):
    """The strength the author's own example graphs use for this file (the most common value).
    A graph with one LoRA is that LoRA; with several, the file name must match."""
    graphs = ((advice or {}).get('workflow') or {}).get('examples') or []
    own = _stem(file_name)
    named, single = [], []
    for graph in graphs:
        loras = [l for l in graph.get('loras') or [] if isinstance(l.get('strength'), (int, float)) and 0.05 <= l['strength'] <= 2]
        match = [l for l in loras if own and (own in _stem(l.get('file')) or _stem(l.get('file')) in own)]
        named += [round(float(l['strength']), 2) for l in match[:1]]
        if len(loras) == 1:
            single.append(round(float(loras[0]['strength']), 2))
    values = named or single
    return Counter(values).most_common(1)[0][0] if values else None


def author_vae(advice):
    from .civitai import _known_vae
    graphs = ((advice or {}).get('workflow') or {}).get('examples') or []
    found = [vae for g in graphs for vae in (_known_vae(re.split(r'[\\/]', name)[-1]) for name in g.get('vae') or []) if vae]
    return Counter(found).most_common(1)[0][0] if found else None


def accelerator(keys, title, file_name):
    """Speed (distillation) LoRAs train the timestep embedding or modulation layers; style LoRAs do not."""
    if not any('time_text_embed' in k or '.modulation.' in k or k.startswith('modulation.') for k in keys):
        return None
    steps = re.search(r'(\d+)[ _-]?steps?', f'{title} {file_name}', re.I)
    return {'steps': int(steps[1]) if steps else None}


def trained_on(advice, metadata, keys, title, file_name, families, made_for=None):
    """The base this LoRA was made for, from the strongest evidence: the trainer of a local LoRA, the base
    file in the author's own example graphs, an explicit Turbo name, then the architecture in the file header."""
    family, evidence = None, None
    if made_for:
        from .training import ARCHS
        family = next((key for key, info in ARCHS.items() if info['label'] == made_for), None)
        evidence = 'trained in this app' if family else None
    wf = (advice or {}).get('workflow') or {}
    if not family and wf.get('family') in families:
        family, evidence = wf['family'], "author's example workflow"
    # "Turbo" in a Qwen LoRA name, not the name of another model ("Z-Image-Turbo + Krea 2").
    named = re.sub(r'z[ _-]?image[ _-]?turbo|krea[ _-]?2?[ _-]?turbo', '', f'{title} {file_name}', flags=re.I)
    if not family and 'qwen-image-2.1-turbo' in families and re.search(r'(?<![a-z])turbo(?![a-z])', named, re.I) \
            and not accelerator(keys, title, file_name):
        family, evidence = next((f for f in families if f.endswith('turbo')), None), 'name says Turbo'
    arch = HEADER_ARCH.get((metadata or {}).get('ss_base_model_version', ''))
    label = FAMILY_LABELS.get(family) or arch
    if not label:
        return None
    settings = None
    if family and wf.get('family') == family:
        settings = {k: v for k, v in {'steps': wf.get('steps'), 'cfg': wf.get('cfg'), 'sampler': wf.get('supported_sampler'),
                                      'scheduler': wf.get('supported_scheduler'), 'vae_id': author_vae(advice)}.items() if v}
    return {'family': family, 'label': label, 'evidence': evidence or 'file header', 'settings': settings or None}


def _tensors(path):
    """The safetensors JSON header only: names, dtypes and shapes, no tensor data."""
    import json, struct
    with open(path, 'rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        found = json.loads(handle.read(length))
    return found.pop('__metadata__', None) or {}, found


def header(path):
    metadata, tensors = _tensors(path)
    return metadata, list(tensors)


def _ranks(path):
    return [t['shape'][0] for k, t in _tensors(path)[1].items() if re.search(r'(?:lora_A|lora_down)\.(?:default\.)?weight$', k)]


def plain_fp8(path):
    """FP8 weights with no scales are cast to BF16 on load; a LoRA on them loses more to rounding."""
    tensors = _tensors(path)[1]
    fp8 = sum(1 for k, t in tensors.items() if k.endswith('.weight') and str(t.get('dtype', '')).startswith('F8'))
    return fp8 > 0 and not any(k.endswith(('scale_weight', 'weight_scale', '.comfy_quant')) for k in tensors)


def describe(entry, path):
    """Fill trained_on, accelerator and the author's example strength for one installed LoRA entry."""
    metadata, keys = header(path)
    advice = entry.get('lora_author') if isinstance(entry.get('lora_author'), dict) else None
    file_name = (entry.get('civitai') or {}).get('file_name') or entry['files'][0]['path']
    fast = accelerator(keys, entry['title'], file_name)
    entry['trained_on'] = trained_on(advice, metadata, keys, entry['title'], file_name, entry.get('families') or [], entry.get('made_for'))
    if fast:
        entry['accelerator'] = fast
        if fast['steps'] and entry['trained_on']:
            entry['trained_on']['settings'] = {'steps': fast['steps'], 'cfg': 1}
    else:
        entry.pop('accelerator', None)
    if entry.get('source') == 'trained':
        rank = min(_ranks(path), default=None)
        if rank is not None and rank <= 4 and re.search(r'\b(?:check|test)\b', entry['title'], re.I):
            entry['test_file'] = f'Test file (rank {rank}), made to check training. Not for real use.'
    value = example_strength(advice, file_name)
    if value is not None:
        entry['default_strength'] = value
        if advice is not None:
            advice['strength'] = value
    return entry

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

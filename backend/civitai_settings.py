"""Read settings for one published version without mixing standard and Turbo sections."""
import re

FIELDS = ('steps', 'cfg', 'sampler', 'scheduler', 'video_shift', 'audio_shift', 'scheduler_video_shift', 'scheduler_audio_shift')


def partition_hint(text):
    ref=bool(re.search(r'ref2v(?:a)?|\br2v\b|\breference (?:model|deltas)\b|\bref/t2va\b|\bdoes t2va and reference\b',text,re.I))
    fl=bool(re.search(r'fl2v(?:a)?',text,re.I))
    return ref if ref != fl else None


def partition_section(text, reference):
    """Do not apply FL2VA instructions to a Ref2VA file on the same page.

    A whole page that only names one partition belongs to that partition.
    Mixed pages can have common settings followed by partition headings.
    Comparisons/ancestry in prose do not establish a new heading.
    """
    hint=partition_hint(text)
    if hint is not None and hint != reference:
        return ''
    lines=text.splitlines()
    markers=[]
    for index,line in enumerate(lines):
        heading=line.strip(' \t\u2022*-:#')
        kind=partition_hint(heading)
        if kind is not None and len(heading)<90 and re.match(
                r'^(?:(?:recommended|inference|sampling|settings|for|using)\s+)*(?:ref2v(?:a)?|fl2v(?:a)?|r2v)\b',heading,re.I):
            markers.append((index,kind))
    if not markers:
        return text
    selected=lines[:markers[0][0]]
    for n,(start,kind) in enumerate(markers):
        if kind==reference:
            selected+=lines[start:markers[n+1][0] if n+1<len(markers) else len(lines)]
    return '\n'.join(selected)


def h3_capabilities(model,version,file=None):
    from .civitai import _text
    text=_text(version.get('description'))+'\n'+version_section(_text(model.get('description')),version)
    # A page can list both partitions. A concrete file/version wins over its page.
    reference=None
    for name in ((file or {}).get('name',''),version.get('name',''),model.get('name','')):
        ref=bool(re.search(r'ref2v(?:a)?|\br2v\b',name,re.I))
        fl=bool(re.search(r'fl2v(?:a)?',name,re.I))
        if ref != fl:
            reference=ref
            break
    if re.search(r'not a (?:native )?ref2v|fl2va.based hybrid',text,re.I):
        reference=False
    if reference is None:
        reference=bool(re.search(r'\breference model\b|\bref2va\b.*compatible|\bdoes t2va and reference\b|\bref/t2va hybrid\b|\breference prompting mode\b',text,re.I))
    if not reference:
        return False,['text','image']
    text_mode=bool(re.search(r'reference model with t2v ability|does t2va and reference|\bt2v(?:a)?\s+(?:ability|support)',text,re.I))
    # Ref2VA references are not FL2VA keyframes. Mentioning both in a comparison
    # is not evidence that this checkpoint supports first/last-frame conditioning.
    return True,['text'] if text_mode else []


def version_section(text, version):
    """Select release headings such as beta 6 / beta 5 before parsing settings."""
    markers=list(re.finditer(r'(?im)^\s*((?:beta|version|v)\s*[_ -]?\d+(?:\.\d+)*)\s*:\s*$',text))
    if not markers:
        return text
    normalize=lambda value: re.sub(r'[^a-z0-9.]','',value.lower())
    names={normalize(match[0]) for match in re.finditer(r'(?:beta|version|v)\s*[_ -]?\d+(?:\.\d+)*',version.get('name',''),re.I)}
    common=text[:markers[0].start()]
    for index,marker in enumerate(markers):
        if normalize(marker[1]) in names:
            return common+text[marker.end():markers[index+1].start() if index+1<len(markers) else len(text)]
    return common


def video_variant(model, version, file=None):
    from .civitai import _text
    name = ((file or {}).get('name','')+' '+version.get('name','')).replace('_',' ').replace('-',' ')
    if re.search(r'non[- ]?(?:distill|turbo)|\b(?:base|standard)\b', name, re.I):
        return False
    if re.search(r'turbo|\b[48][ -]?steps?\b', name, re.I):
        return True
    if file and re.search(r'hybrid',file.get('name',''),re.I) and any(
        re.search(r'turbo',other.get('name',''),re.I) for other in version.get('files',[])):
        return False
    text = _text(version.get('description'))+'\n'+version_section(_text(model.get('description')),version)
    if re.search(r'turbo[^.\n]{0,50}(?:integrated|baked in)|no non-turbo version',text,re.I):
        return True
    if re.search(r'(?:turbo|distill\w*)[^.\n]{0,45}(?:merged|fused|blend)|(?:merged|fused)[^.\n]{0,45}turbo|\b[48][ -]?steps?\s+turbo', text, re.I):
        return True
    # Some pages name the ordinary version "Hybrid v4" and the matching one "Hybrid v4 Turbo".
    normalized = lambda value: re.sub(r'\W+', '', re.sub(r'\bturbo\b', '', value, flags=re.I)).lower()
    if any(re.search(r'\bturbo\b', sibling.get('name',''), re.I) and normalized(sibling['name']) == normalized(name)
           for sibling in model.get('modelVersions', [])):
        return False
    if re.search(r'turbo|\b[48][ -]?steps?\b',model.get('name',''),re.I):
        return True
    return None


def variant_section(text, turbo):
    """Keep common text plus the selected type's section; discard the other type."""
    lines = text.splitlines()
    markers = []
    for i, line in enumerate(lines):
        heading = line.strip(' \t•*-:')
        if len(heading) > 65:
            continue
        if re.match(r'^(?:non[- ]?distilled|standard|base)(?:\s|$)', heading, re.I):
            markers.append((i, False))
        elif re.match(r'^(?:turbo(?:\s*\(distilled\))?|distilled)(?:\s|$)', heading, re.I):
            markers.append((i, True))
    if not markers:
        return text
    common = lines[:markers[0][0]]
    if turbo is None:
        return '\n'.join(common)
    for index, (start, kind) in enumerate(markers):
        if kind == turbo:
            end = markers[index+1][0] if index+1 < len(markers) else len(lines)
            return '\n'.join(common+lines[start:end])
    return '\n'.join(common)


def _video_values(text):
    from .civitai import author_settings, find_sampler, find_scheduler
    # Release names (beta 5, beta6) are not the Beta noise scheduler.
    text=re.sub(r'\bbeta[\s_]*\d+(?:/\d+)*\b','',text,flags=re.I)
    # Benchmarks and training/quantization notes describe observations, not
    # settings to use. Preserve explicit recommendations in the other sentences.
    sentences=re.split(r'(?<=[.!?])\s+|\n+',text)
    text='\n'.join(line for line in sentences if not re.search(
        r'\b(?:benchmark|calibration|quantization error|training steps|warmup|PDD)\b|'
        r'\bsame (?:scene|seed)\b|\bi (?:rendered|tested)\b|\bgoing from\b',line,re.I))
    found = author_settings(text)
    # Sampler and shifts often sit on separate lines from Steps.
    for line in re.split(r'(?<=[.!?])\s+|\n+',text):
        if re.search(r'optional|upscal|with added|example|if you', line, re.I):
            continue
        sampler = find_sampler(line)
        scheduler = find_scheduler(line)
        if sampler:
            found.setdefault('sampler', sampler)
        if scheduler and (sampler or re.search(r'\bschedul(?:e|er)\b',line,re.I)):
            found.setdefault('scheduler', scheduler)
        for kind in ('video', 'audio'):
            match = re.search(rf'(?:shift\s*{kind}|{kind}\s*(?:sigma\s*)?shift)\s*[:=]?\s*(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?', line, re.I)
            if match and 0 < float(match[1]) <= 30:
                found.setdefault(kind+'_shift', float(match[1]))
                if match[2]:
                    found.setdefault(kind+'_shift_range', [float(match[1]), float(match[2])])
        pair = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s+shift', line, re.I)
        if pair and all(0 < float(pair[i]) <= 30 for i in (1,2)):
            found.setdefault('video_shift', float(pair[1]))
            found.setdefault('audio_shift', float(pair[2]))
        pair = re.search(r'\bshift\s*(?:[-:]\s*)?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',line,re.I)
        if pair:
            found.setdefault('video_shift',float(pair[1]));found.setdefault('audio_shift',float(pair[2]))
        pair = re.search(r'\bshift\s*[-:]?\s*(\d+(?:\.\d+)?)\s*video\s*/\s*(\d+(?:\.\d+)?)[+]?\s*audio',line,re.I)
        if pair:
            found.setdefault('video_shift',float(pair[1]));found.setdefault('audio_shift',float(pair[2]))
        if re.search(r'\bno shift\b',line,re.I):
            found.setdefault('video_shift',1.0);found.setdefault('audio_shift',1.0)
    return found


def version_settings(model, version, file=None):
    from .civitai import BASE_MODELS, _text, author_settings, example_settings, step_settings
    info = BASE_MODELS.get(version.get('baseModel'), {})
    own, page = _text(version.get('description')), _text(model.get('description'))
    if info.get('area') != 'video':
        found = author_settings(own, page, arch=info.get('arch'))
        origin = 'author' if found else 'examples'
        found = found or example_settings(version)
        return {**found, 'settings_origin': origin} if found else {}
    turbo = video_variant(model, version, file)
    reference,_=h3_capabilities(model,version,file)
    selected_page = partition_section(variant_section(version_section(page,version), turbo),reference)
    own_values = _video_values(partition_section(variant_section(own, turbo),reference))
    page_values = _video_values(selected_page)
    workflows=[w for w in version.get('_workflow_settings',[]) if w.get('reference')==reference
               and video_variant(model,version,{'name':w.get('checkpoint','')}) in (None,turbo)]
    # Several attached workflows are usable only when their sampling agrees.
    workflow=workflows[0] if workflows and all(all(w.get(k)==workflows[0].get(k) for k in FIELDS) for w in workflows) else {}
    workflow_values={k:workflow[k] for k in FIELDS if k in workflow}
    named = step_settings((file or {}).get('name','').replace('_',' ')) or step_settings(version.get('name',''))
    title_values=step_settings(model.get('name',''))
    if title_values and not own_values.get('steps') and not page_values.get('steps') and not named:
        page_values.update(title_values)
    if re.search(r'(?:sampling[^.\n]{0,120}FastH3|FastH3[^.\n]{0,80}templates)',own+'\n'+selected_page,re.I):
        page_values.setdefault('sampler','res_multistep')
        page_values.setdefault('scheduler','simple')
    # Explicit version/title values outrank generic model recommendations.
    found = {**page_values, **workflow_values, **own_values, **named}
    if 'steps' in own_values or 'steps' in named:
        for key in ('steps_range','step_options'):
            if key not in own_values and key not in named:
                found.pop(key,None)
    origins = {key: 'version' if key in own_values or key in named else 'attached_workflow' if key in workflow_values else 'model' for key in FIELDS if key in found}
    if workflow_values:
        found['workflow_source']=workflow['source']
    # Examples on a mixed FL/Ref version cannot identify the selected file.
    partitions={partition_hint(f.get('name','')) for f in version.get('files',[])}-{None}
    examples = example_settings(version) if len(partitions)<2 else {}
    for key in FIELDS:
        if key not in found and key in examples:
            found[key] = examples[key]
            origins[key] = 'examples'
    if found:
        found.update(settings_origin='author' if own_values or page_values or named else 'examples', field_sources=origins)
    return found


def h3_defaults(model, version, template, url, file=None):
    from .civitai import _text
    settings = version_settings(model, version, file)
    merged = video_variant(model, version, file) is True
    reference = template.get('family') == 'minimax-h3-ref2va'
    defaults = dict(template['generation_defaults'])
    if reference:
        baseline = dict(defaults['profiles']['standard'])
    else:
        baseline = {'steps':20, 'sampler':'res_multistep', 'scheduler':'simple',
                    'guidance':1.0, 'video_shift':12.0, 'audio_shift':3.0}
    # A named Turbo without a stated step count uses the documented H3 Turbo baseline,
    # clearly marked as a default rather than an author recommendation.
    if merged:
        baseline.update(steps=8, sampler='euler')
    if 'FastH3' in ((version.get('description') or '')+version_section(_text(model.get('description')),version)):
        baseline.update(sampler='res_multistep',scheduler='simple')
    applied = {**baseline, **{key:settings[key] for key in FIELDS if key in settings}, 'cfg':1.0, 'guidance':1.0}
    applied.update(source=url, embedded_turbo=merged)
    sources = {key:settings.get('field_sources',{}).get(key,'workflow') for key in FIELDS}
    # H3's released model uses CFG 1. Its video engine has no negative conditioning pass.
    sources['cfg'] = 'workflow'
    applied['field_sources'] = sources
    if settings.get('workflow_source'):
        applied['workflow_source']=settings['workflow_source']
    for key in ('steps_range','step_options','video_shift_range','audio_shift_range'):
        if key in settings:
            applied[key] = settings[key]
    defaults.update(applied, author_settings=settings or None,
                    note='Turbo is included in this checkpoint.' if merged else 'Uses this version\'s settings where stated.')
    if reference:
        if merged:
            defaults.update(quality='turbo', profiles={'turbo':dict(applied)})
        else:
            profiles = {key:dict(value) for key,value in defaults['profiles'].items()}
            profiles['standard'] = dict(applied)
            defaults.update(quality='standard', profiles=profiles)
    return defaults


def ltx_defaults(model,version,url,file=None):
    from .ltx_settings import SIGMAS,REFINE_SIGMAS
    from .civitai import ltx_variant, _text
    settings = version_settings(model,version,file)
    variant = ltx_variant(model.get('name',''),version.get('name',''),settings,_text(version.get('description'))+'\n'+_text(model.get('description')),file)
    strength = 0.5
    strength_source = 'workflow'
    if variant == 'dev':
        for text in (_text(version.get('description')),version_section(_text(model.get('description')),version)):
            match = re.search(r'\bdistill(?:ed|ation)?\s+lora\s*(?:strength|weight|at|@|:|=)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)\b', text, re.I)
            if match:
                strength = float(match[1]);strength_source = 'author';break
    return dict(steps=8,fixed_steps=True,cfg=1.0,sampler='euler_ancestral',scheduler='manual',
                sigmas=SIGMAS,refine_sigmas=REFINE_SIGMAS,source=url,
                author_settings=settings,variant=variant,
                distill_lora_strength=strength if variant=='dev' else None,distill_lora_strength_source=strength_source if variant=='dev' else None,
                note=('Uses the distilled LoRA with this dev checkpoint. ' if variant=='dev' else '')+'Distilled workflow: 8 steps, CFG 1. Spatial refinement adds 3 steps.')

"""Inspect published Comfy API graphs as data. Never execute their nodes."""
import json

MAX_BYTES = 2 * 1024 * 1024


def inspect_graph(raw):
    if isinstance(raw, str):
        if len(raw.encode('utf-8')) > MAX_BYTES:
            return []
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, dict):
        return []
    graph = raw.get('prompt', raw)
    if not isinstance(graph, dict) or len(graph) > 1500:
        return []
    graph = {str(k): n for k, n in graph.items() if isinstance(n, dict) and isinstance(n.get('inputs'), dict)}

    def node(value):
        return graph.get(str(value[0]), {}) if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) else {}

    def ancestors(start):
        found, pending = {}, [start]
        while pending:
            current = pending.pop()
            if not current or id(current) in found:
                continue
            found[id(current)] = current
            pending.extend(node(v) for v in current['inputs'].values())
        return list(found.values())

    outputs = [n for n in graph.values() if n.get('class_type', '').startswith(('SaveImage', 'PreviewImage'))]
    reachable = {id(n) for out in outputs for n in ancestors(out)}
    rows = []
    for sampler in graph.values():
        inp = sampler['inputs']
        if id(sampler) not in reachable or not all(k in inp for k in ('steps', 'cfg', 'model', 'positive', 'negative', 'latent_image')):
            continue
        if not isinstance(inp['steps'], int) or isinstance(inp['steps'], bool) or not isinstance(inp['cfg'], (int, float)):
            continue
        chain, seen, stack, unknown = [], set(), [], []
        current = node(inp['model'])
        while current and id(current) not in seen:
            seen.add(id(current)); chain.append(current)
            kind, values = current.get('class_type', ''), current['inputs']
            entries = []
            if kind in ('LoraLoader', 'LoraLoaderModelOnly'):
                entries = [{'file': values.get('lora_name'), 'strength': values.get('strength_model'), 'clip_strength': values.get('strength_clip', 0)}]
            elif kind == 'Power Lora Loader (rgthree)':
                entries = [{'file': v.get('lora'), 'strength': v.get('strength'), 'clip_strength': v.get('strengthTwo', v.get('strength'))}
                           for v in values.values() if isinstance(v, dict) and v.get('on') is True and v.get('lora')]
            elif kind == 'VisualLoRALoader':
                try:
                    data = json.loads(values.get('lora_data', '[]'))
                    entries = [{'file': v.get('name'), 'strength': v.get('strength_model'), 'clip_strength': v.get('strength_clip')}
                               for v in data if isinstance(v, dict) and v.get('enabled') is True]
                except (ValueError, TypeError):
                    unknown.append(kind)
            elif 'lora' in kind.lower():
                unknown.append(kind)
            # Follow the actual model connection. A clip input is not a model adapter.
            for entry in entries:
                entry['loader'] = kind
                entry['clip_connected'] = bool(node(values.get('clip')))
            stack[0:0] = entries
            current = node(values.get('model'))
        base = next((n['inputs'].get('unet_name') or n['inputs'].get('ckpt_name') for n in chain
                     if n['inputs'].get('unet_name') or n['inputs'].get('ckpt_name')), None)
        positive = node(inp['positive']); negative = node(inp['negative'])
        inputs = positive.get('inputs', {})
        negative_text = '' if negative.get('class_type') == 'ConditioningZeroOut' else negative.get('inputs', {}).get('negative_prompt', negative.get('inputs', {}).get('text'))
        upstream = ancestors(sampler)
        row = dict(steps=inp['steps'], cfg=inp['cfg'], sampler=inp.get('sampler_name'), scheduler=inp.get('scheduler'),
                   denoise=inp.get('denoise'), base_model=base, loras=stack, unknown_lora_loaders=unknown,
                   positive_prompt=inputs.get('prompt', inputs.get('text')), negative_prompt=negative_text,
                   reference_images=sum(1 for k in inputs if k.startswith(('image', 'images.')) and node(inputs[k])),
                   reference_resolution=inputs.get('resolution'), sampler_node=sampler.get('class_type'),
                   custom_sigmas=bool(node(inp.get('sigmas'))),
                   encoders=[n['inputs'].get('clip_name') for n in upstream if n.get('class_type') == 'CLIPLoader'])
        row['sizes'] = [n['inputs'] for n in upstream if n.get('class_type') in ('ResolutionSelector', 'FluxResolutionNode')]
        row['vae'] = [n['inputs'].get('vae_name') for out in outputs for n in ancestors(out) if n.get('class_type') == 'VAELoader']
        row['vae'] = list(dict.fromkeys(row['vae']))
        rows.append(row)
    return rows


def examples(version):
    result = []
    for index, image in enumerate((version.get('images') or [])[:30]):
        if not isinstance(image, dict):
            continue
        meta = image.get('meta') or {}
        for row in inspect_graph(meta.get('comfy')):
            row['example_index'] = index
            row['image_url'] = image.get('url')
            row['prompt_metadata_differs'] = isinstance(meta.get('prompt'), str) and meta['prompt'] != row['positive_prompt']
            result.append(row)
    return result

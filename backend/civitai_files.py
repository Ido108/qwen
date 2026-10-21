"""Exact file identity and settings scope within a Civitai release."""
import re
from pathlib import PurePosixPath


def download_key(version_id, file_id):
    return f'{int(version_id)}:{int(file_id)}'


def file_section(text, file):
    """Keep common instructions and sections explicitly naming this file/precision."""
    if not file:
        return text
    from .civitai import _file_precision
    precision = _file_precision(file).lower()
    name = file.get('name', '').lower()
    selected = {precision, name, PurePosixPath(name).stem}
    if name.endswith('.gguf'):
        selected.add('gguf')
    marker = re.compile(
        r'^(?:(?:recommended\s+)?settings\s+for\s+)?'
        r'((?:nvfp4|mxfp8|fp8_scaled|bf16|fp16|fp32|fp8|int[2-8]|w[2-8]a\d+|gguf|q[2-8](?:_[a-z0-9]+)*)(?:\s*[/,&+]\s*(?:nvfp4|mxfp8|bf16|fp16|fp32|fp8|int[2-8]|gguf))*)'
        r'(?:\s+(?:files?|settings|versions?|models?|quantization))?\s*(?::|$)', re.I)
    common = re.compile(r'^(?:all files|all versions|common|general|shared)(?:\s+settings)?\s*:?\s*$', re.I)
    keep, specific = True, False
    shared_lines, file_lines = [], []
    for line in text.splitlines():
        heading = line.strip(' \t•*-:#')
        match = marker.match(heading)
        filename = re.match(r'^([^\s:]+\.(?:safetensors|gguf))(?:\s*:|$)', heading, re.I)
        if common.fullmatch(heading):
            keep, specific = True, False
        elif match:
            keep = bool(selected.intersection(re.split(r'\s*[/,&+]\s*', match[1].lower())))
            specific = True
        elif filename:
            keep = filename[1].lower() == name
            specific = True
        if keep:
            (file_lines if specific else shared_lines).append(line)
    # Explicit file instructions outrank common recommendations.
    return '\n'.join(file_lines + shared_lines)


def image_file_type(model, version, file):
    """An explicit per-file type wins; quantization alone does not imply Turbo."""
    from .civitai import detect_turbo
    name = (file or {}).get('name', '').replace('_', ' ')
    if version.get('baseModel', '').startswith('Flux.3'):
        if re.search(r'\b(?:distilled|sd)\b', name, re.I):
            return True
        if re.search(r'\bbase\b', name, re.I):
            return False
    return detect_turbo(name, {}) if name else None


def entry_location(catalog, version_id, file, lora):
    """Keep old IDs/paths for exact matches; new files always get distinct IDs/paths."""
    kind = 'lora' if lora else 'model'
    version_id, file_id = int(version_id), int(file['id'])
    expected = (file.get('hashes') or {}).get('SHA256', '').lower()
    for item in catalog:
        saved = item.get('civitai') or {}
        if saved.get('version_id') != version_id:
            continue
        exact = saved.get('file_id') == file_id
        if not exact and saved.get('file_id') is None and expected:
            exact = any((f.get('sha256') or '').lower() == expected for f in item.get('files', []))
        if exact and item.get('files'):
            return item['id'], item['directory'], item['files'][0]['path']
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', file['name'])
    return (f'civitai-{kind}-{version_id}-file-{file_id}',
            f"civitai/{'loras' if lora else 'models'}", f'{version_id}_{file_id}_{name}')


def saved_file(entry, files):
    """Use saved IDs first, then hashes; accept old filenames only when unique."""
    exact = next((f for f in files if f['id'] == entry['civitai'].get('file_id')), None)
    if exact:
        return exact
    hashes = {f['sha256'].lower() for f in entry['files'] if f.get('sha256')}
    matches = [f for f in files if (f.get('hashes', {}).get('SHA256') or '').lower() in hashes]
    if len(matches) == 1:
        return matches[0]
    matches = [f for f in files if any(
        s['path'].endswith(re.sub(r'[^A-Za-z0-9._-]+', '_', f['name'])) for s in entry['files'])]
    return matches[0] if len(matches) == 1 else None


def resume_path(root, relative, version_id, file, siblings):
    """Reuse an unambiguous old partial download without moving or deleting it."""
    clean = lambda name: re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    name = clean(file['name'])
    if sum(clean(f['name']) == name for f in siblings) != 1:
        return relative
    target = root / relative
    legacy = root / f'{version_id}_{name}'
    suffixes = ['', '.part', *[f'.part{i}' for i in range(8)]]
    if not any(target.with_name(target.name + suffix).exists() for suffix in suffixes) and any(
            legacy.with_name(legacy.name + suffix).exists() for suffix in suffixes[1:]):
        return legacy.name
    return relative

from pathlib import Path
import json
import shutil
import sys

from PIL import Image

root = Path(__file__).resolve().parent.parent
source = root / 'reference' / 'space' / 'examples'
destination = root / 'public' / 'examples'
destination.mkdir(parents=True, exist_ok=True)
for path in source.iterdir():
    if path.suffix.lower() in ('.png', '.webp', '.jpg', '.jpeg'):
        shutil.copy2(path, destination / path.name)
cases = json.loads((source / 'cases.json').read_text(encoding='utf-8'))
extra = json.loads((source / 'generated_references.json').read_text(encoding='utf-8'))
examples = []
for i, case in enumerate(cases + extra):
    examples.append({
        'id': str(i), 'title': case['title'], 'title_en': case.get('title_en', case['title']),
        'prompt': case['prompt'], 'prompt_extend': case.get('prompt_extend', True),
        'inputs': case.get('inputs', []), 'outputs': case.get('outputs', []),
        'category': 'edit' if case.get('inputs') else 'text',
    })
    for name in case.get('inputs', []) + case.get('outputs', []):
        if not (destination / name).is_file():
            raise RuntimeError(f'Missing example: {name}')
(root / 'public' / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
branding = root / 'public' / 'branding'
with Image.open(branding / 'icon-source.png') as source_icon:
    icon = source_icon.convert('RGBA')
    if icon.getchannel('A').getextrema()[0] == 255:
        raise RuntimeError('Icon has no transparent background')
    for size in (16, 32, 48, 64, 128, 192, 256, 512):
        icon.resize((size, size), Image.Resampling.LANCZOS).save(branding / f'icon-{size}.png')
    icon.save(branding / 'app.ico', sizes=[(size,size) for size in (16,24,32,48,64,128,256)])
print(f'Prepared {len(examples)} official examples and 8 icon sizes.')

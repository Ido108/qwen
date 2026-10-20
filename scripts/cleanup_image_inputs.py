"""Inspect old, unused per-job PNG copies. --apply removes verified copies only."""
import argparse
import json
from pathlib import Path
import re

from PIL import Image

APP_DIR = Path(__file__).resolve().parent.parent


def candidates(data_dir, examples_dir, only_folder=None):
    root = (Path(data_dir) / 'jobs').resolve(strict=True)
    uploads = (Path(data_dir) / 'uploads').resolve(strict=True)
    examples = Path(examples_dir).resolve()
    for folder in [Path(only_folder)] if only_folder else root.iterdir():
        if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != root:
            continue
        try:
            status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
            request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if status.get('status') not in ('succeeded', 'failed', 'cancelled') or request.get('kind', 'image') != 'image':
            continue
        sources = []
        for image_id in request.get('images', []):
            source_root = examples if image_id.startswith('example:') else uploads
            name = image_id.removeprefix('example:') if image_id.startswith('example:') else image_id
            source = (source_root / name).resolve()
            if not source.is_relative_to(source_root) or not source.is_file():
                break
            sources.append(source)
        else:
            for path in folder.iterdir():
                match = re.fullmatch(r'(input|prepared)-([1-9]\d*)\.png', path.name)
                if not match or path.is_symlink() or not path.is_file() or path.resolve().parent != folder.resolve():
                    continue
                index = int(match[2]) - 1
                if index >= len(sources):
                    continue
                try:
                    with Image.open(path) as duplicate, Image.open(sources[index]) as original:
                        duplicate.load()
                        original.load()
                        if match[1] == 'input':
                            a, b = duplicate.convert('RGBA'), original.convert('RGBA')
                            if a.size != b.size or a.tobytes() != b.tobytes():
                                continue
                except OSError:
                    continue
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    paths = list(candidates(args.data_dir, APP_DIR / 'public' / 'examples'))
    total = sum(p.stat().st_size for p in paths)
    removed, removed_bytes = [], 0
    if args.apply:
        # Recheck eligibility immediately before each deletion.
        for path in paths:
            if path in set(candidates(args.data_dir, APP_DIR / 'public' / 'examples', path.parent)):
                size = path.stat().st_size
                path.unlink()
                removed.append(str(path))
                removed_bytes += size
    print(json.dumps({'applied': args.apply, 'files': len(paths), 'bytes': total,
                      'paths': [str(p) for p in paths], 'removed': removed, 'removed_bytes': removed_bytes}))


if __name__ == '__main__':
    main()

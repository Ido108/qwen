"""Fetch only the pinned, checksum-verified Space examples used by the interface."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
import httpx

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT/'reference'/'example-assets.json').read_text(encoding='utf-8'))
DESTINATION = ROOT/'public'/'examples'

def valid(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']

def fetch(item):
    path = DESTINATION/item['name']
    if path.parent.resolve() != DESTINATION.resolve():
        raise ValueError('Invalid example filename.')
    if valid(path, item):
        return
    url = f"https://huggingface.co/spaces/Qwen/Qwen-Image-2.1/resolve/{MANIFEST['revision']}/examples/{quote(item['name'])}"
    temporary = path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with httpx.stream('GET', url, follow_redirects=True, timeout=90) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for chunk in response.iter_bytes(1024*1024):
                        output.write(chunk)
            if not valid(temporary, item):
                raise ValueError('Example checksum mismatch: '+item['name'])
            temporary.replace(path)
            print('Downloaded example:', item['name'], flush=True)
            return
        except (httpx.HTTPError, OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt+1)

if __name__ == '__main__':
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, MANIFEST['files']))
    print(f"Verified {len(MANIFEST['files'])} example images.")

"""Pictures from a web image search (Google Images through SerpAPI, as in Image Studio and Moise). The key is the
env var SERPAPI_API_KEY or one saved in Settings (Windows DPAPI, current user only)."""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import httpx

from .secure_store import AUTH_DIR, _crypt

KEY_FILE = AUTH_DIR / 'serpapi.dpapi'
MAX_BYTES = 32 * 1024 * 1024


def key() -> str:
    value = os.environ.get('SERPAPI_API_KEY') or os.environ.get('SERPAPI_KEY')
    if value:
        return value.strip()
    try:
        return _crypt(KEY_FILE.read_bytes(), decrypt=True).decode().strip()
    except (FileNotFoundError, RuntimeError):
        return ''


def save_key(value: str | None) -> None:
    if not value:
        KEY_FILE.unlink(missing_ok=True)
        return
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    temporary = KEY_FILE.with_suffix('.tmp')
    temporary.write_bytes(_crypt(value.strip().encode()))
    temporary.replace(KEY_FILE)


def search(query: str, count: int = 24) -> list[dict]:
    """Google Images results: [{url, thumb, title, source, width, height}], the same request as Image Studio."""
    api_key = key()
    if not api_key:
        raise ValueError('Add a SerpAPI key in Settings to search the web.')
    results = []
    with httpx.Client(timeout=20) as client:
        for page in range(3):
            response = client.get('https://serpapi.com/search.json', params={
                'engine': 'google_images', 'q': query, 'api_key': api_key, 'ijn': page, 'safe': 'off', 'filter': 0})
            if response.status_code == 401:
                raise ValueError('The SerpAPI key was rejected. Check it in Settings.')
            response.raise_for_status()
            data = response.json()
            if data.get('error'):
                raise ValueError(f"Web search failed: {data['error']}")
            for item in data.get('images_results') or []:
                url = item.get('original') or item.get('image')
                if url and url.startswith(('http://', 'https://')) and all(r['url'] != url for r in results):
                    results.append({'url': url, 'thumb': item.get('thumbnail') or url, 'title': item.get('title') or '',
                                    'source': item.get('source') or '', 'width': item.get('original_width'),
                                    'height': item.get('original_height')})
            if len(results) >= count or not data.get('images_results'):
                break
    return results[:count]


def _public(url: str) -> None:
    """Refuse addresses on this PC or the home network, after a real DNS lookup (also at each redirect)."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('Only http and https pictures can be imported.')
    for info in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError('This picture address is not on the public internet.')


def download_any(urls: list[str]) -> bytes:
    """The first of these addresses that gives a picture: the full-size file, else the search's own preview copy."""
    from io import BytesIO
    from PIL import Image
    problem = 'Could not download this picture.'
    for url in [u for u in urls if u]:
        try:
            data = download(url)
            with Image.open(BytesIO(data)) as picture:
                picture.verify()
            return data
        except Exception as exc:  # a blocked site, a web page, broken bytes: try the next address
            problem = str(exc) or problem
    raise ValueError(f'{problem} Choose another picture.')


def download(url: str) -> bytes:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36',
               'Accept': 'image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8'}
    with httpx.Client(timeout=20, follow_redirects=False, headers=headers) as client:
        for _ in range(5):
            _public(url)
            with client.stream('GET', url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers['location'])
                    continue
                if response.status_code != 200:
                    raise ValueError(f'The site answered {response.status_code}.')
                # Many sites label pictures "application/octet-stream" or "binary/octet-stream": the bytes decide
                # (the caller opens them with PIL); only a web page instead of the picture is refused here.
                if response.headers.get('content-type', '').startswith('text/html'):
                    raise ValueError('The site sent a web page instead of the picture.')
                data = bytearray()
                for chunk in response.iter_bytes():
                    data += chunk
                    if len(data) > MAX_BYTES:
                        raise ValueError('The picture is over 32 MB.')
                return bytes(data)
    raise ValueError('Too many redirects. Choose another picture.')

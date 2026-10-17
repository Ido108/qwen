"""Remote use from the Moise dashboard (also on the phone) at <Moise>/qwen-studio/.

The app keeps an outbound long poll to Moise (no open ports on this PC). Moise hands over each request of the
remote page; it runs here against this app on 127.0.0.1 and the answer goes back. Moise checks the dashboard login.
Settings: MOISE_SERVER_URL and MOISE_RELAY_TOKEN (Windows user environment variables, as for MoiseGpuServer).
"""
from __future__ import annotations
import json
import os
import threading
import time
import uuid

import httpx

FORWARDED = ('content-type', 'range', 'x-studio-token')
RETURNED = ('Content-Type', 'Content-Range', 'Accept-Ranges', 'Content-Disposition', 'Cache-Control')
# A video asks for "bytes=N-": answer in parts, so one answer never carries a whole large file.
RANGE_PART = 8 * 1024 * 1024


def setting(name: str) -> str:
    value = os.environ.get(name, '')
    if not value and os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                value = str(winreg.QueryValueEx(key, name)[0] or '')
        except OSError:
            pass
    return value.strip()


def server_url() -> str:
    base = setting('MOISE_SERVER_URL').rstrip('/')
    local = base.startswith(('http://127.0.0.1:', 'http://localhost:'))
    return base if base.startswith('https://') or local else ''


def limit_range(value: str) -> str:
    unit, _, spec = value.partition('=')
    start, dash, end = spec.partition('-')
    if unit.strip() != 'bytes' or not dash or end or ',' in spec or not start.isdigit():
        return value
    return f'bytes={start}-{int(start) + RANGE_PART - 1}'


def run_request(client: httpx.Client, app_url: str, job: dict, body: bytes) -> tuple[int, dict, bytes]:
    path = str(job.get('path', ''))
    if not path.startswith('/') or path.startswith('//') or '..' in path.split('?')[0]:
        return 400, {'Content-Type': 'text/plain; charset=utf-8'}, b'Invalid path'
    headers = {k: str(v) for k, v in (job.get('headers') or {}).items() if k.lower() in FORWARDED}
    for name in [k for k in headers if k.lower() == 'range']:
        headers[name] = limit_range(headers[name])
    try:
        answer = client.request(str(job.get('method', 'GET')), app_url + path, content=body or None, headers=headers, timeout=httpx.Timeout(290, connect=5))
    except httpx.HTTPError as error:
        return 502, {'Content-Type': 'text/plain; charset=utf-8'}, f'Qwen did not answer: {type(error).__name__}'.encode()
    return answer.status_code, {k: answer.headers[k] for k in RETURNED if k in answer.headers}, answer.content


def serve(base: str, token: str, app_url: str, stop: threading.Event) -> None:
    worker = str(uuid.uuid4())
    root = base + '/api/qwen-studio-worker'
    auth = {'Authorization': f'Bearer {token}', 'X-Qwen-Worker-Id': worker}
    connected, last_error = False, ''
    with httpx.Client() as moise, httpx.Client() as app:
        def handle(job: dict) -> None:
            try:
                body = b''
                if job.get('hasBody'):
                    got = moise.get(f'{root}/task/{job["id"]}/body', headers=auth, timeout=120)
                    got.raise_for_status()
                    body = got.content
                status, headers, data = run_request(app, app_url, job, body)
                moise.post(f'{root}/task/{job["id"]}/result', content=data, timeout=300,
                           headers={**auth, 'X-Qwen-Result-Status': str(status), 'X-Qwen-Result-Headers': json.dumps(headers),
                                    'Content-Type': 'application/octet-stream'})
            except httpx.HTTPError as error:
                print(f'[Moise] Could not answer a remote request: {type(error).__name__}', flush=True)

        while not stop.is_set():
            try:
                answer = moise.post(root + '/poll', json={'workerId': worker}, headers=auth, timeout=httpx.Timeout(40, connect=10))
                if answer.status_code == 409:
                    raise RuntimeError('another Qwen app is connected to Moise')
                answer.raise_for_status()
                if not connected:
                    print(f'[Moise] Connected. Open {base}/qwen-studio/ to use Qwen from anywhere (also on the phone).', flush=True)
                connected = True
                for job in answer.json().get('jobs') or []:
                    threading.Thread(target=handle, args=(job,), daemon=True, name='moise-request').start()
            except Exception as error:
                if connected:
                    print(f'[Moise] Connection lost ({type(error).__name__}). Trying again.', flush=True)
                elif str(error) != last_error and not isinstance(error, httpx.HTTPError):
                    print(f'[Moise] Not connected: {error}.', flush=True)
                connected, last_error = False, str(error)
                stop.wait(10)


def start(port: int) -> threading.Event | None:
    base, token = server_url(), setting('MOISE_RELAY_TOKEN')
    if not base or not token or setting('QWEN_REMOTE') == '0':
        return None
    stop = threading.Event()
    threading.Thread(target=serve, args=(base, token, f'http://127.0.0.1:{port}', stop), daemon=True, name='moise-remote').start()
    return stop

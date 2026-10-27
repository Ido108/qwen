"""Remote use from the Moise dashboard (also on the phone) at <Moise>/qwen-studio/.

The app keeps an outbound long poll to Moise (no open ports on this PC). Moise hands over each request of the
remote page; it runs here against this app on 127.0.0.1 and the answer goes back. Moise checks the dashboard login.
Settings: MOISE_SERVER_URL and MOISE_RELAY_TOKEN (Windows user environment variables, as for MoiseGpuServer).
Several PCs can be connected to one Moise: each sends its name (COMPUTERNAME), GPU, VRAM and queue with every poll.
"""
from __future__ import annotations
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

import httpx

FORWARDED = ('content-type', 'range', 'x-studio-token')
RETURNED = ('Content-Type', 'Content-Range', 'Accept-Ranges', 'Content-Disposition', 'Cache-Control')
# A video asks for "bytes=N-": answer in parts, so one answer never carries a whole large file.
RANGE_PART = 8 * 1024 * 1024


def error_detail(error: Exception) -> str:
    # Do not print request URLs, response bodies or headers: they may contain secrets.
    if isinstance(error, httpx.HTTPStatusError):
        code = error.response.status_code
        meaning = {401: 'login rejected', 403: 'login rejected', 409: 'another Qwen server is connected',
                   429: 'too many requests', 502: 'gateway could not reach Moise',
                   503: 'Moise is unavailable', 504: 'Moise did not answer in time'}.get(code, 'request rejected')
        return f'HTTP {code}: {meaning}'
    if isinstance(error, httpx.TimeoutException):
        return f'{type(error).__name__}: no answer in time'
    return type(error).__name__


@dataclass
class ConnectionReport:
    base: str
    started: bool = False
    failures: int = 0
    outage: bool = False
    last_report: float = 0
    last_reason: str = ''
    stable_since: float | None = None

    def failed(self, error, now):
        self.failures += 1
        self.stable_since = None
        self.outage = True
        delay = min(30, 2 ** min(self.failures - 1, 5))
        if isinstance(error, httpx.HTTPStatusError):
            retry = error.response.headers.get('retry-after', '')
            if retry.isdigit():
                delay = min(300, max(delay, int(retry)))
            if error.response.status_code in (401, 403, 409):
                delay = max(delay, 10)
        reason = error_detail(error)
        if reason != self.last_reason or now - self.last_report >= 60:
            print(f'[Moise] Remote connection interrupted ({reason}). Retrying in {delay}s. Local jobs continue.', flush=True)
            self.last_report, self.last_reason = now, reason
        return delay

    def succeeded(self, now):
        self.failures = 0
        if not self.started:
            print(f'[Moise] Connected. Open {self.base}/qwen-studio/ to use Qwen from anywhere (also on the phone).', flush=True)
            self.started = True
        if self.outage:
            if self.stable_since is None:
                self.stable_since = now
            elif now - self.stable_since >= 30:
                print('[Moise] Remote connection restored.', flush=True)
                self.outage, self.last_reason = False, ''


def worker_name() -> str:
    return (os.environ.get('COMPUTERNAME') or socket.gethostname() or '').strip()[:64]


def gpu_info(cache={'until': 0.0, 'value': {}}) -> dict:
    """Name and memory of the GPU from nvidia-smi, read at most every 30 seconds."""
    if cache['until'] > time.monotonic():
        return cache['value']
    value = {}
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
        name, total, free = [part.strip() for part in out.splitlines()[0].split(',')]
        value = {'gpu': name, 'vramTotalMb': int(float(total)), 'vramFreeMb': int(float(free))}
    except Exception:
        pass
    cache.update(until=time.monotonic() + 30, value=value)
    return value


def poll_body(worker: str, status) -> dict:
    body = {'workerId': worker, 'name': worker_name(), **gpu_info()}
    try:
        body.update(status() if status else {})
    except Exception:
        pass
    return body


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


def serve(base: str, token: str, app_url: str, stop: threading.Event, status=None) -> None:
    worker = str(uuid.uuid4())
    root = base + '/api/qwen-studio-worker'
    auth = {'Authorization': f'Bearer {token}', 'X-Qwen-Worker-Id': worker}
    report = ConnectionReport(base)
    # Long polling must not wait for upload/result connections in the shared pool.
    with httpx.Client() as poller, httpx.Client() as moise, httpx.Client() as app:
        def handle(job: dict) -> None:
            try:
                body = b''
                if job.get('hasBody'):
                    got = moise.get(f'{root}/task/{job["id"]}/body', headers=auth, timeout=120)
                    got.raise_for_status()
                    body = got.content
                status, headers, data = run_request(app, app_url, job, body)
                result = moise.post(f'{root}/task/{job["id"]}/result', content=data, timeout=300,
                           headers={**auth, 'X-Qwen-Result-Status': str(status), 'X-Qwen-Result-Headers': json.dumps(headers),
                                    'Content-Type': 'application/octet-stream'})
                result.raise_for_status()
            except httpx.HTTPError as error:
                print(f'[Moise] Remote request {job["id"][:8]} failed ({error_detail(error)}).', flush=True)

        while not stop.is_set():
            try:
                answer = poller.post(root + '/poll', json=poll_body(worker, status), headers=auth, timeout=httpx.Timeout(40, connect=10))
                answer.raise_for_status()
                report.succeeded(time.monotonic())
                for job in answer.json().get('jobs') or []:
                    threading.Thread(target=handle, args=(job,), daemon=True, name='moise-request').start()
            except Exception as error:
                stop.wait(report.failed(error, time.monotonic()))


def start(port: int, status=None) -> threading.Event | None:
    """status: returns {'busy': bool, 'queue': int} of this app, sent to Moise for the choice of PC."""
    base, token = server_url(), setting('MOISE_RELAY_TOKEN')
    if not base or not token or setting('QWEN_REMOTE') == '0':
        return None
    stop = threading.Event()
    threading.Thread(target=serve, args=(base, token, f'http://127.0.0.1:{port}', stop, status), daemon=True, name='moise-remote').start()
    return stop

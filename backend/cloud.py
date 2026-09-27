"""RunPod Serverless transport. Media stays on the attached network volume."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote

import httpx

from .common import DATA_DIR, write_json

CONFIG = DATA_DIR / 'runpod.json'
SECRET = DATA_DIR / 'runpod.secret'
SERVICE = 'Qwen RunPod'
FIELDS = ('mode', 'image_endpoint', 'video_endpoint', 'datacenter', 'volume_id')


def _windows_crypt(data: bytes, decrypt: bool = False) -> bytes:
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    function = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise RuntimeError('Could not access the saved RunPod connection.')
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(result.data)


@lru_cache(maxsize=1)
def secrets_read() -> dict:
    if os.environ.get('RUNPOD_API_KEY') and os.environ.get('RUNPOD_S3_ACCESS_KEY') and os.environ.get('RUNPOD_S3_SECRET_KEY'):
        return {'api_key': os.environ['RUNPOD_API_KEY'], 's3_access_key': os.environ['RUNPOD_S3_ACCESS_KEY'], 's3_secret_key': os.environ['RUNPOD_S3_SECRET_KEY']}
    if sys.platform == 'darwin':
        result = subprocess.run(['security', 'find-generic-password', '-a', 'user', '-s', SERVICE, '-w'], capture_output=True, text=True)
        return json.loads(result.stdout) if result.returncode == 0 else {}
    if os.name == 'nt' and SECRET.exists():
        return json.loads(_windows_crypt(SECRET.read_bytes(), decrypt=True))
    return {}


def secrets_write(value: dict) -> None:
    if sys.platform == 'darwin':
        result = subprocess.run(['security', 'add-generic-password', '-a', 'user', '-s', SERVICE, '-w', json.dumps(value), '-U'], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Could not save the RunPod connection in macOS Keychain.')
    elif os.name == 'nt':
        SECRET.parent.mkdir(parents=True, exist_ok=True)
        SECRET.write_bytes(_windows_crypt(json.dumps(value).encode()))
    else:
        raise RuntimeError('Set RUNPOD_API_KEY and RunPod S3 environment variables on this system.')
    secrets_read.cache_clear()


def settings() -> dict:
    try:
        data = json.loads(CONFIG.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    return {'mode': data.get('mode', 'runpod' if sys.platform == 'darwin' else 'local'),
            'image_endpoint': data.get('image_endpoint', ''), 'video_endpoint': data.get('video_endpoint', ''),
            'datacenter': data.get('datacenter', ''), 'volume_id': data.get('volume_id', '')}


def status() -> dict:
    data = settings()
    try:
        secret = secrets_read()
        error = ''
    except (OSError, ValueError, RuntimeError) as exc:
        secret, error = {}, str(exc)
    return {**data, 'platform': sys.platform, 'connected': bool(all(secret.get(k) for k in ('api_key', 's3_access_key', 's3_secret_key'))),
            'api_key_saved': bool(secret.get('api_key')), 's3_key_saved': bool(secret.get('s3_access_key') and secret.get('s3_secret_key')),
            'error': error}


def configure(data: dict) -> dict:
    current = settings()
    for key in FIELDS:
        if key in data:
            current[key] = str(data[key]).strip()
    if current['mode'] not in ('local', 'runpod'):
        raise ValueError('Choose Local or RunPod.')
    if current['mode'] == 'local' and sys.platform == 'darwin':
        raise ValueError('Qwen generation on macOS requires a RunPod GPU.')
    for key in ('image_endpoint', 'video_endpoint', 'volume_id'):
        if current[key] and not re.fullmatch(r'[A-Za-z0-9_-]{3,80}', current[key]):
            raise ValueError(f'Invalid {key.replace("_", " ")}.')
    if current['datacenter'] and not re.fullmatch(r'[A-Z]{2,4}-[A-Z]{2,4}-\d', current['datacenter']):
        raise ValueError('Enter the RunPod datacenter ID, for example EU-RO-1.')
    secret = dict(secrets_read())
    for key in ('api_key', 's3_access_key', 's3_secret_key'):
        if data.get(key):
            secret[key] = str(data[key]).strip()
    if any(data.get(key) for key in ('api_key', 's3_access_key', 's3_secret_key')):
        secrets_write(secret)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, current)
    return status()


def ready(kind: str) -> bool:
    data = status()
    return bool(data['mode'] == 'runpod' and data['connected'] and data[f'{kind}_endpoint'] and data['datacenter'] and data['volume_id'])


def _s3():
    import boto3
    data, secret = settings(), secrets_read()
    return boto3.client('s3', endpoint_url=f"https://s3api-{data['datacenter'].lower()}.runpod.io/",
                        region_name=data['datacenter'], aws_access_key_id=secret['s3_access_key'],
                        aws_secret_access_key=secret['s3_secret_key'])


def upload(path: Path, key: str) -> None:
    _s3().upload_file(str(path), settings()['volume_id'], key)


def download(key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(settings()['volume_id'], key, str(path))


def _url(kind: str, operation: str) -> str:
    endpoint = settings()[f'{kind}_endpoint']
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,80}', endpoint):
        raise RuntimeError(f'Configure a RunPod {kind} endpoint in Settings.')
    return f'https://api.runpod.ai/v2/{quote(endpoint)}/{operation}'


def _request(method: str, kind: str, operation: str, **kwargs) -> dict:
    secret = secrets_read()
    if not secret.get('api_key'):
        raise RuntimeError('Connect RunPod in Settings.')
    with httpx.Client(timeout=40) as client:
        response = client.request(method, _url(kind, operation), headers={'Authorization': f"Bearer {secret['api_key']}"}, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f'RunPod request failed ({response.status_code}): {response.text[:400]}') from exc
    return response.json()


def submit(kind: str, payload: dict) -> str:
    response = _request('POST', kind, 'run', json={'input': payload,
        'policy': {'executionTimeout': 7_200_000, 'ttl': 10_800_000}})
    if not response.get('id'):
        raise RuntimeError('RunPod did not return a job ID.')
    return response['id']


def poll(kind: str, remote_id: str, update, cancelled) -> dict:
    started = time.monotonic()
    while True:
        if time.monotonic() - started > 10800:
            raise RuntimeError('RunPod job timed out after three hours including queue time.')
        if cancelled():
            cancel(kind, remote_id)
            raise RuntimeError('Generation cancelled.')
        try:
            result = _request('GET', kind, f'status/{quote(remote_id, safe="")}')
        except httpx.RequestError:
            update('CONNECTING', {})
            time.sleep(3)
            continue
        state = result.get('status')
        if state == 'COMPLETED':
            output = result.get('output')
            if not isinstance(output, dict):
                raise RuntimeError('RunPod completed without a valid output.')
            return output
        if state in ('FAILED', 'CANCELLED', 'TIMED_OUT', 'ERROR'):
            raise RuntimeError(f'RunPod {state.lower()}: {str(result.get("error") or result.get("output") or "No details")[:500]}')
        update(state or 'Waiting for worker', result)
        time.sleep(2)


def cancel(kind: str, remote_id: str) -> None:
    _request('POST', kind, f'cancel/{quote(remote_id, safe="")}')


def progress_message(result: dict) -> str:
    value = result.get('output')
    if isinstance(value, list) and value:
        value = value[-1]
    if isinstance(value, dict):
        value = value.get('message') or value.get('status')
    return value[:180] if isinstance(value, str) else ''


def progress_details(result: dict) -> dict:
    value=result.get('output')
    if isinstance(value,list) and value:
        value=value[-1]
    if not isinstance(value,dict):
        return {}
    allowed=('stage','stage_started_at','stage_seconds','step','total_steps','phase','chunk','total_chunks','preview_chunk',
             'input_frames_done','input_frames_total','output_frames_done','preview_step','preview_updated_at',
             'preview_unavailable','preview_key','download_completed_bytes','download_total_bytes')
    return {key:value[key] for key in allowed if key in value}


def test_connection() -> dict:
    data = settings()
    if not ready('image') and not ready('video'):
        raise RuntimeError('Enter an endpoint, network volume and both API keys first.')
    _s3().list_objects_v2(Bucket=data['volume_id'], Prefix='app-connection-test/', MaxKeys=1)
    endpoints = {}
    for kind in ('image', 'video'):
        if data[f'{kind}_endpoint']:
            response = _request('GET', kind, 'health')
            endpoints[kind] = response
    return {'ok': True, 'endpoints': endpoints}

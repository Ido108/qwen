"""Moise/Weboss Antigravity contract, adapted to this local Python application.

Same start/poll/exchange/status/disconnect routes and PKCE callback. Tokens stay
on the server, encrypted with Windows DPAPI; workers refresh before each call.
See reference/oauth-integration.md for the exact source implementations.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

import httpx
from .common import DATA_DIR, write_json

CLIENT_ID = '1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com'
# Installed-app client configuration from Moise/Weboss, not a user credential.
CLIENT_SECRET = os.environ.get('ANTIGRAVITY_CLIENT_SECRET', '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')
REDIRECT_URI = 'http://localhost:1455/auth/callback'
TOKEN_URL = 'https://oauth2.googleapis.com/token'
ENDPOINTS = ('https://daily-cloudcode-pa.googleapis.com', 'https://daily-cloudcode-pa.sandbox.googleapis.com', 'https://cloudcode-pa.googleapis.com')
SCOPES = ' '.join(('openid', 'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/cclog', 'https://www.googleapis.com/auth/experimentsandconfigs'))
AUTH_DIR = Path(os.environ.get('QWEN_AUTH_DIR', str(DATA_DIR / 'connections')))
TOKEN_FILE = AUTH_DIR / 'antigravity.dpapi'
KEYCHAIN_SERVICE = 'Qwen Antigravity'
SETTINGS_FILE = AUTH_DIR / 'settings.json'
MODELS = [
    {'id': 'gemini-3.7-flash', 'name': 'Gemini 3.7 Flash', 'wire': 'gemini-3.7-flash-tiered'},
    {'id': 'gemini-3.8-flash', 'name': 'Gemini 3.8 Flash', 'wire': 'gemini-3.8-flash-tiered'},
    {'id': 'gemini-3.6-flash', 'name': 'Gemini 3.6 Flash', 'wire': 'gemini-3.6-flash-tiered', 'enum': 'MODEL_PLACEHOLDER_M196'},
    {'id': 'gemini-3.1-pro-low', 'name': 'Gemini 3.1 Pro', 'wire': 'gemini-3.1-pro-low', 'enum': 'MODEL_PLACEHOLDER_M36', 'thinking': 'low'},
]
_lock = threading.RLock()
_flows_lock = threading.RLock()
_flows: dict = {}
_callback = None

class ConnectionError(RuntimeError):
    pass

def safe_error(value):
    text = re.sub(r'(?:ya29\.[\w.-]+|1//[\w.-]+|Bearer\s+\S+)', '[redacted]', str(value))
    return text[:600]

def _crypt(data: bytes, decrypt=False):
    if os.name != 'nt':
        raise ConnectionError('This connection store requires Windows.')
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    fn = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ConnectionError('Could not read the saved connection. Reconnect Google Account.')
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(result.data)

@contextmanager
def token_lock():
    # Server and generation worker can both refresh; serialize across processes.
    if os.name == 'nt':
        import msvcrt
    else:
        import fcntl
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, (AUTH_DIR / '.connection.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0'); handle.flush()
        deadline = time.monotonic() + 180
        while True:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise ConnectionError('The connection is busy. Try again.')
                time.sleep(.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def load_tokens():
    if sys.platform == 'darwin':
        result = subprocess.run(['security', 'find-generic-password', '-a', 'user', '-s', KEYCHAIN_SERVICE, '-w'], capture_output=True, text=True)
        return json.loads(result.stdout) if result.returncode == 0 else {}
    if not TOKEN_FILE.exists():
        return {}
    return json.loads(_crypt(TOKEN_FILE.read_bytes(), decrypt=True))

def save_tokens(tokens):
    if sys.platform == 'darwin':
        result = subprocess.run(['security', 'add-generic-password', '-a', 'user', '-s', KEYCHAIN_SERVICE, '-w', json.dumps(tokens), '-U'], capture_output=True, text=True)
        if result.returncode:
            raise ConnectionError('Could not save the connection in macOS Keychain.')
        return
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_FILE.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temporary.write_bytes(_crypt(json.dumps(tokens).encode()))
    temporary.replace(TOKEN_FILE)

def selected_model():
    try:
        value = json.loads(SETTINGS_FILE.read_text(encoding='utf-8')).get('model')
        if any(m['id'] == value for m in MODELS):
            return value
    except (OSError, ValueError):
        pass
    return MODELS[0]['id']

def set_model(value):
    if not any(m['id'] == value for m in MODELS):
        raise ValueError('Unknown prompt enhancement model.')
    with token_lock():
        write_json(SETTINGS_FILE, {'model': value})
    return status()

def status():
    try:
        token = load_tokens()
        connected = bool(token.get('accessToken') and token.get('refreshToken') and token.get('projectId'))
        error = ''
    except (ValueError, OSError, ConnectionError) as exc:
        token, connected, error = {}, False, safe_error(exc)
    return {'connected': connected, 'expired': bool(connected and token.get('expiresAt', 0) < time.time()*1000),
            'email': token.get('email', ''), 'model': selected_model(),
            'models': [{'id': m['id'], 'name': m['name']} for m in MODELS], 'error': error}

def _headers(access):
    return {'Authorization': f'Bearer {access}', 'Content-Type': 'application/json',
            'User-Agent': 'antigravity/1.107.0 windows/amd64',
            'X-Goog-Api-Client': 'google-cloud-sdk vscode_cloudshelleditor/0.1',
            'Client-Metadata': json.dumps({'ideType': 'ANTIGRAVITY', 'platform': 'WINDOWS', 'pluginType': 'GEMINI'})}

def discover_projects(access):
    pool = []
    with httpx.Client(timeout=20) as client:
        for endpoint in ENDPOINTS:
            try:
                response = client.post(endpoint + '/v1internal:loadCodeAssist', headers=_headers(access),
                    json={'metadata': {'ide_type': 'ANTIGRAVITY', 'ide_version': '1.19.6', 'ide_name': 'antigravity'}})
                if response.is_success:
                    project = response.json().get('cloudaicompanionProject')
                    project = project.get('id') if isinstance(project, dict) else project
                    if project and project not in pool:
                        pool.append(project)
            except (httpx.HTTPError, ValueError):
                continue
        try:
            response = client.get('https://cloudresourcemanager.googleapis.com/v1/projects', headers=_headers(access))
            if response.is_success:
                for project in response.json().get('projects', []):
                    value = project.get('projectId')
                    if project.get('lifecycleState') == 'ACTIVE' and value and value not in pool:
                        pool.append(value)
        except (httpx.HTTPError, ValueError):
            pass
    if not pool:
        raise ConnectionError('No Antigravity project is available for this account. Reconnect Google Account.')
    return pool

def _token_request(payload, previous=None):
    previous = previous or {}
    try:
        response = httpx.post(TOKEN_URL, data={'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET, **payload}, timeout=30)
        data = response.json()
    except (httpx.HTTPError, ValueError):
        raise ConnectionError('Could not contact Google. Check the internet connection and try again.') from None
    if not response.is_success:
        raise ConnectionError('Google connection failed: ' + safe_error(data.get('error_description') or data.get('error') or response.status_code))
    access, refresh = data.get('access_token'), data.get('refresh_token') or previous.get('refreshToken')
    if not access or not refresh:
        raise ConnectionError('Reconnect Google Account and approve offline access.')
    pool = discover_projects(access)
    token = {**previous, 'accessToken': access, 'refreshToken': refresh,
             'expiresAt': (time.time() + data.get('expires_in', 3600))*1000,
             'projectId': pool[0], 'projectsPool': pool}
    if data.get('id_token'):
        try:
            part = data['id_token'].split('.')[1]
            token['email'] = json.loads(base64.urlsafe_b64decode(part + '='*(-len(part)%4))).get('email', '')
        except (ValueError, IndexError):
            pass
    return token

def ensure_tokens(force=False):
    with token_lock():
        token = load_tokens()
        if not token.get('refreshToken'):
            raise ConnectionError('Connect Google Account in Settings → Prompt enhancement.')
        if not force and token.get('accessToken') and token.get('projectId') and time.time()*1000 < token.get('expiresAt', 0)-300_000:
            return token
        token = _token_request({'grant_type': 'refresh_token', 'refresh_token': token['refreshToken']}, token)
        save_tokens(token)
        return token

def disconnect():
    with token_lock(), _flows_lock:
        if sys.platform == 'darwin':
            subprocess.run(['security', 'delete-generic-password', '-a', 'user', '-s', KEYCHAIN_SERVICE], capture_output=True)
        else:
            TOKEN_FILE.unlink(missing_ok=True)
        _flows.clear()
    return status()

def _cleanup_flows():
    for state, flow in list(_flows.items()):
        if time.time()-flow['createdAt'] > 600:
            del _flows[state]

def poll(state):
    with _flows_lock:
        _cleanup_flows()
        flow = _flows.get(state)
        if not flow:
            return {'status': 'expired'}
        return dict(flow.get('result') or {'status': 'pending'})

def exchange(callback_input):
    parsed = urlsplit(callback_input.strip())
    if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1') or parsed.port != 1455 or parsed.path != '/auth/callback':
        raise ValueError('Paste the full localhost callback URL with code and state.')
    args = parse_qs(parsed.query)
    state, code = args.get('state', [''])[0], args.get('code', [''])[0]
    with _flows_lock:
        _cleanup_flows()
        flow = _flows.get(state)
        if not flow:
            raise ValueError('This connection request expired. Click Connect Google Account again.')
        if flow.get('result', {}).get('success'):
            return {'success': True}
        if flow.get('exchanging') or flow.get('result'):
            raise ValueError('This callback has already been used. Start a new connection if needed.')
        if args.get('error'):
            flow['result'] = {'status': 'completed', 'success': False, 'error': 'Google sign-in was cancelled or denied.'}
            raise ValueError(flow['result']['error'])
        if not code:
            raise ValueError('The callback URL is missing code. Paste the full URL.')
        flow['exchanging'] = True
    try:
        with token_lock():
            try:
                previous = load_tokens()
            except (ValueError, OSError, ConnectionError):
                previous = {}
            token = _token_request({'grant_type': 'authorization_code', 'code': code,
                'code_verifier': flow['verifier'], 'redirect_uri': REDIRECT_URI}, previous)
            save_tokens(token)
        result = {'status': 'completed', 'success': True}
    except Exception as exc:
        result = {'status': 'completed', 'success': False, 'error': safe_error(exc)}
    with _flows_lock:
        flow['result'] = result
        flow.pop('verifier', None)
    if not result['success']:
        raise ConnectionError(result['error'])
    return {'success': True}

def _start_listener(port=1455):
    global _callback
    if _callback:
        return True
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Callback URLs contain authorization codes.
        def do_GET(self):
            if urlsplit(self.path).path != '/auth/callback':
                self.send_error(404); return
            try:
                exchange('http://localhost:1455' + self.path)
                code, title, message = 200, 'Connected', 'Your Google account is connected. You can close this window and return to Qwen.'
            except Exception:
                code, title, message = 400, 'Complete the connection in Qwen', 'If this sign-in belongs to another app, return to that app. Otherwise paste the full address from this window into Qwen Settings.'
            html = f'<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{font:16px Segoe UI,sans-serif;background:#f7f6fc;color:#292439;display:grid;place-content:center;min-height:90vh}}main{{max-width:440px;padding:40px;background:white;border-radius:24px;box-shadow:0 12px 60px #51448118}}h1{{color:#6246db}}</style><main><h1>{title}</h1><p>{message}</p></main>'
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers(); self.wfile.write(html.encode())
    class CallbackServer(ThreadingHTTPServer):
        allow_reuse_address = False
        def server_bind(self):
            if os.name == 'nt':
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()
    try:
        _callback = CallbackServer(('127.0.0.1', port), Handler)
        threading.Thread(target=_callback.serve_forever, daemon=True).start()
        return True
    except OSError:
        return False  # Another local app owns port 1455; manual paste stays available.

def start():
    with _flows_lock:
        _cleanup_flows()
        if len(_flows) >= 20:
            _flows.pop(next(iter(_flows)))
        verifier, state = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        _flows[state] = {'verifier': verifier, 'createdAt': time.time()}
        automatic = _start_listener()
    params = {'client_id': CLIENT_ID, 'redirect_uri': REDIRECT_URI, 'response_type': 'code',
              'scope': SCOPES, 'code_challenge': challenge, 'code_challenge_method': 'S256',
              'state': state, 'access_type': 'offline', 'prompt': 'consent'}
    return {'authUrl': 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params),
            'state': state, 'automaticCallback': automatic, 'expiresIn': 600}

def request_body(model_id, system, parts, project):
    model = next((m for m in MODELS if m['id'] == model_id), None)
    if not model:
        raise ValueError('Unknown prompt enhancement model.')
    trajectory = str(uuid.uuid4())
    labels = {'trajectory_id': trajectory, 'last_step_index': '1', 'used_claude': 'false', 'used_claude_conservative': 'false'}
    if model.get('enum'):
        labels['model_enum'] = model['enum']
    generation = {'temperature': .4, 'maxOutputTokens': 8192}
    if model.get('thinking'):
        generation['thinkingConfig'] = {'thinkingLevel': model['thinking']}
    elif model_id == 'gemini-3.7-flash':
        generation['thinkingConfig'] = {'thinkingBudget': 0}
    return {'project': project, 'model': model['wire'], 'userAgent': 'antigravity', 'requestType': 'agent',
            'requestId': f'agent/{uuid.uuid4()}/{int(time.time()*1000)}/{trajectory}/2',
            'request': {'systemInstruction': {'parts': [{'text': system}]},
                        'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': generation, 'labels': labels}}

def _read_stream(response):
    output, finish = [], ''
    for line in response.iter_lines():
        if not line.startswith('data:') or line[5:].strip() == '[DONE]':
            continue
        data = json.loads(line[5:])
        if data.get('error'):
            raise ConnectionError(safe_error(data['error'].get('message', 'Antigravity stream failed.')))
        data = data.get('response', data)
        if data.get('error'):
            raise ConnectionError(safe_error(data['error'].get('message', 'Antigravity stream failed.')))
        candidates = data.get('candidates') or []
        if data.get('promptFeedback', {}).get('blockReason'):
            raise ConnectionError('The prompt enhancement provider declined this request. You can use your original prompt with enhancement off.')
        for candidate in candidates[:1]:
            finish = candidate.get('finishReason') or finish
            for part in candidate.get('content', {}).get('parts', []):
                if part.get('text') and not part.get('thought'):
                    output.append(part['text'])
    if finish in ('SAFETY', 'PROHIBITED_CONTENT', 'RECITATION', 'BLOCKLIST'):
        raise ConnectionError('The prompt enhancement provider declined this request. You can use your original prompt with enhancement off.')
    if finish == 'MAX_TOKENS':
        raise ConnectionError('Prompt enhancement reached its response limit. Try a shorter request or another model.')
    if not ''.join(output).strip():
        raise ConnectionError('The selected Antigravity model returned no text. Select another model in Settings.')
    return ''.join(output)

def generate_text(model_id, system, parts):
    tokens = ensure_tokens()
    last_error = 'Antigravity is unavailable.'
    refreshed = False
    with httpx.Client(timeout=httpx.Timeout(180, connect=20)) as client:
        for project in tokens.get('projectsPool') or [tokens['projectId']]:
            for endpoint in ENDPOINTS:
                for retry in range(2):
                    body = request_body(model_id, system, parts, project)
                    try:
                        with client.stream('POST', endpoint + '/v1internal:streamGenerateContent?alt=sse',
                                headers={**_headers(tokens['accessToken']), 'Accept': 'text/event-stream'}, json=body) as response:
                            if response.is_success:
                                return _read_stream(response)
                            response.read()
                            try:
                                error = response.json().get('error', {})
                                detail = error.get('message', '') if isinstance(error, dict) else str(error)
                            except ValueError:
                                detail = 'Service unavailable'
                            last_error = f'Antigravity ({response.status_code}): {safe_error(detail)}'
                            if response.status_code == 401 and not refreshed:
                                tokens, refreshed = ensure_tokens(force=True), True
                                continue
                            if response.status_code in (429, 502, 503, 504) and retry == 0:
                                time.sleep(2); continue
                            if response.status_code not in (403, 404, 429, 502, 503, 504):
                                raise ConnectionError(last_error)
                            break
                    except httpx.HTTPError:
                        last_error = 'Could not contact Antigravity. Check your internet connection.'
                        break
    raise ConnectionError(last_error)

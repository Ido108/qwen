"""Start an independent Qwen server and open Chrome or an optional desktop window."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

from windows_processes import Instance, Job, set_event, reset_event, windows, post_message
from watch_logs import watch, viewer_name

APP = Path(__file__).resolve().parents[1]


def get_json(url):
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def matching_server(url, data):
    try:
        health = get_json(url + '/api/health')
        return (health.get('app') == 'image-studio-qwen21'
                and Path(health.get('app_dir', '')).resolve() == APP
                and Path(health.get('data_dir', '')).resolve() == data)
    except (OSError, ValueError):
        return False


def require_idle(url):
    queue = get_json(url + '/api/queue')
    if queue.get('active') or queue.get('pending'):
        raise RuntimeError('Finish or cancel the queued jobs before stopping or restarting Qwen. Nothing was stopped.')
    catalog = get_json(url + '/api/models')
    if (catalog.get('transfer') or {}).get('status') == 'running' or any(item.get('status') == 'running' for item in catalog.get('downloads', [])):
        raise RuntimeError('Model downloads are running. Wait for them to finish before stopping or restarting Qwen. Nothing was stopped.')


def process_inventory():
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
        'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    return json.loads(result.stdout)


def stop_legacy(url, data, saved, *, close_app_windows=False):
    """One-time upgrade of the former detached launcher, scoped by live health.

    Never infer ownership from a saved PID alone (it may have been reused).
    """
    if not matching_server(url, data):
        return
    require_idle(url)
    inventory = process_inventory()
    by_id = {p['ProcessId']: p for p in inventory}
    root = by_id.get(saved.get('pid'), {})
    command = root.get('CommandLine') or ''
    port = str(saved.get('port', ''))
    # The legacy runtime file is repo-owned; verify its command and listening PID.
    listeners = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
        f'Get-NetTCPConnection -State Listen -LocalPort {int(port)} | Select-Object -ExpandProperty OwningProcess'],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    listening = {int(line.strip()) for line in listeners.stdout.splitlines() if line.strip().isdigit()}
    descendants = {root.get('ProcessId')}
    for _ in range(len(inventory)):
        expanded = descendants | {p['ProcessId'] for p in inventory if p['ParentProcessId'] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    if ('uvicorn backend.server:app' not in command or f'--port {port}' not in command
            or not listening.intersection(descendants)):
        raise RuntimeError('Could not verify the previous Qwen process. Nothing was stopped.')
    require_idle(url)
    subprocess.run(['taskkill', '/PID', str(root['ProcessId']), '/T', '/F'],
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    # Close only the old app-mode window, never a general Edge browser window.
    if close_app_windows:
        for hwnd, _, title in windows():
            if title == 'Qwen: Qwen-Image-2.1':
                post_message(hwnd, 0x10, 0, 0)
    deadline = time.monotonic() + 10
    while matching_server(url, data) and time.monotonic() < deadline:
        time.sleep(.1)
    if matching_server(url, data):
        raise RuntimeError('The previous Qwen server did not close.')


def choose_port(first):
    for port in range(first, first + 20):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                pass
    raise RuntimeError('No free local Qwen port is available.')


def browser_executable():
    # Search all Chrome locations before considering Edge.
    for relative in ('Google/Chrome/Application/chrome.exe', 'Microsoft/Edge/Application/msedge.exe'):
        for parent in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA'):
            path = Path(os.environ.get(parent, 'C:/')) / relative
            if path.is_file():
                return path
    raise RuntimeError('Install Google Chrome to open Qwen.')


def read_state(runtime):
    try:
        return json.loads(runtime.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def identity_for(data):
    return hashlib.sha256(f'{APP}|{data}'.lower().encode()).hexdigest()[:24]


def write_state(runtime, state):
    runtime.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(runtime)


def stop_server(data, runtime):
    saved = read_state(runtime)
    if not saved.get('url') or not matching_server(saved['url'], data):
        return
    require_idle(saved['url'])
    instance = Instance('Local\\Qwen-' + identity_for(data))
    try:
        if instance.acquire():
            stop_legacy(saved['url'], data, saved)
        else:
            set_event(instance.stop)
            if not instance.acquire(20000):
                raise RuntimeError('Qwen did not stop. The current server is still in use.')
    finally:
        instance.close()


def serve(args):
    """The background owner controls GPU children, but never owns a browser."""
    data, runtime = args.data_dir.resolve(), args.runtime_file.resolve()
    instance = Instance('Local\\Qwen-' + identity_for(data))
    job = None
    try:
        if not instance.acquire():
            return
        saved = read_state(runtime)
        if saved.get('url') and matching_server(saved['url'], data):
            return
        port = choose_port(args.port)
        url = f'http://127.0.0.1:{port}'
        log_dir = data / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, QWEN_DATA_DIR=str(data), QWEN_SERVER_PORT=str(port), PYTHONUTF8='1')
        job = Job()
        command = [sys.executable, '-u', '-m', 'uvicorn', 'backend.server:app', '--host', '127.0.0.1',
                   '--port', str(port), '--loop', 'backend.windows_loop:new_event_loop']
        with (log_dir/'server.out.log').open('ab') as output, (log_dir/'server.err.log').open('ab') as error:
            server = job.spawn(command, cwd=APP, env=env, stdout=output, stderr=error)
        write_state(runtime, {'port':port, 'pid':server.pid, 'owner_pid':os.getpid(), 'url':url,
                              'app_dir':str(APP), 'data_dir':str(data), 'managed':True, 'independent':True})
        while True:
            if instance.stopped():
                try:
                    require_idle(url)
                    break
                except (OSError, ValueError, RuntimeError) as error:
                    print(f'Qwen stays running: {error}', flush=True)
                    reset_event(instance.stop)
            if server.poll() is not None:
                raise RuntimeError(f'Qwen stopped. See {log_dir / "server.err.log"}')
            time.sleep(.2)
    finally:
        if job:
            job.close()
            if read_state(runtime).get('owner_pid') == os.getpid():
                runtime.unlink(missing_ok=True)
        instance.close()


def ensure_server(args):
    data, runtime = args.data_dir.resolve(), args.runtime_file.resolve()
    saved = read_state(runtime)
    if saved.get('url') and matching_server(saved['url'], data):
        return saved
    log_dir = data / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(APP/'scripts/desktop.py'), '--serve', '--data-dir', str(data),
               '--runtime-file', str(runtime), '--port', str(args.port)]
    # CREATE_NO_WINDOW gives the venv redirector a windowless console which its
    # interpreter inherits. DETACHED_PROCESS loses that console and the Python
    # redirector creates a visible, empty terminal for the background owner.
    # This console is independent of the launcher and the activity viewers.
    with (log_dir/'launcher.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=APP, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        saved = read_state(runtime)
        if saved.get('url') and matching_server(saved['url'], data):
            return saved
        if process.poll() not in (None, 0):
            raise RuntimeError(f'Qwen could not start. See {log_dir / "launcher.log"}')
        time.sleep(.2)
    raise RuntimeError(f'Qwen did not answer. See {log_dir / "server.err.log"}')


def open_browser(args, url):
    if args.desktop_window:
        executable = APP/'node_modules/electron/dist/electron.exe'
        if not executable.is_file():
            raise RuntimeError('The desktop window is not installed. Run npm install in the Qwen folder. The server stays running.')
        command = [str(executable), str(APP/'scripts/electron.cjs'), '--url='+url]
    else:
        command = [str(args.browser or browser_executable()), '--new-tab', url]
    # Browsers are separate clients. They are not children of the server Job Object.
    subprocess.Popen(command, cwd=APP, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)


def open_console(url, data):
    """Open independent, named Terminal tabs. Reopening does not duplicate them."""
    terminal = shutil.which('wt.exe')
    if not terminal:
        print('Windows Terminal is unavailable. Activity appears in this terminal.', flush=True)
        return watch(url, data)
    command = [terminal, '-w', 'Qwen-' + identity_for(data)]
    missing = []
    for channel, title, color in (('activity', 'Qwen - Activity', '#7866DB'),
                                  ('server', 'Qwen - Server', '#399E91')):
        instance = Instance(viewer_name(data, channel))
        try:
            if instance.acquire():
                missing.append(channel)
                if len(command) > 3:
                    command.append(';')
                command.extend(['new-tab', '--title', title, '--suppressApplicationTitle',
                                '--tabColor', color, '-d', str(APP), sys.executable, '-u',
                                str(APP/'scripts/watch_logs.py'), '--url', url,
                                '--data-dir', str(data), '--channel', channel])
        finally:
            instance.close()
    if len(command) > 3:
        command.append(';')
    command.extend(['focus-tab', '-t', '0'])
    subprocess.run(command, cwd=APP, check=True, timeout=15,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    # Catch a failed viewer import/start instead of leaving an empty tab unnoticed.
    deadline = time.monotonic() + 8
    while missing and time.monotonic() < deadline:
        for channel in list(missing):
            instance = Instance(viewer_name(data, channel))
            try:
                if not instance.acquire():
                    missing.remove(channel)
            finally:
                instance.close()
        if missing:
            time.sleep(.1)
    if missing:
        raise RuntimeError('A log tab could not start. See the error in Windows Terminal. The server stays running.')


def run(args):
    if args.serve:
        return serve(args)
    if args.logs_only:
        saved = read_state(args.runtime_file.resolve())
        return open_console(saved.get('url', f'http://127.0.0.1:{args.port}'), args.data_dir.resolve())
    if args.restart or args.stop:
        stop_server(args.data_dir.resolve(), args.runtime_file.resolve())
    if args.stop:
        print('Qwen server stopped.', flush=True)
        return
    state = ensure_server(args)
    if not args.no_browser:
        open_browser(args, state['url'])
    print(f'Qwen is ready: {state["url"]}\nThe server keeps running when you close the browser.', flush=True)
    if not args.no_console:
        open_console(state['url'], args.data_dir.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--runtime-file', type=Path, default=APP/'runtime.json')
    parser.add_argument('--port', type=int, default=47821)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--no-console', action='store_true', help='Do not open the activity and server log tabs')
    parser.add_argument('--logs-only', action='store_true', help='Open the log tabs without starting or stopping the server')
    parser.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--desktop-window', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--browser', type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('Qwen launcher closed.', flush=True)
    except Exception as error:
        print(f'Qwen: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

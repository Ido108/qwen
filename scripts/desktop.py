"""Own the Windows Qwen window, HTTP server and all local worker descendants."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

from windows_processes import Instance, Job, focus_job, set_event, wait_handle, windows, post_message
from watch_logs import watch

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
    queue = get_json(url + '/api/queue')
    if queue.get('active') or queue.get('pending'):
        raise RuntimeError('Finish or cancel the current jobs before restarting Qwen. Nothing was stopped.')
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
    queue = get_json(url + '/api/queue')
    if queue.get('active') or queue.get('pending'):
        raise RuntimeError('A job was added. Finish or cancel it before restarting Qwen.')
    subprocess.run(['taskkill', '/PID', str(root['ProcessId']), '/T', '/F'],
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    # Stop only viewers attached to this repository/data folder. Their original
    # BAT/PowerShell wrappers return normally and close their terminal windows.
    for process in inventory:
        cmd = process.get('CommandLine') or ''
        if str(APP / 'scripts' / 'watch_logs.py').lower() in cmd.lower() and str(data).lower() in cmd.lower():
            subprocess.run(['taskkill', '/PID', str(process['ProcessId']), '/T', '/F'],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    # Close only the old app-mode window, never a general Edge browser window.
    if close_app_windows:
        for hwnd, _, title in windows():
            if title == 'Qwen — Qwen-Image-2.1':
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
    for parent in ('PROGRAMFILES(X86)', 'PROGRAMFILES', 'LOCALAPPDATA'):
        base = Path(os.environ.get(parent, 'C:/'))
        for relative in ('Microsoft/Edge/Application/msedge.exe', 'Google/Chrome/Application/chrome.exe'):
            path = base / relative
            if path.is_file():
                return path
    raise RuntimeError('Install Microsoft Edge or Google Chrome to open the Qwen window.')


def run(args):
    data = args.data_dir.resolve()
    runtime = args.runtime_file.resolve()
    identity = hashlib.sha256(f'{APP}|{data}'.lower().encode()).hexdigest()[:24]
    instance = Instance('Local\\Qwen-' + identity)
    job = None
    log_thread = None
    logs_done = threading.Event()
    try:
        if not instance.acquire():
            if not (args.restart or args.stop):
                set_event(instance.focus)
                print('Qwen is already open. Showing its window.', flush=True)
                return
            # Restart is an explicit close: stop the entire owned app, then wait
            # for the lock so old/new servers cannot overlap.
            set_event(instance.stop)
            if not instance.acquire(15000):
                raise RuntimeError('The previous Qwen window did not close. Close it before restarting.')
        try:
            saved = json.loads(runtime.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            saved = {}
        if saved.get('url'):
            stop_legacy(saved['url'], data, saved, close_app_windows=runtime==APP/'runtime.json')
        if args.stop:
            return
        port = choose_port(args.port)
        url = f'http://127.0.0.1:{port}'
        log_dir = data / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, QWEN_DATA_DIR=str(data), PYTHONUTF8='1')
        job = Job()
        command = [sys.executable, '-u', '-m', 'uvicorn', 'backend.server:app', '--host', '127.0.0.1',
                   '--port', str(port), '--loop', 'backend.windows_loop:new_event_loop']
        with (log_dir/'server.out.log').open('wb') as output, (log_dir/'server.err.log').open('wb') as error:
            server = job.spawn(command, cwd=APP, env=env, stdout=output, stderr=error)
        state = {'port': port, 'pid': server.pid, 'owner_pid': os.getpid(), 'url': url,
                 'app_dir': str(APP), 'data_dir': str(data), 'managed': True}
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_text(json.dumps(state, indent=2), encoding='utf-8')
        print('Starting Qwen...', flush=True)
        deadline = time.monotonic() + 90
        while not matching_server(url, data):
            if instance.stopped():
                return
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'Qwen could not start. See {log_dir / "server.err.log"}')
            time.sleep(.2)
        browser = None
        if not args.no_browser:
            # A dedicated profile prevents an existing Edge background process
            # from taking ownership of the app window outside our Job Object.
            profile = Path(os.environ['QWEN_BROWSER_PROFILE']) if os.environ.get('QWEN_BROWSER_PROFILE') else APP / '.local' / ('browser-' + identity)
            browser = job.spawn([args.browser or browser_executable(), f'--app={url}',
                f'--user-data-dir={profile}', '--no-first-run', '--disable-background-mode',
                '--disable-session-crashed-bubble', '--no-default-browser-check', '--window-size=1440,1000'],
                cwd=APP, hidden=False)
            state['browser_pid'] = browser.pid
            runtime.write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f'Qwen is ready: {url}\nClosing this terminal or the Qwen window stops Qwen and local jobs.\n', flush=True)
        if not args.no_console:
            log_thread = threading.Thread(target=watch, args=(url, data), kwargs={'stop_event': logs_done}, daemon=True)
            log_thread.start()
        while not instance.stopped():
            if server.poll() is not None:
                raise RuntimeError('Qwen stopped. See the server output above.')
            if browser and browser.poll() is not None:
                break
            if wait_handle(instance.focus, 0) == 0:
                focus_job(job)
            time.sleep(.1)
    finally:
        logs_done.set()
        if job:
            job.close()
            try:
                if json.loads(runtime.read_text()).get('owner_pid') == os.getpid():
                    runtime.unlink()
            except (OSError, ValueError):
                pass
        if log_thread:
            log_thread.join(timeout=3)
        instance.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--runtime-file', type=Path, default=APP/'runtime.json')
    parser.add_argument('--port', type=int, default=47821)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--no-console', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--browser', type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nQwen closed.', flush=True)
    except Exception as error:
        print(f'Qwen: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

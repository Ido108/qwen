"""Exercise the actual Windows BAT and app window using isolated data; no renders."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from windows_processes import windows, post_message, open_process, wait_handle, close_handle

APP=Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=30):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            if predicate(): return
        except (OSError,ValueError,KeyError): pass
        time.sleep(.1)
    raise AssertionError('Timed out waiting for desktop lifecycle condition')


def alive(pid):
    handle=open_process(0x100000,False,pid)
    if not handle:return False
    try:return wait_handle(handle,0)==258
    finally:close_handle(handle)


def main():
    with tempfile.TemporaryDirectory(prefix='qwen-desktop-qa-') as folder:
        data=Path(folder);state=data/'runtime.json'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=dict(os.environ,QWEN_DATA_DIR=str(data),QWEN_STATE_FILE=str(state),QWEN_APP_PORT=str(port),QWEN_BROWSER_PROFILE=str(data/'browser'),CUDA_VISIBLE_DEVICES='')
        processes=[]; handles=[]
        def launch(name):
            handle=(data/f'launch-{len(handles)}.txt').open('wb');handles.append(handle)
            process=subprocess.Popen(['cmd.exe','/c',str(APP/name)],cwd=APP,env=env,stdout=handle,stderr=handle,creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(process);return process
        def ready():
            saved=json.loads(state.read_text())
            return saved.get('browser_pid') and any(p==saved['browser_pid'] for _,p,_ in windows())
        def close_window():
            saved=json.loads(state.read_text())
            targets=[hwnd for hwnd,pid,_ in windows() if pid==saved['browser_pid']]
            assert targets,'App window is missing'
            for hwnd in targets:post_message(hwnd,0x10,0,0)
            return saved
        try:
            first=launch('Start Qwen.bat');wait_for(ready,60)
            saved=json.loads(state.read_text())
            with urlopen(saved['url']+'/api/health') as response:assert response.status==200
            duplicate=launch('Start Qwen.bat');assert duplicate.wait(20)==0
            assert json.loads(state.read_text())['owner_pid']==saved['owner_pid']
            print('PASS: actual Start BAT, one app window, duplicate launch reuses it',flush=True)
            restart=launch('Restart Qwen.bat')
            assert first.wait(30)==0
            wait_for(lambda:ready() and json.loads(state.read_text())['owner_pid']!=saved['owner_pid'],60)
            assert not alive(saved['pid']) and not alive(saved['browser_pid'])
            print('PASS: actual Restart BAT closes the old server and browser',flush=True)
            current=close_window()
            assert restart.wait(20)==0
            wait_for(lambda:not alive(current['pid']))
            print('PASS: closing the real app window closes its BAT and server',flush=True)
            last=launch('Start Qwen.bat');wait_for(ready,60)
            current=json.loads(state.read_text())
            subprocess.run(['taskkill','/PID',str(current['owner_pid']),'/F'],capture_output=True,check=True)
            wait_for(lambda:not alive(current['pid']) and not alive(current['browser_pid']))
            # A forced process exit intentionally reaches BAT's error pause.
            subprocess.run(['taskkill','/PID',str(last.pid),'/T','/F'],capture_output=True)
            print('PASS: abrupt owner loss closes the app and server without Python cleanup',flush=True)
            print('PASS: no media generation was requested',flush=True)
        finally:
            subprocess.run([sys.executable,str(APP/'scripts/desktop.py'),'--data-dir',str(data),'--runtime-file',str(state),'--stop'],cwd=APP,capture_output=True,timeout=20)
            for process in processes:
                if process.poll() is None:subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                process.wait(timeout=10)
            for handle in handles:handle.close()


if __name__=='__main__':main()

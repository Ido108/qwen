"""Windows lifecycle tests use isolated data and sleeping processes, never CUDA."""
import ctypes
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen

APP = Path(__file__).resolve().parents[1]
if os.name == 'nt':
    sys.path.insert(0,str(APP/'scripts'))
    from windows_processes import Job, open_process, wait_handle, close_handle


def alive(pid):
    handle=open_process(0x100000,False,pid)
    if not handle:
        return False
    try:
        return wait_handle(handle,0)==258
    finally:
        close_handle(handle)


def eventually(predicate, seconds=15):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:
            if predicate():
                return
        except (OSError,ValueError,KeyError):
            pass
        time.sleep(.1)
    raise AssertionError('Condition not reached before timeout')


@unittest.skipUnless(os.name=='nt','Windows process ownership')
class WindowsLifetimeTests(unittest.TestCase):
    def test_legacy_server_upgrade_stops_verified_wrapper(self):
        from desktop import stop_legacy, matching_server
        with tempfile.TemporaryDirectory(prefix='qwen-legacy-') as folder:
            data=Path(folder)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            url=f'http://127.0.0.1:{port}'
            env=dict(os.environ,QWEN_DATA_DIR=folder,CUDA_VISIBLE_DEVICES='')
            with (data/'server.log').open('wb') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','uvicorn','backend.server:app','--host','127.0.0.1','--port',str(port)],cwd=APP,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    eventually(lambda:matching_server(url,data),30)
                    stop_legacy(url,data,{'pid':process.pid,'port':port})
                    eventually(lambda:not matching_server(url,data))
                    process.wait(timeout=10)
                finally:
                    if process.poll() is None:
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                        process.wait(timeout=10)

    def test_job_close_stops_grandchildren(self):
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/'child.json'
            code='import subprocess,sys,time,json; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(90)"]); Path(sys.argv[1]).write_text(json.dumps([p.pid])); time.sleep(90)'
            job=Job()
            try:
                parent=job.spawn([sys.executable,'-c',code,str(marker)],cwd=APP)
                eventually(marker.exists)
                child=json.loads(marker.read_text())[0]
                self.assertTrue(alive(parent.pid)); self.assertTrue(alive(child))
            finally:
                job.close()
            eventually(lambda:not alive(parent.pid) and not alive(child))

    def test_failed_start_leaves_no_process(self):
        job=Job()
        try:
            with self.assertRaises(OSError):
                job.spawn(['C:/this-does-not-exist/qwen.exe'],cwd=APP)
        finally:
            job.close()

    def test_single_instance_restart_and_abrupt_owner_exit(self):
        with tempfile.TemporaryDirectory(prefix='qwen-lifecycle-') as folder:
            folder=Path(folder); state=folder/'runtime.json'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            args=[sys.executable,'-u',str(APP/'scripts/desktop.py'),'--data-dir',str(folder),
                  '--runtime-file',str(state),'--port',str(port),'--no-browser','--no-console']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            handles=[]
            def start(extra=()):
                handle=(folder/f'launcher{len(handles)}.log').open('wb'); handles.append(handle)
                return subprocess.Popen([*args,*extra],cwd=APP,env=env,stdout=handle,stderr=handle,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            first=start(); second=None
            try:
                def ready():
                    value=json.loads(state.read_text())
                    with urlopen(value['url']+'/api/health',timeout=1) as response:
                        return response.status==200
                eventually(ready,40)
                old=json.loads(state.read_text())
                # Starting Qwen again while it is idle replaces the old instance.
                replacement=start(); self.assertEqual(first.wait(timeout=20),0)
                eventually(lambda:ready() and json.loads(state.read_text())['owner_pid']!=old['owner_pid'],40)
                self.assertFalse(alive(old['pid']))
                first,old=replacement,json.loads(state.read_text())
                # Browser clients can reset sockets; the server must stay usable
                # without Proactor callback errors or exception filtering.
                for _ in range(5):
                    connection=socket.create_connection(('127.0.0.1',old['port']))
                    connection.sendall(b'GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n')
                    connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('hh',1,0))
                    connection.close()
                second=start(['--restart'])
                self.assertEqual(first.wait(timeout=20),0)
                eventually(lambda:ready() and json.loads(state.read_text())['owner_pid']!=old['owner_pid'],40)
                current=json.loads(state.read_text())
                self.assertFalse(alive(old['pid']))
                # Simulate hard console closure: the OS, not a Python finally,
                # must stop the server descendants when the owner disappears.
                subprocess.run(['taskkill','/PID',str(current['owner_pid']),'/F'],capture_output=True,check=True)
                eventually(lambda:not alive(current['pid']))
                second.wait(timeout=10)
                self.assertNotIn('_ProactorBasePipeTransport',(folder/'logs/server.err.log').read_text())
            finally:
                subprocess.run([*args,'--stop'],cwd=APP,env=env,capture_output=True,timeout=20)
                for process in (first,second):
                    if process and process.poll() is None:
                        process.kill(); process.wait()
                for handle in handles:
                    handle.close()


if __name__=='__main__':
    unittest.main()

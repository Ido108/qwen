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
from unittest.mock import patch
from types import SimpleNamespace
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
            env=dict(os.environ,QWEN_DATA_DIR=folder,CUDA_VISIBLE_DEVICES='',QWEN_REMOTE='0',QWEN_NO_WARMUP='1')
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

    def test_busy_stop_refuses_before_signalling_owner(self):
        from desktop import stop_server
        with patch('desktop.read_state',return_value={'url':'http://127.0.0.1:1'}), patch('desktop.matching_server',return_value=True), patch('desktop.get_json',return_value={'active':None,'pending':[{'id':'queued'}]}), patch('desktop.Instance') as instance:
            with self.assertRaisesRegex(RuntimeError,'Nothing was stopped'):
                stop_server(Path('C:/test'),Path('C:/test/runtime.json'))
            instance.assert_not_called()

    def test_browser_launch_is_a_normal_chrome_tab(self):
        from desktop import open_browser
        with patch('desktop.browser_executable',return_value=Path('C:/Chrome/chrome.exe')), patch('desktop.subprocess.Popen') as spawn:
            open_browser(SimpleNamespace(desktop_window=False,browser=None),'http://127.0.0.1:47821')
            command=spawn.call_args.args[0]
            self.assertEqual(command[1:],['--new-tab','http://127.0.0.1:47821'])
            self.assertNotIn('--app',str(command))

    def test_download_stop_refuses_before_signalling_owner(self):
        from desktop import stop_server
        for catalog in ({'downloads':[{'status':'running'}]}, {'transfer':{'status':'running'}}):
            with patch('desktop.read_state',return_value={'url':'http://127.0.0.1:1'}), patch('desktop.matching_server',return_value=True), patch('desktop.get_json',side_effect=[{'active':None,'pending':[]},catalog]), patch('desktop.Instance') as instance:
                with self.assertRaisesRegex(RuntimeError,'Model downloads are running'):
                    stop_server(Path('C:/test'),Path('C:/test/runtime.json'))
                instance.assert_not_called()

    def test_independent_server_reuse_restart_browser_close_and_owner_exit(self):
        with tempfile.TemporaryDirectory(prefix='qwen-lifecycle-') as folder:
            folder=Path(folder); state=folder/'runtime.json'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            args=[sys.executable,'-u',str(APP/'scripts/desktop.py'),'--data-dir',str(folder),
                  '--runtime-file',str(state),'--port',str(port),'--no-browser','--no-console']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QWEN_REMOTE='0',QWEN_NO_WARMUP='1')
            env.pop('ELECTRON_RUN_AS_NODE',None)
            def run(extra=()):
                result=subprocess.run([*args,*extra],cwd=APP,env=env,capture_output=True,text=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return result
            def ready():
                value=json.loads(state.read_text())
                with urlopen(value['url']+'/api/health',timeout=1) as response:
                    return response.status==200
            try:
                run()
                self.assertTrue(ready())
                old=json.loads(state.read_text())
                self.assertTrue(old['independent'])
                # A detached venv redirector used to create a visible blank
                # console for the real interpreter. Inspect the owner's console
                # from a separate process so this test keeps its own handles.
                probe='''import ctypes,json,sys
k=ctypes.WinDLL("kernel32",use_last_error=True)
k.FreeConsole()
attached=bool(k.AttachConsole(int(sys.argv[1])))
k.GetConsoleWindow.restype=ctypes.c_void_p
window=k.GetConsoleWindow()
visible=bool(ctypes.windll.user32.IsWindowVisible(ctypes.c_void_p(window))) if window else False
print(json.dumps({"attached":attached,"visible":visible}),flush=True)
k.FreeConsole()
'''
                inspection=subprocess.run([sys.executable,'-c',probe,str(old['owner_pid'])],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                console=json.loads(inspection.stdout)
                self.assertTrue(console['attached'])
                self.assertFalse(console['visible'],'Background owner must not create a visible blank terminal')
                # The launcher returns. Its server remains alive. Reopening reuses it.
                run()
                self.assertEqual(json.loads(state.read_text())['owner_pid'],old['owner_pid'])
                self.assertTrue(alive(old['pid']))
                # The optional Electron client loads the app and closes its own window.
                electron=APP/'node_modules/electron/dist/electron.exe'
                self.assertTrue(electron.is_file(),'Install the optional desktop client for this integration test')
                result=subprocess.run([str(electron),str(APP/'scripts/electron.cjs'),'--url='+old['url'],'--check-close'],cwd=APP,env=env,capture_output=True,text=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('page loaded',result.stdout)
                self.assertTrue(ready())
                self.assertTrue(alive(old['pid']))
                # Clients can reset sockets without stopping the server.
                for _ in range(5):
                    connection=socket.create_connection(('127.0.0.1',old['port']))
                    connection.sendall(b'GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n')
                    connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('hh',1,0))
                    connection.close()
                run(['--restart'])
                current=json.loads(state.read_text())
                self.assertNotEqual(current['owner_pid'],old['owner_pid'])
                self.assertFalse(alive(old['pid']))
                self.assertTrue(ready())
                # A background-owner crash cleans up only its own server descendants.
                subprocess.run(['taskkill','/PID',str(current['owner_pid']),'/F'],capture_output=True,check=True)
                eventually(lambda:not alive(current['pid']))
                self.assertNotIn('_ProactorBasePipeTransport',(folder/'logs/server.err.log').read_text())
            finally:
                subprocess.run([*args,'--stop'],cwd=APP,env=env,capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)


if __name__=='__main__':
    unittest.main()

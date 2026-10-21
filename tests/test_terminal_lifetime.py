"""Real Windows owner/viewer/process tests with CPU-only fake job children."""
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP/'scripts'))
if os.name == 'nt':
    from desktop import matching_server, open_console, stop_server
    from windows_processes import Instance, close_handle, open_process, wait_handle, windows, post_message
    from watch_logs import viewer_name


def eventually(check, timeout=15):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        try:
            if check():
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.05)
    raise AssertionError('Expected lifecycle state did not arrive')


def alive(pid):
    handle = open_process(0x100000, False, pid)
    if not handle:
        return False
    try:
        return wait_handle(handle, 0) == 258
    finally:
        close_handle(handle)


FAKE_SERVER = r'''
import json, os, subprocess, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
data=Path(os.environ['QWEN_DATA_DIR'])
child=subprocess.Popen([sys.executable,'-c',
    "import json,os,sys,time; from pathlib import Path; held=bytearray(8*1024*1024); Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(120)",
    str(data/'worker.json')])
(data/'server.json').write_text(json.dumps({'pid':os.getpid(),'child':child.pid}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/api/health':
            value={'app':'image-studio-qwen21','app_dir':str(Path.cwd()),'data_dir':str(data)}
        elif self.path=='/api/queue':
            value={'active':{'id':'a'*32,'status':'running'} if (data/'busy').exists() else None,'pending':[],'recent':[]}
        elif self.path=='/api/models':
            value={'models':[],'transfer':{'status':'idle'},'downloads':[]}
        else:
            value={'active':{'id':'a'*32,'status':'running'} if (data/'busy').exists() else None,
                   'pending_count':0,'paused':False,'transfer':{'status':'idle'},'downloads':[]}
        body=json.dumps(value).encode()
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass
HTTPServer(('127.0.0.1',int(os.environ['QWEN_SERVER_PORT'])),Handler).serve_forever()
'''

OWNER = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()/'scripts'))
import desktop
from windows_processes import Job as WindowsJob
fake=Path(sys.argv[1]).read_text()
class TestJob(WindowsJob):
    def spawn(self,command,**kwargs):
        return super().spawn([sys.executable,'-u','-c',fake],**kwargs)
desktop.Job=TestJob
desktop.serve(SimpleNamespace(data_dir=Path(sys.argv[2]),runtime_file=Path(sys.argv[3]),port=int(sys.argv[4])))
'''


@unittest.skipUnless(os.name == 'nt', 'Windows terminal lifetime')
class TerminalLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen-terminal-test-')
        self.data = Path(self.temp.name)
        self.runtime = self.data/'runtime.json'
        self.fake = self.data/'fake-server.py'
        self.fake.write_text(FAKE_SERVER)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{self.port}'
        self.children = []
        self.addCleanup(self.cleanup)
        self.env = dict(os.environ, QWEN_AUTH_DIR=str(self.data/'auth'), QWEN_REMOTE='0',
                        QWEN_NO_WARMUP='1', CUDA_VISIBLE_DEVICES='')
        self.owner = self.spawn(['-c', OWNER, str(self.fake), str(self.data), str(self.runtime), str(self.port)])
        eventually(lambda: matching_server(self.url, self.data))
        eventually(lambda: (self.data/'worker.json').exists())
        self.state = json.loads(self.runtime.read_text())
        self.worker_pid = json.loads((self.data/'worker.json').read_text())['pid']
        self.server_pid = json.loads((self.data/'server.json').read_text())['pid']

    def spawn(self, args):
        process = subprocess.Popen([sys.executable, '-u', *args], cwd=APP, env=self.env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        self.children.append(process)
        return process

    def end_tree(self, process):
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        # Closing one tab can stop another between poll() and taskkill().
        # A confirmed exit is success; a process that stays alive fails wait().
        process.wait(timeout=10)

    def cleanup(self):
        for child in reversed(self.children):
            self.end_tree(child)
        if hasattr(self, 'worker_pid'):
            eventually(lambda: not alive(self.worker_pid) and not alive(self.server_pid))
        self.temp.cleanup()

    def viewer(self, channel, session=None):
        session = session or self.state['terminal_session']
        process = self.spawn([str(APP/'scripts/watch_logs.py'), '--url', self.url,
                              '--data-dir', str(self.data), '--channel', channel,
                              '--owner-session', session, '--owner-pid', str(self.state['owner_pid'])])
        def connected():
            control = Instance(viewer_name(self.data, channel, session))
            try:
                return not control.acquire()
            finally:
                control.close()
        eventually(connected)
        return process

    def test_closing_activity_stops_active_worker_and_other_tab(self):
        (self.data/'busy').write_text('active test job')
        activity = self.viewer('activity')
        server_tab = self.viewer('server')
        unrelated = self.spawn(['-c', 'import time; time.sleep(120)'])
        with self.assertRaisesRegex(RuntimeError, 'Nothing was stopped'):
            stop_server(self.data, self.runtime)
        self.end_tree(activity)  # Simulates abrupt tab closure; no Python cleanup runs.
        eventually(lambda: not alive(self.state['owner_pid']))
        eventually(lambda: not alive(self.worker_pid) and not alive(self.server_pid))
        eventually(lambda: server_tab.poll() is not None)
        self.assertIsNone(unrelated.poll())
        self.assertFalse(self.runtime.exists())
        self.assertFalse(matching_server(self.url, self.data))

    def test_closing_server_tab_also_stops_the_app(self):
        activity = self.viewer('activity')
        server_tab = self.viewer('server')
        self.end_tree(server_tab)
        eventually(lambda: not alive(self.worker_pid) and not alive(self.server_pid))
        eventually(lambda: activity.poll() is not None)

    def test_old_session_and_unopened_tabs_do_not_stop_new_owner(self):
        stale = self.viewer('activity', session='old-session')
        self.end_tree(stale)
        time.sleep(.4)
        self.assertTrue(alive(self.state['owner_pid']))
        self.assertTrue(alive(self.worker_pid))
        current = self.viewer('activity')
        self.end_tree(current)
        eventually(lambda: not alive(self.worker_pid))

    def test_owner_exit_stops_viewers_and_its_worker_tree(self):
        activity = self.viewer('activity')
        self.end_tree(self.owner)
        eventually(lambda: activity.poll() is not None)
        eventually(lambda: not alive(self.worker_pid) and not alive(self.server_pid))

    def test_reopening_terminal_reuses_both_tabs(self):
        launched = []

        def launch_terminal(command, **kwargs):
            launched.append(command)
            for channel in ('activity', 'server'):
                if '--channel' in command and channel in command:
                    self.viewer(channel)
            return subprocess.CompletedProcess(command, 0)

        with patch('desktop.shutil.which', return_value='wt.exe'), \
                patch('desktop.subprocess.run', side_effect=launch_terminal):
            open_console(self.state, self.data)
            open_console(self.state, self.data)
        self.assertEqual(launched[0].count('new-tab'), 2)
        self.assertNotIn('new-tab', launched[1])
        self.assertTrue(alive(self.worker_pid))

    @unittest.skipUnless(shutil.which('wt.exe'), 'Windows Terminal is not installed')
    def test_actual_terminal_window_close_stops_busy_server_and_worker(self):
        title = 'Qwen close check '+self.state['terminal_session'][:8]
        (self.data/'busy').write_text('active test job')
        command = [shutil.which('wt.exe'), '-w', title, 'new-tab', '--title', title,
                   '--suppressApplicationTitle', '-d', str(APP), sys.executable, '-u',
                   str(APP/'scripts/watch_logs.py'), '--url', self.url, '--data-dir', str(self.data),
                   '--owner-session', self.state['terminal_session'],
                   '--owner-pid', str(self.state['owner_pid'])]
        subprocess.run(command, check=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        found = []
        def find_window():
            found[:] = [hwnd for hwnd, pid, text in windows() if title in text]
            return len(found) == 1
        eventually(find_window)
        def connected():
            control = Instance(viewer_name(self.data, 'activity', self.state['terminal_session']))
            try:
                return not control.acquire()
            finally:
                control.close()
        eventually(connected)
        self.assertTrue(alive(self.worker_pid))
        post_message(found[0], 0x10, 0, 0)  # WM_CLOSE on this test window only.
        eventually(lambda: not alive(self.server_pid) and not alive(self.worker_pid))
        eventually(lambda: not any(title in text for _, _, text in windows()))


if __name__ == '__main__':
    unittest.main()

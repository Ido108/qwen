import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import httpx

from backend import moise_remote


class App(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *args):
        pass

    def answer(self):
        size = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(size) if size else b''
        App.seen.append({'method': self.command, 'path': self.path, 'token': self.headers.get('x-studio-token'),
                         'range': self.headers.get('range'), 'cookie': self.headers.get('cookie'), 'body': body})
        data = json.dumps({'ok': True}).encode()
        self.send_response(206 if self.headers.get('range') else 200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('X-Private', 'not returned')
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = answer


class MoiseRemoteTests(unittest.TestCase):
    def test_short_failures_are_grouped_and_status_is_reported_without_secrets(self):
        response=httpx.Response(502, request=httpx.Request('POST','https://secret.example/?key=private'))
        error=httpx.HTTPStatusError('private body',request=response.request,response=response)
        report=moise_remote.ConnectionReport('https://moise.example')
        with mock.patch('builtins.print') as printed:
            report.succeeded(0)
            self.assertEqual(report.failed(error,1),1)
            report.succeeded(2)
            report.failed(error,4)
            report.succeeded(5)
            self.assertEqual(printed.call_count,2)
            report.succeeded(36)
            self.assertEqual(printed.call_count,3)
            messages=' '.join(str(call.args) for call in printed.call_args_list)
            self.assertIn('HTTP 502',messages)
            self.assertNotIn('private',messages)
            self.assertNotIn('secret.example',messages)

    def test_retry_backoff_respects_server_limits(self):
        report=moise_remote.ConnectionReport('https://moise.example')
        response=httpx.Response(429,headers={'Retry-After':'45'},request=httpx.Request('POST','https://moise.example'))
        error=httpx.HTTPStatusError('',request=response.request,response=response)
        with mock.patch('builtins.print'):
            self.assertEqual(report.failed(error,0),45)
            timeout=httpx.ReadTimeout('secret')
            self.assertEqual([report.failed(timeout,i) for i in range(5)],[2,4,8,16,30])

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), App)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        App.seen.clear()

    def test_open_video_range_is_answered_in_parts(self):
        self.assertEqual(moise_remote.limit_range('bytes=0-'), f'bytes=0-{moise_remote.RANGE_PART - 1}')
        self.assertEqual(moise_remote.limit_range('bytes=100-'), f'bytes=100-{100 + moise_remote.RANGE_PART - 1}')
        for kept in ('bytes=0-99', 'bytes=-500', 'bytes=0-1,5-9', 'items=0-'):
            self.assertEqual(moise_remote.limit_range(kept), kept)

    def test_request_reaches_the_app_with_token_body_and_range_only(self):
        with httpx.Client() as client:
            status, headers, data = moise_remote.run_request(client, self.url, {
                'method': 'POST', 'path': '/api/uploads',
                'headers': {'content-type': 'multipart/form-data; boundary=x', 'x-studio-token': 't', 'cookie': 'moise', 'host': 'evil'}}, b'PNG')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data), {'ok': True})
            self.assertEqual(headers, {'Content-Type': 'application/json'})
            self.assertEqual(App.seen[-1], {'method': 'POST', 'path': '/api/uploads', 'token': 't', 'range': None, 'cookie': None, 'body': b'PNG'})
            status, _, _ = moise_remote.run_request(client, self.url, {'method': 'GET', 'path': '/api/videos/a.mp4', 'headers': {'range': 'bytes=0-'}}, b'')
            self.assertEqual(status, 206)
            self.assertEqual(App.seen[-1]['range'], f'bytes=0-{moise_remote.RANGE_PART - 1}')

    def test_paths_outside_the_app_are_refused(self):
        with httpx.Client() as client:
            for path in ('api/x', '//evil.example/x', '/api/../../secret', 'http://evil/x'):
                status, _, _ = moise_remote.run_request(client, self.url, {'method': 'GET', 'path': path}, b'')
                self.assertEqual(status, 400, path)
        self.assertEqual(App.seen, [])

    def test_starts_only_with_https_moise_and_a_token(self):
        def settings(values):
            return mock.patch.object(moise_remote, 'setting', side_effect=lambda name: values.get(name, ''))
        with settings({'MOISE_SERVER_URL': 'http://moise.example', 'MOISE_RELAY_TOKEN': 'x'}):
            self.assertIsNone(moise_remote.start(1))
        with settings({'MOISE_SERVER_URL': 'https://moise.example'}):
            self.assertIsNone(moise_remote.start(1))
        with settings({'MOISE_SERVER_URL': 'https://moise.example', 'MOISE_RELAY_TOKEN': 'x', 'QWEN_REMOTE': '0'}):
            self.assertIsNone(moise_remote.start(1))
        with settings({'MOISE_SERVER_URL': 'https://moise.example/', 'MOISE_RELAY_TOKEN': 'x'}), \
                mock.patch.object(moise_remote, 'serve') as serve, mock.patch.object(moise_remote.threading, 'Thread') as thread:
            stop = moise_remote.start(47821)
            self.assertIsNotNone(stop)
            args = thread.call_args.kwargs['args']
            self.assertEqual(args[:3], ('https://moise.example', 'x', 'http://127.0.0.1:47821'))


    def test_poll_tells_moise_the_pc_name_gpu_and_queue(self):
        smi = mock.Mock(stdout='NVIDIA GeForce RTX 5090, 32607, 30001\n')
        cache = {'until': 0.0, 'value': {}}
        with mock.patch.object(moise_remote.subprocess, 'run', return_value=smi) as run:
            info = moise_remote.gpu_info(cache)
            self.assertEqual(moise_remote.gpu_info(cache), info)
        self.assertEqual(run.call_count, 1, 'read once, then from the cache')
        self.assertEqual(info, {'gpu': 'NVIDIA GeForce RTX 5090', 'vramTotalMb': 32607, 'vramFreeMb': 30001})
        with mock.patch.dict(moise_remote.os.environ, {'COMPUTERNAME': 'PC-5090'}), \
                mock.patch.object(moise_remote, 'gpu_info', return_value=info):
            body = moise_remote.poll_body('w1', lambda: {'busy': True, 'queue': 3})
            self.assertEqual(body, {'workerId': 'w1', 'name': 'PC-5090', **info, 'busy': True, 'queue': 3})
            self.assertEqual(moise_remote.poll_body('w1', lambda: 1 / 0)['name'], 'PC-5090', 'a broken status never stops the poll')
        with mock.patch.object(moise_remote.subprocess, 'run', side_effect=FileNotFoundError):
            self.assertEqual(moise_remote.gpu_info({'until': 0.0, 'value': {}}), {}, 'no nvidia-smi: nothing is sent')


if __name__ == '__main__':
    unittest.main()

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


if __name__ == '__main__':
    unittest.main()

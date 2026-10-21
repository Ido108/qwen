"""Pinned llama.cpp for the local prompt rewriters. Its generation loop measured about 120 tokens per second for
the 9B prompt models on an RTX 5090, against about 11 in ComfyUI's TextGenerate (2026-10-21)."""
from pathlib import Path
import atexit
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from .common import DATA_DIR

RELEASE = 'b11489'
ASSETS = {  # ggml-org/llama.cpp release b11489 (2026-10-08), Windows CUDA 13.4 build and its CUDA runtime
    f'llama-{RELEASE}-bin-win-cuda-13.4-x64.zip': 'f2c407f114c312fab3411156dbc5cd3153b4fc6ee51812ad844718846b95a952',
    'cudart-llama-bin-win-cuda-13.4-x64.zip': '738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668',
}
ROOT = Path(os.environ.get('QWEN_RUNTIME_DIR') or DATA_DIR) / 'tools' / 'llama.cpp' / RELEASE
SERVER = ROOT / 'bin' / 'llama-server.exe'


def ensure(progress=lambda **_: None):
    """Download and verify the pinned release once; the files stay for later jobs."""
    if SERVER.is_file():
        return SERVER
    progress(stage='installing_tools', message='Installing the local prompt rewriter tools…')
    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT.parent) as folder:
        for name, expected in ASSETS.items():
            archive = Path(folder) / name
            urllib.request.urlretrieve(f'https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{name}', archive)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f'The downloaded llama.cpp file {name} does not match its published checksum.')
            with zipfile.ZipFile(archive) as source:
                source.extractall(Path(folder) / 'bin')
        shutil.move(str(Path(folder) / 'bin'), str(ROOT / 'bin'))
    return SERVER


def free_port():
    """A free local port in 20000 to 60000 (common development ports stay untouched)."""
    for _ in range(50):
        port = random.randint(20000, 60000)
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError('No free local port for the prompt rewriter.')


class Server:
    """One llama-server for one rewrite: it holds the GPU only while it runs, then exits."""
    def __init__(self, model, mmproj=None, context=32768):
        self.port = free_port()
        command = [str(ensure()), '-m', str(model), '--host', '127.0.0.1', '--port', str(self.port), '-ngl', '999',
                   '-c', str(context), '--jinja', '-fa', 'on', '--no-webui', '--no-warmup']
        if mmproj:
            command += ['--mmproj', str(mmproj)]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('The local prompt rewriter stopped while loading.')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2) as reply:
                    if json.load(reply).get('status') == 'ok':
                        return
            except OSError:
                time.sleep(0.3)
        self.close()
        raise RuntimeError('The local prompt rewriter did not start.')

    def chat(self, body):
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3600) as reply:
            return json.load(reply)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# One rewriter can stay loaded between jobs (measured 2026-10-21: starting Qwen PE Q8 took 5.6 to 6.2 s of a 10.3 s
# Enhance, writing 3.3 to 3.6 s). The caller decides after each rewrite whether the GPU has room to keep it.
_resident = None


def is_resident(model, mmproj=None):
    return bool(_resident and _resident[0] == (str(model), str(mmproj) if mmproj else None) and _resident[1].process.poll() is None)


def shared(model, mmproj=None):
    """The loaded rewriter for these files, started if needed (any other one is closed first)."""
    global _resident
    if is_resident(model, mmproj):
        return _resident[1]
    release()
    server = Server(model, mmproj)
    _resident = ((str(model), str(mmproj) if mmproj else None), server)
    return server


def release():
    global _resident
    if _resident:
        _resident[1].close()
        _resident = None


# The worker exits through its stop message; a hung worker is ended with its whole process tree.
atexit.register(release)

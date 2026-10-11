"""One GPU owner: reuse image jobs, release before video/model changes."""
import json
import os
import subprocess
import sys
import time

from .common import APP_DIR
from .models import required_image


def stop_process(process, timeout=30):
    """Wait for a worker to exit; end the whole process tree if it hangs.

    On Windows the venv python.exe is a launcher with a child interpreter that
    holds the GPU memory, so the tree must be ended, not only the launcher.
    """
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(process.pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.kill()
        process.wait(timeout=30)


class ImageWorker:
    def __init__(self):
        self.process = None
        self.key = None

    def status(self):
        process, key = self.process, self.key
        loaded = bool(key and process is not None and process.poll() is None)
        return {'loaded': loaded, 'model_id': key[0] if loaded else None,
                'text_encoder_id': key[1] if loaded else None}

    def close(self):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write('{"stop":true}\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            # Only called while idle or after a finished job. Never overlap GPUs.
            stop_process(process)
        try:
            process.stdin.close()
        except OSError:
            pass
        self.process = self.key = None

    def start(self, directory, request):
        # The worker keeps model, encoder and VAE loaded; tool LoRAs are applied per job.
        key = tuple(required_image(request)[:3])
        if self.process is not None and (self.key != key or self.process.poll() is not None):
            self.close()
        if self.process is None:
            with (directory / 'worker.log').open('a', encoding='utf-8') as log:
                self.process = subprocess.Popen(
                    [sys.executable, '-X', 'faulthandler', '-u', '-m', 'backend.worker', '--persistent'],
                    cwd=APP_DIR, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
            self.key = key
        (directory / 'worker.done.json').unlink(missing_ok=True)
        self.process.stdin.write(json.dumps({'directory': str(directory)}) + '\n')
        self.process.stdin.flush()
        return self.process

    def wait(self, directory, process):
        complete = directory / 'worker.done.json'
        while True:
            if complete.exists():
                code = json.loads(complete.read_text(encoding='utf-8'))['returncode']
                if code:
                    # A failed worker exits; a CUDA error can leave it hanging on exit.
                    stop_process(process)
                    self.close()
                return code
            code = process.poll()
            if code is not None:
                self.close()
                return code
            time.sleep(.1)

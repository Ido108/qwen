"""Read-only console for an existing Qwen server and its current worker."""
from __future__ import annotations

import argparse
import codecs
import json
import re
from pathlib import Path
import time
from urllib.request import urlopen


class LogTail:
    def __init__(self, path: Path, label: str):
        self.path, self.label = path, label
        self.offset = None
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.pending = ''

    def read(self):
        try:
            with self.path.open('rb') as file:
                file.seek(0, 2)
                size = file.tell()
                if self.offset is None or size < self.offset:
                    self.offset = max(0, size-12000)
                    self.decoder.reset()
                file.seek(self.offset)
                data = file.read(128*1024)
                self.offset = file.tell()
            if data:
                content = self.decoder.decode(data)
                if self.label == 'Server':
                    # Keep worker progress and all errors; routine polling would
                    # otherwise push the useful generation output off screen.
                    lines = (self.pending + content).splitlines(keepends=True)
                    self.pending = lines.pop() if lines and not lines[-1].endswith(('\n','\r')) else ''
                    content = ''.join(line for line in lines if not re.search(r'"GET .* HTTP/\S+" (200|206|304)\b', line))
                if content:
                    print(f'\n[{self.label}]', flush=True)
                    print(content, end='', flush=True)
        except (FileNotFoundError, PermissionError):
            pass


def watch(url, data_dir, *, once=False, stop_event=None):
    tails=[LogTail(data_dir/'logs/server.out.log','Server'),LogTail(data_dir/'logs/server.err.log','Server')]
    worker=None
    active_id=None
    last_state=None
    while not stop_event or not stop_event.is_set():
        try:
            with urlopen(url+'/api/queue',timeout=2) as response:
                queue=json.load(response)
            active=queue.get('active') or {}
            if active.get('id')!=active_id:
                if worker:
                    worker.read()
                active_id=active.get('id')
                worker=LogTail(data_dir/'jobs'/active_id/'worker.log','Generation') if active_id else None
            state=(active_id,active.get('stage'),active.get('step'),active.get('message'),len(queue.get('pending') or []))
            if state!=last_state:
                if active_id:
                    unit='batches' if active.get('kind')=='upscale' else 'steps'
                    progress=f"| {active.get('step',0)}/{active['total_steps']} {unit} " if active.get('total_steps') else ''
                    print(f'\n[{time.strftime("%H:%M:%S")}] {active.get("message","Working")} '
                          f'{progress}'
                          f'| {state[-1]} queued | Job {active_id[:8]}',flush=True)
                else:
                    print(f'\n[{time.strftime("%H:%M:%S")}] Ready | {state[-1]} queued',flush=True)
                last_state=state
        except (OSError,ValueError):
            if last_state!='offline':
                print('\nWaiting for the Qwen server...',flush=True)
                last_state='offline'
        for tail in tails:
            tail.read()
        if worker:
            worker.read()
        if once:
            return
        if stop_event:
            stop_event.wait(1)
        else:
            time.sleep(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:47821')
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    print(f'Qwen activity viewer\n{args.url}\nThis extra viewer does not control Qwen. Close the main Qwen window to stop the app.\n',flush=True)
    watch(args.url,args.data_dir,once=args.once)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nActivity viewer closed.')

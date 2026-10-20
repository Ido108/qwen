"""Pinned FLUX.3 support, isolated from the existing Qwen/video runtime."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from .common import DATA_DIR

COMMIT = '6607f36b53c7b695b160ac5b52d595e9e3752a2d'
PACKAGES = ['comfy-kitchen==0.2.38', 'comfy-aimdo==0.5.6']
ROOT = Path(os.environ.get('QWEN_RUNTIME_DIR') or DATA_DIR) / 'flux3'
COMFY = ROOT / ('ComfyUI-' + COMMIT)
LIB = ROOT / 'packages'


def ensure(progress=lambda **_: None):
    stamp=ROOT/'ready.json'
    expected={'comfy':COMMIT,'packages':PACKAGES}
    try:
        if json.loads(stamp.read_text())==expected and (COMFY/'comfy/sd.py').is_file():return
    except (OSError,ValueError):pass
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (COMFY/'comfy/sd.py').is_file():
        progress(stage='installing_tools',message='Installing FLUX.3 image tools…')
        archive=ROOT/(COMMIT+'.zip')
        if not archive.is_file():
            urllib.request.urlretrieve('https://codeload.github.com/Comfy-Org/ComfyUI/zip/'+COMMIT,archive)
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()):raise ValueError('Invalid runtime archive path.')
            source.extractall(ROOT)
    progress(stage='installing_tools',message='Installing FLUX.3 image tools…')
    uv=shutil.which('uv')
    if not uv:
        bundled=ROOT.parent/'tools/uv/uv.exe'
        if bundled.is_file():uv=str(bundled)
    installer=[uv,'pip','install','--python',sys.executable] if uv else [sys.executable,'-m','pip','install','--disable-pip-version-check']
    with (ROOT/'setup.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([*installer,'--no-deps',
            '--upgrade','--target',str(LIB),*PACKAGES],stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError('FLUX.3 tools could not install. See '+str(ROOT/'setup.log'))
    stamp.write_text(json.dumps(expected),encoding='utf-8')


def activate(progress):
    ensure(progress)
    if 'comfy' in sys.modules:
        # Comfy is a namespace package in this revision, so __file__ is None.
        paths={Path(path).resolve().parent for path in sys.modules['comfy'].__path__}
        if paths!={COMFY.resolve()}:
            raise RuntimeError('FLUX.3 needs its own image worker. Select the model and retry.')
    for folder in (COMFY,LIB):
        if str(folder) not in sys.path:sys.path.insert(0,str(folder))

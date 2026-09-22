"""Exercise the actual model downloader on a small pinned model config, in isolation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from backend import models
from backend.common import DATA_DIR, MODELS_DIR

source=models.entry('official-bf16')
file=next(f for f in source['files'] if f['path']=='transformer/config.json').copy()
file['sha256']=hashlib.sha256((MODELS_DIR/source['directory']/file['path']).read_bytes()).hexdigest()
fixture={**source,'id':'download-test','directory':'config-only','files':[file],'bytes':file['bytes'],'dependencies':[]}
parent=DATA_DIR/'validation'
parent.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='download-check-',dir=parent) as directory:
    root=Path(directory).resolve()
    assert root.is_relative_to(parent.resolve())
    with patch.multiple(models,DATA_DIR=root,MODELS_DIR=root/'models',INSTALL_PATH=root/'installed.json'),patch.object(models,'catalog',return_value=[fixture]):
        models.transfer.update(status='running')
        models.install('download-test')
        assert models.transfer['status']=='succeeded',models.transfer
        assert models.is_installed('download-test')
        downloaded=models.local_path(fixture,file)
        assert hashlib.sha256(downloaded.read_bytes()).hexdigest()==file['sha256']
        models.remove('download-test')
        assert not downloaded.exists()
        print('PASS: real Hugging Face download, checksum, installed state, and isolated removal.',flush=True)

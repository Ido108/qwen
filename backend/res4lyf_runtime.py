"""Pinned RES4LYF algorithms without ComfyUI's web-server/node startup.

Only the UI logging/config interface is replaced. The numerical sampler and
its helper modules run unchanged from the verified upstream source.
"""
from functools import lru_cache
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COMMIT = '56e44755d866c87ff9cb4b6608c8e42155db6ecc'
SAMPLERS = ('res_2m', 'res_2s', 'res_2m_ode', 'res_2s_ode', 'deis_2m')  # deis_2m: the BFS Qwen swap workflows
MANIFEST = Path(__file__).resolve().parent.parent / 'reference' / 'res4lyf-source.json'
PACKAGE = '_qwen_res4lyf_' + COMMIT


def source_root():
    from .common import DATA_DIR
    return Path(os.environ.get('QWEN_RUNTIME_DIR') or DATA_DIR) / 'samplers' / ('RES4LYF-' + COMMIT)


def source_files():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['commit'] != COMMIT:
        raise RuntimeError('The sampler source list does not match this app version.')
    return manifest['files']


def valid_file(root, item):
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Invalid sampler source path.')
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']


def install():
    """Install verified source only. Python dependencies belong in lock.txt."""
    root = source_root()
    def fetch(item):
        if valid_file(root, item):
            return
        url = f'https://raw.githubusercontent.com/ClownsharkBatwing/RES4LYF/{COMMIT}/{item["path"]}'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Sampler download verification failed: ' + item['path'])
        path = root / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.download')
        temp.write_bytes(data)
        temp.replace(path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, source_files()))
    return root


def _log(*args, level='info', debug=None):
    logger = logging.getLogger('qwen.res4lyf')
    severity = logging.DEBUG if debug else getattr(logging, str(level).upper(), logging.INFO)
    logger.log(severity, '%s', ' '.join(map(str, args)))


@lru_cache(maxsize=1)
def algorithm():
    root = source_root()
    if any(not valid_file(root, item) for item in source_files()):
        raise RuntimeError('Sampler files are missing or damaged. Run Qwen setup to repair them.')
    # Namespace packages bypass extension entry points that start a web server
    # and load unrelated image/model nodes. No sampling code is replaced.
    for name, folder in ((PACKAGE, root), (PACKAGE + '.beta', root / 'beta')):
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module
    config = types.ModuleType(PACKAGE + '.res4lyf')
    config.RESplain = _log
    config.is_debug_logging_enabled = lambda: logging.getLogger('qwen.res4lyf').isEnabledFor(logging.DEBUG)
    config.get_display_sampler_category = lambda: False
    sys.modules[config.__name__] = config
    try:
        return importlib.import_module(PACKAGE + '.beta.rk_sampler_beta').sample_rk_beta
    except ImportError as error:
        raise RuntimeError('A sampler library is missing. Run Qwen setup to repair it.') from error


def _wrapper(name):
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        options = {'eta': 0.0, 'eta_substep': 0.0} if name.endswith('_ode') else {}
        def progress(event):
            # RES4LYF also emits a post-loop preview. Comfy's callback treats
            # every event as a completed step, so omit that duplicate event.
            if callback is not None and not event.get('final'):
                callback(event)
        # Match beta/__init__.py exactly, including its SDE defaults.
        return algorithm()(model, x, sigmas, None, extra_args, progress if callback else None, disable,
                           rk_type=name.removesuffix('_ode'), **options)
    sample.__name__ = 'sample_' + name
    return sample


def ensure_sampler(name):
    """Register the exact requested algorithm; never substitute another one."""
    if name not in SAMPLERS:
        return
    algorithm()
    import comfy.samplers
    for sampler in SAMPLERS:
        setattr(comfy.samplers.k_diffusion_sampling, 'sample_' + sampler, _wrapper(sampler))
        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            comfy.samplers.KSampler.SAMPLERS.append(sampler)


if __name__ == '__main__':
    print('Ready: RES4LYF source at', install())

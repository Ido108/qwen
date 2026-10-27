"""LoRA training with ostris ai-toolkit, run as a command-line process in its own Python environment.

ai-toolkit (https://github.com/ostris/ai-toolkit) trains LoRAs for every image architecture this app runs:
Qwen-Image-2.1 (and Turbo), Krea 2 (Raw and Turbo) and Z-Image (Turbo and Base). Its requirements pin other
versions of torch, diffusers and transformers than the app, so it gets its own environment in the runtime folder
(installed once, from the Train page). Each training is a queue job: it never shares the GPU with a generation.
The settings per architecture are the defaults of the ai-toolkit UI (extensions_built_in/diffusion_models/ui.tsx).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import APP_DIR, DATA_DIR, MODELS_DIR, write_json

AITK_REPO = 'https://github.com/ostris/ai-toolkit.git'
AITK_COMMIT = 'dcebc7442df70e5239c68167919fd2909c615769'  # 2026-10-16
# The versions the ai-toolkit README installs on Windows (CUDA 13.0, which the RTX 50 series needs).
TORCH = ['torch==2.13.0', 'torchvision==0.28.0', 'torchaudio==2.11.0']
TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'
# ConvRot's fused activation-quant kernel and torchao need Triton; the ai-toolkit requirements do not install it on Windows.
TRITON = 'triton-windows>=3.6,<3.9' if os.name == 'nt' else None
# A tiny compiled CUDA kernel: an install that does not work with this torch build is removed, and the trainer runs without it.
TRITON_CHECK = 'import torch; f=torch.compile(lambda x: x*2+1); assert f(torch.ones(8, device="cuda")).sum().item()==24'
RUNTIME_DIR = Path(os.environ.get('QWEN_RUNTIME_DIR') or DATA_DIR).resolve()
TRAINER_DIR = RUNTIME_DIR / 'trainer'
TOOLKIT_DIR = TRAINER_DIR / 'ai-toolkit'
TRAINER_PYTHON = TRAINER_DIR / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
SETUP_STAMP = TRAINER_DIR / 'setup-complete.json'
SETUP_LOG = TRAINER_DIR / 'setup.log'
TRAINED_DIR = MODELS_DIR / 'trained'
TRAININGS_DIR = DATA_DIR / 'trainings'

_setup = {'status': 'idle'}
_setup_lock = threading.Lock()


def setup_status() -> dict:
    """'ready' when the pinned ai-toolkit and its environment are installed."""
    try:
        stamp = json.loads(SETUP_STAMP.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        stamp = {}
    # 'triton' is recorded by the current setup; an older installation runs the setup once more to add it.
    ready = stamp.get('commit') == AITK_COMMIT and (TRITON is None or 'triton' in stamp) and TRAINER_PYTHON.is_file() and (TOOLKIT_DIR / 'run.py').is_file()
    with _setup_lock:
        state = dict(_setup)
    if ready and state.get('status') != 'running':
        state = {'status': 'ready'}
    return {**state, 'ready': ready, 'directory': str(TRAINER_DIR), 'commit': AITK_COMMIT}


def _uv() -> str | None:
    found = shutil.which('uv')
    local = RUNTIME_DIR / 'tools' / 'uv' / ('uv.exe' if os.name == 'nt' else 'uv')
    return found or (str(local) if local.is_file() else None)


def _base_python() -> str:
    """The interpreter the app's own environment was made from (Python 3.12)."""
    return getattr(sys, '_base_executable', None) or sys.executable


def start_setup() -> dict:
    with _setup_lock:
        if _setup.get('status') == 'running':
            return dict(_setup)
        _setup.clear()
        _setup.update(status='running', stage='starting', message='Starting the trainer installation…', started_at=time.time())
    threading.Thread(target=_run_setup, name='trainer-setup', daemon=True).start()
    return setup_status()


def _step(stage: str, message: str) -> None:
    with _setup_lock:
        _setup.update(stage=stage, message=message)


def _run(command: list[str], log, cwd: Path | None = None, env: dict | None = None) -> None:
    log.write(f'\n$ {" ".join(command)}\n'); log.flush()
    result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name} {command[1] if len(command) > 1 else ""} failed (code {result.returncode}). See {SETUP_LOG}.')


def _run_setup() -> None:
    try:
        TRAINER_DIR.mkdir(parents=True, exist_ok=True)
        git = shutil.which('git')
        if not git:
            raise RuntimeError('Git is needed to install the trainer. Install Git for Windows, then try again.')
        with SETUP_LOG.open('a', encoding='utf-8') as log:
            log.write(f'\n==== {time.strftime("%Y-%m-%d %H:%M:%S")} install ai-toolkit {AITK_COMMIT}\n')
            _step('downloading', 'Downloading ai-toolkit…')
            if not (TOOLKIT_DIR / '.git').is_dir():
                _run([git, 'clone', '--filter=blob:none', AITK_REPO, str(TOOLKIT_DIR)], log)
            else:
                _run([git, 'fetch', '--filter=blob:none', 'origin'], log, cwd=TOOLKIT_DIR)
            _run([git, '-c', 'advice.detachedHead=false', 'checkout', '--force', AITK_COMMIT], log, cwd=TOOLKIT_DIR)
            _run([git, 'submodule', 'update', '--init', '--recursive'], log, cwd=TOOLKIT_DIR)
            uv = _uv()
            _step('environment', 'Creating the trainer Python environment…')
            if not TRAINER_PYTHON.is_file():
                if uv:
                    _run([uv, 'venv', '--python', _base_python(), str(TRAINER_DIR / 'venv')], log)
                else:
                    _run([_base_python(), '-m', 'venv', str(TRAINER_DIR / 'venv')], log)
            pip = [uv, 'pip', 'install', '--python', str(TRAINER_PYTHON)] if uv else [str(TRAINER_PYTHON), '-m', 'pip', 'install']
            # The app's uv cache is on the data drive, which can be exFAT: git refuses repositories there ("dubious
            # ownership", exFAT records no owner) and exFAT has no links. The trainer uses its own cache next to it.
            env = {'UV_LINK_MODE': 'copy', 'UV_CACHE_DIR': str(TRAINER_DIR / 'uv-cache'),
                   'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'safe.directory', 'GIT_CONFIG_VALUE_0': '*'}
            _step('torch', 'Installing PyTorch for CUDA 13 (about 3 GB)…')
            _run(pip + TORCH + ['--index-url', TORCH_INDEX], log, env=env)
            _step('requirements', 'Installing the trainer packages…')
            _run(pip + ['-r', str(TOOLKIT_DIR / 'requirements.txt')] + (['--extra-index-url', TORCH_INDEX, '--index-strategy', 'unsafe-best-match'] if uv else []), log, env=env)
            # requirements.txt may move torch; put the pinned CUDA build back if it did.
            _run(pip + TORCH + ['--index-url', TORCH_INDEX], log, env=env)
            triton = None
            if TRITON:
                _step('triton', 'Installing Triton GPU kernels…')
                try:
                    _run(pip + [TRITON], log, env=env)
                    _run([str(TRAINER_PYTHON), '-c', TRITON_CHECK], log)
                    triton = TRITON
                except RuntimeError:
                    log.write('Triton does not work with this PyTorch build; the trainer runs without it.\n')
                    uninstall = [uv, 'pip', 'uninstall', '--python', str(TRAINER_PYTHON)] if uv else [str(TRAINER_PYTHON), '-m', 'pip', 'uninstall', '-y']
                    subprocess.run(uninstall + ['triton-windows'], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            _step('checking', 'Checking the trainer…')
            _run([str(TRAINER_PYTHON), '-c', 'import torch, diffusers, transformers, toolkit; assert torch.cuda.is_available(), "CUDA is not available"; '
                  'print("torch", torch.__version__, "cuda", torch.version.cuda, "diffusers", diffusers.__version__)'], log, cwd=TOOLKIT_DIR)
        write_json(SETUP_STAMP, {'commit': AITK_COMMIT, 'installed_at': time.time(), 'torch': TORCH, 'triton': triton})
        with _setup_lock:
            _setup.clear()
            _setup.update(status='ready', message='The trainer is installed.')
    except Exception as exc:
        with _setup_lock:
            _setup.update(status='failed', message=str(exc))


# ---- architectures ----------------------------------------------------------
# key: the family the finished LoRA works with in this app. A LoRA lists only the type it was trained on: a test LoRA
# trained on Z-Image Turbo (with the Turbo training adapter) did nothing on Z-Image Base. The values are the "selected" defaults of the
# ai-toolkit UI for that model (extensions_built_in/diffusion_models/ui.tsx at AITK_COMMIT).
ARCHS = {
    'flux3-klein-base': {'label':'FLUX.3 Image Klein Base','arch':'flux3',
        'name_or_path':'black-forest-labs/flux-3-image-klein-base','qtype':'convrot8','qtype_te':'convrot8',
        'timestep_type':'shift','guidance':3.5,'sample_steps':28,'model_kwargs':{'match_target_res':True},
        # Measured on an RTX 5090 (512 px, rank 4): cached embeddings 1.21 s per step, without 0.55 s; this runner reuses the prompt itself.
        'cache_text_embeddings':False,
        'gate':'https://huggingface.co/black-forest-labs/flux-3-image-klein-base',
        'families':['flux3-klein','flux3-klein-base']},
    'qwen-image-2.1': {'label': 'Qwen-Image-2.1', 'arch': 'qwen_image_2', 'name_or_path': 'Comfy-Org/Qwen-Image-2.1',
                       # "2.1 is meant to be sampled without guidance" (ai-toolkit qwen_image_2/src/pipeline.py:403).
                       'qtype': 'convrot8', 'qtype_te': 'convrot8', 'timestep_type': 'shift', 'guidance': 1.0, 'sample_steps': 25,
                       'model_kwargs': {'rgba': False}, 'families': ['qwen-image-2.1']},
    'qwen-image-2.1-turbo': {'label': 'Qwen-Image-2.1 Turbo', 'arch': 'qwen_image_2',
                             'name_or_path': 'Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors',
                             'assistant_lora_path': 'ostris/qwen_image_2_turbo_training_adapter/qwen_image_2_turbo_training_adapter_v1.safetensors',
                             'qtype': 'convrot8', 'qtype_te': 'convrot8', 'timestep_type': 'shift', 'guidance': 1.0, 'sample_steps': 8,
                             'model_kwargs': {'turbo': True, 'rgba': False}, 'families': ['qwen-image-2.1-turbo']},
    'krea-2': {'label': 'Krea 2 Raw', 'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw', 'timestep_type': 'linear', 'guidance': 4.0, 'low_vram': True,
               # krea-ai/krea-2 README: "TRAIN on Raw and RUN on Turbo".
               'sample_steps': 25, 'gate': 'https://huggingface.co/krea/Krea-2-Raw', 'families': ['krea-2', 'krea-2-turbo']},
    'krea-2-turbo': {'label': 'Krea 2 Turbo', 'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Turbo', 'timestep_type': 'linear', 'guidance': 1.0, 'low_vram': True,
                     'sample_steps': 9, 'gate': 'https://huggingface.co/krea/Krea-2-Turbo',
                     'assistant_lora_path': 'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
                     'families': ['krea-2-turbo']},
    'z-image-turbo': {'label': 'Z-Image Turbo', 'arch': 'zimage', 'name_or_path': 'Tongyi-MAI/Z-Image-Turbo', 'qtype': 'qfloat8', 'quantize': False,
                      'timestep_type': 'weighted', 'guidance': 1.0, 'sample_steps': 9,
                      'assistant_lora_path': 'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors',
                      'families': ['z-image-turbo']},
    'z-image': {'label': 'Z-Image Base', 'arch': 'zimage', 'name_or_path': 'Tongyi-MAI/Z-Image', 'qtype': 'qfloat8', 'quantize': False, 'timestep_type': 'weighted',
                'guidance': 4.0, 'sample_steps': 30, 'families': ['z-image']},
    # ai-toolkit's "MOSS-Image Base" defaults (ui.tsx moss_image:base): the int8 ConvRot Base attaches as-is, CFG 3 samples with
    # layer 22 skipped on the negative branch and the LoRA kept off layer 22. OpenMOSS: train LoRAs on Base, text encoder frozen.
    # Base-trained LoRAs apply to SFT and to Base finetunes such as Serenity, not to the distilled Turbo.
    'moss-image': {'label': 'MOSS Image', 'arch': 'moss_image', 'name_or_path': 'OpenMOSS-Team/MOSS-Image-Base', 'qtype': 'convrot8', 'qtype_te': 'convrot8',
                   'timestep_type': 'shift', 'guidance': 3.0, 'sample_steps': 30, 'model_kwargs': {'uncond_skip_layers': [22]},
                   'network_kwargs': {'ignore_if_contains': ['layers.22.']}, 'gate': 'https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base',
                   'families': ['moss-image-base', 'moss-image']},
}
# civitai.inspect() names for the trained file, to check it before it is added.
INSPECT_ARCH = {'flux3':'flux3-klein','qwen_image_2': 'qwen-image-2.1', 'krea2': 'krea-2', 'zimage': 'z-image', 'moss_image': 'moss-image'}
# The app's own files for a training, so the trainer reads them instead of downloading its defaults.
LOCAL_FILES = {'flux3-klein-base': ['flux3-base-int8', 'flux3-encoder-int8', 'flux3-vae'], 'moss-image': ['moss-base-int8', 'moss-encoder-int8', 'flux3-vae']}


def slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip()).strip('_')[:60] or 'lora'


def output_dir(request: dict) -> Path:
    """One folder per name and base model, not per job: ai-toolkit continues from the last saved step in it,
    so Continue after Stop (a new queue job) picks up where the training stopped."""
    return TRAININGS_DIR / f"{slug(request['name'])}_{request['arch']}"


def preservation(request: dict) -> dict:
    """Differential output preservation for a character with a trigger word: the same image and caption with the trigger
    replaced by its class word ("photo of idoperson, a man" -> "photo of man, a man") keeps the base model's idea of the
    class (Krea's LoRA workflow; MOSS Serenity's prior preservation). The class comes from the captions, else "person"."""
    if request.get('subject') == 'style' or not request.get('trigger_word'):
        return {'diff_output_preservation': False}
    text = ' '.join(i.get('caption', '') for i in request.get('images', [])).lower()
    women, men = len(re.findall(r'\b(?:woman|women|girl|lady)\b', text)), len(re.findall(r'\b(?:man|men|boy|guy)\b', text))
    word = 'woman' if women > men else 'man' if men > women else 'person'
    return {'diff_output_preservation': True, 'diff_output_preservation_multiplier': 1.0, 'diff_output_preservation_class': word}


def build_config(request: dict, dataset_dir: Path, output_dir: Path) -> dict:
    """The ai-toolkit job for one training, as its UI would write it."""
    info = ARCHS[request['arch']]
    trigger = request.get('trigger_word') or None
    prompts = request.get('sample_prompts') or []
    # 32 GB GPU: only Krea 2 (24.5 GB BF16, quantized while loading) keeps low_vram's CPU-first loading; smaller GPUs keep it for every model.
    # Z-Image's 12 GB BF16 transformer trains unquantized; FLUX.3 and Qwen use pre-quantized ConvRot files as they are.
    from .gpu_memory import large_vram
    model = {'name_or_path': info['name_or_path'], 'arch': info['arch'], 'quantize': info.get('quantize', True), 'quantize_te': True,
             'qtype': info.get('qtype', 'qfloat8'), 'qtype_te': info.get('qtype_te', 'qfloat8'), 'low_vram': info.get('low_vram', False) or not large_vram(),
             'model_kwargs': dict(info.get('model_kwargs', {}))}
    if info['arch']=='flux3':
        from .models import entry, local_path
        def path(model_id):
            item=entry(model_id)
            return str(local_path(item,item['files'][0]))
        # Explicit Base weights avoid both stale upstream repack filenames and
        # accidental use of the distilled flow-map inference checkpoint.
        model['model_kwargs'].update(model_path=path('flux3-base-int8'),ae_path=path('flux3-vae'),
                                     text_encoder_path=path('flux3-encoder-int8'),
                                     config_path=str(APP_DIR/'reference/flux3-base-config.json'))
        # A local package bypasses the toolkit's search for older repack names.
        # Its explicit model_path must always select the downloaded Base file.
        model['name_or_path']=str(Path(path('flux3-base-int8')).parent)
    if info['arch']=='moss_image':
        from .models import entry, local_path
        def path(model_id):
            item=entry(model_id)
            return str(local_path(item,item['files'][0]))
        # Single files: the int8 ConvRot Base, the Qwen3.5-9B int8 encoder and the full FLUX.2 VAE (the toolkit's default VAE file).
        model.update(name_or_path=path('moss-base-int8'),te_name_or_path=path('moss-encoder-int8'),vae_path=path('flux3-vae'))
    if info.get('assistant_lora_path'):
        model['assistant_lora_path'] = info['assistant_lora_path']
    process = {
        'type': 'diffusion_trainer', 'training_folder': str(output_dir), 'device': 'cuda', 'trigger_word': trigger,
        'performance_log_every': 10,
        'network': {'type': 'lora', 'linear': int(request['rank']), 'linear_alpha': int(request['rank']), **({'network_kwargs': info['network_kwargs']} if info.get('network_kwargs') else {})},
        # Every checkpoint stays: likeness often peaks before the last step, and the user picks the best one.
        'save': {'dtype': 'bf16', 'save_every': int(request['save_every']), 'max_step_saves_to_keep': int(request['steps']) // int(request['save_every']) + 2, 'save_format': 'diffusers', 'push_to_hub': False},
        'datasets': [{'folder_path': str(dataset_dir), 'caption_ext': 'txt', 'caption_dropout_rate': 0.05, 'cache_latents_to_disk': True,
                      'resolution': list(request['resolutions']), 'num_repeats': 1}],
        'train': {'batch_size': 1, 'steps': int(request['steps']), 'gradient_accumulation': 1, 'train_unet': True, 'train_text_encoder': False,
                  'gradient_checkpointing': True, 'noise_scheduler': 'flowmatch', 'optimizer': 'adamw8bit', 'timestep_type': info['timestep_type'],
                  'content_or_style': 'balanced', 'optimizer_params': {'weight_decay': 1e-4}, 'unload_text_encoder': False,
                  # Captions are encoded once to disk (with the blank caption for dropout); the text encoder then leaves the GPU.
                  'cache_text_embeddings': info.get('cache_text_embeddings', True), 'lr': float(request['learning_rate']), 'ema_config': {'use_ema': False, 'ema_decay': 0.99},
                  'skip_first_sample': False, 'disable_sampling': not prompts, 'dtype': 'bf16', **preservation(request),
                  'loss_type': 'mse'},
        'logging': {'log_every': 1, 'use_ui_logger': False},
        'model': model,
        'sample': {'sampler': 'flowmatch', 'sample_every': int(request['save_every']), 'width': 1024, 'height': 1024,
                   'samples': [{'prompt': prompt} for prompt in prompts], 'neg': '', 'seed': 42, 'walk_seed': False,
                   'guidance_scale': info['guidance'], 'sample_steps': info['sample_steps']},
    }
    return {'job': 'extension', 'config': {'name': slug(request['name']), 'process': [process]}, 'meta': {'name': request['name'], 'version': '1.0'}}


# ---- progress -----------------------------------------------------------------

# Timing tables, library warnings and stack frames are not status messages.
NOISE = re.compile(r'^(- [\d.]+s avg|Traceback|File |WARNING|Warning:|warnings\.warn|[\w.]+Warning:)|avg - ')
PROGRESS = re.compile(r'(\d+)/(\d+) \[([\d:]+)<([\d:?]+)[^\]]*?(?:loss: ([\d.e+-]+))?\]')


def _seconds(clock: str) -> int | None:
    if not re.fullmatch(r'[\d:]+', clock or ''):
        return None
    total = 0
    for part in clock.split(':'):
        total = total * 60 + int(part)
    return total


def read_progress(log_path: Path, total_steps: int) -> dict:
    """The newest step, loss and time left from the trainer log (tqdm lines end with a carriage return)."""
    try:
        with log_path.open('rb') as handle:
            handle.seek(max(0, log_path.stat().st_size - 65536))
            text = handle.read().decode('utf-8', 'replace')
    except OSError:
        return {}
    lines = [line.strip() for line in re.split(r'[\r\n]+', text) if line.strip()]
    found = {}
    progress_index=-1
    for index in range(len(lines)-1,-1,-1):
        line=lines[index]
        match = PROGRESS.search(line)
        if match and int(match.group(2)) == total_steps:
            progress_index=index
            found = {'step': int(match.group(1)), 'total_steps': total_steps, 'eta_seconds': _seconds(match.group(4))}
            if match.group(5):
                found['loss'] = float(match.group(5))
            break
    found.update(stage='training' if 'step' in found else 'loading',
                 message=f"Training: step {found['step']} of {total_steps}" if 'step' in found else 'Loading the training tools...')
    stages=[('loading transformer','loading_model','Loading the base model...'),
            ('loading qwen','loading_encoder','Loading the text encoder...'),
            ('loading image vae','loading_vae','Loading the image encoder...'),
            ('caching latents','encoding','Preparing the training images...'),
            ('generating samples','sampling','Creating sample images...'),
            ('saving model','saving','Saving the LoRA...')]
    for index in range(len(lines)-1,progress_index,-1):
        match=next(((stage,message) for text,stage,message in stages if text in lines[index].lower()),None)
        if match:
            found.update(stage=match[0],message=match[1]);break
    # The last line that is not a progress bar says what the trainer does now (loading, caching, sampling).
    status = next((line for line in reversed(lines) if not PROGRESS.search(line) and not NOISE.search(line)), '')
    found['trainer_message'] = re.sub(r'\x1b\[[0-9;]*m', '', status)[:200]
    return found


def sample_files(output_dir: Path, name: str) -> list[str]:
    folder = output_dir / slug(name) / 'samples'
    files = sorted(folder.glob('*.jpg'), key=lambda path: path.stat().st_mtime) if folder.is_dir() else []
    return [path.name for path in files]


def final_lora(output_dir: Path, name: str) -> Path | None:
    path = output_dir / slug(name) / f'{slug(name)}.safetensors'
    return path if path.is_file() else None


# ---- running a training ---------------------------------------------------------

def prepare_dataset(request: dict, directory: Path) -> Path:
    """The images as numbered PNG files with a .txt caption each, as ai-toolkit reads a dataset folder."""
    from PIL import Image, ImageOps
    from .common import resolve_image
    dataset = directory / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(request['images'], 1):
        with Image.open(resolve_image(item['id'])) as opened:
            target = dataset / f'{index:03d}.png'
            if not target.exists():
                ImageOps.exif_transpose(opened).convert('RGB').save(target)
        (dataset / f'{index:03d}.txt').write_text(item.get('caption', ''), encoding='utf-8')
    return dataset


def trainer_env() -> dict:
    env = {key: value for key, value in os.environ.items() if key not in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME')}
    # HF_HUB_DISABLE_SYMLINKS: the data drive can be exFAT, where huggingface_hub's symlink probe passes but the cache's
    # blob links fail ("[WinError 1] Incorrect function" for the MOSS tokenizer, 2026-10-21); it copies instead.
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HF_HOME=str(DATA_DIR / 'hf-cache'), HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_SYMLINKS='1',
               NO_ALBUMENTATIONS_UPDATE='1', TQDM_MININTERVAL='2')
    if not env.get('HF_TOKEN'):
        # The app keeps its Hugging Face cache on the data drive; the account token stays in the user's home cache.
        token = Path.home() / '.cache' / 'huggingface' / 'token'
        if token.is_file():
            env['HF_TOKEN'] = token.read_text(encoding='utf-8').strip()
    return env


def untrained(path: Path) -> bool:
    """True when every lora_B / lora_up tensor is all zeros: the LoRA starts there and did not move."""
    from safetensors import safe_open
    try:
        with safe_open(str(path), 'pt') as handle:
            ups = [key for key in handle.keys() if re.search(r'lora_(?:B|up)\b', key)]
            return bool(ups) and all(not handle.get_tensor(key).any() for key in ups)
    except Exception:
        return False  # unreadable values: the architecture check below reports the file


def register(job_id: str, request: dict, source: Path) -> dict:
    """Copy the finished LoRA to the models folder and add it to the LoRA list of its architecture."""
    from . import civitai, models
    info = ARCHS[request['arch']]
    found = civitai.inspect(source)
    if untrained(source):
        raise RuntimeError('The training finished, but the LoRA did not learn: all its up weights are still zero '
                           '(a known ai-toolkit issue, #1054 and #1075). Run the training again, or lower the learning rate.')
    if found['kind'] != 'lora' or found['arch'] != INSPECT_ARCH[info['arch']]:
        raise RuntimeError(f"The trained file is not a {info['label']} LoRA ({found['kind']}, {found['arch']}).")
    relative = f"{job_id[:8]}_{slug(request['name'])}.safetensors"
    # The model library's loras/trained folder when its drive has room (models.local_path finds it), else the data folder.
    target = models.download_target({'directory': 'trained', 'kind': 'image-lora'}, {'path': relative, 'bytes': source.stat().st_size}, TRAINED_DIR / relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    size = target.stat().st_size
    entry = {'id': f'trained-lora-{job_id[:12]}', 'title': request['name'], 'kind': 'image-lora', 'format': 'LoRA', 'repo': 'trained',
             'source': 'trained', 'revision': job_id, 'directory': 'trained', 'files': [{'path': relative, 'bytes': size, 'sha256': None}],
             'bytes': size, 'dependencies': [], 'families': list(info['families']), 'made_for': info['label'],
             'trained_words': [request['trigger_word']] if request.get('trigger_word') else [], 'default_strength': 1.0,
             'summary': f"Trained in this app on {info['label']} from {len(request['images'])} images, {request['steps']} steps, rank {request['rank']}.",
             'training_job': job_id}
    from .lora_author import describe
    describe(entry, target)
    civitai._register(entry)
    with models.lock:
        saved = models.installed_manifest()
        saved[entry['id']] = {'revision': job_id, 'verified': True}
        write_json(models.INSTALL_PATH, saved)
    return entry


def run(job_id: str, request: dict, directory: Path, on_process, cancelled) -> None:
    """Train one LoRA. Writes progress into the job's status.json; returns when the trainer ends."""
    from .common import read_json
    status_path = directory / 'status.json'

    def update(**values):
        job = read_json(status_path)
        now=time.time()
        if values.get('stage',job.get('stage'))!=job.get('stage'):
            previous=job.get('stage')
            if previous and previous!='queued' and job.get('stage_started_at'):
                elapsed=round(now-job['stage_started_at'],3)
                job.setdefault('stage_seconds',{})[previous]=round(job.get('stage_seconds',{}).get(previous,0)+elapsed,3)
                job.setdefault('stage_history',[]).append({'stage':previous,'started_at':job['stage_started_at'],'seconds':elapsed})
            values['stage_started_at']=now
        job.update(values)
        write_json(status_path, job)

    if not setup_status()['ready']:
        raise RuntimeError('Install the trainer on the Train page first.')
    started = time.time()
    update(status='running', stage='preparing', message='Preparing the images…', started_at=started)
    if request.get('auto_caption', False) and any(not i.get('caption', '').strip() for i in request['images']):
        from . import training_captions
        captions = training_captions.run(request, directory, on_process, cancelled, update)
        if cancelled(): return
        request = {**request, 'images': [dict(item, caption=item['caption'] if item.get('caption', '').strip() else captions.get(item['id'], {}).get('caption', '')) for item in request['images']]}
        missing = sum(not i['caption'].strip() for i in request['images'])
        if missing:
            raise RuntimeError(f'{missing} images need captions. Completed captions are saved. Retry or edit them before training.')
        write_json(directory / 'prepared-dataset.json', {'images': request['images']})
    if cancelled(): return
    if request['arch'] in LOCAL_FILES:
        from . import models
        models.ensure_installed(LOCAL_FILES[request['arch']],owner=job_id,cancelled=cancelled,
            progress=lambda transfer:update(stage='downloading_models',message='Downloading the training model files...',transfer=transfer))
    dataset = prepare_dataset(request, directory)
    output = output_dir(request)
    output.mkdir(parents=True, exist_ok=True)
    config = directory / 'train.yaml'
    import yaml
    config.write_text(yaml.safe_dump(build_config(request, dataset, output), sort_keys=False, allow_unicode=True), encoding='utf-8')
    log_path = directory / 'worker.log'
    with log_path.open('ab') as log:
        runner=str(APP_DIR/'scripts/train_flux3.py') if request['arch']=='flux3-klein-base' else 'run.py'
        process = subprocess.Popen([str(TRAINER_PYTHON), '-u', runner, str(config)], cwd=TOOLKIT_DIR, stdout=log, stderr=subprocess.STDOUT,
                                   env=trainer_env(), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    on_process(process)
    update(stage='loading', message='Loading the base model (the first training downloads it)…', step=0, total_steps=int(request['steps']))
    while process.poll() is None:
        time.sleep(2)
        if cancelled():
            continue  # the server stops the process
        progress = read_progress(log_path, int(request['steps']))
        stage=progress.get('stage','loading')
        message=progress.get('message','Loading the base model...')
        update(stage=stage, message=message, samples=sample_files(output, request['name']),
               **{key: value for key, value in progress.items() if key not in ('trainer_message','stage','message')}, trainer_message=progress.get('trainer_message', ''))
    if cancelled():
        return
    result = final_lora(output, request['name'])
    if process.returncode or not result:
        tail = read_progress(log_path, int(request['steps'])).get('trainer_message', '').rstrip('.')
        gate = ARCHS[request['arch']].get('gate')
        hint = f' Open {gate}, accept the terms with your Hugging Face account, then retry.' if gate and re.search(r'401|403|gated|restricted', tail, re.I) else ''
        raise RuntimeError(f'The trainer stopped (code {process.returncode}): {tail or "see the log"}.{hint}')
    entry = register(job_id, request, result)
    update(status='succeeded', stage='complete', message='LoRA trained and added.', finished_at=time.time(), lora_id=entry['id'],
           elapsed_seconds=round(time.time() - started, 1), samples=sample_files(output, request['name']), step=int(request['steps']))

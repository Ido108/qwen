"""Single-image upscaling: SeedVR2 (the runtime and models the video upscaler already ships) or a fast 4x GAN-style model."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from .common import JOBS_DIR, OUTPUTS_DIR, read_json, resolve_image, write_json

# The longest output side and pixel count one job may produce. The DiT pass holds about 0.75 GB per output
# megapixel on top of the 3B FP16 weights (17 MP peaked at 21.5 GB on the 5090), so 24 MP stays inside 32 GB.
MAX_SIDE = 8192
MAX_PIXELS = 24_000_000
IMAGE_KINDS = ('image', 'image-upscale')
# SeedVR2 phases as the CLI reports them, and the stage names and labels the queue shows.
PHASES = {'encoding': ('encoding', 'Reading the image'),
          'upscaling': ('upscaling', 'Upscaling the image'),
          'decoding': ('decoding', 'Decoding the result'),
          'post-processing': ('saving', 'Matching colors')}
SEED_STAGES = ('loading_model', 'encoding', 'upscaling', 'decoding', 'saving')
STAGE_LABELS = {'loading_model': 'Loading the model', 'encoding': 'Reading the image', 'upscaling': 'Upscaling the image',
                'decoding': 'Decoding the result', 'saving': 'Saving the result'}
# numz ComfyUI-SeedVR2_VideoUpscaler defaults (README and inference_cli.py): tile 1024, overlap 128, noise 0, seed 42.
# The NVIDIA kernel cache (%APPDATA%\NVIDIA\ComputeCache) was full at 952 MB, so every new process compiled its
# kernels again: the first model call took 6 to 16 s. With a 4 GiB limit it took about 1 s (2026-10-22).
CUDA_CACHE = str(4 * 1024**3)
TILE_DEFAULT, OVERLAP_DEFAULT, GAN_TILE_DEFAULT = 1024, 128, 1024


class ImageUpscaleRequest(BaseModel):
    model_id: str = 'seedvr2-3b-fp16'
    # An uploaded input image, or a finished gallery image (the server copies it to the uploads first).
    source_image: str = Field(default='', max_length=200)
    source_job: str = Field(default='', max_length=32)
    size_mode: Literal['scale', 'long_side'] = 'scale'
    scale: Literal[2, 4] = 2
    long_side: int = Field(default=4096, ge=256, le=MAX_SIDE)
    color_correction: Literal['lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none'] = 'lab'
    seed: int = Field(default=42, ge=0, le=2147483647)
    input_noise_scale: float = Field(default=0.0, ge=0, le=1)
    latent_noise_scale: float = Field(default=0.0, ge=0, le=1)
    vae_tile_size: Literal[512, 768, 1024, 1536, 2048] = TILE_DEFAULT
    vae_tile_overlap: Literal[32, 64, 128, 256] = OVERLAP_DEFAULT
    memory_mode: Literal['auto', 'low_memory'] = 'auto'
    gan_tile_size: Literal[256, 384, 512, 768, 1024] = GAN_TILE_DEFAULT
    output_dir: str = Field(default='', max_length=1000)


def upscaler(model_id):
    from .models import entry
    item = entry(model_id)
    if (item['kind'], item.get('family')) not in (('upscale', 'seedvr2'), ('image-upscale', 'spandrel')):
        raise ValueError('Choose an image upscaler.')
    return item


def output_dimensions(width, height, request):
    if width <= 0 or height <= 0:
        raise ValueError('The source image has no valid size.')
    factor = request['scale'] if request.get('size_mode', 'scale') == 'scale' else request['long_side'] / max(width, height)
    if factor <= 1:
        raise ValueError(f'Choose a long side larger than the source ({max(width, height)} px).')
    limit = 8
    if request.get('model_id'):
        item = upscaler(request['model_id'])
        limit = item.get('scale') or limit  # a fixed-scale model does not upscale twice
    if factor > limit:
        raise ValueError(f'This upscaler goes up to {limit} times the source size.')
    result = (max(1, round(width * factor)), max(1, round(height * factor)))
    if max(result) > MAX_SIDE or result[0] * result[1] > MAX_PIXELS:
        raise ValueError(f'The result would be {result[0]} × {result[1]}. Choose a smaller size '
                         f'(up to {MAX_SIDE} px on the long side and {MAX_PIXELS // 1_000_000} megapixels).')
    return result


def required(request):
    item = upscaler(request['model_id'])
    return [item['id'], *item['dependencies']]


def job_image(job_id):
    """The saved file of a finished image result."""
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Saved image not found.')
    try:
        job = read_json(JOBS_DIR / job_id / 'status.json')
    except (OSError, ValueError):
        raise ValueError('Saved image not found.')
    path = Path(job.get('output_path', ''))
    if job.get('kind', 'image') not in IMAGE_KINDS or job.get('status') != 'succeeded' or not path.is_file():
        raise ValueError('Saved image not found.')
    return path


def plan(path, request, strict=True):
    """Source and result size, read from the file header only. With strict=False a size problem is returned
    as 'error' beside the source size, so the form can show both."""
    from PIL import Image
    with Image.open(path) as picture:
        width, height = picture.size
        if (picture.getexif() or {}).get(0x0112) in (5, 6, 7, 8):
            width, height = height, width  # shown rotated, as ImageOps.exif_transpose stores it
    sizes = {'source_width': width, 'source_height': height}
    try:
        out_width, out_height = output_dimensions(width, height, request)
    except ValueError as exc:
        if strict:
            raise
        return {**sizes, 'width': None, 'height': None, 'error': str(exc)}
    return {**sizes, 'width': out_width, 'height': out_height}


def seed_estimates(megapixels, billions=3):
    """Expected seconds per stage, fitted to the 3B FP16 runs on the 5090 (2.8 to 24 MP, 2026-10-21).
    The UI moves the progress bar inside a stage by elapsed time over this estimate and stops short of the end."""
    return {'loading_model': 8.0, 'encoding': 3.5 + .2 * megapixels,
            'upscaling': (1 + .55 * megapixels) * (2.5 if billions >= 7 else 1),
            'decoding': .5 + .45 * megapixels, 'saving': 1.2 + .12 * megapixels}


class Progress:
    """Stage, overall fraction and an estimate per stage, written with the job status."""
    def __init__(self, update, estimates):
        self.update, self.estimates = update, estimates
        total = sum(estimates.values())
        self.start, done = {}, 0.0
        for stage, seconds in estimates.items():
            self.start[stage] = done / total
            done += seconds
        self.end = {stage: self.start[stage] + seconds / total for stage, seconds in estimates.items()}
        self.update(stage_estimates={k: round(v, 1) for k, v in estimates.items()}, estimated_seconds=round(total, 1))

    def stage(self, stage, message=None, **values):
        index = list(self.estimates).index(stage)
        self.update(stage=stage, message=message or STAGE_LABELS[stage], step=index + 1, total_steps=len(self.estimates),
                    progress=round(self.start[stage], 4), progress_stage_end=round(self.end[stage], 4),
                    progress_estimate_seconds=round(self.estimates[stage], 1), progress_at=time.time(), **values)

    def within(self, stage, fraction, next_fraction, seconds, **values):
        """Progress inside one stage (GAN tiles): fraction done now, the fraction at the next update, its expected time."""
        span = self.end[stage] - self.start[stage]
        self.update(progress=round(self.start[stage] + span * fraction, 4), progress_stage_end=round(self.start[stage] + span * next_fraction, 4),
                    progress_estimate_seconds=round(seconds, 2), progress_at=time.time(), **values)

    def seed_events(self, **values):
        """run_seed reports each SeedVR2 phase as 'batch 1 of 1': show it as a stage of the whole job instead."""
        stage = values.get('stage')
        if stage in self.estimates and 'status' not in values:
            for key in ('stage', 'message', 'step', 'total_steps'):
                values.pop(key, None)
            self.stage(stage, **values)
        else:
            self.update(**values)


def source_picture(request, directory):
    from PIL import Image, ImageOps
    with Image.open(resolve_image(request['source_image'])) as opened:
        picture = ImageOps.exif_transpose(opened)
        picture.load()
    alpha = picture.mode in ('RGBA', 'LA', 'PA') or 'transparency' in picture.info
    if alpha:
        picture = picture.convert('RGBA')
        alpha = picture.getchannel('A').getextrema()[0] < 255
    return picture.convert('RGBA' if alpha else 'RGB'), alpha


def destination_for(request, job):
    folder = Path(request.get('output_dir') or OUTPUTS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '_upscaled.png')


def run(request, job, directory, update):
    selected = upscaler(request['model_id'])
    picture, alpha = source_picture(request, directory)
    width, height = output_dimensions(picture.width, picture.height, request)
    update(width=width, height=height, source_width=picture.width, source_height=picture.height, actual_steps=1,
           model_id=selected['id'], model_format=selected['format'], upscaler=selected['title'])
    destination = destination_for(request, job)
    if selected.get('family') == 'spandrel':
        run_gan(selected, picture, alpha, (width, height), request, destination, update)
    else:
        run_seedvr2(selected, picture, alpha, (width, height), request, directory, destination, update)
    update(status='succeeded', stage='complete', message='Upscaled image saved.', output_path=str(destination),
           image_url=f'/api/jobs/{job["id"]}/image', finished_at=time.time(), progress=1, progress_stage_end=1,
           elapsed_seconds=round(time.time() - job['started_at'], 1), width=width, height=height,
           has_transparency=alpha, model_revision=selected['revision'])
    write_json(destination.with_suffix('.json'), job)


def seed_options(request, selected, free):
    """Command-line knobs from the form: noise, VAE tiles (the VAE always tiles, see VAE_TILES) and memory."""
    from .upscale_settings import seed_memory_options
    tile, overlap = request.get('vae_tile_size', TILE_DEFAULT), request.get('vae_tile_overlap', OVERLAP_DEFAULT)
    common = ['--input_noise_scale', str(request.get('input_noise_scale', 0.0)), '--latent_noise_scale', str(request.get('latent_noise_scale', 0.0)),
              *VAE_TILES, '--vae_encode_tile_size', str(tile), '--vae_encode_tile_overlap', str(overlap),
              '--vae_decode_tile_size', str(tile), '--vae_decode_tile_overlap', str(overlap)]
    memory = {'resolution': request['_short'], 'batch_size': 1}
    safe = common + seed_memory_options(selected, {**memory, 'memory_mode': 'low_memory'}, free)
    fast = safe if request.get('memory_mode') == 'low_memory' else common + seed_memory_options(selected, memory, free)
    return fast, safe


# The VAE always tiles: on one 5090 frame it was as fast or faster than untiled and used less memory
# (2048 x 2048 result, 2026-10-21: decode 1.8 s and 12.2 GB peak tiled, 4.8 s and 18.1 GB untiled; 2048 px tiles gave no gain at 4x).
VAE_TILES = ['--vae_encode_tiled', '--vae_decode_tiled']


def run_seedvr2(selected, picture, alpha, size, request, directory, destination, update):
    from PIL import Image
    from .gpu_memory import dedicated_memory_snapshot
    from .upscale_worker import run_seed, seed_model_dir
    width, height = size
    prepared = directory / 'source.png'
    # Opaque pictures go in as RGB: SeedVR2 upscales a real alpha channel separately, which costs time.
    picture.save(prepared, format='PNG')
    billions = selected.get('upscale_defaults', {}).get('parameters_billion', 3)
    progress = Progress(update, seed_estimates(width * height / 1e6, billions))
    progress.stage('loading_model', 'Loading the SeedVR2 upscaling model…')
    model_dir, model_name = seed_model_dir(selected)
    raw = directory / 'upscaled_raw.png'
    short = min(width, height)
    cmd = [sys.executable, '-u', '-m', 'backend.seedvr2_cli', str(prepared),
           '--output', str(raw), '--model_dir', str(model_dir), '--dit_model', model_name,
           '--resolution', str(short), '--batch_size', '1', '--seed', str(request['seed']),
           '--color_correction', request.get('color_correction', 'lab')]
    gpu = dedicated_memory_snapshot()
    free = gpu.get('free_gb')
    fast, safe = seed_options({**request, '_short': short}, selected, free)
    if gpu.get('available'):
        update(dedicated_free_gb=free, gpu_total_gb=gpu['total_gb'])
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1', PYTHONUTF8='1')
    environment.setdefault('CUDA_CACHE_MAXSIZE', CUDA_CACHE)
    update(upscale_options=' '.join(fast))
    code = run_seed(cmd + fast, environment, progress.seed_events, PHASES)
    if code and fast != safe:
        raw.unlink(missing_ok=True)
        progress.stage('loading_model', 'Retrying with GPU memory saving…', upscale_options=' '.join(safe), upscale_retry=True)
        code = run_seed(cmd + safe, environment, progress.seed_events, PHASES)
    if code:
        raise RuntimeError(f'SeedVR2 stopped with code {code}. See the job log for details.')
    if not raw.is_file() or raw.stat().st_size == 0:
        raise RuntimeError('SeedVR2 finished without an output image.')
    progress.stage('saving', 'Saving the upscaled image…')
    temporary = destination.with_name(destination.stem + '.partial.png')
    try:
        with Image.open(raw) as opened:
            exact = opened.size == (width, height) and opened.mode == ('RGBA' if alpha else 'RGB')
        if exact:
            # The PNG SeedVR2 wrote is the result: move it instead of encoding 4 to 24 MP again.
            shutil.move(raw, temporary)
        else:
            with Image.open(raw) as opened:
                result = opened.convert('RGBA' if alpha else 'RGB')
            # SeedVR2 sets the short side; the long side can differ by a pixel from the exact aspect ratio.
            result.resize((width, height), Image.Resampling.LANCZOS).save(temporary, format='PNG', compress_level=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    prepared.unlink(missing_ok=True)


def tile_starts(length, tile, overlap):
    """Tile origins that cover 0..length with at least `overlap` shared pixels."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    return starts + [length - tile]


def run_gan(selected, picture, alpha, size, request, destination, update):
    """A fixed 4x model (spandrel), run in overlapping tiles with feathered seams, then resized to the chosen size."""
    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ModelLoader
    from .models import local_path
    width, height = size
    tile = request.get('gan_tile_size', GAN_TILE_DEFAULT)
    overlap = 32
    ys, xs = tile_starts(picture.height, tile, overlap), tile_starts(picture.width, tile, overlap)
    tiles = [(y, x) for y in ys for x in xs]
    megapixels = picture.width * picture.height / 1e6
    progress = Progress(update, {'loading_model': 3.0, 'upscaling': 1 + .5 * megapixels, 'saving': 1 + .1 * width * height / 1e6})
    progress.stage('loading_model', f"Loading {selected['title']}…")
    os.environ.setdefault('CUDA_CACHE_MAXSIZE', CUDA_CACHE)  # before the first CUDA call in this process
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Restart after checking the graphics driver.')
    model = ModelLoader().load_from_file(str(local_path(selected, selected['files'][0])))
    dtype = torch.bfloat16 if model.supports_bfloat16 else torch.float16 if model.supports_half else torch.float32
    model = model.to('cuda', dtype).eval()
    scale = model.scale
    source = torch.from_numpy(np.asarray(picture.convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1)
    out = torch.zeros(3, picture.height * scale, picture.width * scale)
    weight = torch.zeros(1, picture.height * scale, picture.width * scale)
    started = time.time()
    progress.stage('upscaling', f'Upscaling tile 1 of {len(tiles)}')
    with torch.inference_mode():
        for index, (y, x) in enumerate(tiles):
            part = source[:, y:y + tile, x:x + tile]
            result = model(part[None].to('cuda', dtype)).float().clamp(0, 1)[0].cpu()
            h, w = result.shape[1:]
            # Linear ramps over the overlap on every side; never zero, so a lone border tile keeps full weight.
            ramp = overlap * scale
            ry = torch.clamp((torch.arange(h) + 1) / ramp, max=1) * torch.clamp((h - torch.arange(h)) / ramp, max=1)
            rx = torch.clamp((torch.arange(w) + 1) / ramp, max=1) * torch.clamp((w - torch.arange(w)) / ramp, max=1)
            mask = (ry[:, None] * rx[None, :])[None]
            out[:, y * scale:y * scale + h, x * scale:x * scale + w] += result * mask
            weight[:, y * scale:y * scale + h, x * scale:x * scale + w] += mask
            done = index + 1
            seconds = (time.time() - started) / done
            progress.within('upscaling', done / len(tiles), min(1, (done + 1) / len(tiles)), seconds,
                            message=f'Upscaling tile {min(done + 1, len(tiles))} of {len(tiles)}')
    del model
    torch.cuda.empty_cache()
    progress.stage('saving', 'Saving the upscaled image…')
    array = (out / weight).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    result = Image.fromarray(array, 'RGB')
    if result.size != (width, height):
        result = result.resize((width, height), Image.Resampling.LANCZOS)
    if alpha:
        result.putalpha(picture.getchannel('A').resize((width, height), Image.Resampling.LANCZOS))
    temporary = destination.with_name(destination.stem + '.partial.png')
    try:
        result.save(temporary, format='PNG', compress_level=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

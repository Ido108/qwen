from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field, field_validator

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('QWEN_DATA_DIR', 'E:/AI/qwen-image-studio')).resolve()
MODELS_DIR = DATA_DIR / 'models'
JOBS_DIR = DATA_DIR / 'jobs'
UPLOADS_DIR = DATA_DIR / 'uploads'
OUTPUTS_DIR = DATA_DIR / 'images'
VIDEOS_DIR = DATA_DIR / 'videos'
EXAMPLES_DIR = APP_DIR / 'public' / 'examples'
for directory in (JOBS_DIR, UPLOADS_DIR, OUTPUTS_DIR, VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def read_json(path: Path):
    # Atomic replacement on Windows/exFAT can briefly deny a reader's open.
    # Retry that sharing race instead of returning an HTTP 500 during polling.
    for attempt in range(25):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 24:
                raise
            time.sleep(min(.005 * (attempt + 1), .05))

def write_json(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers can briefly hold a non-delete-sharing handle. Keep the
    # old complete document until replacement succeeds; never truncate it.
    try:
        for attempt in range(25):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt == 24:
                    raise
                time.sleep(min(.01 * (attempt + 1), .1))
    finally:
        temp.unlink(missing_ok=True)

class GenerationRequest(BaseModel):
    model_id: str = 'gguf-q8_0'
    text_encoder: str = 'auto'
    prompt: str = Field(min_length=1, max_length=16000)
    images: list[str] = Field(default_factory=list, max_length=10)
    enhance: bool = True
    custom_size: bool = False
    auto_resolution: Literal[1024, 2048] = 2048
    width: int = Field(default=2688, ge=256, le=2688)
    height: int = Field(default=1536, ge=256, le=2688)
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    negative_prompt: str = Field(default='', max_length=8000)
    output_dir: str = str(OUTPUTS_DIR)
    # None follows the selected model's documented default; integers override it.
    steps: int | None = Field(default=None, ge=1, le=80)

    @field_validator('prompt')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Enter a prompt first.')
        return value.strip()

    @field_validator('width', 'height')
    @classmethod
    def valid_size(cls, value):
        if value % 8:
            raise ValueError('Dimensions must be multiples of 8.')
        return value

class VideoRequest(BaseModel):
    model_id: str = 'fasth3-q4_k_m'
    prompt: str = Field(min_length=1, max_length=16000)
    mode: Literal['text', 'image'] = 'text'
    first_frame: str | None = None
    last_frame: str | None = None
    width: int = Field(default=1344, ge=256, le=1344)
    height: int = Field(default=768, ge=256, le=1344)
    frames: int = Field(default=124, ge=22, le=362)
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    quality: Literal['standard', 'turbo'] = 'standard'
    steps: int = Field(default=20, ge=10, le=60)
    audio: bool = True
    enhance: bool = False
    output_dir: str = str(VIDEOS_DIR)

    @field_validator('prompt')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Enter a prompt first.')
        return value.strip()

    @field_validator('width', 'height')
    @classmethod
    def valid_size(cls, value):
        if value % 32:
            raise ValueError('Video dimensions must be multiples of 32.')
        return value

    @field_validator('frames')
    @classmethod
    def valid_frames(cls, value):
        if value % 17 != 5:
            raise ValueError('Choose a supported video duration.')
        return value


def resolve_image(image_id: str) -> Path:
    if image_id.startswith('example:'):
        root, name = EXAMPLES_DIR, image_id.removeprefix('example:')
    else:
        root, name = UPLOADS_DIR, image_id
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Input image not found. Upload it again.')
    if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
        raise ValueError('Unsupported image file.')
    return path

def model_status() -> dict:
    manifest_path = DATA_DIR / 'model-manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        manifest = {}
    status = {}
    for name in ('Qwen-Image-2.1', 'Qwen-Image-2.1-PE-T2I', 'Qwen-Image-2.1-PE-I2I'):
        entry = manifest.get(name, {})
        status[name] = bool(entry.get('verified') and all(
            (MODELS_DIR / name / f['path']).is_file() and
            (MODELS_DIR / name / f['path']).stat().st_size == f['bytes']
            for f in entry.get('files', [])
        ))
    return status

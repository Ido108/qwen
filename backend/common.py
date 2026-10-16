from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('QWEN_DATA_DIR', str(Path(os.environ.get('SystemDrive', 'C:') + '/AI/qwen-image-studio')) if os.name == 'nt' else str(Path.home() / 'Library/Application Support/Qwen'))).resolve()
MODELS_DIR = DATA_DIR / 'models'
JOBS_DIR = DATA_DIR / 'jobs'
UPLOADS_DIR = DATA_DIR / 'uploads'
VIDEO_UPLOADS_DIR = UPLOADS_DIR / 'videos'
AUDIO_UPLOADS_DIR = UPLOADS_DIR / 'audio'
OUTPUTS_DIR = DATA_DIR / 'images'
VIDEOS_DIR = DATA_DIR / 'videos'
EXAMPLES_DIR = APP_DIR / 'public' / 'examples'
for directory in (JOBS_DIR, UPLOADS_DIR, VIDEO_UPLOADS_DIR, AUDIO_UPLOADS_DIR, OUTPUTS_DIR, VIDEOS_DIR):
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

class LoraChoice(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    strength: float = Field(default=1.0, ge=0, le=2)


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
    cfg: float | None = Field(default=None, ge=1, le=15)
    vae_id: str = Field(default='auto', max_length=120)
    # Optional LoRA tool (see reference/image-tools.json) and its options.
    tool_id: str | None = Field(default=None, max_length=40)
    lora_strength: float | None = Field(default=None, ge=0, le=1.5)
    tool_options: dict[str, str] = Field(default_factory=dict)
    # Extra LoRAs, for example ones added from Civitai: [{"id": ..., "strength": ...}].
    loras: list['LoraChoice'] = Field(default_factory=list, max_length=4)

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
    # MiniMax H3 allows up to 1344 px per side; LTX 2.5 up to 1536. models.required_video
    # applies each family's own size, frame and pixel-count rules.
    width: int = Field(default=1344, ge=256, le=1536)
    height: int = Field(default=768, ge=256, le=1536)
    frames: int = Field(default=124, ge=22, le=362)
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    quality: Literal['standard', 'turbo'] = 'standard'
    # LTX 2.5 is distilled and always runs its fixed schedule; this applies to MiniMax H3.
    steps: int = Field(default=20, ge=10, le=60)
    audio: bool = True
    enhance: bool = False
    output_dir: str = str(VIDEOS_DIR)
    loras: list[LoraChoice] = Field(default_factory=list, max_length=4)

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
        # MiniMax H3 uses 17n+5 frames and LTX 2.5 uses 8n+1 frames.
        if value % 17 != 5 and value % 8 != 1:
            raise ValueError('Choose a supported video duration.')
        return value


class SourcePerson(BaseModel):
    id: str = Field(pattern=r'^person-(?:[1-9]|1[0-2])$')
    time: float = Field(ge=0)
    box: tuple[float,float,float,float] | None = None


class ReferencePerson(BaseModel):
    id: str = Field(pattern=r'^ref-person-[A-Za-z0-9_-]{1,80}$')
    images: list[str] = Field(min_length=1, max_length=9)


class ReferenceVideoRequest(BaseModel):
    workflow: Literal['faceswap', 'crossview', 'reference'] = 'faceswap'
    model_id: str = 'h3-ref-turbo-int8'
    source_video: str
    source_start_seconds: float = Field(default=0, ge=0)
    source_end_seconds: float | None = Field(default=None, gt=0)
    reference_images: list[str] = Field(default_factory=list, max_length=9)
    reference_people: list[ReferencePerson] = Field(default_factory=list, max_length=9)
    source_people: list[SourcePerson] = Field(default_factory=list, max_length=12)
    person_assignments: dict[str, str | None] = Field(default_factory=dict)
    replace_subjects_first: bool = False
    replacement_prompt: str = Field(default='',max_length=16000)
    warp_video: str | None = None
    camera_azimuth: float = Field(default=-30, ge=-90, le=90)
    camera_elevation: float = Field(default=10, ge=-45, le=60)
    camera_distance: float = Field(default=1, ge=0.5, le=2)
    camera_fov: float = Field(default=50, ge=20, le=100)
    camera_vertical_shift: float = Field(default=0, ge=-0.5, le=0.5)
    camera_aim_lock: bool = False
    prompt: str = Field(default='', max_length=16000)
    width: int = Field(default=768, ge=256, le=1344)
    height: int = Field(default=448, ge=256, le=1344)
    frames: int = Field(default=124, ge=22, le=362)
    steps: int | None = Field(default=None, ge=1, le=60)
    quality: Literal['standard', 'turbo'] | None = None
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    reference_image_size: Literal['match', 'max'] = 'match'
    source_audio: bool = True
    output_audio: Literal['generated', 'source', 'mute'] = 'generated'
    enhance: bool = False
    output_dir: str = str(VIDEOS_DIR)
    loras: list[LoraChoice] = Field(default_factory=list, max_length=4)

    @model_validator(mode='after')
    def grouped_references(self):
        if self.reference_people:
            ids=[person.id for person in self.reference_people]
            images=[image for person in self.reference_people for image in person.images]
            if len(set(ids))!=len(ids) or len(images)>9 or len(set(images))!=len(images):
                raise ValueError('Use unique people and up to nine images in total.')
            if self.reference_images and self.reference_images!=images:
                raise ValueError('Reference images do not match the selected people.')
            self.reference_images=images
            owners={image:person.id for person in self.reference_people for image in person.images}
            self.person_assignments={key:owners.get(value,value) for key,value in self.person_assignments.items()}
        return self

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


class UpscaleRequest(BaseModel):
    model_id: str = 'seedvr2-3b-fp8'
    source_video: str
    resolution: int = Field(default=1080, ge=480, le=4320)
    size_mode: Literal['short_side', 'scale'] = 'short_side'
    scale: Literal[2, 4] = 2
    memory_mode: Literal['auto', 'low_memory', 'gpu'] = 'auto'
    color_correction: Literal['lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none'] = 'lab'
    clip_len: int = Field(default=8, ge=4, le=32)
    flash_chunk_size: int = Field(default=33, ge=9, le=129)
    flash_sampling: Literal['streaming_faithful_lowvram', 'streaming_faithful_full', 'streaming'] = 'streaming_faithful_lowvram'
    flash_sparse_ratio: float = Field(default=2.0, ge=1, le=4)
    flash_local_range: Literal[3, 5, 7, 9, 11, 13, 15] = 11
    flash_conditioning: float = Field(default=1.0, ge=0, le=2)
    flash_cache_format: Literal['int8', 'hybrid', 'float'] = 'int8'
    flash_decoder: Literal['tiny', 'full'] = 'tiny'
    flash_color: Literal['off', 'adain', 'wavelet_quarter_res', 'wavelet_full_res'] = 'off'
    batch_size: int = Field(default=5, ge=5, le=33)
    chunk_size: int = Field(default=100, ge=5, le=1000)
    temporal_overlap: int = Field(default=1, ge=0, le=32)
    seed: int = Field(default=100, ge=0, le=2147483647)
    preserve_audio: bool = True
    output_dir: str = str(VIDEOS_DIR)

    @field_validator('resolution')
    @classmethod
    def valid_resolution(cls, value):
        if value % 2:
            raise ValueError('Target resolution must be an even number.')
        return value

    @field_validator('clip_len')
    @classmethod
    def valid_clip_len(cls, value):
        if value % 4:
            raise ValueError('SwiftVR frame batch must be a multiple of 4.')
        return value

    @field_validator('batch_size')
    @classmethod
    def valid_batch_size(cls, value):
        if value % 4 != 1:
            raise ValueError('Frame batch size must follow 4n+1: 5, 9, 13, 17…')
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


def resolve_video(video_id: str) -> Path:
    if not isinstance(video_id, str) or len(video_id) < 34:
        raise ValueError('Input video not found. Upload it again.')
    path = (VIDEO_UPLOADS_DIR / video_id).resolve()
    if not path.is_relative_to(VIDEO_UPLOADS_DIR.resolve()) or not path.is_file():
        raise ValueError('Input video not found. Upload it again.')
    if path.suffix.lower() not in {'.mp4', '.mov', '.mkv', '.webm'}:
        raise ValueError('Unsupported input video.')
    return path

def resolve_audio(audio_id: str) -> Path:
    path = (AUDIO_UPLOADS_DIR / audio_id).resolve()
    if not path.is_relative_to(AUDIO_UPLOADS_DIR.resolve()) or not path.is_file():
        raise ValueError('Voice reference not found. Upload it again.')
    if path.suffix.lower() not in {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}:
        raise ValueError('Unsupported audio file.')
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

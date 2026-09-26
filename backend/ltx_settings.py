"""LTX-2.5 MSR V2 contract; independent from MiniMax video replacement."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from .common import VIDEOS_DIR

SIGMAS = [1.0, .99375, .9875, .98125, .975, .909375, .725, .421875, 0.0]
REFINE_SIGMAS = [.85, .725, .4219, 0.0]


class MultiReferenceRequest(BaseModel):
    model_id: str = 'ltx25-msr-q4_k_m'
    prompt: str = Field(min_length=1, max_length=16000)
    reference_images: list[str] = Field(min_length=1, max_length=4)
    background: str | None = None
    audio_references: list[str | None] = Field(default_factory=list, max_length=2)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=448, ge=256, le=1536)
    frames: int = Field(default=121, ge=33, le=361)
    fps: Literal[24, 25] = 24
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    reference_strength: float = Field(default=1, ge=.1, le=1)
    lora_strength: float = Field(default=1, ge=.1, le=1.5)
    reference_frames: Literal[25, 33] = 33
    refine: bool = False
    audio: bool = True
    enhance: Literal[False] = False
    output_dir: str = str(VIDEOS_DIR)

    @field_validator('output_dir')
    @classmethod
    def output_folder(cls, value):
        return value.strip() or str(VIDEOS_DIR)

    @field_validator('prompt')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('Describe the video you want to create.')
        return value.strip()

    @field_validator('width', 'height')
    @classmethod
    def size(cls, value):
        if value % 32:
            raise ValueError('LTX dimensions must be multiples of 32.')
        return value

    @field_validator('frames')
    @classmethod
    def duration(cls, value):
        if value % 8 != 1:
            raise ValueError('LTX video length must use 8n+1 frames.')
        return value


def required_ltx(request):
    from .models import entry
    item = entry(request.model_id)
    if item['kind'] != 'multi-reference' or item['family'] != 'ltx25-msr':
        raise ValueError('Choose an LTX 2.5 multi-reference model.')
    if request.width * request.height > 1024 * 1024:
        raise ValueError('The first pass must not exceed one megapixel.')
    for index, audio in enumerate(request.audio_references):
        if audio and index >= len(request.reference_images):
            raise ValueError('Each voice reference needs its matching subject image.')
    ids = [item['id'], *item['dependencies']]
    if request.refine:
        ids.append('ltx25-spatial-upscaler')
    return ids


def media_inputs(request):
    """One ordered transfer contract shared by desktop and RunPod."""
    items = [('image', value) for value in request['reference_images']]
    if request.get('background'):
        items.append(('image', request['background']))
    items.extend(('audio', value) for value in request.get('audio_references', []) if value)
    return items

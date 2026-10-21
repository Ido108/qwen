"""Reference images contract (LTX-2.5 MSR V2 or MiniMax H3 Ref2VA) and LTX-2.5 video rules."""
from __future__ import annotations
import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from .common import VIDEOS_DIR, LoraChoice
from .prompt_mentions import ImageLabels

# Official LTX-2.5 distilled schedule (ComfyUI blueprints "Text to Video (LTX-2.5)",
# "Image to Video (LTX-2.5)" and "First & Last Frame to Video (LTX-2.5)").
SIGMAS = [1.0, .99375, .9875, .98125, .975, .909375, .725, .421875, 0.0]
REFINE_SIGMAS = [.85, .725, .4219, 0.0]
# Official negative prompt. CFG is 1, so ComfyUI skips the negative pass.
LTX_VIDEO_NEGATIVE = 'pc game, console game, video game, cartoon, childish, ugly'
# Official first-frame strengths: 0.7 for the half-size pass, 1.0 after the 2x upscale.
LTX_FIRST_FRAME_STRENGTH = (.7, 1.0)
# Official first and last frame guide strength (single full-size pass).
LTX_KEYFRAME_STRENGTH = .7
LTX_REFINE_SEED = 42  # The official refinement pass uses a fixed noise seed.
LTX_FPS = 24


class MultiReferenceRequest(ImageLabels):
    model_id: str = 'ltx25-msr-q4_k_m'
    prompt: str = Field(min_length=1, max_length=16000)
    reference_images: list[str] = Field(min_length=1, max_length=4)
    background: str | None = None
    audio_references: list[str | None] = Field(default_factory=list, max_length=2)
    width: int = Field(default=768, ge=256, le=1536)
    height: int = Field(default=448, ge=256, le=1536)
    frames: int = Field(default=121, ge=22, le=362)
    fps: Literal[24, 25] = 24
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    reference_strength: float = Field(default=1, ge=.1, le=1)
    lora_strength: float = Field(default=1, ge=.1, le=1.5)
    reference_frames: Literal[25, 33] = 33
    refine: bool = False
    audio: bool = True
    enhance: Literal[False] = False
    # MiniMax H3 Ref2VA only. None follows the model's documented profile.
    steps: int | None = Field(default=None, ge=1, le=60)
    quality: Literal['standard', 'turbo'] | None = None
    loras: list[LoraChoice] = Field(default_factory=list)
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
            raise ValueError('Video dimensions must be multiples of 32.')
        return value

    @field_validator('frames')
    @classmethod
    def duration(cls, value):
        # LTX uses 8n+1 frames and MiniMax H3 uses 17n+5 frames; required_ltx checks the model.
        if value % 8 != 1 and value % 17 != 5:
            raise ValueError('Choose a supported video duration.')
        return value


def is_ref2va(item):
    return item['kind'] == 'reference-video' and item.get('family') == 'minimax-h3-ref2va'


def required_ltx(request):
    """All files for a Reference images job, for either supported model family."""
    from .models import entry, video_loras
    item = entry(request.model_id)
    for index, audio in enumerate(request.audio_references):
        if audio and index >= len(request.reference_images):
            raise ValueError('Each voice reference needs its matching subject image.')
    if is_ref2va(item):
        if request.frames % 17 != 5:
            raise ValueError('MiniMax H3 uses 17n+5 frames. Choose a duration from the list.')
        if request.fps != 24:
            raise ValueError('MiniMax H3 creates 24 fps video.')
        if request.refine:
            raise ValueError('Double width and height is available for LTX 2.5 only.')
        if request.width > 1344 or request.height > 1344 or request.width * request.height > 1344 * 768:
            raise ValueError('The video size must not exceed 1344 × 768 pixels.')
        if not 0.25 <= request.width / request.height <= 4:
            raise ValueError('MiniMax H3 supports output aspect ratios from 1:4 to 4:1.')
        from .reference_sampling import reference_sampling
        profile = reference_sampling(item, request.quality, request.steps)
        ids = [item['id'], *item['dependencies']]
        if profile.get('lora_id'):
            ids.append(profile['lora_id'])
        return list(dict.fromkeys(ids + video_loras(item, request)))
    if item['kind'] not in ('video', 'multi-reference') or item['family'] not in ('ltx25', 'ltx25-msr'):
        raise ValueError('Choose an LTX 2.5 or MiniMax H3 Ref2VA model for reference images.')
    if request.frames % 8 != 1:
        raise ValueError('LTX video length must use 8n+1 frames.')
    if request.width * request.height > 1024 * 1024:
        raise ValueError('The first pass must not exceed one megapixel.')
    ids = [item['id'], *item['dependencies'], 'ltx25-msr-v2']
    if request.refine:
        ids.append('ltx25-spatial-upscaler')
    return list(dict.fromkeys(ids + video_loras(item, request)))


def ltx_video_two_pass(request):
    """Text and first-frame video use the official half-size pass plus 2x refinement.
    First and last frame video uses the official single full-size pass."""
    return not getattr(request, 'last_frame', None)


def check_ltx_video(request):
    """Create video rules for LTX 2.5, from the official ComfyUI workflows."""
    if request.frames % 8 != 1:
        raise ValueError('LTX 2.5 uses 8n+1 frames, for example 121 frames for 5 seconds.')
    if ltx_video_two_pass(request):
        if request.width % 64 or request.height % 64:
            raise ValueError('LTX 2.5 generates at half size and then doubles it, so width and height must be multiples of 64.')
        if request.width * request.height > 1536 * 1024:
            raise ValueError('The LTX 2.5 video size must not exceed 1,572,864 pixels.')
    elif request.width * request.height > 1024 * 1024:
        raise ValueError('With a last frame, LTX 2.5 generates at full size in one pass. Use up to one megapixel.')


IMAGE_TAG = re.compile(r'\bImage\s+(\d+)\b', re.IGNORECASE)


def ref2va_prompt(request):
    """Turn the tab's "Image N" labels into MiniMax H3 media tags (<Picture N>, <Audio N>).

    Reference order matches the official r2v template: subject images, then the
    background image, then standalone voice audio.
    """
    subjects = len(request['reference_images'])
    total = subjects + (1 if request.get('background') else 0)

    def tag(match):
        index = int(match.group(1))
        if not 1 <= index <= total:
            raise ValueError(f'{match.group()} has no uploaded image.')
        return f'<Picture {index}>'

    notes = ['Use ' + ' and '.join(f'<Picture {i+1}>' for i in range(subjects)) +
             (' as the subject reference.' if subjects == 1 else ' as the subject references.') +
             ' Keep each subject\'s appearance consistent in every frame.']
    if request.get('background'):
        notes.append(f'Use <Picture {total}> as the background and setting.')
    audio = 0
    for index, voice in enumerate(request.get('audio_references') or []):
        if voice:
            audio += 1
            notes.append(f'<Audio {audio}> is the voice of the subject in <Picture {index+1}>.')
    return IMAGE_TAG.sub(tag, request['prompt']).strip() + '\n\n' + ' '.join(notes)


def media_inputs(request):
    """One ordered transfer contract shared by desktop and RunPod."""
    items = [('image', value) for value in request['reference_images']]
    if request.get('background'):
        items.append(('image', request['background']))
    items.extend(('audio', value) for value in request.get('audio_references', []) if value)
    return items

"""Requests of the portrait tools: Animate image, Lip sync, Extract audio and Face restore.

The model options are the portrait engine's own request fields (src/portraitControls.ts shows them). Only the keys
of each tool are passed on; the engine checks every value range and choice again when the job starts.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PORTRAIT_KINDS = ('animate', 'lipsync', 'extract-audio', 'face-restore')
AUDIO_KINDS = {'extract-audio'}

ANIMATION_OPTIONS = {
    'animation_engine', 'perform_mode', 'perform_expression', 'mrfa_checkpoint', 'mrfa_alignment', 'mrfa_relative',
    'mrfa_relative_jacobian', 'mrfa_adapt_scale', 'fsrt_relative', 'fsrt_adapt_scale', 'fsrt_alignment', 'fsrt_chunk_size',
    'imtalker_crop_context', 'imtalker_face_fit', 'personalive_crop_context', 'tps_mode', 'tps_adapt_scale',
    'tps_preserve_detail', 'tps_alignment', 'tps_output_aspect', 'tps_sampling', 'tps_checkpoint', 'mode', 'motion', 'lip',
    'eye', 'smoothing', 'adapt', 'relative_motion', 'stitching', 'normalize_lip', 'source_crop_scale', 'driving_crop_scale',
    'driving_crop_mode', 'crop_vy', 'redetect_interval', 'reference_samples', 'scene_handling', 'composition',
    'composition_blend', 'face_analysis', 'mesh_shape', 'face_stretch', 'lipsync'}
SYNC_OPTIONS = {
    'sync_model', 'sync_steps', 'sync_guidance', 'sync_seed', 'wav2lip_pad_bottom', 'wav2lip_face_box', 'sync_batch_size',
    'sync_crop_shift', 'sync_crop_mode', 'sync_area', 'sync_blend', 'sync_margin', 'sync_feather'}
CLIP_OPTIONS = {'start', 'duration', 'audio_start'}
Option = bool | int | float | str | None


def checked_options(options: dict, allowed: set[str]) -> dict:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f'Unknown settings: {", ".join(unknown)}')
    for key, value in options.items():
        if isinstance(value, str) and len(value) > 40:
            raise ValueError(f'The value of {key} is too long.')
    return options


class AnimateRequest(BaseModel):
    source_image: str = Field(min_length=1, max_length=200)
    driving_video: str = Field(min_length=1, max_length=200)
    audio_mode: Literal['driving', 'file', 'video'] = 'driving'
    audio_file: str = Field(default='', max_length=200)
    audio_video: str = Field(default='', max_length=200)
    preview: bool = False
    options: dict[str, Option] = Field(default_factory=dict)
    output_dir: str = Field(default='', max_length=1000)

    @field_validator('options')
    @classmethod
    def known(cls, value):
        return checked_options(value, ANIMATION_OPTIONS | SYNC_OPTIONS | CLIP_OPTIONS)


class LipSyncRequest(BaseModel):
    source_video: str = Field(min_length=1, max_length=200)
    audio_mode: Literal['own', 'file', 'video'] = 'file'
    audio_file: str = Field(default='', max_length=200)
    audio_video: str = Field(default='', max_length=200)
    options: dict[str, Option] = Field(default_factory=dict)
    output_dir: str = Field(default='', max_length=1000)

    @field_validator('options')
    @classmethod
    def known(cls, value):
        return checked_options(value, SYNC_OPTIONS | CLIP_OPTIONS)


class ExtractAudioRequest(BaseModel):
    source_video: str = Field(min_length=1, max_length=200)
    audio_format: Literal['wav', 'mp3'] = 'wav'
    start: float = Field(0, ge=0, le=7200)
    duration: float | None = Field(None, gt=0, le=7200)
    output_dir: str = Field(default='', max_length=1000)


class FaceRestoreRequest(BaseModel):
    source_video: str = Field(min_length=1, max_length=200)
    enhance_blend: float = Field(1, ge=0, le=1)
    enhance_scale: Literal[1, 2] = 1
    start: float = Field(0, ge=0, le=7200)
    duration: float | None = Field(None, gt=0, le=7200)
    output_dir: str = Field(default='', max_length=1000)

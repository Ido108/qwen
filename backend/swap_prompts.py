"""Durable per-tool prompt versions. Jobs keep their own immutable text snapshot."""
from __future__ import annotations
import threading
import time
import uuid
from .common import DATA_DIR, read_json, write_json
from .prompt_mentions import image_mentions

PATH = DATA_DIR / 'swap-prompts.json'
LOCK = threading.RLock()


def style_instructions(mode: str) -> str:
    if mode != 'non-realistic':
        return ''
    return (
        'Recreate the replacement person entirely in the exact visual language of <image1>. '
        'Translate the recognizable facial features from <image2> into the same degree of stylization, '
        'facial proportions and shape simplification as the original person in <image1>. '
        'Match the original contour shapes, line weight, edge softness, surface texture, mark-making, '
        'color palette, color variation, contrast, shading method, shadow shapes and highlight treatment. '
        'Give the eyes, nose, mouth, skin and hair the same rendering method and level of detail as the source character. '
        'Match the source character’s perspective, scale, placement, depth and overlap with nearby objects. '
        'Blend the face boundary, hairline, ears, neck and clothing transitions into the same visual treatment. '
        'Make the replacement read as part of the same original artwork.'
    )


class PromptConflict(ValueError):
    pass


def default_prompt(tool: dict) -> str:
    body = tool['id'] == 'body-swap'
    part = 'person' if body else 'head'
    features = 'identity, clothing, body shape and proportions' if body else 'identity, facial features and hair'
    preserved = 'background and text' if body else 'body, clothing, background and text'
    return (f'{"body_swap" if body else "head_swap"}: Use <image1> as the base. '
            f'Give the selected {part} the {features} from <image2>. '
            'Render these features in the exact visual style of <style_source>, matching its use of shape, '
            'edges, texture, color, lighting and detail. '
            f'Preserve the exact framing, composition, camera perspective, {preserved} of <image1>. '
            "Match the original person's scale, placement and overlap with the scene. "
            'Keep the body pose from <image1> and the head angle, gaze and expression from <pose_source>.')


def _scene_prompt_v2(tool: dict) -> str:
    """Keep the prior built-in version intact for explicit history restoration."""
    body = tool['id'] == 'body-swap'
    part = 'person' if body else 'head'
    features = 'identity, clothing, body shape and proportions' if body else 'identity, facial features and hair'
    unchanged = 'background and text' if body else 'body, clothing, background and text'
    return (f'{"body_swap" if body else "head_swap"}: start with <image1> as the base image. '
            f'Replace only the selected {part} with the {features} from <image2>. '
            f'Keep the exact framing, composition, camera perspective, {unchanged} of <image1>. '
            'Integrate the replacement into the scene with the same scale, placement and occlusion as the original person. '
            'Keep the body pose from <image1>. Keep the exact head angle, gaze and expression from <pose_source>. '
            'Match the coloring, lighting, contrast, texture and visual style of <style_source> exactly, including how the person blends into the image. '
            'Transfer identity without importing the reference photo\'s setting or composition.')


def validate(tool: dict, text: str) -> str:
    if not tool.get('swap'):
        raise ValueError('Choose a Swap tool.')
    text = image_mentions(text.strip(), 2).replace('@PoseSource', '<pose_source>').replace('@StyleSource', '<style_source>')
    trigger = 'body_swap:' if tool['id'] == 'body-swap' else 'head_swap:'
    if not text.startswith(trigger):
        raise ValueError(f'Keep "{trigger}" at the start of the prompt.')
    if len(text) > 6000 or not all(f'<image{i}>' in text for i in (1, 2)):
        raise ValueError('Use both image mentions and no more than 6,000 characters.')
    return text


def _initial(tool):
    validate(tool, tool['prompt'])
    return {'current': 'style-v3', 'versions': [
        {'id': 'style-v3', 'text': default_prompt(tool), 'created_at': None, 'label': 'Match scene style'},
        {'id': 'scene-v2', 'text': _scene_prompt_v2(tool), 'created_at': None, 'label': 'Scene preservation'},
        {'id': 'author-v1', 'text': tool['prompt'], 'created_at': None, 'label': 'Original prompt'}]}


def get(tool: dict) -> dict:
    with LOCK:
        data = read_json(PATH) if PATH.exists() else {}
        state = data.get(tool['id']) or _initial(tool)
        current = next(v for v in state['versions'] if v['id'] == state['current'])
        return {**state, 'text': current['text'], 'tool_id': tool['id']}


def save(tool: dict, text: str, expected: str, restore_id: str | None = None) -> dict:
    with LOCK:
        state = get(tool)
        if state['current'] != expected:
            raise PromptConflict('This prompt changed on another screen. Reload the saved version before saving. Your text stays in the editor.')
        if restore_id:
            original = next((v for v in state['versions'] if v['id'] == restore_id), None)
            if not original:
                raise ValueError('This prompt version is no longer available.')
            text = original['text']
        text = validate(tool, text)
        if text == state['text']:
            return state
        version = {'id': uuid.uuid4().hex, 'text': text, 'created_at': time.time(),
                   'label': 'Restored version' if restore_id else 'Saved version'}
        if restore_id:
            version['restored_from'] = restore_id
        data = read_json(PATH) if PATH.exists() else {}
        data[tool['id']] = {'current': version['id'], 'versions': [version, *state['versions']]}
        write_json(PATH, data)
        return get(tool)

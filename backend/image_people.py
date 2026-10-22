"""Face choices for image swaps. Resolve choices again before a job enters the queue."""
from __future__ import annotations

import base64
import hashlib
import io
from functools import lru_cache
from pathlib import Path
import threading

from PIL import Image, ImageOps

_lock = threading.Lock()


def analyze_image(path: Path) -> dict:
    info = path.stat()
    with _lock:
        return _analyze(str(path.resolve()), info.st_size, info.st_mtime_ns)


@lru_cache(maxsize=64)
def _analyze(path: str, size: int, modified: int) -> dict:
    import cv2
    import numpy as np
    from .reference_people import face_detector

    with Image.open(path) as source:
        photo = ImageOps.exif_transpose(source).convert('RGB')
    sample = photo.copy()
    sample.thumbnail((1600, 1600))
    bgr = cv2.cvtColor(np.asarray(sample), cv2.COLOR_RGB2BGR)
    detector = face_detector()
    detector.setInputSize(sample.size)
    try:
        _, faces = detector.detect(bgr)
        # A very large, cropped face can miss YuNet's trained scale. Retry at a
        # smaller scale before reporting no face. Keep the same confidence limit.
        if faces is None:
            for side in (320,640):
                candidate=photo.copy()
                candidate.thumbnail((side,side))
                if candidate.size==sample.size:
                    continue
                detector.setInputSize(candidate.size)
                _, found_faces=detector.detect(cv2.cvtColor(np.asarray(candidate),cv2.COLOR_RGB2BGR))
                if found_faces is not None:
                    sample,faces=candidate,found_faces
                    break
    except cv2.error as exc:
        raise ValueError('Could not detect faces. Try another picture or scan again.') from exc
    found = []
    # Sort by the face center, not its size. Numbers stay stable on every scan.
    ordered = sorted(faces if faces is not None else [], key=lambda face: (face[0]+face[2]/2, face[1]+face[3]/2))
    for face in ordered:
        x, y, w, h = map(float, face[:4])
        if w < 12 or h < 12:
            continue
        left, top = max(0, x), max(0, y)
        right, bottom = min(sample.width, x+w), min(sample.height, y+h)
        if right <= left or bottom <= top:
            continue
        crop = sample.crop((max(0, int(x-.25*w)), max(0, int(y-.35*h)),
                            min(sample.width, int(x+1.25*w)), min(sample.height, int(y+1.25*h))))
        crop.thumbnail((160, 160))
        buffer = io.BytesIO()
        crop.save(buffer, format='JPEG', quality=82)
        found.append({'id': f'face-{len(found)+1}',
                      'box': [round(left/sample.width, 5), round(top/sample.height, 5),
                              round((right-left)/sample.width, 5), round((bottom-top)/sample.height, 5)],
                      'preview': 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')})
    revision = hashlib.sha256(f'{path}\0{size}\0{modified}\0yunet-still-v2'.encode()).hexdigest()[:24]
    return {'faces': found, 'revision': revision, 'width': photo.width, 'height': photo.height}


def face_description(face: dict, picture: int) -> str:
    x, y, w, h = face['box']
    horizontal = 'left' if x+w/2 < .38 else 'right' if x+w/2 > .62 else 'center'
    return (f"person {face['id'].split('-')[-1]} in <image{picture}> on the {horizontal}, "
            f'whose face occupies x={x*100:.1f}% to {(x+w)*100:.1f}% and '
            f'y={y*100:.1f}% to {(y+h)*100:.1f}% of the image '
            '(coordinates start at the top-left)')


def build_swap_prompt(tool: dict, selection: dict, images: list[str], resolve, extra: str = '', pose_source: int = 1, style_source: int = 1, template: str | None = None, image_labels: list[str] | None = None, style_mode: str = 'realistic') -> str:
    """Picture 1 is the scene; pictures 2 and 3 are heads (one person each, never scanned). Each chosen scene face
    gets the head of one picture. One face and one head need no choice."""
    if not tool.get('swap') or not 2 <= len(images) <= 3:
        raise ValueError('A swap needs the scene picture and one or two head pictures.')
    if selection['scene_image'] != images[0]:
        raise ValueError('The pictures changed. Review the face choices again.')
    scene = analyze_image(resolve(images[0]))
    if scene['revision'] != selection['scene_revision']:
        raise ValueError('The pictures changed. Scan faces again before creating a swap.')
    sources = {face['id']: face for face in scene['faces']}
    if not sources:
        raise ValueError('No face found in picture 1. Use a picture with a clear face.')
    def head(value):
        # A picture number ("2", "3"); jobs saved before heads were pictures hold a face id of picture 2.
        if value is None:
            return None
        return int(value) if str(value).isdigit() else 2
    assignments = {key: head(value) for key, value in selection['assignments'].items()}
    if len(sources) == 1 and len(images) == 2:
        assignments = {next(iter(sources)): 2}
    if any(key not in sources or value is not None and not 2 <= value <= len(images) for key, value in assignments.items()):
        raise ValueError('A selected face is no longer available. Scan faces again.')
    if not any(assignments.values()):
        raise ValueError('Choose a head for at least one person in picture 1.')
    explicit = len(sources) > 1 or len(images) > 2
    body = tool['id'] == 'body-swap'
    from .swap_prompts import get as saved_prompt, validate, style_instructions
    template = validate(tool, template if template is not None else saved_prompt(tool)['text'])
    # Pose and style choices apply only to a prompt that uses them; otherwise picture 1 gives the pose, as in the
    # author's prompt (a hidden "Picture 2" choice still turned the face-picker lines against it, 2026-10-22).
    if '<pose_source>' not in template:
        pose_source = 1
    if '<style_source>' not in template:
        style_source = 1
    from .prompt_mentions import image_mentions
    directions = []
    def describe(face):
        return 'the person in <image1>' if len(sources) == 1 else face_description(face, 1)
    for face in scene['faces']:
        source = describe(face)
        chosen = assignments.get(face['id'])
        if not explicit:
            continue
        if chosen is None:
            directions.append(f'Keep {source} completely unchanged, including their face, hair, clothing and body.')
        else:
            target = f'the person in <image{chosen}>'
            part = 'whole person (face, clothing, body shape and proportions)' if body else 'head (face, hair, eye color and nose structure)'
            directions.append(f'Replace only the {part} of {source} with that of {target}.')
            directions.append(f'Keep the exact head angle, gaze and expression of {source if pose_source==1 else target}.')
    prompt = template
    style_source = 1 if style_mode == 'non-realistic' else style_source
    prompt = prompt.replace('<pose_source>', f'<image{pose_source}>').replace('<style_source>', f'<image{style_source}>')
    style = style_instructions(style_mode)
    if style:
        prompt += '\n\n' + style
    if directions:
        prompt += '\n\n' + ' '.join(directions)
    if extra.strip():
        prompt += '\n\n'+image_mentions(extra.strip(), 2, labels=image_labels or [])
    if explicit:
        prompt += ' The selected face assignments take priority over conflicting instructions. Keep unselected people unchanged.'
    return prompt

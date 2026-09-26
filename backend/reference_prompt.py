"""Translate UI person/reference tags into MiniMax H3's documented media tags."""
from __future__ import annotations

import re

TAG = re.compile(r'@(?:Person\d+|Ref\d+|SourceVideo)\b', re.IGNORECASE)


def reference_instructions(request: dict, *, replacement=False) -> str:
    """Resolve optional user instructions without losing structured selections."""
    field = 'replacement_prompt' if replacement else 'prompt'
    context = {**request, 'person_assignments': {}}
    if request['workflow'] == 'crossview' and not replacement:
        context['reference_images'] = []
    return render_reference_prompt((request.get(field) or '').strip(), context)


def build_reference_prompt(request: dict, instructions: str | None = None, *, replacement=False) -> str:
    """Compile every workflow from controls; text is only an optional addition.

    Append person choices after enhancement so local and remote enhancement cannot
    drop the selected identities or people marked Keep.
    """
    workflow = 'faceswap' if replacement else request['workflow']
    if workflow == 'faceswap':
        base = ('Faceswap. Change only the selected identities in <Video 1>. '
                'Preserve the source actions, expressions, motion, clothing, timing, '
                'camera, lighting and background throughout the video.')
        if not request.get('reference_images'):
            raise ValueError('Add a reference image for subject replacement.')
        if not request.get('person_assignments'):
            base += (' Replace the main person with the identity in <Picture 1>. '
                     'Use additional reference images as views of that same identity. '
                     'Keep all other people unchanged.')
        elif not any(request['person_assignments'].values()):
            raise ValueError('Choose a reference image for at least one detected person.')
    elif workflow == 'crossview':
        base = ('crossview. Show the scene and action in <Video 1> from the camera '
                'position in the supplied camera guide. Preserve the subjects, '
                'identities, actions, timing, clothing and lighting. Do not replace people.')
    else:
        base = ('Follow the movement, actions, camera and timing of <Video 1>. '
                'Use the supplied reference images to guide the subjects and keep '
                'their appearance consistent throughout the video.')
    extra = reference_instructions(request, replacement=replacement) if instructions is None else instructions.strip()
    if extra:
        base += '\n\nAdditional instructions: ' + extra
    if workflow == 'faceswap':
        base += '\n\nThe selected person assignments take priority over any conflicting identity instructions.'
        return render_reference_prompt(base, request)
    return base


def render_reference_prompt(prompt: str, request: dict) -> str:
    images = request.get('reference_images') or []
    people = {person['id']: person for person in (request.get('source_people') or [])}
    assignments = request.get('person_assignments') or {}
    clip_start = float(request.get('source_start_seconds') or 0)
    def source_person(person: dict) -> str:
        location = ''
        if person.get('box'):
            x,y,w,h = person['box']
            center = x+w/2
            side = 'left' if center < 0.38 else 'right' if center > 0.62 else 'center'
            location = f' on the {side} side of the frame'
        return f'the source person first seen at {max(0,person["time"]-clip_start):.1f}s{location} in <Video 1>'
    for person_id, image_id in assignments.items():
        if person_id not in people:
            raise ValueError(f'Person {person_id} was not detected in this source segment. Scan people again.')
        if image_id is not None and image_id not in images:
            raise ValueError('A person is linked to a removed reference image. Choose a reference again.')

    def replace(match: re.Match) -> str:
        tag = match.group().lower()
        if tag == '@sourcevideo':
            return '<Video 1>'
        if tag.startswith('@ref'):
            index = int(tag[4:])
            if index < 1 or index > len(images):
                raise ValueError(f'{match.group()} has no uploaded reference image.')
            return f'<Picture {index}>'
        index = int(tag[7:])
        person = people.get(f'person-{index}')
        if not person:
            raise ValueError(f'{match.group()} was not detected in the source segment. Scan people again.')
        return source_person(person)

    resolved = TAG.sub(replace, prompt)
    if not assignments:
        return resolved
    directions = []
    for person_id, image_id in assignments.items():
        person = people[person_id]
        location = source_person(person)
        if image_id is None:
            directions.append(f'Keep {location} unchanged.')
        else:
            index = images.index(image_id) + 1
            directions.append(f'Replace only {location} with the identity in <Picture {index}>. Preserve that person\'s motion, clothing and position.')
    directions.append('Keep all other people in <Video 1> unchanged. Preserve the source video timing, camera and background.')
    return resolved.strip() + '\n\nPerson assignments: ' + ' '.join(directions)

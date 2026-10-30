"""Translate UI person/reference tags into MiniMax H3's documented media tags."""
from __future__ import annotations

import re

TAG = re.compile(r'@(?:Person\d+|Ref\d+|Source\d+|SourceVideo)\b', re.IGNORECASE)


def reference_instructions(request: dict, *, replacement=False) -> str:
    """Resolve optional user instructions without losing structured selections."""
    field = 'replacement_prompt' if replacement else 'prompt'
    context = {**request, 'person_assignments': {}}
    if request['workflow'] == 'crossview' and not replacement:
        context['reference_images'] = []
        context['reference_people'] = []
    return render_reference_prompt((request.get(field) or '').strip(), context, include_groups=False)


def build_reference_prompt(request: dict, instructions: str | None = None, *, replacement=False) -> str:
    """Compile every workflow from controls; text is only an optional addition.

    Append person choices after enhancement so local and remote enhancement cannot
    drop the selected identities or people marked Keep.
    """
    workflow = 'faceswap' if replacement else request['workflow']
    if workflow == 'faceswap' and single_identity(request):
        # The FaceSwap LoRA author's workflows send only the trigger word. With the
        # longer identity text the swap faded back to the source face mid-shot and
        # <Picture 1> appeared as a shot of its own (measured 2026-10-30).
        extra = reference_instructions(request, replacement=replacement) if instructions is None else instructions.strip()
        return 'Faceswap' + ('\n\n' + extra if extra else '')
    if workflow == 'faceswap':
        base = ('Faceswap. Change only the selected identities in <Video 1>. '
                'Preserve the source actions, expressions, motion, clothing, timing, '
                'camera, lighting and background throughout the video.')
        if not request.get('reference_images'):
            raise ValueError('Add a reference image for subject replacement.')
        if not request.get('person_assignments'):
            if request.get('reference_people'):
                pictures=group_pictures(request['reference_people'][0],request['reference_images'])
                base += f' Replace the main person with the identity in {pictures}. Keep all other people unchanged.'
            else:
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
        base += (' Scene preservation takes priority over any conflicting additional instructions: '
                 'retain the exact source camera angle, camera movement, framing, perspective, '
                 'shot boundaries, subject positions and background. Do not introduce a new '
                 'viewpoint, zoom, pan, crop, cut or camera motion. Change identities only.')
        return render_reference_prompt(base, request)
    return render_reference_prompt(base,request) if workflow=='reference' else base


def single_identity(request: dict) -> bool:
    """One source person (or none detected) becomes one reference identity; nobody to keep."""
    if not request.get('reference_images'):
        raise ValueError('Add a reference image for subject replacement.')
    assignments = request.get('person_assignments') or {}
    if assignments and not any(assignments.values()):
        raise ValueError('Choose a reference image for at least one detected person.')
    if len(assignments) > 1:
        return False
    groups = request.get('reference_people') or []
    if assignments:
        render_reference_prompt('', request)  # validates the assignment
        chosen = next(iter(assignments.values()))
        pictures = next((person['images'] for person in groups if person['id'] == chosen), [chosen])
        # Every picture must show the chosen person; otherwise the model sees two identities.
        return set(pictures) == set(request['reference_images'])
    return len(groups) <= 1 if groups else len(request['reference_images']) == 1


def group_pictures(person:dict,images:list[str])->str:
    if any(image not in images for image in person['images']):
        raise ValueError('A person has a removed reference image. Add the image again.')
    return ' and '.join(f'<Picture {images.index(image)+1}>' for image in person['images'])


def render_reference_prompt(prompt: str, request: dict, *, include_groups=True) -> str:
    images = request.get('reference_images') or []
    people = {person['id']: person for person in (request.get('source_people') or [])}
    assignments = request.get('person_assignments') or {}
    groups=request.get('reference_people') or []
    groups_by_id={person['id']:person for person in groups}
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
        if image_id is not None and image_id not in groups_by_id and image_id not in images:
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
        if tag.startswith('@person') and groups:
            index=int(tag[7:])
            if not 1<=index<=len(groups):
                raise ValueError(f'{match.group()} has no uploaded person.')
            return 'the person shown in '+group_pictures(groups[index-1],images)
        index = int(tag[7:])
        person = people.get(f'person-{index}')
        if not person:
            raise ValueError(f'{match.group()} was not detected in the source segment. Scan people again.')
        return source_person(person)

    resolved = TAG.sub(replace, prompt)
    if groups and include_groups:
        descriptions=[f'Person {index+1}: {group_pictures(person,images)} show different views of the same identity.' for index,person in enumerate(groups)]
        resolved+='\n\nReference people: '+' '.join(descriptions)+' Each person group is a separate identity. Use all images within a group together, not as separate people.'
    if not assignments:return resolved
    directions = []
    for person_id, image_id in assignments.items():
        person = people[person_id]
        location = source_person(person)
        if image_id is None:
            directions.append(f'Keep {location} unchanged.')
        else:
            pictures=group_pictures(groups_by_id[image_id],images) if image_id in groups_by_id else f'<Picture {images.index(image_id)+1}>'
            directions.append(f'Replace only {location} with the identity in {pictures}. Preserve that person\'s motion, clothing and position.')
    directions.append('Keep all other people in <Video 1> unchanged. Preserve the source video timing, camera and background.')
    return resolved.strip() + '\n\nPerson assignments: ' + ' '.join(directions)

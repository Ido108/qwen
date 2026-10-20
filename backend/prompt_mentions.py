"""Resolve UI mentions against the inputs of this request, before queueing."""
import re
from pydantic import BaseModel, Field, field_validator


def subject_label(value: str) -> str:
    label = re.sub(r'[^\w\s-]', ' ', value, flags=re.UNICODE).replace('_', ' ').replace('-', ' ')
    label = ' '.join(label.split()).lower()[:48].strip()
    return label if any(letter.isalpha() for letter in label) else 'person'


class ImageLabels(BaseModel):
    image_labels: list[str] = Field(default_factory=list, max_length=10)

    @field_validator('image_labels')
    @classmethod
    def clean_labels(cls, values):
        if any(len(value) > 200 for value in values):
            raise ValueError('Image labels must be 200 characters or fewer.')
        return [subject_label(value) for value in values]

IMAGE = re.compile(r'(?<![\w@])@(?:Image|Picture|Ref)(\d+)\b|<\s*(?:image|picture)\s*(\d+)\s*>', re.I)


def image_mentions(text: str, count: int, notation: str = 'qwen', labels: list[str] | None = None) -> str:
    """Expand friendly @ tags when labels are provided; raw model tokens keep their meaning.

    Templates pass no labels because their tokens can refer to the whole scene.
    Existing 'the person from @Image1' wording already supplies its subject.
    """
    def replace(match):
        index = int(match[1] or match[2])
        if not 1 <= index <= count:
            raise ValueError(f'Image {index} is not selected. Add it or remove its mention.')
        reference = f'<image{index}>' if notation == 'qwen' else f'Image {index}'
        if labels is not None and match[1] and not re.search(r'\bthe\s+[\w -]{1,48}\s+from\s*$', text[:match.start()], re.I):
            label = subject_label(labels[index-1]) if index <= len(labels) else 'person'
            return f'the {label} from {reference}'
        return reference
    return IMAGE.sub(replace, text)


def frame_mentions(text: str, first: bool, last: bool) -> str:
    def replace(match):
        is_first = match[1].lower() == 'firstframe'
        if not (first if is_first else last):
            raise ValueError(f'Add the {"first" if is_first else "last"} frame or remove its mention.')
        return 'the first frame' if is_first else 'the last frame'
    return re.sub(r'(?<![\w@])@(FirstFrame|LastFrame)\b', replace, text, flags=re.I)

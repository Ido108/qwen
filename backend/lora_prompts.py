"""Apply explicitly declared LoRA activation words without editing the user's draft."""
from functools import lru_cache
import json
import re
import struct


def _words(value):
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = value
        value = decoded if isinstance(decoded, list) else [value]
    if not isinstance(value, list):
        return []
    # Keep complete phrases intact. A comma can be part of a trained phrase.
    return [word.strip() for word in value if isinstance(word, str) and 0 < len(word.strip()) <= 4000]


def description_words(text):
    """Read an explicit activation label; do not invent words from examples."""
    words = []
    text = (text or '').replace('**', '').replace('“', '"').replace('”', '"')
    for match in re.finditer(r'(?im)^\s*(?:[-*#•]+\s*)?(?:(?:recommended|required)\s+)?(?:t(?:rigger|igger)(?:\s+words?)?|activation\s+(?:words?|text))\s*[*`]*\s*[:=][ \t]*([^\n]*)', text):
        value = re.sub(r'^\s*-\s*', '', match[1]).strip(' *`"\' .')
        if not value or re.match(r'(?i)(?:none\b|no\b|not (?:needed|required)\b|optional\b|the only\b)', value):
            continue
        words.extend(part.strip(' *`"\' .') for part in value.split(',') if part.strip(' *`"\' .'))
    for match in re.finditer(r'(?i)\b(?:the\s+)?(?:only\s+)?trigger word(?:\s+you\s+(?:truly\s+)?need(?:\s+for\s+this\s+model)?)?\s+is\s+["`\']([^"`\'\n]+)["`\']', text):
        words.append(match[1])
    return _words(words)


@lru_cache(maxsize=64)
def _file_words(path, size, modified):
    # Header only. Never infer triggers from training captions or tag frequency.
    try:
        with open(path, 'rb') as source:
            length = struct.unpack('<Q', source.read(8))[0]
            if not 0 < length <= min(size - 8, 16 * 1024 * 1024):
                return []
            metadata = json.loads(source.read(length)).get('__metadata__', {})
        return [word for key in ('trigger_word', 'trigger_words', 'activation_text')
                for word in _words(metadata.get(key))]
    except (OSError, ValueError, struct.error, AttributeError):
        return []


def required_words(request):
    from .models import entry, local_path
    records = []
    for choice in request.get('loras') or []:
        if choice.get('enabled') is False or float(choice.get('strength', 1)) == 0:
            continue
        item = entry(choice['id'])
        words = _words(item.get('trained_words'))
        if not words and 'lora_author' not in item:
            words = description_words(item.get('summary', ''))
        if not words:
            for file in item.get('files', []):
                if not file['path'].lower().endswith('.safetensors'):
                    continue
                path = local_path(item, file)
                try:
                    stat = path.stat()
                except OSError:
                    continue
                words.extend(_file_words(str(path), stat.st_size, stat.st_mtime_ns))
        unique = list(dict((word.casefold(), word) for word in words).values())
        addition = choice.get('prompt_addition')
        if addition is None:
            addition = item.get('prompt_addition', '')
        if unique or addition.strip():
            records.append({'id': choice['id'], 'title': item['title'], 'words': unique, 'prompt_addition': addition.strip()})
    if request.get('tool_id') and request.get('lora_strength', 1) != 0:
        from .models import image_tool
        tool = image_tool(request['tool_id'])
        if not tool.get('lora'):
            return records
        item = entry(tool['lora'])
        addition = request.get('tool_prompt_addition')
        addition = (item.get('prompt_addition', '') if addition is None else addition).strip()
        if addition:
            records.append({'id': item['id'], 'title': item['title'], 'words': [], 'prompt_addition': addition})
    return records


def contains_word(prompt, word):
    # Whole phrase matching avoids mistaking 'art' in 'cart' for an activation.
    pattern = r'\s+'.join(re.escape(part) for part in word.split())
    return bool(re.search(r'(?<!\w)' + pattern + r'(?!\w)', prompt, re.IGNORECASE))


def internal_prompt(prompt, request):
    records = required_words(request)
    result = prompt
    for record in records:
        added = []
        for word in record['words']:
            if not contains_word(result, word):
                # Preserve the exact activation text; add no explanatory prompt prose.
                result = (result.rstrip() + (' ' if result.rstrip().endswith(('.', '!', '?')) else ', ') if result.strip() else '') + word
                added.append(word)
        record['added'] = added
        addition = record.get('prompt_addition', '')
        if addition:
            from .prompt_mentions import image_mentions
            from .models import entry
            model = entry(request['model_id']) if request.get('model_id') else {}
            notation = 'qwen' if model.get('family', '').startswith('qwen-image') else 'plain'
            addition = image_mentions(addition, len(request.get('images') or []), notation, request.get('image_labels'))
            if not contains_word(result, addition):
                result = result.rstrip() + ('\n' if result.strip() else '') + addition
                record['addition_applied'] = addition
    return result, records

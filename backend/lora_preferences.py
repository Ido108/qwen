"""User-written LoRA additions, separate from imported model metadata."""
import threading
from .common import DATA_DIR, read_json, write_json

PATH = DATA_DIR / 'lora-settings.json'
lock = threading.RLock()


def read():
    with lock:
        return read_json(PATH) if PATH.exists() else {}


def save(model_id, text):
    from .models import entry
    if 'lora' not in entry(model_id).get('kind', ''):
        raise ValueError('Select a LoRA.')
    with lock:
        values = read()
        values[model_id] = {'prompt_addition': text.strip()}
        write_json(PATH, values)
    return values[model_id]


def snapshot(request):
    """A queued job keeps its text if preferences change before it starts."""
    values = read()
    for choice in request.get('loras') or []:
        if choice.get('prompt_addition') is None:
            choice['prompt_addition'] = values.get(choice['id'], {}).get('prompt_addition', '')
    if request.get('tool_id') and request.get('tool_prompt_addition') is None:
        from .models import image_tool
        request['tool_prompt_addition'] = values.get(image_tool(request['tool_id'])['lora'], {}).get('prompt_addition', '')

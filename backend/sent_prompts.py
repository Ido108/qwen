"""What the app actually sent to a model, for the result's "Sent to the model" details.

Workers add labelled texts with progress(sent_prompt={'label','text'}); the image worker keeps them as sent_prompts.
install() wraps ComfyUI's CLIP.tokenize once per process and decodes the tokens back to text, so the record holds
the exact encoder input with its hidden chat template (Qwen, Krea, Z-Image, FLUX.3 and its prompt model, MOSS, video).
"""
from __future__ import annotations

_captured: list[tuple[str, str]] = []
_installed: set[int] = set()
RUN = 8  # a token repeated more than this (image placeholders) is shown once with its count


def _inner(tokenizer, key):
    for name in (key, 'clip_' + key):
        found = getattr(tokenizer, name, None)
        if found is not None and hasattr(getattr(found, 'tokenizer', None), 'decode'):
            return found.tokenizer
    for value in vars(tokenizer).values():
        if hasattr(getattr(value, 'tokenizer', None), 'decode'):
            return value.tokenizer
    return None


def decode(tokenizer, tokens):
    """{encoder: [[(id, weight, ...)]]} to text; embeddings (images) become a short marker."""
    texts = []
    for key, batches in tokens.items():
        inner = _inner(tokenizer, key)
        if inner is None:
            continue
        for batch in batches:
            ids, parts = [], []
            def flush():
                if ids:
                    parts.append(inner.decode(ids)); ids.clear()
            index = 0
            while index < len(batch):
                token = batch[index][0]
                if not isinstance(token, int):
                    flush(); parts.append('[image]'); index += 1; continue
                end = index
                while end < len(batch) and batch[end][0] == token:
                    end += 1
                if end - index > RUN:
                    flush(); parts.append(f'{inner.decode([token])} x {end - index}')
                else:
                    ids.extend(item[0] for item in batch[index:end])
                index = end
            flush()
            text = ''.join(parts).strip()
            if text and text not in texts:
                texts.append(text)
    return '\n'.join(texts)


def install(sd_module):
    """Record every ComfyUI tokenize call of this process (once per comfy.sd module, both vendored copies)."""
    if id(sd_module) in _installed:
        return
    _installed.add(id(sd_module))
    original = sd_module.CLIP.tokenize

    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokens = original(self, text, return_word_ids, **kwargs)
        try:
            decoded = decode(self.tokenizer, tokens)
            if decoded:
                _captured.append((type(self.tokenizer).__name__, decoded))
        except Exception:
            pass  # the record is a diagnostic; it never stops a job
        return tokens
    sd_module.CLIP.tokenize = tokenize


_memory: dict = {}


def flush(progress, label='Text encoder input (with its template)', key=None):
    """Report what the encoders received since the last flush. With a key, a job that reused saved conditioning
    (no encoder call) reports the text recorded for that key earlier in this worker."""
    texts = []
    while _captured:
        _, text = _captured.pop(0)
        if text not in texts:
            texts.append(text)
    if key is not None:
        if texts:
            _memory[key] = texts
            while len(_memory) > 64:
                _memory.pop(next(iter(_memory)))
        elif key in _memory:
            texts, label = _memory[key], label + ' (reused from an earlier identical request)'
    for text in texts:
        progress(sent_prompt={'label': label, 'text': text})


def clear():
    _captured.clear()


def add(job, value):
    """Append one record to job['sent_prompts'] (the worker's update calls this)."""
    record = {'label': str(value.get('label', '')), 'text': str(value.get('text', ''))}
    if record['text'] and record not in job.setdefault('sent_prompts', []):
        job['sent_prompts'].append(record)
    return job.get('sent_prompts', [])

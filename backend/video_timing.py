"""Observe existing Comfy loads without changing placement or tensor values."""
from contextlib import contextmanager


@contextmanager
def gpu_load_stage(memory, progress, synchronize, load_stage, load_message, resume_stage, resume_message):
    original = memory.load_models_gpu

    def observed(*args, **kwargs):
        progress(stage=load_stage, message=load_message)
        result = original(*args, **kwargs)
        synchronize()
        progress(stage=resume_stage, message=resume_message)
        return result

    memory.load_models_gpu = observed
    try:
        yield
    finally:
        memory.load_models_gpu = original

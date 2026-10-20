"""Compatibility with the pinned trainer and the official INT8 text encoder."""

def complete_tied_head_quantization(state):
    """The trainer copies the embedding to lm_head; copy its quantization too.

    Without the marker, the loader treats the copied INT8 bytes as ordinary
    trainable floating weights. The two tensors use the same scale/rotation.
    Leave independently stored heads and non-quantized files unchanged.
    """
    prefix='model.language_model.embed_tokens'
    weight=state.get(prefix+'.weight')
    if weight is not None and state.get('lm_head.weight') is weight and prefix+'.comfy_quant' in state:
        for key,value in list(state.items()):
            if key.startswith(prefix+'.'):
                state.setdefault('lm_head'+key[len(prefix):],value)
    return state

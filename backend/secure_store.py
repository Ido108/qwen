"""Windows DPAPI encryption for the saved Civitai, Hugging Face and People secrets (current Windows user only)."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
from .common import DATA_DIR

AUTH_DIR = Path(os.environ.get('QWEN_AUTH_DIR', str(DATA_DIR / 'connections')))


def _crypt(data: bytes, decrypt=False):
    if os.name != 'nt':
        raise RuntimeError('This connection store requires Windows.')
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    fn = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise RuntimeError('Could not read a saved secret. Save it again in Settings.')
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(result.data)

"""Woman or man for a detected face: FairFace gender (ViT, onnx-community/fairface_gender_image_detection-ONNX),
run on the CPU by OpenCV like the face detector. Tested on 6 photos, 6 correct at 0.98 or more (2026-10-22)."""
from __future__ import annotations

import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

from .common import LIBRARY_DIR, MODELS_DIR
from .reference_people import _sha256

MODEL = (LIBRARY_DIR / 'onnx' if LIBRARY_DIR and LIBRARY_DIR.is_dir() else MODELS_DIR) / 'fairface-gender-vit.onnx'
MODEL_URL = ('https://huggingface.co/onnx-community/fairface_gender_image_detection-ONNX/resolve/'
             '8e2913896361751b45f598ab1bcd914541a51ee2/onnx/model.onnx')
MODEL_SHA256 = 'f9a361ef12ea495caf10c7ca1aff24b5fad21a4bd77f0e363597129779be836d'
_lock = threading.RLock()  # classify() holds it while _net() may call ensure(), which takes it again


def ensure() -> Path:
    with _lock:
        if MODEL.is_file() and _sha256(MODEL) == MODEL_SHA256:
            return MODEL
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        temporary = MODEL.with_suffix('.download')
        try:
            request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Qwen-Image-Studio/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if _sha256(temporary) != MODEL_SHA256:
                raise ValueError('The gender model failed its checksum.')
            temporary.replace(MODEL)
        finally:
            temporary.unlink(missing_ok=True)
        return MODEL


def available() -> bool:
    """True when the model is on disk; otherwise it starts downloading once in the background (a scan never waits)."""
    if MODEL.is_file():
        return True
    global _fetching
    if not _fetching:
        _fetching = True
        threading.Thread(target=lambda: _safe_ensure(), daemon=True).start()
    return False


_fetching = False


def _safe_ensure():
    global _fetching
    try:
        ensure()
    except Exception:
        _fetching = False


@lru_cache(maxsize=1)
def _net():
    import cv2
    return cv2.dnn.readNetFromONNX(str(ensure()))


def classify(crop_bgr) -> tuple[str, float]:
    """('female' or 'male', probability) for a face crop with some margin, as the model was trained (224 px, mean 0.5)."""
    import cv2
    import numpy as np
    blob = cv2.dnn.blobFromImage(crop_bgr, 1 / 127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)
    with _lock:
        net = _net()
        net.setInput(blob)
        logits = net.forward()[0]
    odds = np.exp(logits - logits.max())
    odds /= odds.sum()
    return ('female', 'male')[int(odds.argmax())], round(float(odds.max()), 3)

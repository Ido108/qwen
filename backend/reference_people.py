"""Identify visible faces in a selected source segment for explicit reference mapping."""
from __future__ import annotations

import base64
import hashlib
import io
import threading
import urllib.request
from pathlib import Path

from .common import APP_DIR, MODELS_DIR

DETECTOR = APP_DIR / 'reference' / 'face_detection_yunet_2023mar.onnx'
RECOGNIZER = MODELS_DIR / 'face-recognition-sface-2021dec.onnx'
RECOGNIZER_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
                  'models/face_recognition_sface/face_recognition_sface_2021dec.onnx')
RECOGNIZER_SHA256 = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
_download_lock = threading.Lock()


def ensure_recognizer() -> Path:
    with _download_lock:
        if RECOGNIZER.is_file() and _sha256(RECOGNIZER) == RECOGNIZER_SHA256:
            return RECOGNIZER
        RECOGNIZER.parent.mkdir(parents=True, exist_ok=True)
        temporary = RECOGNIZER.with_suffix('.download')
        request = urllib.request.Request(RECOGNIZER_URL, headers={'User-Agent': 'Qwen-Image-Studio/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if _sha256(temporary) != RECOGNIZER_SHA256:
                raise ValueError('The face recognition model failed its checksum. Try detecting people again.')
            temporary.replace(RECOGNIZER)
        except OSError as exc:
            raise ValueError('Could not download the face recognition model. Check the connection and scan again.') from exc
        finally:
            temporary.unlink(missing_ok=True)
        return RECOGNIZER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_people(path: Path, start: float, end: float, max_people: int = 12) -> list[dict]:
    import av
    import cv2
    import numpy as np
    from PIL import Image

    if end <= start:
        raise ValueError('Select a source segment before detecting people.')
    recognizer = cv2.FaceRecognizerSF.create(str(ensure_recognizer()), '')
    detector = cv2.FaceDetectorYN.create(str(DETECTOR), '', (320, 320), score_threshold=0.65,
                                          nms_threshold=0.3, top_k=500)
    targets = np.linspace(start + min(0.15, (end-start)/4),
                          max(start + min(0.15, (end-start)/4), end-0.15), 12)
    groups: list[dict] = []
    with av.open(str(path)) as media:
        stream = next((s for s in media.streams if s.type == 'video'), None)
        if stream is None:
            raise ValueError('The source file has no video stream.')
        origin = float(stream.start_time*stream.time_base) if stream.start_time is not None and stream.time_base else 0.0
        if start > 0 and stream.time_base:
            media.seek((stream.start_time or 0) + int(max(0, start-0.5)/float(stream.time_base)),
                       stream=stream, backward=True)
        target_index = 0
        for frame in media.decode(stream):
            seconds = (frame.time-origin) if frame.time is not None else 0.0
            if seconds + 1/48 < targets[target_index]:
                continue
            bgr = frame.to_ndarray(format='bgr24')
            height, width = bgr.shape[:2]
            scale = min(1.0, 1280/max(height,width))
            sample = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
            detector.setInputSize((sample.shape[1],sample.shape[0]))
            _, faces = detector.detect(sample)
            if faces is not None:
                for face in sorted(faces, key=lambda item: item[2]*item[3], reverse=True):
                    if len(groups) >= max_people and all(g['hits'] > 0 for g in groups):
                        break
                    x,y,w,h = face[:4]
                    if w < 16 or h < 16:
                        continue
                    aligned = recognizer.alignCrop(sample, face)
                    embedding = recognizer.feature(aligned).flatten().astype(np.float32)
                    norm = np.linalg.norm(embedding)
                    if not norm:
                        continue
                    embedding /= norm
                    matched = None
                    best = 0.363
                    for group in groups:
                        score = float(np.dot(embedding, group['embedding']))
                        if score > best:
                            matched, best = group, score
                    if matched is not None:
                        matched['hits'] += 1
                        matched['embedding'] = matched['embedding']*0.8 + embedding*0.2
                        matched['embedding'] /= np.linalg.norm(matched['embedding'])
                        continue
                    if len(groups) >= max_people:
                        continue
                    left,top = max(0,int(x-0.2*w)),max(0,int(y-0.25*h))
                    right,bottom = min(sample.shape[1],int(x+1.2*w)),min(sample.shape[0],int(y+1.2*h))
                    crop = cv2.cvtColor(sample[top:bottom,left:right],cv2.COLOR_BGR2RGB)
                    if crop.size == 0:
                        continue
                    photo = Image.fromarray(crop)
                    photo.thumbnail((180,180))
                    buffer = io.BytesIO()
                    photo.save(buffer,format='JPEG',quality=82)
                    groups.append({'id':f'person-{len(groups)+1}','time':round(seconds,2),
                                   'preview':'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'),
                                   'box':[round(float(x/scale/width),4),round(float(y/scale/height),4),
                                          round(float(w/scale/width),4),round(float(h/scale/height),4)],
                                   'hits':1,'embedding':embedding})
            while target_index < len(targets) and seconds + 1/48 >= targets[target_index]:
                target_index += 1
            if target_index == len(targets):
                break
    return [{key:value for key,value in group.items() if key != 'embedding'} for group in groups]

"""Input preparation for LoRA image tools (reference/image-tools.json)."""
from __future__ import annotations
import math

from PIL import Image

GRAY = (128, 128, 128, 255)  # Outpaint LoRA fill colour, #808080


def fit_area(image: Image.Image, area: float) -> Image.Image:
    """Scale to about `area` pixels with both sides a multiple of 32."""
    scale = math.sqrt(area / (image.width * image.height))
    size = (max(32, round(image.width * scale / 32) * 32), max(32, round(image.height * scale / 32) * 32))
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def budget(request: dict) -> int:
    """Output pixel budget: Auto 1K ≈ 1 MP, Auto 2K ≈ 2 MP (the size these LoRAs were tested at)."""
    if request.get('custom_size'):
        return min(request['width'] * request['height'], 2048 * 2048)
    return 1024 * 1024 if request.get('auto_resolution', 2048) == 1024 else 2_000_000


def outpaint_canvas(image: Image.Image, direction: str, amount: str, area: int):
    """Pad with flat gray. Returns the canvas and the box of the original picture on it."""
    extra = int(amount) / 100
    w, h = image.size
    left = right = top = bottom = 0.0
    if direction in ('all', 'horizontal'):
        left = right = w * extra / 2
    if direction in ('all', 'vertical'):
        top = bottom = h * extra / 2
    if direction == 'left': left = w * extra
    if direction == 'right': right = w * extra
    if direction == 'top': top = h * extra
    if direction == 'bottom': bottom = h * extra
    full_w, full_h = w + left + right, h + top + bottom
    scale = math.sqrt(area / (full_w * full_h))
    canvas_w, canvas_h = max(32, round(full_w * scale / 32) * 32), max(32, round(full_h * scale / 32) * 32)
    # Keep the original's aspect ratio; place it where the requested padding puts it.
    inner_w = min(canvas_w, max(1, round(w * scale)))
    inner_h = min(canvas_h, max(1, round(h * scale)))
    x = round((canvas_w - inner_w) * (left / (left + right))) if left + right else (canvas_w - inner_w) // 2
    y = round((canvas_h - inner_h) * (top / (top + bottom))) if top + bottom else (canvas_h - inner_h) // 2
    canvas = Image.new('RGBA', (canvas_w, canvas_h), GRAY)
    original = image.convert('RGBA').resize((inner_w, inner_h), Image.Resampling.LANCZOS)
    canvas.paste(original, (x, y))
    return canvas, (x, y, original)


def paste_back(result: Image.Image, box, feather: int = 24) -> Image.Image:
    """Put the untouched original back, blending only the edges that meet new content."""
    import numpy as np
    x, y, original = box
    w, h = original.size
    cols = np.ones(w, np.float32)
    rows = np.ones(h, np.float32)
    ramp_w = np.clip((np.arange(w) + 1) / feather, 0, 1)
    ramp_h = np.clip((np.arange(h) + 1) / feather, 0, 1)
    if x > 0: cols = np.minimum(cols, ramp_w)
    if x + w < result.width: cols = np.minimum(cols, ramp_w[::-1])
    if y > 0: rows = np.minimum(rows, ramp_h)
    if y + h < result.height: rows = np.minimum(rows, ramp_h[::-1])
    mask = Image.fromarray((np.outer(rows, cols) * 255).round().astype(np.uint8), 'L')
    output = result.convert('RGBA').copy()
    region = output.crop((x, y, x + w, y + h))
    output.paste(Image.composite(original, region, mask), (x, y))
    return output


def prepare(tool: dict, request: dict, images: list[Image.Image]):
    """Return (images, output_size or None, paste-back box or None)."""
    area = budget(request)
    box = None
    if tool.get('prepare') == 'restore':
        images = [fit_area(images[0], area)]
    elif tool.get('prepare') == 'outpaint':
        options = request.get('tool_options', {})
        canvas, box = outpaint_canvas(images[0], options['direction'], options['amount'], area)
        images = [canvas]
    if tool.get('fit_inputs'):
        # Each picture at a set area, scaled up or down (the BFS swaps: the author feeds both at about 2 MP;
        # Body swap gives the reference person about 0.59 MP). The output keeps the shape of image 1.
        images = [fit_area(img, area if size == 'budget' else size) for img, size in zip(images, tool['fit_inputs'])]
        return images, (request['width'],request['height']) if request.get('custom_size') else images[0].size, box
    if tool['reference_resolution'] == 0:
        # References keep their size, so the output matches image 1 exactly.
        images = [images[0]] + [img if img.width * img.height <= area else fit_area(img, area) for img in images[1:]]
        if tool.get('prepare') not in ('restore', 'outpaint'):
            images[0] = fit_area(images[0], min(area, images[0].width * images[0].height))
        return images, images[0].size, box
    return images, None, box

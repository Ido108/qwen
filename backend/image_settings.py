"""Image output sizing, independent of model loading and GPU inference."""
import math
import re


def output_dimensions(request, images, rewrite):
    if request['custom_size']:
        return request['width'], request['height']
    match = re.fullmatch(r'(\d+):(\d+)', rewrite.get('wh_ratio', ''))
    follow = re.fullmatch(r'<image(\d+)>', rewrite.get('ratio_follow', ''))
    if match and int(match[1]) > 0 and int(match[2]) > 0:
        ratio = int(match[1]) / int(match[2])
    elif images:
        index = int(follow[1]) - 1 if follow and 0 < int(follow[1]) <= len(images) else len(images) - 1
        ratio = images[index].width / images[index].height
    else:
        ratio = 1
    ratio = max(1 / 3, min(3, ratio))
    # Retain 2K behavior for old jobs that predate this setting.
    resolution = request.get('auto_resolution', 2048)
    return (round(math.sqrt(resolution * resolution * ratio) / 32) * 32,
            round(math.sqrt(resolution * resolution / ratio) / 32) * 32)

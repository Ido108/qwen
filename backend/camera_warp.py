"""Generate CrossView's actual depth-aware guide from a source clip and camera pose."""
from __future__ import annotations

import importlib.util
import sys
from .common import APP_DIR
from .models import entry,local_path


def build_camera_guide(frames,request,progress):
    """ComfyUI must already be on sys.path and initialized by video_engine."""
    import torch
    import comfy.utils
    import comfy.model_management as memory
    from comfy.ldm.moge.model import MoGeModel
    from comfy_extras.nodes_moge import MoGeInference

    item=entry('moge2-geometry')
    path=local_path(item,item['files'][0])
    progress(stage='camera_depth',message='Measuring scene depth for the camera view…')
    weights=comfy.utils.load_torch_file(str(path),safe_load=True)
    model=MoGeModel(weights)
    del weights
    with torch.inference_mode():
        geometry=MoGeInference.execute(model,frames,5,
            request['camera_fov'],2,True,True,0)[0]
    del model
    memory.soft_empty_cache()

    module_path=APP_DIR/'vendor'/'CrossViewWarp'/'crossview_warp_node.py'
    spec=importlib.util.spec_from_file_location('qwen_crossview_warp',module_path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    progress(stage='camera_warp',message='Building the new camera view…')
    warp,orbit=module.CrossViewWarp().build(
        frames,
        azimuth=request['camera_azimuth'],
        elevation=request['camera_elevation'],
        distance=request['camera_distance'],
        hfov=request['camera_fov'],
        vertical_shift=request['camera_vertical_shift'],
        depth_ratio=6.0,
        smooth_depth=False,
        invert_depth=False,
        pivot_override=False,
        keep_source_aim=request['camera_aim_lock'],
        moge_geometry=geometry,
    )
    del geometry,orbit
    return warp

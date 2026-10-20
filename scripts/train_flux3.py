"""Run the pinned trainer with the FLUX.3 INT8 tied-head compatibility fix."""
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0,os.getcwd())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from toolkit.win_console import suppress_child_consoles
suppress_child_consoles()
from backend.flux3_training import complete_tied_head_quantization
from extensions_built_in.diffusion_models.flux3.src.text_encoder import Flux3TextEncoder

original=Flux3TextEncoder.convert_state_dict_on_load
Flux3TextEncoder.convert_state_dict_on_load=classmethod(
    lambda cls,state:complete_tied_head_quantization(original(state)))
runpy.run_path(str(Path.cwd()/'run.py'),run_name='__main__')

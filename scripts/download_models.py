"""Compatibility entry point: Q8 + Q4_K_M and their image companions.
Enhance prompt uses Qwen-Image PE on this computer; its files download with the first job that uses it.
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('install_models.py')), run_name='__main__')

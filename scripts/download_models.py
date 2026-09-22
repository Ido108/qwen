"""Compatibility entry point: Q8 + Q4_K_M and their image companions.
Prompt enhancement uses Antigravity, without downloading local LLMs.
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('install_models.py')), run_name='__main__')

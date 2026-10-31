import os

# Tests never use the real model library (QWEN_MODELS_LIBRARY from local-settings.json): a fake download from
# test_krea landed there on 2026-10-21 and the app listed it as a broken Krea 2 Raw model. Tests that need a
# library patch model_library.LIBRARY_DIR to a temporary folder.
os.environ.pop('QWEN_MODELS_LIBRARY', None)

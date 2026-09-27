# FlashVSR Stock source

Upstream: https://github.com/Pizzawookiee/ComfyUI-FlashVSR-Stock
Commit: c4855c12ab91b5947568076cd24a51dcc20493a4
License: GPL-3.0-only, as declared in the upstream README and NOTICE.
Per-file attribution and the upstream NOTICE are retained.

Python runtime files are unchanged. Assets, workflows and .git are excluded.
Qwen uses the stock masked attention path, INT8 ConvRot weights, BasicGuider,
one-step sigmas [1,0], and the TCDecoder. Optional external attention binaries
are not installed or required. Weights are managed separately by the model manager.

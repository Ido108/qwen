# RTX 5090 optimization: 2026-10-21

Measured on the user's app (port 47821): RTX 5090 32 GB, 64 GB RAM, i7-13700F, Windows 11. Same prompt and seed for each pair. Times are from each job's `status.json`.

## Image models

| Model | Before | After | Change |
|---|---|---|---|
| Qwen Turbo, Q8_0 GGUF to official INT8 ConvRot | 0.38 to 0.48 s/step | 0.134 s/step | Catalog entries `turbo-int8-convrot`, `int8-convrot` |
| Qwen Uncensored (default), Q8_0 GGUF to INT8 ConvRot | 0.48 s/step | 0.15 s/step (40 steps) | `uc-int8-convrot` is the default; saved drafts move once |
| Qwen warm job, new prompt, 1024 px | about 14 s | 2.2 s (Turbo), 7.3 s (40 steps) | INT8 encoder for every Qwen model |
| Qwen Turbo INT8, 2048 px | 2.8 s/step (Q8_0) | 0.72 s/step, 7.6 s per image | |
| Noct Q FP8 (no scales) | 0.45 to 0.50 s/step | 0.21 s/step | Native FP8 matmul |
| FLUX.3 Klein INT8 | 0.21 s/step | 0.156 s/step | |
| Krea 2 Turbo FP8 / NVFP4 | not measured | 0.437 / 0.376 s/step | Official files on disk registered |
| Z-Image Turbo BF16 | not measured | 0.274 s/step | |
| MOSS Image SFT / Turbo INT8 | new | 0.30 / 0.107 s/step | New architecture |
| Offload to RAM during sampling | 3 to 12x slower in some jobs | 0 GB in every test | Encoder unloads before sampling when VRAM is short; placement is logged |

## Video (MiniMax H3, DaSiWa v4 Turbo, Orbit LoRA, 73 frames, 768 px)

| Stage | Before (job a18b8842) | After |
|---|---|---|
| Text encoder to GPU | 23.4 s | 13.8 s |
| Model and LoRA to GPU | 29.8 s | 13.0 s |
| Prepare decoder | 7.0 s | 0.003 s (first VAE reused) |
| Total | 116.9 s | 78.9 s |
| Same input, new seed | 116.9 s | 38.6 s (prompt and keyframe cache) |

## Model files

- 470 GB moved from the external USB drive E: (1.4 to 1.6 GB/s sequential, 0.5 to 0.65 GB/s through the mapped load) to `D:\models` (NVMe) in ComfyUI type folders. The app reads `D:\models` first and E: second. Cold Qwen job: 37.5 s, encoder load 3.5 s.
- Large background reads (`backend/file_prefetch.py`) fill the Windows cache before the mapped load.
- `scripts/move_models_to_library.py` copies with `robocopy /J` (unbuffered). A buffered copy filled RAM while jobs ran.

## Training (FLUX.3 Klein Base, 512 px, rank 4, 50 steps)

| Setting | Step | Sample image | Total |
|---|---|---|---|
| Before | 0.78 s | 37 s | 163 s |
| `cache_text_embeddings` on | 1.21 s | 20 s | 209 s |
| `low_vram` off, Triton installed, cache off for FLUX.3 | 0.55 s | 20 s | 136 s |

The FLUX.3 runner reuses the prompt itself; cached embeddings doubled the step time there. Qwen and Z-Image keep the cache (ai-toolkit examples); not measured.

## Measured and reverted

- FLUX.3 prompt model: MTP drafting and the official sampled decoding (top_k 1, presence penalty 1.5) were slower than greedy decoding (11 to 21 tokens/s against 28). Greedy without MTP stays.

## Found, not fixed in code

- FLUX.3 prompt decoding runs with the GPU in power state P3 (memory clock 7001 MHz, half of full) at 13 to 20% use. Setting "Power management mode: Prefer maximum performance" for the app's Python in the NVIDIA settings is a user action.
- Driver "CUDA Sysmem Fallback Policy": "Prefer No Sysmem Fallback" for the app's Python stops silent VRAM spill to RAM. A user action.
- The first keyframe VAE encode of an H3 job took 12 s (1.3 s before); the second encode took 0.8 s. Cause not proven.

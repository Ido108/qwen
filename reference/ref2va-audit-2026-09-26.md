# Ref2VA sampling, performance and interface audit

## Sampling contracts

The model list is the source of default steps, quality mode, sampler, schedule and
video/audio sigma shifts. The API and worker both resolve that contract. An explicit
step count is preserved; selecting another model resets to that model's default.

| Checkpoint | Default steps | Sampler | Video/audio shift | Evidence |
| --- | ---: | --- | --- | --- |
| MiniMax Ref2VA standard INT8 | 20 | res_multistep / simple | 12 / 3 | [Comfy R2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json): standard switch selects PrimitiveInt 143, value 20 |
| MiniMax Ref2VA Pruned GGUF, all seven quantizations | 20 | res_multistep / simple | 12 / 3 | Same base model contract; [quantizer](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) does not specify another schedule |
| MiniMax Ref2VA full GGUF, all ten quantizations | 25 | res_multistep / simple | 12 / 3 | [Quantizer's R2V workflow](https://huggingface.co/Abiray/MiniMax-H3-GGUF/blob/main/minimax_ref2va_gguf_workflow.json), BasicScheduler 124 |
| PulpCut Ref2VA Turbo INT8 | 8 | euler / simple | 12 / 3 | [Publisher](https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot) recommends eight passes with Euler; turbo is already merged |
| Standard INT8 with the optional Ref2V 8-step v1.0 768p adapter | 8 | euler / simple | 6 / 3 | [Exact artifact contract](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#lora), [Comfy sampler instructions](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/COMFYUI_SETUP_AND_INFERENCE.md) |
| 10Eros-Max Ref2VA Beta2 Q4_K_M / Q8_0 | 25 | euler / simple | 12 / 3 | [Quantizer](https://huggingface.co/Abiray/10Eros-Max-ref2va-Beta2-GGUF) recommends a range of 20–30; 25 is the app's midpoint, not an exact publisher default |

GGUF is a weight format, not step distillation. Standard GGUF variants are not
assigned an eight-step schedule. All existing model IDs and pinned file revisions
are retained. New files have pinned revisions, byte sizes and SHA-256 checksums.
Downloads reuse the H3 INT4 conditioning encoder, FP16 video VAE and FP32 audio VAE.
Only the selected transformer is added; no second copy of shared components is needed.

FaceSwap and CrossView are adapters, not independent base checkpoints. FaceSwap
uses strength 1 and its trigger. CrossView uses strength 0.8 and a separate camera
guide. Its author's eight-step example uses a **DMD adapter**, while the app's
optional eight-step adapter is **LightX2V**; those recipes are not interchangeable.
The chosen base/distillation profile determines sampling in both stages.

The Eros quantizer suggests CFG 3.5–4.5, while the original H3 checkpoint is
CFG-distilled and the native reference workflows use BasicGuider (CFG 1).
This integration retains CFG 1 with the native H3 conditioning contract. It does
not manufacture a negative branch or claim to reproduce the quantizer's CFG
suggestion. This discrepancy needs a separately authorized inference comparison.

## Observed performance, read-only

The user's Turbo INT8 job `b535f7b4` completed in 2065.8 seconds: 14.375 seconds of
output, 345 frames, 576 × 1024, eight steps. Stage timings: input preparation 57.2s,
conditioning 99.6s, sampling 1852.1s, decode 35.5s and save 9.1s, plus model loading.
The sampling log loaded 5882 MiB of DiT weights on GPU and offloaded 14114 MiB.
GPU utilization reached 100%; there was no evidence of CPU-only inference or
thermal throttling in the sampled measurements.

A subsequent user-started Eros Q4 run at 448 × 800 loaded all 11.06 GiB of its
weights under a 7.71 GiB activation estimate. Windows process counters then showed
about 7.2–7.4 GiB of shared GPU memory. No other Python inference worker was found.
The conditioning encoder had already been released (CUDA allocated: 0.01 GiB).
This distinguishes activation pressure from a leaked text encoder.

The pinned Comfy H3 memory estimate sees packed output latents, but omits the
reference video/image/audio tokens and camera guide. The app now includes these
in its working-memory budget before loading DiT weights. This preserves source
duration and dimensions, and favors deliberate weight offload over unplanned
Windows shared-memory spill. It remains an estimate, not a guaranteed VRAM cap.
No generation benchmark was run after this fix, so no speedup is claimed.

Source decoding now resizes with libswscale before expanding 4K frames to RGB.
Progress separately reports source decoding, each VAE encoding, Qwen conditioning,
model loading, sampling, output decoding and saving, with stage and total time.

## Controls and previews

Person-to-reference clicks compile the internal prompt. Additional instructions
are optional and enhancement cannot remove structured assignments. CrossView
remains independent; optional replacement is completed before the camera stage.

One thumbnail dropdown offers references and Keep. Model selection is at the top;
trim controls are inside the source player card. Optional instructions and advanced
settings collapse. Queue submission stays available during generation.

Live preview uses actual per-step H3 denoised latent estimates and Comfy's published
RGB projection. It is a low-detail animated draft (up to 12 sampled times), not
finished streamed MP4 chunks. It first appears after a completed denoising step.
Preview failure cannot fail generation. RunPod forwards stage/step progress and
copies the same preview through the configured volume; no paid cloud run was made.

## Verification and activation

- CPU unit tests cover prompt compilation, shared dependencies, model profiles,
  reference memory accounting, preview output, queue behavior and cloud progress.
- `scripts/audit_ref2va_gguf.py` checked all 17 pinned remote GGUF headers using
  bounded Range requests: H3 tensor signatures, logical shapes, channel counts,
  50 transformer blocks and advertised file sizes. This does not load model weights.
- CloakBrowser checks intercept every generation request. They exercise thumbnail
  selection, Keep, keyboard input, optional prompts, independent camera stages,
  queue submission, trim seek requests, model defaults/custom steps/reset, live
  preview switching and desktop/tablet/mobile layout. No inference is submitted.
- Windows launch retains a visible activity terminal. `Show Qwen Logs.bat` attaches
  read-only. `Restart Qwen.bat` refuses to stop a running or queued job.
- Existing active generation was not interrupted. The new backend routes require
  restarting the idle server. A deployed RunPod image must also be rebuilt from
  this commit before its new progress/preview code is active.

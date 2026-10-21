# Video implementation and model selection

Reviewed 2026-09-22 for the local RTX 4090 24 GB / 64 GB RAM installation.

## Sources and compatibility

- [FastVideo](https://github.com/hao-ai-lab/FastVideo) is the inference/distillation project. The selected accelerated model is **FastH3 8-Step V2**, not the older 4-step preview. [Upstream model card](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) limits the distilled task to text-to-audio-video; first/last-frame and reference conditioning were not distilled. Although the [Comfy guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fastvideo) exposes first/last-frame examples, Qwen routes image inputs to the base MiniMax H3 model, following the upstream scope.
- The working official converted download repository is [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy), not the unavailable `Comfy-Org/FastVideo-FastH3-Comfy` URL found in prose documentation. Official template file URLs point to the correct organization.
- [MiniMax H3 native workflows](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native) supply text-to-video and first/last-frame image-to-video. A separate Ref2VA model would be needed for arbitrary identity/style references; the first/last-frame model is not represented as that model.
- [Official workflow templates](https://github.com/Comfy-Org/workflow_templates/tree/main/templates) use `MiniMaxH3ImageToVideo`, `BasicGuider`, `res_multistep`, `simple`, and joint video/audio latent tensors. BasicGuider means **CFG 1**, including the standard base model. A CFG 6 validation render showed severe artifacts and was rejected.
- [FastH3 pruned GGUF](https://huggingface.co/vantagewithai/FastVideo-FastH3-8-Step-V2-ComfyUI-GGUF) Q4_K_M is 13.61 GB and retains all 50 learned `to_gate_compress` layers. Its metadata says `wan`; actual tensor structure is detected as MiniMax H3. The older unpruned GGUF releases are larger and do not use the same checkpoint/schedule.
- [MiniMax H3 GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF) Q4_0 is 11.38 GB. Its `minimax_h3` architecture label is accepted in the app's isolated GGUF loader, while ComfyUI still validates its tensor structure. Q8_0 and official INT8 remain optional downloads.
- [FastWan 2.2 TI2V 5B FullAttn](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers) was also reviewed. Its distinct three-step DMD schedule is not interchangeable with H3; it would require separate weights, encoder/VAE and inference integration. The current FastVideo tab uses FastH3 for shared H3 compatibility and native audio.

## Runtime contract

- All model entries pin repository revisions, sizes and SHA-256 hashes in `model-catalog.json`. The installer does not download weights. Settings downloads a chosen model and missing shared files, supports resume/cancel, and verifies every model file.
- Both families share the H3-specific Qwen3-VL-32B NVFP4 encoder, FP16 video VAE and FP32 audio VAE. This encoder is required video conditioning, not a local prompt rewriting LLM. Prompt rewriting runs on this computer (Qwen-Image PE, with the official LTX enhancer prompts for LTX).
- FastH3 uses exactly 8 steps, video/audio shifts 10/3 and VSA attention with 10% retained cubes from 20% of the schedule. Dense fallback uses Comfy Kitchen INT8 attention. The installed CUDA sparse kernels support the RTX 4090.
- Base H3 uses shift 12/3 and 20 steps by default. Optional official Turbo LoRA uses 8 steps at strength 1. Sampling guidance stays at 1 for both families.
- Duration follows the actual 17k+5 frame grid at 24 fps. Native landscape size is 1344 × 768. The first uploaded image determines the initial aspect ratio; users may adjust dimensions in multiples of 32. Lower-resolution presets are labeled Draft.
- MP4 contains H.264 video and optionally AAC stereo audio at 32 kHz. Audio can be omitted from the export; the model still denoises its joint audio/video latent. Poster and JSON provenance are saved alongside the job. Image and video histories and output folders are separate.
- A persistent shared queue accepts image and video requests at any time and runs one GPU worker at a time. Pending jobs preserve their submitted settings, can be reordered/cancelled, and survive restart. Cancel stops only the owned active worker process tree and releases its allocations before the next job starts.
- On Windows, the worker compares CUDA free memory to NVIDIA dedicated VRAM before loading models. It reserves the virtual-budget gap, a 2 GiB safety margin, and encoder activation space (4 GiB with images, 2 GiB for text). The one-use conditioning encoder is destroyed before the upstream node loads the keyframe VAE, avoiding the observed native crash while copying quantized weights back to CPU. Requested resolution, duration, and sampling steps are preserved.
- MiniMax H3 and derived weights retain their upstream community license; see the linked model repositories for the exact terms. Source licenses remain with the vendored inference code.

## Validation

Real generation results and browser evidence are kept on the local data drive under `validation`; contact images and generated media are excluded from Git. Benchmarks include loading and encoding, not just sampling. See `validation.md` for the final accepted runs and limitations.

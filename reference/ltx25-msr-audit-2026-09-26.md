# LTX 2.5 MSR V2 integration

## Upstream contract

- [MSR model and sample workflow](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference): V2 LoRA, four image slots plus background, optional voices paired with image slots 1/2. Voice duration is limited to five seconds per reference.
- [Publisher's native nodes](https://github.com/liconstudio/ComfyUI-LTX2.5-MSR), pinned at `03061c18f8a79d26f3c63931dd8e28f5e5d872a6`, Apache 2.0. Vendored with license and provenance. The loader applies learned image/audio slot embeddings and native audio conditioning patches; loading just a generic LoRA is insufficient.
- [Distilled GGUF files](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF): Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0. All seven use the same joint audio/video architecture and the same distilled schedule.
- [Official components](https://huggingface.co/Lightricks/LTX-2.5): distilled INT8 base, Gemma 4 12B INT8 encoder, video VAE, audio VAE and optional spatial upscaler. Every catalog artifact has a pinned revision, size and SHA-256. Access to these files requires the user's Hugging Face account.

The sample workflow is saved in `ltx25-msr-v2-workflow.json`. First-pass Euler
ancestral sigmas are `[1,.99375,.9875,.98125,.975,.909375,.725,.421875,0]` (8 steps,
CFG 1). Optional spatial refinement doubles both dimensions and uses
`[.85,.725,.4219,0]` (3 more steps). The app does not substitute a generic 20-step
scheduler. Reference frame count is 25 or 33, default 33; output length uses 8n+1.
The app caps the first pass at one megapixel and 361 frames. FPS is 24 or 25.

## Pipeline

Text encoding completes before loading the transformer. Comfy manages CPU/GPU
placement; CUDA is required locally and macOS submits to RunPod. GGUF loading
preserves embedded LTX 2.5 configuration and quantized model patching. The native
MSR guide receives video latents, then video/audio are concatenated with their
noise masks before sampling. Guide slots are separated and cropped before
decoding; refinement upsamples the cropped target, reapplies reference guides,
and runs its separate sigma sequence. Optional preview projects actual denoised
latents; it is not finished streamed video.

Image/voice uploads, queue submission, gallery restore, model downloads and cloud
dispatch share the main application. Required models download when a job starts,
under the same transfer lock as Settings downloads. Explicit image encoder choices
do not accidentally download the default encoder too. A failed gated-access
preflight prevents large unrelated downloads. Retry resumes available cache data.

## Verification and limits

- Seven bounded remote GGUF header checks confirm native LTXAV model detection,
  embedded transformer configuration, matching 4,349 tensor shapes, 48 blocks,
  128 video channels, pinned revisions and advertised sizes. Evidence is in
  `ltx25-gguf-header-audit.json`; reproduce with `scripts/audit_ltx_gguf.py`.
- `scripts/verify_ltx_native.py` exercises the real publisher guide and native
  concat/separate/crop nodes with tiny CPU tensors and a fake VAE, including all
  five image slots and sparse voice slot 2. It loads no checkpoint and performs no
  diffusion inference.
- CPU unit tests cover all eight INT8/GGUF model dependency sets and schedules,
  dimensions, duration, voice pairing, gated access, API queueing, automatic
  download failures/cancellation and cloud file ordering. Audio upload tests
  decode a small WAV fixture and reject invalid input without leaving a file.
- Mocked browser tests exercise image/audio uploads, model switching, image tags,
  custom dimensions, refinement, fixed seed, download progress, continuing queue
  submission, preview updates and desktop/tablet/phone layouts. All API mutations
  are intercepted; no real generation is submitted.

No LTX checkpoint inference, RTX 4090 speed benchmark, physical Mac run, deployed
RunPod generation or paid cloud test has been performed for this integration.
The current Hugging Face account returned gated-access denial for required base
components during the read-only access check. The user must grant model access
before a full local or cloud run can complete. Code and contract tests do not
certify generated quality, scene preservation or a particular VRAM footprint.

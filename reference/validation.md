# Validation on the local Windows / RTX 4090 installation

Date: 2026-09-22. GPU: RTX 4090 24 GB; RAM: 64 GB; Python 3.12.10; Torch 2.11 / CUDA 13.0. Model files are pinned and SHA-256 verified. Runtime environments live on NTFS; model weights live on the E: data drive.

## Images

The Qwen image engine was validated before adding video. Real inference covered Q8_0, Q4_K_M, official BF16, transparency, negative prompting, custom dimensions, 2K output, a single contact photo, two contact photos and ten simultaneous reference images. Original contact photos were not modified. The Windows read-only model loader, CPU encoder initialization and GPU prefix cache avoid the observed host commit/transfer failures.

| Accepted workflow | Time | Evidence job |
|---|---:|---|
| Q8 image edit through the browser, 1024², 40 steps | 46.7 s | `224c844b5cc04017ba6f46d1a0a2ca17` |
| Q4 two-contact edit, 1184 × 864, 40 steps | 78.6 s | `5d9dada566f44182bbd016e55c68fe1a` |
| Ten reference images, 1024², 40 steps | 454 s | `344ea47aa76a4c439e65a37b753c5688` |

Visual checks confirmed faces remained recognizable. In the two-contact test, the model retained an indoor setting despite the requested outdoor cafe; this is recorded as a prompt-adherence limitation, not claimed as exact fulfillment.

## Video

Times include loading, prompt/image encoding, sampling, VAE decoding and MP4 export. Peak values below are Torch allocated VRAM, not total driver-reserved VRAM. Different sizes and prompts are not a controlled speed comparison.

| Accepted workflow | Output | Time | Peak allocated VRAM | Evidence job |
|---|---|---:|---:|---|
| FastH3 Q4_K_M text-to-video | 768², 56 frames, audio | 58.5 s | 15.13 GB | `23894e938fbf4b3baedcf4cc50c699f6` |
| FastH3 Q4_K_M, native landscape, VSA active | 1344 × 768, 124 frames, audio | 183.8 s | 17.09 GB | `93e05fa12565445bba6484fcb411ae5e` |
| MiniMax H3 Q4_0, one contact photo, 20 steps | 768², 56 frames, audio | 121.8 s | 17.79 GB | `374add3ee75f43e5917de660cf235f20` |
| MiniMax H3 Q4_0 Turbo, first + last images, 8 steps | 768², 56 frames, audio | 79.4 s | 17.79 GB | `0541d0c042064bbe96150a4233340ee3` |
| MiniMax H3 Q4_0 Turbo, text-to-video, audio off | 768², 56 frames | 62.3 s | 15.13 GB | `41023255060148688ab5b60ae2a36b73` |

All accepted MP4s decode as H.264 at exactly 24 fps with the expected number of frames. Audio-enabled files have AAC stereo at 32 kHz and duration differences below 1 ms. Per-frame checks found no black frames; contact sheets were visually inspected. The standard contact clip shows a natural blink and head turn while preserving identity. The two-frame Turbo clip transitions from the supplied studio background to the supplied garden background. Waveforms and container timing were checked; semantic audio content was not manually auditioned.

The first base-model I2V experiment incorrectly used CFG 6. It produced visible artifacts, was rejected from normal history and retained only as failed QA evidence. The shipped workflow uses the official BasicGuider equivalent, CFG 1. FastH3's 50 learned sparse-attention gate tensors are checked before loading; the long native-resolution run logged the actual VSA producer path.

## Browser and installer

- Actual browser workflows cover image upload/reorder/preview, 37 examples, model selection, model remove confirmation, generation, PNG/MP4 download, cancellation, history, and restoring a running job after refresh.
- Video workflows cover automatic MiniMax selection for image inputs, first/last-frame upload, preview, swap, Standard/Turbo selection, aspect-ratio sizing, native playback, and restoring input images after refresh. Desktop 1536 px and mobile 390 px have no horizontal overflow or JavaScript errors. The final screen was compared against the generated video concept.
- Cross-tab cancellation was exercised through the UI. A concurrent image request returned HTTP 409; cancelling the owned video worker released the GPU. A subsequent Q4 image generation completed in 47.6 seconds (`dfecbda02ebd40daaad17cafa7a9f9f6`) with no browser errors.
- Model download/checksum/select/confirmed removal was exercised through the UI using a small isolated Hugging Face fixture. The same model-manager transport installs video weights and their shared dependencies. Existing user model files were not deleted during testing.
- A fresh checkout with Python, Node and uv absent from PATH completed automated setup, fetched the pinned runtimes/dependencies and 77 sample assets, built the interface, checked CUDA, created the shortcut and launched on a free local port. The latest runtime check additionally verifies H.264, AAC and the CUDA sparse-attention kernel.
- The app starts from `Setup.bat` or `Start Qwen.bat`; the branded `Qwen.lnk` shortcut lives in the REPOS folder. Repeated launch reuses the same app/data server. A separate checkout cannot accidentally reuse a different checkout's server.

## Boundaries

- 29 automated tests pass. They cover validation/security, upload safety, shared model dependencies/checksums, image/video history separation, actual MP4 encoding with and without audio, OAuth PKCE/state/refresh/DPAPI and real local callback HTTP handling.
- Google login itself is user-controlled. Live Antigravity enhancement remains unverified until the user connects their account. The normal local callback is `http://localhost:1455/auth/callback`; manual paste is only a fallback.
- Optional video INT8 and additional GGUF quantizations are offered with pinned hashes and compatible companion files. They were not all downloaded or benchmarked. The real video acceptance runs use the installed Q4 variants above.
- Existing redundant image weights have not been deleted without approval. The proposed cleanup retains the installed image Q8/Q4 models and INT8 encoder/VAE. Temporary fresh-install environments also remain pending cleanup approval after automatic approval review rejected that directory cleanup.
- This certifies the tested Windows/RTX 4090 workflows, not every possible prompt, duration, GPU or operating system.

Private input photos, generated outputs, local account tokens and full QA artifacts stay outside Git under the local data directory.

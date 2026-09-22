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
- Video workflows cover explicit MiniMax selection for image inputs, first/last-frame upload, preview, swap, Standard/Turbo selection, aspect-ratio sizing, native playback, and restoring input images after refresh. Desktop and mobile 390 px have no horizontal overflow or JavaScript errors. The model selector identifies text-only and image-capable variants without silently changing modes or dropping the uploaded image.
- The original release exercised cross-tab cancellation and GPU release. Its rejection of concurrent submissions (HTTP 409) has been replaced by the shared persistent queue. Cancelling still stops only the owned worker before advancing the queue.
- Model download/checksum/select/confirmed removal was exercised through the UI using a small isolated Hugging Face fixture. The same model-manager transport installs video weights and their shared dependencies. Existing user model files were not deleted during testing.
- A fresh checkout with Python, Node and uv absent from PATH completed automated setup, fetched the pinned runtimes/dependencies and 77 sample assets, built the interface, checked CUDA, created the shortcut and launched on a free local port. The latest runtime check additionally verifies H.264, AAC and the CUDA sparse-attention kernel.
- The app starts from `Setup.bat` or `Start Qwen.bat`; the branded `Qwen.lnk` shortcut lives in the REPOS folder. Repeated launch reuses the same app/data server. A separate checkout cannot accidentally reuse a different checkout's server.

## Boundaries

- 54 automated tests pass. They cover validation/security, Windows status-read/write sharing races, upload safety, shared model dependencies/checksums, documented model defaults, dedicated GPU memory budgeting, conditioning encoder lifetime, retained CPU weight views, conditioning cache invalidation, resident process reuse/switch/crash recovery, image/video history separation, actual MP4 encoding with and without audio, OAuth PKCE/state/refresh/DPAPI and real local callback HTTP handling.
- Google login itself is user-controlled. The normal local callback is `http://localhost:1455/auth/callback`; manual paste is only a fallback. The memory regression runs below have prompt enhancement disabled to isolate local model behavior.
- Optional video INT8 and additional GGUF quantizations are offered with pinned hashes and compatible companion files. They were not all downloaded or benchmarked. The real video acceptance runs use the installed Q4 variants above.
- Existing redundant image weights have not been deleted without approval. The proposed cleanup retains the installed image Q8/Q4 models and INT8 encoder/VAE. Temporary fresh-install environments also remain pending cleanup approval after automatic approval review rejected that directory cleanup.
- This certifies the tested Windows/RTX 4090 workflows, not every possible prompt, duration, GPU or operating system.

Private input photos, generated outputs, local account tokens and full QA artifacts stay outside Git under the local data directory.

## Follow-up: documented defaults and Windows video memory

The image catalog now uses **25 steps**, matching both pinned official ComfyUI workflows (`371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4`). The former 40-step default came from the Diffusers example. Historical 40-step measurements above remain accurate for those runs. MiniMax H3 Standard remains 20; H3 Turbo and FastH3 V2 use 8.

The user's failed H3 job `81a8175cf65c43e497e7d6c3c79d843b` exhausted GPU memory in Qwen3-VL image/text conditioning before diffusion. On this WDDM installation, CUDA reported 22.5 GiB free while NVIDIA reported about 17.2 GiB of free dedicated VRAM. The video engine now reserves that gap plus working memory and streams some weights from RAM. Releasing the single-use encoder before keyframe VAE encoding also avoids the observed Windows native crash in partial GPU-to-CPU unloading. Other applications were left running, and the original video settings were preserved.

| Accepted regression | Output | Time | Torch peak VRAM | Job |
|---|---|---:|---:|---|
| Exact failed H3 Q4 I2V request, same prompt/input/seed, Standard 20 | 768 × 1024, 124 frames, stereo audio | 291.7 s | 12.71 GiB | `d3cd0503a6d84a5a9b6f46a3b613341f` |
| Q8 image edit submitted through browser with `steps: null` | 1024², resolved 25 steps | 42.9 s | — | `77242a5bf26048b6961e459144ddf6c8` |

The 124-frame video decodes at exactly 24 fps, has no black frames or timestamp errors, and its audio/video duration difference is 0.33 ms. The contact sheet shows the supplied person raising his arms and moving within the portrait framing. Image output and desktop/mobile UI were visually reviewed. Browser checks covered all seven video catalog choices, preserved image mode and upload on model changes, missing-download restrictions, image default reset on model change, and persisted 25-step defaults. Artifacts: `validation/model-defaults-video-memory.*` and `validation/memory-fix/` on the data drive.


## Persistent queue and gallery

Six scheduler/API regression tests exercise mixed image/video FIFO execution with controlled worker processes (maximum concurrent workers: one), failure/cancel recovery, pause/resume, pending cancellation/reordering, restart persistence, fixed-seed reruns and pagination over 125 results. These tests replace GPU inference with controlled subprocesses; they do not claim an actual GPU benchmark.

The real browser/API check on a separate paused queue accepted three mixed requests (HTTP 202 for each), kept prompt/settings/model controls editable, preserved independent request snapshots, reordered and cancelled waiting jobs, restored both input images and an unsent draft after refresh, searched/filtered the gallery, restored image/video settings, and re-enqueued saved settings with the original seed. Desktop 1440px and mobile 390px screenshots were reviewed; no overflow or browser errors occurred. The isolated server used copies of existing result metadata while the user's live long video continued separately. Artifacts: `validation/queue-gallery-ui.*`, `gallery-*.png` and `queue-*.png` on the data drive.

## Resident image worker and conditioning reuse

The old server started a new Python process for every image, and the engine explicitly discarded the encoder and diffusion model. The image worker now accepts successive jobs, retaining the matching encoder, diffusion model and VAE. A different model/encoder, a video job, cancellation or an inference error releases it. CPU fallback retains immutable tensor views, including independent GGUF wrappers, rather than copying GPU tensors back into read-only Windows mappings.

An empty negative prompt skips the unused encoder pass at CFG 1. A single-entry conditioning cache keys on the actual positive/negative text and every ordered input image's mode, dimensions and pixels. The unchanged-input Q8 result at seed 66954407 is **pixel-for-pixel identical** to the accepted pre-change output `77242a5bf26048b6961e459144ddf6c8` (mean and maximum channel difference both zero).

Measured on the existing RTX 4090 workstation, with cloud enhancement disabled. "New worker" means a new Python process; the OS file cache was not cleared. The first-step column measures when the first sampling step finishes, not a predicted start time. Other applications remained open.

| Model / input | Size / steps | Worker | Processing inputs | First step completed | Total |
|---|---|---|---:|---:|---:|
| Q8, one portrait | 1024² / 25 | New | 4.284 s | 13.250 s | 28.1 s |
| Q8, same input and prompt, different seed | 1024² / 25 | Reused | 0.055 s | 1.959 s | 17.3 s |
| Q8, two different input images and a negative prompt | 1024² / 25, CFG 2 | Reused | 5.063 s | 9.933 s | 44.8 s |
| Q4_K_M, one portrait | 2048² / 12 | New | 6.623 s | 24.475 s | 59.7 s |
| Q4_K_M, same inputs, different seed | 2048² / 12 | Reused | 0.036 s | 4.917 s | 39.9 s |
| Q8, one portrait | 2048² / 25 | New | 6.209 s | 21.071 s | 92.5 s |
| Q8, same inputs, different seed | 2048² / 25 | Reused | 0.030 s | 3.271 s | 76.5 s |

An additional Q8 single-input run with a changed positive prompt retained the same worker: input processing 3.581 s, first step 5.734 s, total 21.2 s. The mixed queue also released the image worker before an H3 Q4 Turbo first/last-frame video: 56 frames, 768², 24 fps, stereo audio, 126.6 s, no black frames or frame-clock errors, and 0.67 ms audio/video duration difference. During polling, an exFAT status-file sharing race exposed an HTTP 500; status reads now retry that temporary Windows error.

The installed official BF16 model and its BF16 encoder also passed consecutive generation with a changed prompt: 55.8 s for a new worker and 7.4 s when reused (first step 54.885 s / 6.718 s; peak 16.53 GiB). This was a **512², four-step lifecycle smoke test**, not a quality or default-speed benchmark. Q8 2K and BF16 artifacts are under `validation/resident-large-1790088070/`.

Local artifacts: `validation/resident/`, `validation/resident-api-1790087796/`, and the resident benchmark/QA scripts on the data drive. Private photos and outputs are not committed.

The final mixed GPU queue (`validation/resident-api-1790088313/`) completed all five requests in FIFO order without overlapping inference: Q8 image 51.8 s, cached-input Q8 image 16.7 s, changed two-image/negative conditioning 43.2 s, H3 Turbo first/last-frame video 121.4 s, then Q8 text-to-image 43.5 s. The first three shared one worker PID; the video and subsequent image each had a new PID. The cached image's first step finished in 1.863 s, input processing took 0.042 s. Startup variability is visible across runs; warm reuse is the relevant comparison, not a promise of fixed generation times.

The browser check confirmed an editable draft during the real queue and released the idle model with **Settings → Release model memory**. Dedicated GPU use returned to about 1.9 GiB. Gallery timing details, the reuse indicator, full images and desktop/mobile layout were verified with no browser errors or mobile overflow. The gallery check was replayed against the completed results after correcting a test locator that matched both the hidden image result and the gallery dialog; no GPU work was repeated for that selector correction.

# Video upscaling audit (2026-09-27)

## Models and exact execution paths

The catalogue has 26 selectable upscalers. The existing three IDs and installed
files are preserved. Dependencies have pinned Hugging Face revisions, byte
counts and SHA-256 values in `reference-video-catalog.json`. Selection queues
the required downloads; no large weights were downloaded during this audit.

| Family | Choices | Sampling and companions |
|---|---|---|
| SeedVR2 3B | FP16, FP8, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 | One diffusion step; shared SeedVR2 VAE |
| SeedVR2 7B | FP16, FP8, FP8 mixed, all five GGUF levels above | One step; same VAE |
| SeedVR2 7B Sharp | FP16, FP8, FP8 mixed, all five GGUF levels above | One step; same VAE; sharper edges are a tradeoff, not universally better |
| FlashVSR 1.1 | INT8 ConvRot and original FP32 checkpoint | BasicGuider / CFG 1, sigmas [1,0], flow shift 5; LQ projector + stored prompt embedding + TCDecoder, or optional full Wan 2.1 VAE |
| SwiftVR | Original checkpoint, BF16 compute | One forward pass at the upstream fixed timestep 1000; ReAE + stored prompt embedding + transformer configuration |

The FP32 labels describe the distributed weights. SwiftVR is cast to BF16 while
loading; FlashVSR uses ComfyUI's GPU compute dtype. Neither runs a prompt LLM.
SeedVR2 GGUF stays quantized in the matching upstream GGUF layers.

### Sources

- [SeedVR2 runtime and parameters](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- [SeedVR2 original models](https://huggingface.co/numz/SeedVR2_comfyUI)
- [SeedVR2 GGUF and mixed precision](https://huggingface.co/AInVFX/SeedVR2_comfyUI)
- [Additional SeedVR2 quantizations](https://huggingface.co/cmeka/SeedVR2-GGUF)
- [FlashVSR official implementation](https://github.com/OpenImagingLab/FlashVSR)
- [FlashVSR stock ComfyUI adapter](https://github.com/Pizzawookiee/ComfyUI-FlashVSR-Stock), pinned to `c4855c12ab91b5947568076cd24a51dcc20493a4`
- [FlashVSR INT8 weights and companions](https://huggingface.co/pizzawookiee/FlashVSR-1.1)
- [FlashVSR original precision and Wan VAE](https://huggingface.co/1038lab/FlashVSR)
- [SwiftVR official implementation](https://github.com/H-oliday/SwiftVR), pinned to `dbec2f993ab5b0abf9a2ae2eae67117d5cadb8a7`
- [SwiftVR original model bundle](https://huggingface.co/H-oliday/SwiftVR)

These families cover native temporal restoration with quality/size choices and
two streaming alternatives. There is no claim that every published upscaler is
integrated, or that one model wins on every source. GGUF support is verified at
the loader/format contract level for SeedVR2. No verified native SwiftVR GGUF
release was found. An unverified FlashVSR GGUF dataset mirror is not advertised
as working. DOVE and older restoration research are not silently aliased to an
existing model. Proprietary services such as Topaz are not downloadable weights.

## Parameters and memory

- SeedVR2 defaults: five-frame batch, 100-frame sections, one-frame overlap,
  source-color LAB correction. The batch must be 4n+1. VAE tiling is enabled
  from 1080px short side. Automatic block swapping estimates weight and working
  memory against currently free dedicated VRAM, for every 3B/7B precision.
  Users can choose lower memory or full GPU residence. This is a memory policy,
  not a measured promise of speed or that every size fits 24 GB.
- FlashVSR defaults: faithful low-memory streaming, two new latent frames,
  sparse ratio 2, local range 11, conditioning strength 1, INT8 CPU-backed
  history, 33-frame outer sections with eight frames of context. Each section
  drops only the duplicated context; the source frame count is retained. Full
  history is selectable. The LCSA topology runs through stock masked attention;
  optional external sparse CUDA binaries are not assumed to be installed. The
  original paper's kernel speed cannot be inferred from this adapter.
- FlashVSR tiny decoding uses one latent timestep/output frame at a time for
  bounded VRAM. Full Wan VAE decoding is tiled. Optional AdaIN and wavelet color
  corrections operate in bounded CUDA chunks and are off by default.
- SwiftVR defaults to eight frames per batch for a consumer-GPU profile; 4, 12,
  16, 24 and 32 are selectable. The authors use 24 in their server benchmark.
  Qwen uses SDPA and no first-run compilation. The streaming state is retained
  across sections, without DiT overlap. Three causal tail frames are supplied,
  flushed and removed from the result so arbitrary input lengths are preserved;
  the final video is not truncated to 4k+1 frames.

Output can use a target short side or 2x/4x source size. Both dimensions follow
the source ratio, rounded only to the nearest two pixels for H.264. Internal
32/128-pixel model padding is not allowed to become the final aspect ratio.
The limits are 4320px short side / 8192px long side. Frame rate is retained.
All paths write H.264 yuv420p MP4 and retain source audio by default.

## Saving and previews

The reported SeedVR2 job completed model processing in 279.49 seconds. The
following audio muxer stalled with unbounded `apad` plus `-shortest` and video
stream copy. Its existing raw output was recovered, without another inference.
The recovered file contains 124 frames, 1080x1928, 24 fps, AAC audio, and H.264
video. The earlier raw MP4 used MPEG-4 Part 2, which the browser could not play.

Audio padding now has an explicit finite length matching the output video and
an explicit output `-t`; muxing has progress heartbeats and a bounded timeout.
An error retains the raw output for recovery. See [FFmpeg apad documentation](https://ffmpeg.org/ffmpeg-filters.html#apad)
and [FFmpeg output duration](https://ffmpeg.org/ffmpeg.html).

Upscale previews are animated samples of actual decoded completed sections.
They appear only after a section exists, with a changing preview revision.
They do not decode the VAE again or present the source as a generated result.
Create-video previews use the existing intermediate latent draft renderer.
The previous result stays separately accessible while a new job runs. Reloads
reattach to the active job, and another queued request preserves its preview.

## Verification boundary

CPU contract tests cover all catalogue dependencies, one-step defaults, output
geometry, memory policy, arbitrary-length streaming tails, overlap removal,
animated output previews, finite audio muxing, codec conversion, reuse routes,
person grouping, prompt expansion and RunPod input mapping. Browser fixtures
exercise all families and responsive widths without real generation requests.
SwiftVR/FlashVSR imports and the expanded SeedVR2 CLI were checked locally.

Final checks passed: 127 Python tests, six JavaScript session tests and the
TypeScript/Vite production build. Isolated browser checks passed in
`scripts/verify_reference_workbench.py`, `scripts/verify_people_and_send.py` and
`scripts/verify_upscale_ui.py`. They cover grouped person references and optional
prompts, all video Send to destinations, model-specific controls, queued jobs,
preview revisions, reconnects and layouts at 390/768/1280/1440px. At 1280x720 the
upscale queue button remains visible while advanced controls scroll internally.
The five output types and nested gallery run-details dialog were also checked.

The browser suites intercept generation and upload requests. Their fixture
results are interface evidence, not generated media or quality benchmarks.

The recovered real video was played through the actual local HTTP video route
in Chromium, with advancing playback time and no media error. No new model
inference, GPU quality/speed benchmark, full model download, physical Mac test,
or live RunPod generation was performed. Published 5090/H100/A100 results must
not be described as this machine's RTX 4090 performance.

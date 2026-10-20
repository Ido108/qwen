# RTX 5090 audit — 2026-10-18

## Measured stage separation, 18:04 local time

Controlled repeat `a18b8842ce8b4a0fbd4e37d0f923ca5f` uses the same astronaut input, prompt, seed, DaSiWa v4 Turbo INT8, Orbit LoRA, 768-square size, 73 frames, four steps and no audio as `0e389edd2e4443598a71b6d76624ffa9`. The output contains 73 frames at 24 fps (3.0417 seconds). The inspected contact sheet shows a complete front/side/back/side/front orbit.

| Measured operation | Seconds |
| --- | ---: |
| Initial process, imports, file readers and input VAE setup | 18.232 |
| Load text encoder into GPU | 23.415 |
| Encode prompt and release encoder | 6.509 |
| Load input VAE into GPU | 0.940 |
| Encode the two keyframes | 1.257 |
| Create the video model from its mapped file | 0.439 |
| Load video model into GPU and apply LoRA | 29.760 |
| Sample four steps, including previews | 17.677 |
| Release video model | 3.907 |
| Prepare video decoder from file | 6.976 |
| Load decoder into GPU | 0.928 |
| Decode frames and copy them to CPU | 4.989 |
| Encode and save MP4 | 1.890 |
| Total (rounded by worker) | 116.9 |

Preview work totals 0.547 seconds inside sampling. The old first preview time included pending CUDA work. It was not proof of an eight-second preview bottleneck. The new observer synchronizes the existing load and completed-step boundaries; it does not alter tensors, sampling parameters, attention, placement policy or memory reserve. A load includes lazy file reads, patch application and transfer; the measurement does not isolate PCIe transfer speed.

One-second `nvidia-smi` samples reach 99% during generation and decode. Loading stages average 18–21%, generation averages 55% including first-step setup, and decode averages 82%. Sampling peak physical use is 26,786 MiB; peak PyTorch allocation is 21.3 GiB. All 19.53 GiB of diffusion weights fit on GPU, with 0.0 GiB offloaded. Background GPU activity was 12–20%, so these utilization figures are device-wide, not exclusively Qwen. CPU and process I/O samples are retained with the GPU samples at `E:/AI/qwen-image-studio/validation/orbit/turbo4-telemetry.jsonl`.

The main measured overhead is loading: 53.175 seconds for the text and diffusion model GPU load calls alone. The video worker is single-use and releases the encoder before diffusion and diffusion before decoding. It does not reuse a warm video model across jobs. This explains repeated loads; it does not prove a hardware fault or establish optimal performance. A persistent video cache needs a separate lifecycle/memory design. The current change preserves the existing image cache and video placement safeguards.

For context, the [Nuos first-hand 5090 benchmark](https://github.com/Nuos/rtx509032g-minimax-h3-comfyui) reports 53.03 seconds cold for a three-second T2V clip and 27.47 seconds warm for I2V at approximately 0.65 MP and ten steps. It uses Linux/container, 125 GiB RAM, a different text-encoder precision, SageAttention, no enabled LoRA and cached models. Those are not matched workloads. They show why 117 seconds must not be called the best achievable time for a 5090.

The Activity tab now retains the completed stage table, exact stage times, total time and output path. Measurable stages show counters and a bar. Unknown progress shows a running indicator without invented percentages. Long histories scroll with Up/Down; current progress stays visible. The Server tab follows worker diagnostics separately and filters successful HTTP requests. Real Windows Terminal UI Automation confirmed the completed 1m 56.9s table. The server owner has no visible console, and closing a log tab does not stop it.

## Local evidence

This audit reads the existing run and its source. It does not change GPU memory settings, install attention kernels, or unload an idle model for a benchmark.

- GPU: RTX 5090, 32,607 MiB dedicated memory. An idle sample showed 24,268 MiB used, 1% activity and 40.49 W. Qwen's resident image worker held about 21.07 GiB. Loaded weights stay allocated so the next image can reuse them. Per-process WDDM counters can overlap; do not add them as unique physical memory.
- Runtime: PyTorch 2.11 with CUDA 13.0. ComfyUI revision `078dd3ddc607bc71de684f09d6779229a6852ba4`. The H3 path selects Comfy Kitchen attention. The startup line about PyTorch attention occurs before this per-model selection. It does not prove that the later model uses the wrong attention backend.
- Video job `8d88cd9fac494a279f552322ca849a56`: DaSiWa Hybrid v4 Turbo INT8, 1344×768, 124 frames, 10 steps. It completed in 177.2 seconds. Sampling took about 72.2 seconds. Later steps took about 6.8–7 seconds. The UI estimate of 15 seconds included preparation inside the generating stage.
- The model had 19.53 GiB of weights. Comfy loaded 18.09 GiB on the GPU and kept 1.43 GiB in RAM. The working-memory estimate was 8 GiB. The reserve was 3.92 GiB: a 2 GiB safety reserve plus the 1.92 GiB difference between CUDA and physical free-memory reports. The screenshot's 97% activity confirms active GPU work. It does not measure the penalty from offloading.
- `peak_vram_gb=22.41` measures peak PyTorch allocated memory, not total physical GPU use. It cannot be compared directly with the Task Manager peak of 27.3 GiB.

## Confirmed fixes

1. The selected DaSiWa version explicitly specifies 4-step Turbo. The old importer assigned 20 steps and the Create video control rejected values below 10. Import and queued sampling now use the version's 4-step default, Euler/simple and the page's video/audio shift choices. An explicit user step override remains available.
2. Eros beta 6 specifies 8–12 steps and shifts 10/3. It must not inherit beta 5's schedule, or treat the release name “beta 5” as a scheduler. Release sections and separate standard/Turbo sections are now parsed separately. A checkpoint with Turbo included does not receive another automatic Turbo LoRA.
3. The worker measures completed step intervals. The first model-placement interval does not enter the step estimate. The estimate resets between passes. The RAM notice describes measured weight placement without claiming a measured speed loss.
4. Image weights and conditioning remain cached. Video continues to use the existing isolated worker. This change does not introduce a persistent video model cache.

## External evidence and its limits

### Live check after the settings repair

Job `a23e968df8254dc987d13e6ae91941fe` completed with DaSiWa v4 Turbo INT8. The engine used four steps, CFG 1, Euler/simple and video/audio shifts 6/3. No extra Turbo LoRA was applied. The MP4 contains 124 frames at 24 fps, 768 by 448 pixels, and one audio stream. Its duration is 5.167 seconds. Total execution time was 75.4 seconds. All 19.53 GiB of model weights fit on the GPU for this workload. The last measured step interval was 1.385 seconds.

This run verifies the repaired settings and a valid output file. It is not a controlled speed comparison with the earlier 1344 by 768 run. Resolution, step count and sampling settings differ. Do not infer an end-to-end speed ratio from these two runs.

A separate image Swap check compared installed Turbo Q8, Standard Q8 and original BF16 configurations on the same two inputs and seed. BF16 took 590.7 seconds at 1408 by 1408 pixels and reached about 30.4 GiB of dedicated memory. It did not visibly improve preservation of this scene's color or head angle. The BF16 configuration also uses a larger encoder, so the comparison does not isolate model precision. See [Swap behavior audit](swap-behavior-audit.md). No memory safety setting was reduced to force this workload into VRAM.

### Source comparison

- [PyTorch CUDA memory notes](https://docs.pytorch.org/docs/stable/notes/cuda): `empty_cache()` releases unused allocator blocks. It does not free live model tensors. The current worker does not expose enough allocator data to quantify idle cache-only savings. No idle flush was added.
- [Comfy H3 model card](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/README.md): INT8 ConvRot with CUDA 13 is a supported path. The local INT8/CUDA 13 choice is consistent with this recommendation.
- [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3): the released model is self-guided. The local BasicGuider path uses CFG 1. The model's advertised resolution is not the same as this app's supported workflow limit; 2K regeneration is a separate model path.
- [FastH3 V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2): a separate eight-step text-to-video model with sparse attention. It is not an interchangeable speed toggle for arbitrary H3 checkpoints.
- [LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5): the distilled model uses eight steps and CFG 1. The local two-pass workflow adds three refinement steps. A first-and-last-frame job uses the single-pass workflow. A dev checkpoint uses the existing distillation LoRA conversion; the app does not claim to run the author's full dev schedule.
- [ComfyUI issue 15889](https://github.com/Comfy-Org/ComfyUI/issues/15889): another 5090 H3 reference run reports about 14.72 seconds per step. The workload differs, so this is not a speed target for this PC.
- [ComfyUI issue 14250](https://github.com/Comfy-Org/ComfyUI/issues/14250): Windows H3 reports include pinned-memory/host-buffer failures. The local conservative flags address this class of failure. This does not establish that every flag is optimal.
- [Sol attention implementation](https://github.com/Saganaki22/ComfyUI-sol-attn): the author reports exact INT8 feed-forward chunking and fused modulation options. The reported memory reduction is for an MLP measurement, not the whole video process. Sparse attention is a separate approximation. No new kernel was installed or enabled.
- [5090 workflow notes](https://github.com/Nuos/rtx509032g-minimax-h3-comfyui): useful hardware-specific workflow evidence, but model, precision, size and duration must match before comparing times.
- [Reddit: smaller H3 weights and encoder](https://www.reddit.com/r/comfyui/comments/1vk4ib7/if_you_are_vram_limited_you_need_to_try_this/): a first-hand report claims large memory savings with different weights, but similar speed. This changes the model/conditioning choice and does not prove an equal-quality free speedup.
- [Reddit: 5090 at 2 MP](https://www.reddit.com/r/StableDiffusion/comments/1w2hovr/minimax_h3_on_rtx_5090_32gb_2mp_is_87_slower_than/): a larger reference workload. It cannot be compared directly with this 1 MP, five-second run.
- [Reddit: measured H3 speedups](https://www.reddit.com/r/StableDiffusion/comments/1wp392j/minimax_h3_speedups_measured/): community timing reports vary with model, LoRA and attention settings. They do not establish a controlled result for this machine.

## Optimization decision

Keep the existing loaded-image cache. Keep the Windows memory safety reserve until a controlled run records allocated, reserved and physical peaks through encoding, sampling and decoding. Reducing the reserve merely because final allocated memory looks low can cause a later decode failure. Correct model steps first. Do not promise a proportional end-to-end speedup: encoding, loading and decoding still take time.

Potential later work: measure an exact INT8 feed-forward chunking implementation with the same checkpoint, seed, prompt, resolution, duration and output comparison. A new attention kernel or a smaller encoder requires a separate quality check. This audit does not claim those changes are verified on this PC.

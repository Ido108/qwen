# LoRA application and result prompts — 2026-10-19

## Confirmed defects and fixes

1. Qwen fused MLP aliases were incomplete. The app splits LoRA `img_mlp.gate_up` into `gate_layer` and `proj`. Comfy mapped the halves under bare, transformer and LyCORIS names, but not under native `diffusion_model` or Kohya names. Native adapters therefore lost those parts on fused checkpoints. `backend/lora_loading.py` adds aliases for the actual mapped targets. The change applies by architecture and tensor names, not by a list of model IDs.
2. Trigger words previously entered the editable prompt only when the user first selected a LoRA. Editing the prompt or turning a saved LoRA back on could omit them. `backend/lora_prompts.py` now adds explicit required words for each active, nonzero LoRA inside the worker, after prompt enhancement. It does not change the saved draft. Whole-phrase matching prevents duplicates. Image, keyframe video, reference video and LTX workers use the same resolver. FLUX prompt rewriting preserves the words.
3. The first size validator incorrectly used GGUF's `state_dict`. GGUF exports packed byte dimensions there for memory estimates. This rejected valid Q8 LoRAs in user jobs `e6b10ff6c5384075b4a06c496fe882b2`, `ac549775bb8d4a7f972145087cf712f9` and `ba4f17ceab99426a8518cea34f973f43`. The corrected validator reads the live parameter's logical dimensions without dequantizing large weights. A regression test now uses real Q8 GGUF layers and proves numerical patch application. This was a defect in the new check, not evidence of incompatible LoRAs.
4. Result summaries omitted the saved prompt. `ResultPrompt` now shows it below results with Copy and Show more/less. It also appears in the mobile image viewer. Details and Copy details include the submitted prompt, the effective prompt when different, and a saved negative prompt. Changing the editor or selecting another result does not change the displayed result's saved text. Long text stays bounded and scrolls when expanded.

## Loading and cache checks

- Every Qwen LoRA now receives checked adapter mapping, target existence, standard LoRA matrix dimensions and patch acceptance checks. No matched weights or an unsupported partial attachment stops the job before sampling. The job saves patch counts, tensor counts and strengths as `lora_checks`.
- The loaded base remains unpatched. Per-job clones retain distinct strengths. CPU numerical tests cover off/on, zero strength, repeated strengths, fused and split weights, native and Kohya aliases, and stacked LoRAs.
- Qwen's prefix KV cache resets at each sampling entry/cleanup. Diffusion-only LoRAs do not change text encoder outputs. Their unchanged text/image conditioning can stay cached. Added trigger words change the conditioning fingerprint.
- The user-facing saved LoRA names remain in `applied_lora`. They are not used as proof that tensor patches attached.

## Author activation metadata

Civitai imports now retain explicit `trainedWords`, or an explicit Trigger/Activation label in the author description when the API field is empty. Local safetensors can provide explicit `trigger_word`, `trigger_words` or `activation_text` header fields. Captions, tag frequencies and general example prompts are not treated as required words.

An audit fetched author metadata for 11 installed Civitai Qwen LoRAs. One entry contained a description-only activation label missing from its saved metadata. Only that entry's `trained_words` field changed. The original catalog is backed up at `E:/AI/qwen-image-studio/validation/lora-application/user-models-before-words.json`. No model files changed or downloaded. This does not infer undocumented instructions or guarantee that all words listed by an author are sufficient for a desired visual style.

The original request, effective prompt and activation record are saved separately. PNGs retain Prompt, Actual prompt and LoRA prompt words.

## Evidence

File-header audits cover 20 installed Qwen LoRA files on each of these paths:

| Base model | Unmatched adapter tensors before alias fix | After fix |
|---|---:|---:|
| `turbo-bf16` (fused) | 1,664 across 14 LoRAs | 0 |
| `civitai-model-3370753` (split) | 0 | 0 |
| `gguf-q8_0` | 0 | 0 |
| `turbo-gguf-q8_0` | 0 | 0 |

These audits use actual file headers and meta tensors. They check names and logical matrix dimensions, not image quality. The final GGUF tests apply real Q8 patches and dequantize only tiny CPU tensors. They confirm zero, half and full strengths on fused and split weights, including restoration after repeated runs. CUDA initialization is blocked in these tests.

The selected split-weight model already mapped its adapters before this fix. The alias defect cannot explain every weak result on that model.

Eight existing user-generated result pairs share all saved request settings except LoRAs. Their seed is 1777380000. Each LoRA result differs from its corresponding baseline. Mean absolute RGB differences range from 35.88 to 67.61 on a 0–255 scale. Three newer Turbo runs record 224 accepted patches and 448 adapter tensors each for GrainScape, 2000s Analog Core and NiceGirls. These comparisons show that the saved outputs changed. They do not measure whether the intended style or identity matches the author's examples. No new GPU generation was started for this audit.

Final automated checks: 63 CPU tests and six Node tests pass. The production build passes with the existing large-chunk warning. Remote Chromium and iPhone WebKit checks cover LoRA add/strength/off/on/reload with intercepted generation requests, plus saved prompt display/copy/expansion, exact prompt details, result switching, Gallery and 320-pixel width. Physical iPhone hardware is not tested.

Artifacts and scripts: `E:/AI/qwen-image-studio/validation/lora-application/`. Important outputs are `audit-*.json`, `saved-comparisons.json`, `ui.json`, `prompt-ui.json` and the browser screenshots.

## Delivery

The restart guard first refused while user jobs were active. It stopped nothing. A later guarded restart completed with no active or queued job and no download. Actual listening server PID is 52172. The Windows venv launcher PID in `runtime.json` is its parent, not the listener itself.

The delivery instance uses the existing `QWEN_NO_WARMUP=1` option to prevent its automatic one-step GPU warmup from running during validation. No persistent Windows environment setting or launch default changed. The next requested image loads its model normally and retains it for later images. This instance also skips automatic warmup after video; normal launch restores the usual warmup behavior.

The first remote checks overlapped relay reconnection and received HTTP 503. The relay subsequently reported Connected and Remote connection restored. Authenticated live page and models requests returned 200.

Final post-restart checks pass in Chromium and iPhone WebKit. The LoRA test preserves one WebKit fetch access-control diagnostic during explicit document reload, separately from stable-page errors. It concerns an intercepted job request in the departing document. Post-reload queue requests return 200; selection, strength, bypass and submitted payloads pass. There are no stable-page errors or HTTP failures. The saved-prompt test passes with no page or HTTP errors in either browser. It also verifies Gallery, Copy details, result switching, mobile navigation and 320 px width. Desktop and mobile screenshots were inspected. Final read-only state shows no active job, pending job or running download.

Build: `index-3IAndSrq.js` / `index-CR8p3BBi.css`. Setup fingerprint matches. Model binaries, previous images, active work and the existing dirty worktree remain intact.

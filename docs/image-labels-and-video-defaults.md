# Image labels, saved CFG and reference video

Date: 2026-10-19.

## Delivered behavior

- Image inputs, Swap inputs and reference-video inputs have an editable subject label. Person is the default; background slots default to Background. Presets and a short custom label are available. This follows Weboss's separation between a subject label, a file name and a person's identity.
- A friendly `@Image1` tag compiles to `the person from <image1>` for Qwen, or the selected label. LTX receives `the person from Image 1`; MiniMax Ref2VA receives `the person from <Picture 1>`. The native keyframe tags and video person-group tags retain their distinct meanings. Raw tokens in internal Swap templates still refer to whole images.
- The queued request saves the original editor text and ordered labels. Saved settings restore these values. Photo cycling keeps the label; replacement starts with the new image's label. Shared images do not imply a person unless the owner is explicit or unique.
- Result summaries and details show the engine's saved CFG. The mobile image viewer also shows CFG. The current form and mutable model defaults never supply this value. Screenshot result `e34a1ebc695842aebaa4091cd6f19de5`, seed 1653952953, records CFG 1 in both engine fields.
- Video's default entry and `#video` open Image to video with subject references. Text or keyframes remains available at `#keyframes`. Saved video settings still restore their original workflow. Civitai selections route reference-capable models and LoRAs to the reference screen.

## Model contract audit

The read-only audit covers 38 configured video models, 39 sampling profiles and 46 model/support files. The pinned revision, size and SHA-256 match Hugging Face for 44 files. The two installed Civitai model hashes match their exact published version files. This verifies metadata, not a second full hash read of all local weights or inference output.

| Workflow | Files and sampling |
| --- | --- |
| MiniMax Ref2VA Standard | Ref2VA transformer, H3 encoder, video VAE, audio VAE. The official baseline uses 20 steps, res_multistep/simple, CFG 1 and video/audio shifts 12/3. Imported author settings apply per version and file. |
| MiniMax Ref2VA optional Turbo | Adds the matching Ref2VA eight-step LoRA. Uses Euler, CFG 1 and shifts 12/3. The FL2VA adapter is not substituted. |
| MiniMax merged Turbo | Uses the checkpoint's own profile. No second acceleration LoRA is added. Installed DaSiWa uses four steps; Eros beta 6 uses its eight-step starting value. |
| LTX 2.5 reference images | Transformer, Gemma 4 encoder, video/audio VAEs and MSR V2. Eight fixed sigma intervals, Euler ancestral and CFG 1. Optional 2x refinement adds the spatial upscaler and three sigma intervals. |
| Imported LTX 2.5 dev | Adds the official distillation LoRA for the supported distilled workflow. A stated author strength overrides the workflow starting value. Distilled files do not receive this adapter. |

Imported LTX models previously appeared only in the keyframe screen. The reference page now accepts them by architecture. Its dependency resolver adds MSR V2, including for future matching imports. Keyframe jobs exclude MSR V2. File names take priority over mixed dev/distilled page titles. A low step count alone no longer proves that a checkpoint is distilled. An unidentified variant or an incompatible custom distilled schedule is rejected before a large download instead of silently using eight steps.

Civitai details also use the selected file's variant and distillation strength. A mixed Dev/Distilled page no longer shows the page-level variant for both files. The refinement description now states that the extra three steps are optional.

Source checks:

- [Comfy-Org MiniMax Ref2VA workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json): baseline sampling and shared components.
- [LightX2V Ref2VA V1.0 release](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51): eight steps, Euler and shifts 12/3.
- [PulpCut merged Ref2VA](https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot): the exact output-major checkpoint and eight-step recipe. Its separate input-major file is not used.
- [Lightricks LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5): split transformers, Gemma encoder, video/audio VAEs, distillation adapter and upscaler.
- [Licon MSR V2 model card](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference) and [sample workflow](https://github.com/liconstudio/ComfyUI-LTX2.5-MSR/blob/main/LTX2.5-MSR-sample-workflow-V2.json): ordered visual/voice references, `Image N` prompt convention, native reference conditioning and the 8+3 schedules.

LTX uses the user's compiled description without an added image-editing prompt. H3 uses native Picture/Audio references with concise subject/background/voice binding. First/last-frame conditioning remains a separate path. These rules apply to supported architectures and known formats. Unknown future architectures, custom nodes, quantizers or schedules need a matching implementation; the audit does not prove their compatibility.

## Validation and limits

- 72 CPU contract tests pass. The last backend-only importer refinement also passes its 40 affected tests. Six Node tests cover mention labels, ambiguous photo ownership, actual CFG and architecture-based routing. TypeScript and the production build pass; Vite retains its existing large-chunk notice.
- Chromium desktop and iPhone 13 WebKit emulation pass local label editing, custom labels, autocomplete, exact intercepted submissions, photo cycling, reload, Swap labels, reference-video labels, default video navigation and 320 px overflow checks. No page or HTTP errors occur. Mobile CFG is checked in the full image viewer.
- Browser submissions are intercepted. API compilation is tested with scheduling disabled. No new GPU generation or model-weight download runs. Existing result and prompt history data remain unchanged. Physical iPhone hardware and the visual quality of newly generated output are not tested.
- Evidence: `E:/AI/qwen-image-studio/validation/image-labels/`. `video-audit.json` records each profile and file check. `local-ui.json` records browser checks. Live delivery evidence follows in the task queue.

## Live delivery

- The guarded launcher restarted the backend only while the queue and downloads were idle. The new server PID was 10656. The final frontend bundle is `index-pDy6oyHL.js`; CSS is `index-DfAAFcRk.css`.
- A live, read-only Swap preview compiled `Keep @Image1 beside @Image2.` with Person/Statue labels into `Keep the person from <image1> beside the statue from <image2>.` The saved template tokens remained unchanged. Evidence: `live-backend.json`.
- Chromium desktop and iPhone 13 WebKit emulation passed the same interactions through the real Railway relay. The final run has no HTTP errors, no errors in the loaded page and no horizontal overflow at 320 px. WebKit reported one cancelled queue request during an intentional reload; the new page loaded and passed its assertions. Evidence: `live-ui.json`.
- Opening the mention list exposed an eager-avatar request burst that exceeded the relay's pending-request limit. Mention photos now use lazy loading. The final remote check reports no 429 responses. A separate test fixture used a root upload URL instead of the relay prefix; that fixture is corrected. No Moise authentication or rate limit policy changed.
- A mocked Civitai response with two files verifies Dev at strength 0.7, Distilled without the extra adapter, and switching back. Both desktop and mobile pass. No installation runs. Evidence: `civitai-files.json` and screenshots.
- Test artifacts occupy 209.3 MiB. They remain in the validation folder. The isolated CPU test server on port 47829 is stopped after identity verification. No existing result, prompt version or model file was removed.

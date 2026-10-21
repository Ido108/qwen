# Qwen delivery queue

## Swap and output actions, 2026-10-21

98. [x] Restore the full Swap scene-preservation prompt. Retain all saved versions. Keep style-mode instructions additive. Verify the compiled prompt without inference.
99. [x] Show the selected Realistic / Non-realistic mode clearly on desktop and mobile.
100. [x] Add a shared Use as menu to image outputs and previews. Route to every image-input workflow and open it immediately. Preserve other draft inputs and settings.
101. [x] Add durable per-model Save default actions, including Swap. Keep save and reset separate. Verify persistence, model isolation, restored jobs and local/remote delivery.

Items 98-101 are delivered and verified locally and through Moise. See `swap-output-defaults.md` for the exact changes and evidence. Item 96 visual comparisons still require the pending permission. No QA generation is authorized.

## Compact LoRA controls and current execution check (2026-10-20)

89. [x] Reduce LoRA row height. Keep the strength slider, numeric value, bypass and prompt editor. Add a searchable replacement picker on each row, preserving its slot and strength.
90. [x] Remove the four-LoRA limit from all image/video selection paths and server schemas. Preserve compatibility checks and reference-image limits.
91. [x] Explain the reported weak style effect from current source and available saved runs. Separate earlier loader faults, runtime execution and visual style fidelity. Do not run QA inference.
92. [x] Verify replacement, more than four selections, bypass, prompt ownership, desktop/mobile layout, request validation and guarded live delivery.
93. [x] Show the saved actual sampler and schedule under results and in Details/Copy details, including fixed Turbo schedules. Preserve unknown values in older records.
94. [x] Move all preview text and controls outside the image. Inspect Studio Image's viewer pattern. Verify long prompts, zoom, navigation, desktop/mobile/landscape geometry and local/remote delivery.
95. [x] Correct Civitai file-level choices within each version. Verify the reported Qwen 2.1 FinalCut files and all import paths. Keep each file's format, size, support, saved identity, selected settings and download progress distinct. Preserve active downloads and installed variants. Use exact author/file evidence for settings; do not infer different steps from quantization alone. Verify desktop/mobile selection, request identity, concurrent downloads, installed state and guarded delivery.
96. [ ] Reopen the unresolved LoRA style-effect defect. Audit actual installed LoRA files and recent inference records against current ComfyUI, Diffusers, Qwen and exact Civitai author sources. Trace format conversion, layer mapping, alpha/rank scaling, quantized patch execution, resident model reuse, conditioning and injected prompt text. Fix verified faults. Expose valid sampler and scheduler controls for Turbo models and record the actual choices. Verify math and application paths; keep visual quality claims separate from structural checks. Preserve active jobs and downloads.
97. [x] Add a reset button for each generation setting. Restore the selected model's current defaults, including sampler, schedule, sizes, steps and guidance. Restore tool and LoRA defaults in their own controls. Cover image, Swap and video paths. Verify that reset changes only the chosen setting and keeps user inputs.

Current item 96 status: source research, installed-weight checks, sampling corrections, loader corrections and live delivery are complete. The visible style comparison remains open. The prior no-QA-generation instruction is still active. A concrete four-image Standard/Turbo, LoRA-off/on comparison is prepared at `E:/AI/qwen-image-studio/validation/sampling-resets/proposed-style-comparison.json`. The user input request is pending. Do not generate these images until the user permits it. The unrelated game-quality message was withdrawn; it adds no Qwen task.

Delivery evidence for 96-97, 2026-10-20:

- Compact per-field reset controls cover image and Swap sampling, output dimensions, seed, VAE, negative text, tool options, LoRA strength, all video pages, Orbit, upscaling and training settings. Model sampling fields reset to automatic selection, so they follow the selected model. Dimensions, seed, folders and other fields without published model values use the existing workflow defaults; they are not presented as author recommendations. Training resets wait for the current model/dataset advice. A reset does not clear the prompt or inputs. Coupled controls, such as seed/random and quality/steps, return together to their valid default state.
- Turbo sampling choices now reach Qwen, Krea, Z-Image and FLUX.3 inference. Qwen keeps its trained sigma schedule when only the sampler changes. An explicit schedule replaces it. Reset Schedule restores the trained schedule. Z-Image and FLUX.3 no longer ignore explicit choices. Actual sampler/schedule metadata remains separate from requested automatic values. This does not claim every custom sampler/schedule has equal image quality.
- Diffusers PEFT metadata alpha/rank and rsLoRA scaling now reach the checked loader. Existing alpha tensors take priority. This closes a general format gap; the checked NiceGirls, Lenovo, GrainScape, Analog and Casual Snapshot files do not contain a hidden metadata scale, so that correction does not explain their reported weak style. Z-Image and FLUX.3 now also reject missing or invalid adapter targets instead of silently accepting unchecked loads. See `lora-author-audit.md` for fresh source and tensor evidence.
- 68 distinct focused Python tests pass. Checks cover actual CPU LoRA math, fused/split targets, GGUF logical shapes, stacking, metadata scale, Turbo sampler dispatch, API values, model defaults and Civitai file identity. TypeScript and production build pass. The existing bundle-size warning remains. One old FLUX catalog test lacked any file identity; its fixture now uses an exact file, as required by the earlier per-file repair. Production code did not weaken that repair.
- Local and authenticated Moise browser checks pass in Chromium 1440x1000 and iPhone WebKit 390x844 and 320x740. They verify model changes, independent sampler/schedule resets, LoRA strength, Swap negative text, all video pages, upscale audio, training advice, preserved prompts and no horizontal overflow. Four Turbo-family generation requests per browser are intercepted; no QA image is generated. No page errors occur. Physical iPhone hardware remains untested.
- The guarded launcher restarted only after a fresh idle/no-download check. Server PID 52604 serves the update. Local and remote JavaScript/CSS bytes match `index-DgyuAHVt.js` and `index-DKVchAEe.css`; the setup fingerprint matches. New user jobs arrived after delivery and continue without interruption. Their requests are not QA requests.
- One post-update user run, `804e2b60f0114625b86f94eb04e80e76`, succeeds with 224 of 224 runtime adapter parts executed and 20 model calls. First-layer samples show nonzero applied RMS changes in attention, MLP input and MLP output. These measurements verify numerical contribution in this run, not the visible style or other models. No user image was opened for this check.
- Evidence: `E:/AI/qwen-image-studio/validation/sampling-resets/{local.json,remote.json,delivery.json}`, browser screenshots, `check_sampling_resets.py` in the parent validation directory, and the fresh author audit in `validation/lora-author-pages/2026-10-20/`.

Clarification: the LoRA picker must be an anchored dropdown with a scrollable list on desktop and mobile, not a large dialog. `res_2m` and `res_2s` remain installed; all 82 pinned source hashes pass a fresh read-only check. Turbo can use LoRAs; author examples with different sampling settings do not establish that Turbo blocks style adapters.

Delivery evidence for 89-94:

- LoRA rows measure 68 px on desktop and 94 px on phones. The anchored list supports search, same-slot replacement, bypass, strength and the per-LoRA prompt editor. Replacement keeps strength and bypass but does not carry the old adapter's prompt addition. Browser checks keep ten selected LoRAs and send nine enabled choices in an intercepted request. All four request schemas accept 100 choices; reference-image and strength bounds remain.
- The full-image viewer now uses a header, image area, zoom bar and compact details strip. The prompt appears once. Expanded text and loading messages stay outside the image. Studio Image's source shows the same separation pattern. The pattern was inspected, not copied as a component.
- Chromium 1440x1000 and WebKit 390x844, 320x740 and 844x390 pass local geometry, long-prompt, zoom, navigation, settings restore and metadata-copy checks. Chromium 1440x1000 and WebKit 390x844 pass the same preview and LoRA flows through Moise. Final runs report no page, console or HTTP errors. The Browser plugin is absent; installed Python Playwright runs the checks. Physical iPhone hardware is untested. Fixture data and intercepted generation requests avoid AI inference and user data changes.
- Fifty-two affected Python tests, seven CPU LoRA math/loading tests, twelve JavaScript tests, TypeScript and production build pass. The actual sampler/schedule are saved by image/video engines and read from each result. Older unknown values stay unknown. No new style-quality A/B generation is claimed.
- The guarded restart ran with no jobs/downloads. The first remote check overlapped relay replacement (409 then temporary 503); the relay recovered. A separate browser fixture initially used local preview URLs on the remote host; matching Moise's URL prefix and ending the simulated job before reload resolved that test-only failure. Local/remote assets match `index-Ds5RCCAQ.js` and `index-Dhv0BVgx.css`. Both live schemas have no LoRA count cap. Civitai remains connected; idle release remains 600 seconds. Evidence: `E:/AI/qwen-image-studio/validation/compact-loras/`.

## Terminal closure stops Qwen (2026-10-20)

86. [x] Bind the server lifetime to its Qwen terminal tabs. Closing either tab or the terminal window must stop the server and its worker tree, including active work. Browser closure remains separate. Keep the 10-minute idle policy.
87. [x] Verify real Windows close/exit handling, worker descendants, server restart, duplicate launch and unrelated process preservation without AI inference.
88. [x] Apply the change after existing work finishes. Open the connected terminal tabs and verify the local and remote app. Correct the previous terminal instructions.

Delivery: [Terminal shutdown](terminal-shutdown.md). Seventy-four tests pass; the optional Electron test is skipped. A real Windows Terminal `WM_CLOSE` stops an isolated busy server and its CPU worker. Either tab closure, owner loss, new sessions, duplicate launch and unrelated processes pass. No AI inference or model download ran. The guarded restart ran only after jobs/downloads finished. Both production tabs now hold their current owner-session connection. Local and Moise routes pass with the unchanged frontend and 600-second idle policy. This replaces the earlier behavior where log-tab closure left Qwen running.

## Idle memory release (2026-10-20)

83. [x] Release the image worker after 600 seconds without an active or waiting job. Retain reuse before the deadline. Prevent background warm-up from loading an old model on startup or after release.
84. [x] Test the deadline, new jobs, active and waiting jobs, manual release, shutdown and process exit without model inference. Explain the separate terminal viewer and server.
85. [x] Deliver through the idle job/download guards. Verify local and remote status, memory policy and process state. Preserve other applications and saved files.

Delivery: [Idle memory release](idle-memory-release.md). Sixty CPU/API/process checks pass. The optional Electron test is skipped because its binary is absent; normal Windows server lifecycle checks pass. The controlled-clock test covers the exact 600-second deadline and real worker exit/restart without AI inference. The guarded background owner restarted only with no active/waiting jobs or downloads. Local and Moise routes report `idle_release_seconds=600`, no loaded image model and connected Civitai. The existing frontend `index-BXlQ034p.js` remains unchanged. Windows reports no image worker and about 36.5 GB free RAM at the final snapshot. Evidence: `E:/AI/qwen-image-studio/validation/idle-memory/`.

## Required sampler dependencies (2026-10-19)

79. [x] Audit the exact RES4LYF sampler code, dependencies and connected Civitai sampler settings. Pin a compatible source revision and install required runtime files without changing active jobs or downloading model weights.
80. [x] Integrate res_2m/res_2s into the actual generation path, validation, import metadata and controls. Preserve Turbo schedules and exact sampler names. Report missing prerequisites before generation instead of substituting another algorithm.
81. [x] Verify real sampler execution with small CPU tensors, model/request integration and desktop/mobile controls. Deliver with idle guards and record the precise supported scope and remaining source limitations.
82. [x] Restore Civitai sign-in after a test used the real account file. The user completed sign-in. A live GET /api/civitai/status now reports connected=true and method=account. The test isolates account and model-list writes.

- [Sampler runtime and checks](res4lyf-runtime.md): 82 verified source files, PyWavelets installed, all four RES 2m/2s variants pass CPU execution in both Comfy runtimes, and 52 focused tests pass. Setup/build and dependency checks pass. Main GPU packages stay unchanged. Desktop Chromium 1440x1000 and mobile WebKit 390x844/320x740 pass selection, exact request values and Turbo reset. Requests are intercepted; no image generation or model-weight downloads occur. The local and authenticated Moise routes serve `index-1neiYMMN.js`. Restart used idle job/download guards; the remote route briefly returned 503 while reconnecting, then passed asset and queue checks.

## Mobile image settings restore (2026-10-19)

71. [x] Add Use settings to mobile result thumbnails and the full-image viewer. Reuse the desktop restore path and return to the correct image or Swap form.
72. [x] Restore the actual seed, steps and CFG with the saved prompt, inputs, model, LoRAs and output options. Verify mobile, narrow layouts and desktop without inference; deliver the frontend without restarting active work.
73. [x] Investigate the new report that LoRAs still do not change images. Trace the latest real jobs, their exact model and adapter paths, effective requests, runtime patch application and output differences. Audit every installed LoRA's author page and exact version for triggers, strengths, settings and use limits. Do not treat accepted patch counts as proof of visible style. Preserve user runs; no QA inference.
74. [x] Add an editable negative prompt to Swap and a Realistic / Non-realistic switch. Add precise source-style instructions and negative defaults for Non-realistic mode. Preserve saved prompts, seeds, history and restore. Show the CFG 1 limitation. Verify on desktop and mobile without inference.
75. [x] Training: import complete folders and ZIP datasets; pair and preserve existing captions; automatically caption only missing images with the supplied subject name/trigger. Research primary GitHub/model training guidance for every supported architecture and adapt recommendations to dataset size and subject/style training. Verify archive limits, paths, duplicates, caption pairing, edit/retry flow, training configuration and mobile/desktop UI. Do not run QA model training or inference or download model weights.
76. [x] Add a persistent per-LoRA prompt addition editor, including multiple enabled LoRAs, bypass and zero strength, queued snapshots and result metadata. User confirms visible style effect at strength 1.5 plus related wording; do not claim universal style fidelity.
77. [x] Remove incompatible LoRAs from the active selection on architecture changes across image and video screens. Preserve installed files and compatible settings.
78. [x] Permit image-to-image with empty text; reject an empty text-only request. Verify request, metadata and desktop/mobile behavior without inference.

### LoRA and dataset follow-up delivery

- [Exact author audit](lora-author-audit.md) now follows connected sampler/model/conditioning paths in published Comfy graphs. Thirteen installed Civitai versions yield 47 recognized paths across eight versions. Five have no embedded API graph. The High Resolution graph contains additional unparsed custom branches. Six installed Qwen files match published SHA-256 hashes. Standard 50-step workflows, companion adapters, prompts, VAE and unsupported custom sampler names remain distinct from recent Turbo runs. No silent sampler alias or companion activation is used.
- Quantized Qwen/Krea LoRA application uses checked runtime branches for supported ordinary linear adapters. Each completed job records actual executed branches. CPU math and header checks pass. This does not establish visible style fidelity; no new image A/B run is claimed. Normal Qwen has no hidden source-preservation instruction. Swap and the official FLUX.3 expansion use their own paths.
- Saved prompt additions work from active LoRAs and Settings > LoRAs. Enabled additions combine with declared triggers. Bypass and zero strength skip additions. Queue snapshots preserve the text used for each job; editing a restored selection uses the new saved text on its next job. Image-to-image accepts blank text, while text-only generation rejects it. Standard-model sampler/schedule controls retain explicit supported choices.
- Image and video architecture changes remove incompatible active selections. Adding a compatible video LoRA removes incompatible entries before the four-adapter limit. A catalog refresh no longer re-adds a removed reference-video LoRA. Installed files and completed-job metadata remain unchanged.
- [Dataset and caption workflow](training-dataset-workflow.md) documents folder/ZIP imports, caption pairing, bounded upload staging, local missing-caption jobs, resume/edit behavior and architecture-specific training configuration. Recommended settings are labeled starting values, not optimum values. Caption and training quality remain unmeasured because QA inference, training and model downloads are excluded.
- Validation: 77 focused Python tests and 10 Node tests pass. TypeScript and the production build pass. The existing main-bundle size warning remains. Authenticated Railway Chromium 1440x1000 and WebKit 390x844, 320x740 and 844x390 pass prompt editor persistence, nested Settings dialogs, sampler payloads, empty image prompts, bypass, image/video architecture changes, ZIP/folder selection, preserved captions, missing-caption jobs, manual edits and custom training requests. Generation and training calls are intercepted; CPU API tests exercise real import/persistence handlers. Screenshots were inspected. No page/console/HTTP errors or horizontal overflow occur in the completed controls runs. Physical iPhone hardware is not tested.
- Swap/restore checks separately pass on desktop and all three WebKit sizes. Intentional reload cancellation messages are isolated from stable-page errors; post-reload queue requests return HTTP 200. Evidence: `E:/AI/qwen-image-studio/validation/lora-training-controls/` and `validation/swap-style/`.
- Backend delivery used the guarded launcher only after the queue and downloads were idle. Frontend: `index-BGvXIL5P.js` / `index-CxXsJpeE.css`. No active user job was stopped. The current frontend needs no further server restart. The setup fingerprint matches. Local served JavaScript matches the built SHA-256; remote JavaScript matches after the relay's documented `/qwen-studio` URL prefix rewrite. Both final queue reads show no active or pending jobs. Evidence: `validation/lora-training-controls/delivery.json`.


Mobile restore delivery: thumbnail buttons and full-image Use settings are live in `index-DdZIzOzU.js`. Normal image restore now turns Random seed off and uses the saved seed, step count and CFG. Swap restores the selected tool, both images, face mapping, strength, style/pose choices and saved prompt template. Live Railway UI checks with neutral fixtures pass in Chromium 1440 x 1000 and iPhone WebKit at 390 x 844, 320 x 740 and 844 x 390. Checks compare intercepted generation payloads, cover restoring the currently viewed image after Next, verify viewer closure and correct form, and report no console/page/HTTP errors or page overflow. Browser plugin not available; installed Playwright used. Evidence: `E:/AI/qwen-image-studio/validation/mobile-restore/`. Build, diff check and setup fingerprint pass. No server restart, inference or model download. Physical iPhone hardware is not tested.

## LoRA application and required prompt words (2026-10-19)

67. [x] Trace Qwen LoRA weight mapping, clone reuse and cache invalidation. Check applied patches instead of trusting a saved LoRA name. Use CPU checks and file headers; preserve active work.
68. [x] Add required LoRA words to the internal prompt for active selections. Keep the user's text clean, avoid duplicates and preserve the exact effective prompt with each run.
69. [x] Verify loading, strengths, bypass, repeated jobs and prompt changes with safe tests. Deliver through idle guards and verify the remote controls. State limits on untested output quality.
70. [x] Show each saved result's prompt below the image with copy and expand controls. Include the exact effective prompt in Details and copied metadata. Verify desktop, mobile and result switching without inference.

Delivery: `docs/lora-application-audit.md`. Fixed missing native/Kohya aliases for fused Qwen MLP weights. Corrected the new validator's GGUF packed-shape error after it rejected three valid user jobs. All 20 installed Qwen LoRA files pass header checks on each of four base-model paths. Sixty-three CPU tests, six Node tests and the production build pass. Remote Chromium and iPhone WebKit checks pass LoRA strength/bypass/reload, saved prompt copy/expand, Details, Gallery and 320 px width. One WebKit fetch diagnostic occurs only during document reload; post-reload queue requests return 200 and all interactions pass. Prompt tests have no page or HTTP errors. The guarded restart completed while idle; listener PID 52172 serves the corrections. No QA inference or model download ran. Eight existing user-result pairs differ with LoRAs, but intended visual style remains unmeasured. Physical iPhone hardware is not tested. Setup fingerprint matches.

## Result LoRA metadata (2026-10-19)

64. [x] Trace the saved applied LoRA stack for each image engine. Preserve its names and strengths, including tool LoRAs and known runs without LoRAs.
65. [x] Show the saved LoRAs beside each result and in Details, copied details, Gallery and the mobile full-image viewer. Keep long names readable without horizontal overflow.
66. [x] Test saved values, missing metadata and switching between results. Build and verify the live remote app without inference, downloads or a server restart.

Delivery: `docs/result-lora-metadata.md`. Saved LoRA names and strengths now appear under completed images, in Details and Copy details. The mobile full-image viewer has a Details button. Six Node checks and the production build pass. Live Railway checks pass in Chromium and iPhone WebKit for three actual saved runs, with additional mocked stack/missing-data cases. Gallery, result switching, copied details, saved JSON and 320 px layouts pass with zero page or HTTP errors. Test image bytes are synthetic; no new inference runs. The server stays at PID 37092. No restart or worker unload occurs. Physical iPhone hardware is not tested.

## Repeated image inputs (2026-10-19)

60. [x] Trace all readers of per-job input/prepared PNGs. Remove unused duplicate writes while preserving uploads, results, comparison and saved settings.
61. [x] Reuse decoded and prepared image inputs in bounded memory. Invalidate changed files and settings. Preserve the model's prompt/image conditioning cache.
62. [x] Remove verified old input duplicates from finished jobs. Keep originals and files without enough evidence for safe removal.
63. [x] Verify repeated inputs, changed inputs, cache limits, text-only jobs and status labels with CPU tests and desktop/mobile checks. Deliver while idle without running GPU inference.

Delivery: `docs/image-input-reuse.md`. New image jobs no longer write input/prepared PNG copies. Decoded and prepared images use a shared 256 MiB / 24-entry RAM cache. Eighteen CPU tests pass. Live page checks with mocked job progress pass on desktop and iPhone WebKit. Cleanup removed 25 verified duplicate inputs (43.74 MiB); all 21 original images passed unchanged SHA-256 checks. The old image worker was released only while the queue/downloads were idle. The server stays running. The next user job loads the updated worker; no QA inference or model download ran. GPU speed and physical iPhone behavior remain unmeasured.

## Image labels and result CFG (2026-10-19)

53. [x] Inspect Weboss label handling and Qwen's prompt and result paths. Keep a subject label separate from the image name and person identity.
54. [x] Add editable labels with Person as the default. Expand friendly image mentions to the selected subject in the model's notation. Preserve drafts, photo cycling, Swap templates, saved settings and video reference formats.
55. [x] Show the actual saved CFG in the result summary and details. Never substitute the current form or model defaults.
56. [x] Test compiled requests and desktop/mobile interactions without inference. Build, deliver with idle guards, and verify the remote app.
57. [x] Make reference-based video the default image-to-video entry. Keep keyframe generation explicit and preserve saved work.
58. [x] Audit supported video architectures, imported Turbo/standard versions, exact dependencies, sampling settings and internal prompts. Check future-import rules and reject unsupported combinations clearly. Use primary model sources; no inference or model downloads.
59. [x] Verify video navigation and dependency/settings contracts across current families and representative new imports. Deliver and document the supported limits.

Delivery: `docs/image-labels-and-video-defaults.md`. Reference video is the default; keyframes remain a separate mode. Image labels compile to model-native subject references. Result summaries show saved CFG. The audit covers 38 configured video models, 39 profiles and 46 exact model/support file records. Imported LTX models receive MSR V2 for references; each file retains its Dev/Distilled type and adapter strength. Seventy-two CPU tests, six Node tests, local desktop/mobile checks, and final live Railway interactions pass. Civitai mixed-file selection passes with mocked metadata. The mention menu now loads photos lazily after the remote check exposed an avatar request burst. Final remote checks have no HTTP errors or horizontal overflow. One WebKit queue request was cancelled during intentional reload; the loaded page passes. Build and setup fingerprint pass. No inference or weight download ran. Physical iPhone hardware, new output quality and unknown future formats remain outside this verification. Items 1–59 are complete within these recorded limits.

## Active prompt follow-up (2026-10-19)

42. [x] Audit the authoritative Swap prompt and author guidance. Remove unnecessary sharpening and single-face coordinates. Preserve scene style, framing, angle and integration while transferring identity. Correct strength guidance.
43. [x] Add a pencil editor with persistent per-tool prompt versions, restore, conflict protection and exact queued-job snapshots.
44. [x] Add shared image/person mention suggestions in image, Swap, video and reference prompts. Translate friendly mentions to each model's own notation. Preserve video person groups and reject missing inputs.
45. [x] Verify CPU contracts, desktop/mobile editing, remote submission formats and restoration. Build and deliver when idle. Do not run inference or download weights.
46. [x] Add visible Seed and Random seed controls in Swap. Preserve manual seeds in submitted requests and restore the original seed with saved results.

Prompt follow-up delivery: `docs/swap-prompt-audit.md`. Thirty CPU tests and three mention tests pass. Chromium and iPhone 13 WebKit checks cover one-tap person insertion, keyboard choices, every relevant prompt screen, exact queue payloads, template preview/save/restore, manual seed and reload. Narrow 320 px and landscape checks show no page overflow. A WebKit blur/click race and a zero-width seed checkbox were reproduced and fixed.

Live Railway checks use the real server for opening the editor and saving unchanged text. The saved prompt stays unchanged. Loading an actual saved Swap restores seed 387775514 with Random seed off. Both browsers report no HTTP, console or page errors. Screenshots/results: `E:/AI/qwen-image-studio/validation/swap-prompts/`. The launcher restarted only after the prior job and download finished; server PID 54260 then served the new backend. A later user job remains running. No QA inference or weight download occurred. Physical iPhone hardware and improved generated-image fidelity were not tested. The production build passes with the existing large-chunk notice; setup fingerprint matches.

## Remote result navigation (2026-10-19)

47. [x] Measure first and repeated image selection through Railway. Separate UI delay, image bytes, cache policy, relay delay and decoding.
48. [x] Make image selection immediate. Add bounded display-image loading and reuse cached completed assets. Preserve full-resolution downloads, zoom, comparison, transparency and saved results.
49. [x] Verify desktop rail, mobile viewer and gallery through Railway. Compare measured timings. Deliver with active job/download guards and record limits.

Delivery: `docs/remote-result-navigation.md`. The old middleware forced no-store, so each result click downloaded a 1.28–1.37 MB PNG again. Completed assets now use private browser caching; a 152–166 KB viewing copy follows the immediate thumbnail. The same live Railway selections show the selected image in 7–11 ms; uncached sharp copies take 488–628 ms, and revisits take 6–7 ms. Original PNGs remain available for zoom, comparison and download. Desktop Chromium and iPhone WebKit emulation pass result navigation, cache reuse after reload, gallery, comparison, original zoom and narrow layouts. Final runs report no HTTP, console or page errors. The guarded restart ran only while jobs and downloads were idle. No inference or weight download ran. Physical iPhone hardware was not tested. Items 1–49 are complete within the documented verification limits.

## Swap style follow-up (2026-10-19)

50. [x] Inspect the exact saved prompt for the user's result (seed 1451816053). Check style source, current templates and added instructions.
51. [x] Use concise positive instructions for adapting identity to the selected source style. Preserve prior prompt versions, user edits, face choices and queued snapshots.
52. [x] Verify the compiled prompt and saved version on the live app without inference or a disruptive restart. State the limit on untested visual quality.

The exact screenshot result used style source 1 and the previous 99-word prompt. The new 79-word head prompt explicitly renders the transferred identity in the selected image's visual treatment. It uses positive wording and no fixed style name. All three live Swap prompts now use it; earlier versions and job snapshots remain intact. Nineteen CPU tests, six live compiler previews and remote desktop/mobile editor checks pass. No restart or inference ran. Visual output quality is not newly tested. See the latest section in `docs/swap-prompt-audit.md`. Items 1–52 are complete within their stated limits.

Target: https://moise.up.railway.app/qwen-studio/

1. [x] Trace the live page, browser inputs, remote relay, and local upload handler.
2. [x] Make photo controls use direct native file input taps. Support HEIC/HEIF photos in code.
3. [x] Test image uploads, replacement people, and named people through Railway. Confirm local files and previews.
4. [x] Fix the oversized prompt, page scrolling, and desktop/mobile layout. Test all main screens.
5. [x] Open Chrome by default. Keep an optional app window. Keep the server alive after browser closure.
6. [x] Show persistent, non-blocking progress for active and queued jobs. Test pause, order, cancellation, completion, and failure.
7. [x] Check the source of repeated approval errors. Do not change security controls. External review errors remain outside the app.
8. [x] Build and serve all changes. Preserve the active queue and saved data. Record test limits.
9. [x] Configure Windows startup for the Qwen server and MoiseGpuServer. Inspect and preserve their real launchers and data paths. Final runtime check remains in item 18.
10. [x] Disable OpenClaw auto-start entries. Manual Gateway starts use a visible terminal. The final inventory found no running Gateway, so no running Gateway was stopped.
11. [x] Add detected face choices to image Swap. Reuse the video detector. Map each source face to a selected reference face or Keep person. Build and validate the internal prompt on the server. Test multiple faces, missing faces, stale choices, saved jobs, desktop and mobile.
12. [x] Inspect idle VRAM without unloading or changing models. Explain safe optimization limits.
13. [x] Compare the actual RTX 5090 video path with model authors, ComfyUI source, GitHub issues and first-hand Reddit reports. See gpu-performance-audit.md. No GPU memory policy changed.
14. [x] Audit video model parameters across imported versions and all video screens: resolution rules, steps, CFG, sampler, scheduler, shifts, frames, duration, and supported workflows. Keep model recommendations separate from runtime limits. Keep UI, saved drafts, API validation and actual engine values consistent. Repair installed models without redownloading their files. Live engine verification remains in item 18.
15. [x] Add Swap output settings: Auto aspect ratio at 1 MP or 2 MP, custom dimensions, and image steps/CFG. Save and restore them. Custom dimensions now reach the engine.
16. [x] Verify the stronger-expression LoRA against its author. The intended expression source is picture 1. Add separate picture 1/2 choices for head pose/expression and color/style. Save and validate them in the server prompt. Do not promise exact generative preservation.
17. [x] Add durable Civitai download recovery, guards for stop/restart, exact file identity and separate entries for additional files of one version. Show each process in its own terminal progress row. Remove successful HTTP/progress spam.
18. [x] Finish the live four-step video, final Railway controls, startup launcher runtime checks and real Swap comparisons. Record generative preservation limits instead of claiming exact output fidelity.

## Delivered follow-up queue (2026-10-18)

19. [x] Diagnose the repeated Moise HTTPStatusError reconnect loop with Railway HTTP evidence. Fix the cause and error reporting. Verify the real remote path without interrupting jobs or downloads.
20. [x] Add a dedicated MiniMax H3 360 Orbit page using the author's FL2VA model, same first/last image, exact sampling profile, automatic dependencies, queue/progress, history and mobile controls. Test a real result.
21. [x] Add FLUX.3 Image Klein as a distinct image architecture. Verify the official inference and training variants, exact auxiliary models and recommended settings. Integrate inference, LoRA training and Civitai classification/import. Test the actual supported paths.
22. [x] Build, validate and safely deliver all follow-ups through the local server and Railway page. Keep active work intact and record any actual external blockers.
23. [x] Urgent priority from the user: re-audit every video model/version, add size/aspect and duration presets across all video pages, remove misleading free-VRAM warnings and unrelated model-switch suggestions, clarify keyframes versus references, and research additional Civitai video models. Complete this before the remaining FLUX integration. Also test installed FL2VA Q4 with Orbit before offering it as a supported option.

24. [x] Highest priority: fix remote 401/502/503 recovery and stale asset 404; remove obsolete global Antigravity polling; replace mobile stacked workspace with direct Edit/Result navigation; eliminate page overflow; verify fresh and resumed iPhone WebKit sessions.

25. [x] Immediate priority: fix the blank Python terminal from the detached venv owner. Open named Windows Terminal Activity and Server tabs on normal launch, startup, and Show Logs. Reuse tabs. Verify visible text, independent lifetime, and absence of the blank owner console. Preserve active jobs and downloads.

26. [x] Current priority: replace the noisy Activity stream with a persistent stage table, per-stage elapsed times, real measurable progress bars, retained completion summary, and separate technical Server output. Audit job 0e389edd stage timing, GPU load boundaries, previews, encoding and MP4 output. Explain measured utilization limits. Do not infer performance from Task Manager spikes alone.

27. [x] Verify Turbo Orbit with installed DaSiWa v4 (four steps) and expose the verified model profile. The actual test 0e389edd completed in 122.1 seconds with 73 frames, 768 square, 24 fps, no audio. Its contact sheet shows front-side-back-side-front. Preserve the original 28-step and Q4 choices.

The receipt-app terminology message was explicitly withdrawn by the user. It is outside this queue.

Items 1–27 are delivered. The new mobile correction below supersedes the Edit/Result navigation in item 24. HF gates for FLUX Comfy and Base were accepted and verified (302/200) on 2026-10-18. Physical iPhone hardware and every future third-party model export are not claimed as tested.

## Delivered mobile correction (2026-10-18)

28. [x] Make a person tap select one photo and close the picker. Plus adds; an occupied input replaces only that slot. Open a person's photos directly and allow photo cycling in selected inputs. Preserve bulk selection in Training.
29. [x] Remove mobile Edit/Result navigation and automatic navigation after submission. Keep prompt, inputs, queue feedback and nearby results on the same page. Preserve drafts, scroll and queued requests.
30. [x] Make completed images large on mobile. Add pinch, double-tap, zoom buttons, pan, reset and image navigation. Check both the workspace and gallery.
31. [x] Verify desktop, narrow mobile, landscape and iPhone WebKit interactions locally and through Railway. Build and deliver without interrupting active work.

Design: retain Qwen's white (#ffffff), pale violet (#f7f5fc), dark ink (#211b35), muted ink (#756e87), violet (#684ce6) and line (#e1dcec) palette and existing typeface. Use a single creation page. A compact result filmstrip sits beside the prompt flow. A bottom action bar stays short. The photo picker becomes a bounded bottom sheet with large tap targets. Full-screen viewing gives the image most of the display. No new decorative cards or separate creation/result modes.

Reference review: Studio Image's MobileInputStrip uses explicit 44 px input actions and a bounded sheet. MobileCreateTab keeps recent results in the creation flow. MobileImageViewer supports pinch and panning. Qwen keeps its own model and queue layout; no source is copied. Browser plugin is not available; use installed Playwright Chromium and WebKit.

## Civitai mobile follow-up (2026-10-19)

32. [x] Replace oversized mobile Civitai cards with a compact responsive grid. Preserve filters, search and card identity.
33. [x] Reproduce and fix unstable model details on remote mobile. Check image loading, version changes, repeated taps, scrolling, close/reopen and background polling. Deliver and verify through Railway without a server restart.
34. [x] Verify Z-Image and Krea 2 image-input capabilities against official model cards and implementation. Distinguish redraw, style references and instruction-based editing. Correct misleading product copy if found. Do not download or run additional models for this audit.

Civitai follow-up evidence: `docs/civitai-mobile-qa.md`. Capability findings and primary sources: `docs/image-input-capabilities.md`. Items 1–34 are delivered within the stated test limits. The server remains PID 32972; no restart, GPU generation, installation, account change or user-data change occurred in this follow-up.

## Ref2VA settings correction (2026-10-19)

35. [x] Audit all 22 current Ref2VA catalog entries and all 31 MiniMax H3 checkpoint pages exposed by Civitai. Compare distinct profiles with publisher workflows. Do not generate or download model weights.
36. [x] Fix general per-file, per-partition, per-version setting extraction. Reject quantization tolerances, training counts, benchmark counts and optional second-pass settings. Correct the official Ref2VA 8-step LoRA shift.
37. [x] Reconcile installed metadata and reference-image/video controls. Show the effective steps, preserve deliberate overrides, and distinguish author settings from workflow defaults. Keep queued/running requests unchanged.
38. [x] Validate import and queued-request contracts without inference, plus desktop/mobile controls. Deliver only when jobs and downloads are idle. Record source evidence and unresolved publisher ambiguity.
39. [x] Add the shared People/Library drawer to every video image input: first/last frames, reference subjects/background, video-to-video people/views and Orbit. Preserve slot-specific replacement, single-tap selection/close, direct uploads and photo cycling. Test desktop and mobile without generation.
40. [x] Make Unload model clear and accessible in Settings. Keep active and queued jobs safe. Prevent idle warm-up from undoing a manual unload until the next local image request. Test the endpoint, real lightweight worker lifecycle, desktop/mobile controls and remote delivery without GPU generation or weight downloads.
41. [x] Add LoRA strength sliders and per-LoRA On/Off controls to image and video stacks. Preserve disabled choices and strength in drafts. Exclude disabled LoRAs from new requests and required downloads. Verify independent controls, re-enable, reload, mobile input and outgoing requests without inference.

### Model memory and LoRA controls delivery (2026-10-19)

- Settings > Models now always shows the local Unload model control. It shows the loaded model, loading/unloading status, empty state, success and failure. Active jobs and pending jobs disable the control; the server independently rejects release while either exists. The worker exits without deleting model files. Manual unloading suspends automatic image warm-up for this server session until the next local image starts.
- Each user LoRA has a strength slider, numeric input, On/Off switch and separate Remove button. The same control serves Images, Swap, Create video, Reference images and Video to video. Off choices retain their ID and strength in the browser draft. Only enabled, compatible choices enter generation requests or image dependency checks. Existing drafts with no enabled field stay enabled. Dedicated workflow add-ons such as Orbit remain part of their workflow.
- Fourteen CPU scheduler/worker tests pass, including an actual HTTP release request stopping a real lightweight child process and loading a new child for the next request. Three LoRA request-contract tests pass. TypeScript and the production build pass. Existing build-size and Starlette deprecation warnings remain. No model-weight downloads or GPU generation were started for these checks. Actual VRAM savings were not measured by manually unloading the user's resident model.
- Authenticated Railway browser checks pass in Chromium 1440x1000 and iPhone WebKit 390x664, plus 320x700 and 844x390 layouts. Verified sliders with pointer input, numeric changes, independent strengths, Off/re-enable, draft reload, compatibility filtering, Remove/Add, and intercepted requests for image, Swap, video, reference images and video-to-video. Unload checks include in-flight feedback, paused pending jobs, active jobs, 409 failure/retry and no model loaded. Fixture mutations never reach the real generation/download endpoints. Physical iPhone hardware is not available.
- The final Chromium run has no page, console or HTTP errors. WebKit reports one old-document model-list read as an access-control error at reload; the new document returns 200, restores the draft and passes all controls with no HTTP or console errors. A separate test using real, unmodified remote GETs passes two reloads and reports the analogous old-document queue read. This navigation-only event is retained in the QA evidence, not counted as a current-page failure. An earlier 401 came from an incomplete test video fixture calling the Moise API instead of the Qwen API; correcting the fixture removes it.
- The launcher refused two restart attempts while user jobs ran. It restarted only after its job, pending-queue and download guards passed. Server PID 51628 serves the backend fix. Remote fresh-page and reload checks see the new controls. Build: index-DNH49742.js / index-B0NgBRFu.css; setup fingerprint matches. Scripts, JSON results and screenshots: E:/AI/qwen-image-studio/validation/model-unload.
- Krea LoRA compatibility answer uses the official https://github.com/krea-ai/krea-2 and https://www.krea.ai/krea-2-open-source . Krea 2 uses Qwen auxiliary components but its own main image transformer. Krea RAW LoRAs are designed for Krea Turbo, not direct use with Qwen Image 2.1.

The prior video audit missed Ref2VA-specific errors. Items 36-38 reopen that part of item 14. The active generations observed on 2026-10-19 finished without intervention. The launcher refused to restart during active work. Later restarts passed its job, queue and download guards.

### Ref2VA and video People delivery evidence

- Audit scope: 22 current Ref2VA catalog entries, 31 Civitai H3 checkpoint pages, and 52 Ref2VA weight files across 24 versions. Fourteen versions lack an explicit step recommendation. They retain a labeled workflow default, not an invented author recommendation. No weight files were downloaded and no generation was started for this audit.
- Import rules now separate selected files, FL2VA/Ref2VA partitions, Standard/Turbo releases and exact beta numbers. Calibration tolerances, training counts and benchmark runs no longer become generation defaults. Optional Turbo LoRA settings remain separate from the standard model settings. These are general parsing rules, not a list of model IDs.
- The importer also reads bounded JSON Config attachments as data. It follows the connected sampler and model graph, ignores disconnected examples, and rejects ambiguous multi-pass or extra-LoRA graphs. It does not execute workflow nodes. The Molbal Ref2VA workflow specifies four Euler/simple steps with model shift 12/3 and a separate schedule shift 6/3. Both values now reach the saved request and the engine. The same contract applies to FL2VA imports.
- Official Ref2VA eight-step LoRA: Euler, video shift 12, audio shift 3. Source: https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51 . Standard Comfy Ref2VA uses 20 steps: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json . Full GGUF uses 25 in its supplied workflow: https://huggingface.co/Abiray/MiniMax-H3-GGUF/blob/main/minimax_ref2va_gguf_workflow.json . Publisher JSON for the four-step import: https://civitai.red/api/download/models/3243021?fileId=3125842 .
- The Eros Beta2 GGUF card recommends 20-30 steps, so 25 is the selected midpoint. Its CFG 3.5-4.5 claim conflicts with this app's native H3 BasicGuider/CFG 1 path. It is not represented as a verified CFG recommendation. Source: https://huggingface.co/Abiray/10Eros-Max-ref2va-Beta2-GGUF . The original author also calls beta3/beta4 corrupted test releases; correct imported settings do not establish model quality. Source: https://huggingface.co/TenStrip/10Eros-Max . Entries remain available. This audit does not claim inference quality or equivalence for every third-party export.
- Reference settings appear before the advanced section. Steps show the actual number, with a custom-value state and a reset to model settings. Model changes reset to that model's profile. Saved automatic requests remain automatic; deliberate step overrides survive reload. Size and duration remain tied to each supported architecture's frame and dimension rules.
- Shared People/Library selection now covers first and last frames, four reference subjects, their background, video-to-video people and individual views, plus Orbit. One tap selects and closes. Replacement and cycling affect only the selected slot. Plus opens one picker that also contains native Upload. Selecting a new first frame from the library fits its aspect ratio, like direct upload; reloading an existing draft retains its chosen size.
- Authenticated Railway checks use Chromium at 1440x1000 and iPhone 13 WebKit at 390x664. Narrow 320x700 and landscape 844x390 checks pass. Browser fixtures provide neutral people/photos and an empty queue/history, so real user data and GPU work stay untouched. Model settings and Civitai details come from the real server. Four-reference and nine-view limits, replacement at the limits, photo cycling, modal bounds, scroll restoration and custom-step reload pass. Physical iPhone hardware is not available.
- The live Civitai API confirms the attached workflow, corrected beta4 classification, rejection of the one-step tolerance, and rejection of the ten-step benchmark as a recommendation. Installed H3 metadata is version 8. Model file records match the pre-repair backup exactly. Backup and QA artifacts: E:/AI/qwen-image-studio/validation/ref2va-settings.
- Final fresh Railway runs return HTTP 200 in both browsers and pass page identity, nonblank content, no framework overlay, zero console/page errors, interaction and screenshot checks. One check immediately after the server update failed to find the app document; the later full runs pass. The final server is PID 45004. New user work after delivery stays running.
- 56 focused CPU tests pass. TypeScript, production build and whitespace checks pass. The existing Vite warning about a main chunk over 500 kB remains. Build: index--qM3YXo3.js / index-DknQpabv.css. Setup fingerprint matches. No new GPU inference or model weight download was performed.

### Mobile correction evidence

- Cause of two taps: the People card reveals its action buttons with a CSS hover rule. Touch layouts now show those actions without hover. Selection is separate from bulk toggling. Exact-slot replacement no longer removes an existing occurrence of the person in another slot.
- A shared CSS name from video face matching also fixed photo tiles at 55 by 62 px. Scoped library styles restore full photo tiles. The photo list replaces the people list in the sheet, instead of opening below it.
- Person photo controls preserve the chosen slot and explicit person identity after reload. Ambiguous shared-photo ownership is not guessed. Swap uses the same photo controls. Training retains its multi-selection mode.
- The shared API callback no longer switches the mobile view after any generation POST. The Edit/Result navigation is removed. Recent images sit below the image prompt. The viewer retains Download and Use as input, without replacing the current prompt or clearing other inputs.
- Chromium desktop 1440 by 1000 and authenticated Railway iPhone WebKit 390 by 664 pass the complete flow. Narrow 320 by 700 and landscape 844 by 390 pass width and sheet checks. Real people, photos, uploads, slot replacement and saved drafts use the running server. Generation, queue, history and result images use controlled browser responses to avoid GPU work. Two queued requests preserve their distinct prompts and inputs. No new inference or physical iPhone test is claimed.
- Full-screen viewing passes zoom buttons, fit, double-tap, pointer pinch, pan, image navigation, download links and Use as input. Chromium CDP touch events also pass pinch, pan, fit and double-tap on a real uploaded photo. A suppressed compatibility click after a pan is handled directly through pointer-up. Gallery images use the full mobile width; its old floating close button no longer shrinks them.
- Page identity, nonblank content, overlay absence, console/page errors, screenshot review and target interactions pass. One earlier remote login returned HTTP 502 before the app opened. Subsequent complete authenticated runs pass with no HTTP errors. A WebKit pan assertion initially read before the render; the check now waits for the actual transform change.
- TypeScript and production build pass. Nine session/upload regression tests pass. The ten-input limit still blocks adding and permits replacement. Whitespace checks pass. Build size warning remains: the main JS exceeds Vite's 500 kB warning threshold.
- Final build: index-BzZhQgBs.js / index-CEVQVkZb.css. Setup fingerprint matches. Server PID 32972 remains unchanged; no restart or interruption occurred. QA scripts, JSON results and screenshots are in E:/AI/qwen-image-studio/validation/mobile-correction. The live Studio Image guest mobile page on port 4157 was inspected as a reference.

## Final FLUX and shared checks

- Real FLUX.3 text-to-image and two-reference editing pass. The trained Base LoRA loads into distilled inference. A second 50-step training generates baseline and final samples. See [FLUX.3 validation](flux3-validation.md) for job IDs, exact files, timings, sources and limits.
- The full Python suite passes 247 tests. The final 45 affected tests also pass after adding a precision/naming regression case. Nine JavaScript session/upload tests, TypeScript and production build pass. Git whitespace checks pass.
- Railway Chromium and iPhone WebKit verify the FLUX picker, saved settings, Base training and live Civitai version details. Civitai local-LoRA browsing, the FLUX architecture filter, incorrect generic Base tags and INT6 metadata labels are corrected. No checkpoint-ID allowlist is used for Orbit or FLUX imports.
- Final build: index-B0RALBlC.js / index-CnUffR7r.css. The setup fingerprint matches. Qwen serves the build locally and through Railway. Final queue and downloads are idle. The image model remains loaded for reuse.
- Actual Windows Terminal text confirms the refreshed Activity/Server tabs and retained stage durations. Existing protected legacy viewers are left untouched. Scheduled-task XML still has no OpenClaw automatic trigger; Qwen and MoiseGpuServer retain login triggers. No active user work or download was stopped during these checks.

## Follow-up delivery evidence (18:15 local time)

- Moise session persistence commit b35f0ccf0ec062c5c6853cdea4bc46bd621df17c reached Railway in deployment 43cf23fa-4392-4d09-bcfd-228f705305ff. Nineteen focused tests pass. Browser cookie-only close/reopen and refresh pass. A production restart was not forced solely to test session restoration. Expected relay replacement gaps no longer masquerade as generic HTTPStatusError spam.
- Railway iPhone WebKit checks pass at 320x700, 375x812 and 844x390: no horizontal overflow, visible action/result controls, queue dialog bounds. Injected 503 startup retries and later reconnection preserve the draft. The previous first attempt overlapped the guarded restart; the repeated stable test passes.
- Actual Orbit outputs pass for INT8, Q4 and DaSiWa v4 Turbo. The four-step repeat a18b8842 completes in 116.9 seconds. See gpu-performance-audit.md for measured load/compute/decode times and limits.
- User correction: Orbit must accept every compatible present and future checkpoint. There is no checkpoint allowlist. Server capability metadata selects H3 FL2VA models compatible with the Orbit LoRA and keyframe inputs. Explicit model-author steps win; otherwise Orbit-author steps apply. The live list includes Q4, Q8, INT8 and imported DaSiWa. Railway Chromium and iPhone WebKit verify every choice, step display and saved selection after reload. The generic capability test uses a new ID and checks both standard and Turbo settings.
- Named Activity/Server tabs run separately from the hidden server owner. Real Terminal text confirms retained stage times and total time. Stage history keeps separate two-pass phases. Twelve timing/activity/preview tests and 40 API/lifecycle/video-setting tests pass; the integrated Orbit/activity/timing set also passes (15 tests).
- Backend restart used the guarded launcher only after jobs, queue and downloads were idle. No active task or download was stopped. Current frontend is served from dist through Railway; final shared fingerprint validation remains item 22.

## Current delivery evidence

- The guarded live restart succeeded after a fresh check found no active job, no queued job and no download. The setup fingerprint matches the build.
- 111 focused Python tests, seven JavaScript session tests, TypeScript and production build pass. Earlier Windows lifecycle tests also pass.
- Post-update Railway PNG/JPEG/HEIC uploads pass in Chromium and iPhone 13 WebKit. Named people, references, replacements and training inputs pass. Physical iPhone hardware is not available.
- Real CPU Swap browser checks pass: two/four-face scans, crossed choices, rescan, reload, no-face rejection, custom size, and pose/style source values in the queued request.
- Video browser checks pass: stale 20-step draft repair, actual queued four-step request, Eros defaults, LTX 8+3 and Civitai per-version settings.
- Local models/config responses are 0.080/0.123 seconds. Railway models/config responses are 0.529/0.442 seconds. The earlier remote model response took about 1.94 seconds. These are samples, not latency guarantees.
- Installed DaSiWa v4 Turbo now uses four steps, Euler/simple, CFG 1 and shifts 6/3. Eros beta 6 uses eight steps from its 8–12 range, CFG 1 and shifts 10/3. The final version/file audit resolves res_multistep/simple for Eros beta 6. Its metadata records the source.
- Actual video check: a23e968df8254dc987d13e6ae91941fe, DaSiWa v4 Turbo, 768×448, 124 frames, four steps, generated audio. This checks execution, not speed against the earlier larger run.
- The video completed in 75.4 seconds. PyAV verified 24 fps, 124 frames, a 5.167-second duration and one audio stream. No model weights were offloaded for this smaller workload.
- The actual close-up in the user's Swap screenshot was missed at its original scale. A 320-pixel fallback with the same confidence threshold detects it. The live app now detects one face in each original input. The new unit test passes.
- Final Railway Swap checks pass in Chromium and iPhone WebKit: original close-up photos, automatic one-to-one selection, pose/style source choices, 1 MP/2 MP, saved choices after reload and no horizontal overflow. A fast reload during unfinished requests produced WebKit access-control cancellation messages. After waiting for requests to finish, the same check passes with no page errors. No application authentication error was reproduced.
- Real Swap checks use the user's two inputs, seed 1768185292, alternate BFS LoRA and 1408 by 1408 output. Turbo Q8 job 625b051e46cb414d9a0245d75084c17a completed in 33.4 seconds. Standard Q8 job bee46a6e9b034479a6b630055f40aa93 completed in 113.9 seconds. Original BF16 job 30f4fb2868be4918be424da0b75ba1b5 completed in 590.7 seconds. All three still change the scene's blue face color and pose. Source order and saved prompts are correct. Do not claim that the controls guarantee exact preservation. See swap-behavior-audit.md.
- Direct mode comparison: regular Head swap on Turbo Q8, job ada3a614fa6f499eb23c74d043060eda, uses the same pictures, seed, prompt and source choices. It retains the scene's microphone; the alternative mode omits it. Neither retains the blue face color. The UI now explains the limited meaning of stronger expression. Turbo Q8 and the INT8 encoder are loaded again. The queue and all downloads are idle.
- Qwen Server and MoiseGpuServer Server each have one enabled login trigger with a 30-second delay. OpenClaw Gateway has zero automatic triggers and allows manual start. Its task calls gateway.cmd. Hidden launcher mode is 0. The PowerShell wrapper reapplies the rule after install/update and before start/restart. Backups: C:/Users/ido/.openclaw/manual-start-backup-20261018. Windows was not rebooted.
- Moise's scheduled launcher was tested. Native warning output initially stopped the wrapper. The wrapper now preserves the canonical launcher's warning handling and saves a transcript. The running port-4000 server reports moise-gpu, status ok and gpu_ready true. Existing isolated Python environments remain in use.
- The Qwen launcher reuse check leaves the existing server PID unchanged. The separate CPU-only QA server on port 47829 stopped after its checks. Production Qwen and Moise remain running.
- Final build and setup fingerprint match. The final nine Swap tool/detector tests pass. The live activity viewer shows one row for the running image job with its stage, step count and percentage, without successful HTTP or raw progress spam.

## Earlier evidence and work history

The sections below record earlier states. The current checklist and delivery evidence above supersede them.

## Latest verified state (2026-10-18)

- Current implementation item: image Swap face selection. Startup and Civitai work remain active next items.
- Live PNG/JPEG/HEIC uploads through Railway pass in Chromium and iPhone 13 WebKit emulation. Physical iPhone testing is not available.
- Desktop/mobile compact prompt, scrolling, and queue progress pass browser checks. Latest backend changes still need a safe delivery.
- The server now runs independently. Normal launch opens Chrome and a live activity terminal. Optional Electron and lifecycle tests pass.
- Both interrupted Civitai downloads resumed from their partial files and finished: 3393047 / 40,222,950,568 bytes; 3392061 / 20,967,668,616 bytes.
- New Swap tests: 42 tests pass across API, Swap, real image detection contracts and Civitai regression tests. TypeScript passes. Browser tests against a separate CPU-only server are in progress.
- Video job 8d88cd9fac494a279f552322ca849a56 completed without intervention in 177.2 seconds. DaSiWa Hybrid v4 Turbo INT8, 1344x768, 124 frames, 10 steps, CFG 1, res_multistep/simple, shifts 12/3. Model size 19.53 GiB; 18.09 GiB loaded, 1.43 GiB offloaded; inference estimate 8 GiB; reserve 3.92 GiB (2 GiB safety plus WDDM free-memory difference). This run proves the current import settings bug: the downloaded version describes 4-step Turbo, while the UI/API still permit only 10+ steps in this path.
- Timing defect: VideoProgress divides full generating-stage time (including first model placement) by completed steps. Actual later step intervals are about 6.8-7 seconds, while the screenshot estimate says 15 seconds. Fix using step interval measurements and restrict the offload warning to its relevant stage. Do not claim an exact speed penalty from the weight split alone.
- GPU read-only check: queue idle, resident turbo-gguf-q8_0 with encoder-int8, warming false. NVIDIA reports 24,268 MiB used, 7,920 MiB free, 1% GPU activity, 40.49 W. Windows reports about 21.07 GiB dedicated usage for Qwen persistent worker PID 55696. Windows per-process counters must not be summed as unique physical use.
- Existing caches preserve the model, encoder, VAE, one conditioning entry, and selected LoRAs. No GPU memory action was taken. PyTorch reserved-versus-allocated bytes are not exposed by the current worker, so a possible cache-only saving is not yet measured. Preserve loaded weights; RAM offload would need a transfer back to VRAM.

## Earlier state before delivery

- Priority correction: the first live restart checked jobs but missed two running Civitai downloads. Their partial files remain. Restore both downloads first. Do not restart while any download runs.
- Added request: fix Civitai model classification, image/video support labels, and exact per-version generation settings, including installed and downloading models. Inspect MiniMax H3 versions 3393047 and 3392061 and adjacent versions.
- Add a download guard to Stop and Restart, with tests. Add durable download recovery so interrupted downloads can resume after process failure.
- Restore the activity terminal on normal launch. Keep it independent from the server. Replace repeated HTTP success and percentage lines with separate live progress rows for each process and stage; retain errors and stage summaries.
- Measure remote click latency. Model state currently takes 1.6–2 seconds; a single-read implementation takes about 0.05 seconds in local checks. Validate full remote behavior after safe delivery.

- Build and 24 upload/API tests pass. Four relay tests and seven session tests pass.
- The old live handler rejects a real HEIC file with HTTP 400. The new handler passes HEIC and EXIF tests.
- Chromium completes real PNG/JPEG uploads through Railway, including named people, replacement people, references, and training images. Local files exist.
- The server restart command was rejected by the tool layer with `blocked by policy`. It did not run. Manual restart is pending.
- The local Codex config already sets `approval_policy = "never"` and `sandbox_mode = "danger-full-access"`. This does not establish the cause of the external rejection.
- User screenshot shows prompt flex growth and inaccessible scrolling. Remove prompt growth and verify bounded editor sizes.
- User requests Chrome tabs by default, an optional app window, and a server that does not stop when the browser closes.

## Evidence before the fix

- The Railway page serves assets from the local Qwen app through Moise's outbound relay.
- The live page and local queue answer. The queue is idle at the first check.
- Chromium and WebKit with the iPhone 13 preset open the main picker. This does not prove behavior on a physical iPhone.
- All photo selectors accept only PNG, JPEG, and WebP. The runtime has no HEIC decoder.
- Existing photo buttons call `.click()` on `display:none` file inputs.
- No Moise source changes are necessary for the current photo-upload work.
- Browser plugin is not available. Validation uses Python Playwright with installed Chromium and WebKit.
- Temporary browser scripts and screenshots: `E:\AI\qwen-image-studio\validation\mobile-upload`.

## Preservation

- Leave the existing untracked `.a5c/` directory unchanged.
- Use generated test images. Do not change user photos or existing people.
- Check the queue again before any restart. Do not stop a running job.

- The new live activity viewer opens separately. Automatic approval review rejected closing old viewer PID 37176 with only "blocked by policy". Do not retry closing it by another mechanism. The server and downloads stay running.


## Civitai file identity delivery, 2026-10-20

- Item 95 complete. Each file has exact type, precision, format, size, installed state, supported state, sampling profile, download ID, progress and cancellation. New file IDs/paths cannot collide; existing installed IDs/paths and unambiguous old partial downloads stay in place. Both checkpoints and LoRAs expose their files. Shared quantization variants retain shared author settings unless explicit file/type instructions differ. Mixed image Base/Turbo, H3 partitions and LTX dev/distilled are scoped per file.
- FinalCut exact version 3390745 exposes NVFP4 file 3280828 (4,101,174,822 bytes), FP8 3280147 (7,115,156,456), INT8 3282512 (7,183,949,080) and FP32 3282658 (28,460,530,000). All retain the author's 25-step recommendation. The user's FP32 download completed before restart. Its legacy installed ID remains civitai-model-3390745.
- 57 affected Python tests pass. Production build passes with the existing chunk-size notice. Browser interactions use intercepted install/cancel calls, not model downloads: local Chromium 1440 and WebKit 390/320; remote Chromium 1440 and WebKit 390. Tests cover independent simultaneous progress, exact payload/cancel, exact installed model use, file-specific settings, LoRA files and overflow.
- Guarded restart found no active jobs or downloads. Local and remote assets match the build (index-CcY5aBN8.js / index-CIAZXMkp.css). Real FinalCut detail APIs and cancel schemas pass on both URLs. All 31 existing user catalog IDs and paths stay intact. Auth remains connected. An initial remote 503 during relay handover cleared; later browser and API checks pass. No QA generation, training or model-weight download ran.
- Evidence: E:/AI/qwen-image-studio/validation/civitai-files/{local-ui.json,remote-ui.json,live-delivery.json,finalcut-model.json,finalcut-version.json}. Active next item: 96, unresolved LoRA style effects and Turbo sampling controls.

## Swap and model-default delivery, 2026-10-21

- Items 98-101 complete. The full scene-preserving prompt is saved for all three Swap tools. All prior versions remain. Realistic and Non-realistic share that base; Non-realistic adds its style instructions. The active radio control has a visible fill and outline.
- Use as appears in results, full-screen previews, Gallery, desktop Recent and mobile thumbnails. Ten destinations set the input and open the target immediately. Existing draft text and other inputs remain. No generation starts.
- Set as default persists each setting by model. Swap-specific controls also use the Swap type. Seed, image dimensions and sampling values are shared between Create and Swap for that model. Reset uses the saved value. Use recommended default removes the override. Restoring a result keeps its snapshot.
- 29 focused Python tests and the production build pass. Local and authenticated remote Chromium 1440 and WebKit 390/320 checks pass. Each viewport verifies ten destinations and 52 save/remove actions. The final runs report no page errors, console errors or horizontal overflow. Save writes and media requests use isolated fixtures. The real persistence module receives the browser payloads; real HTTP endpoints are also covered by isolated tests.
- The live API and three prompt versions are checked on both URLs. Assets match `index-Azi0eNI6.js` and `index-Bok462yL.css`. The setup fingerprint matches. The backend restart occurred only with no active jobs or downloads. The final frontend-only change required no second restart. The temporary Vite server was stopped.
- The first remote opening caught the relay reconnecting after restart. Later checks passed. A test-only request for a nonexistent input image initially returned 400; the final test supplies its face-data fixture and reports no error. No QA inference, training or model download ran. Item 96 remains pending the already-requested permission for the four-image quality comparison.

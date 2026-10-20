# Swap prompts and image mentions — 2026-10-19

## Positive style adaptation correction

The user's comparison shows result `864b701320ed4b1185e6741284f15808`, seed 1451816053. Its saved request uses `scene-v2`, style source 1, pose source 1 and no extra instructions. The exact request includes a general visual-style instruction, but also the negative clause `Transfer identity without importing the reference photo's setting or composition.` The screenshot still shows a photographic-looking replacement inside a drawn scene. The presence of a style instruction does not prove effective style transfer.

The new default uses positive, source-relative instructions. Its central instruction is: `Render these features in the exact visual style of <style_source>, matching its use of shape, edges, texture, color, lighting and detail.` Identity comes from image 2. Visual treatment comes from the selected style source (image 1 by default). The text imposes no named style such as cartoon, illustration or photorealism. A realistic source and a drawn source therefore use the same rule. Head prompts shrink from 99 to 79 words. Required triggers, source controls, framing, pose, identity and face assignments remain intact.

Prompt-writing rule for new templates: state the desired result directly, use positive instructions, reference the selected image's visual properties, and omit redundant or unrelated quality commands. Historical versions and user-authored text remain intact for reproducibility.

`style-v3` is the new built-in default. The exact `scene-v2` text and original author prompt remain in history. Existing custom saved text takes precedence. The live editor API saved the new text for all three Swap types, with optimistic version checks and all prior versions preserved. It now serves those saved versions without a server restart:

- Head swap: `70e2becd69164197bb0b9bf4c85b71bf`.
- Head swap, stronger expression: `4ffd59bb8d384343a8bf6e656c0c9c00`.
- Body swap: `c9382c08d1464167ba58f1a152c08d29`.

Nineteen CPU tests pass. Six actual live preview requests pass: all three tools, each with style source 1 and 2. These previews use the same compiler as generation and submit no generation job. A byte hash confirms that the existing result record remains unchanged. Authenticated Railway checks show the new text and the prior versions in the editor in Chromium and iPhone WebKit emulation. The server remains PID 7944. No model download, restart or inference ran. New visual output quality is not tested, so exact adaptation remains a model-dependent outcome.

Restoring settings from an old result intentionally restores its saved prompt snapshot. That reproducibility behavior remains unchanged; the new shared default applies when no old snapshot is selected. Evidence: `E:/AI/qwen-image-studio/validation/swap-style/`.

## Cause and change

The previous server compiler replaced the catalog prompt with a longer prompt. It described face coordinates even for one face, repeated scene instructions, and always added `High quality, sharp details, 4k`. That last instruction can conflict with a soft or stylized source. The user's observed result does not isolate a single cause; prompt wording, LoRA variant, strength, base model and seed can all affect the result.

The new default keeps the trained `head_swap:` or `body_swap:` trigger. Image 1 supplies composition, perspective, placement, scale, occlusion, body pose and surroundings. Image 2 supplies identity and the features relevant to the selected swap type. The chosen pose and style sources supply angle, expression, coloring, lighting, contrast and texture. Style is described by reference to the selected image; no style name or sharpening command is imposed. Single-face requests do not add coordinates. Multiple-face requests retain validated assignments and instructions to preserve unselected people.

The author recommends starting at strength 1.0. The author describes early MLP contributions as important for source-pose and expression transfer. This supports correcting the previous claim that increasing LoRA strength simply increases image-2 resemblance. It does not establish a monotonic blend or a universal optimum at 1.45. The UI now explains this distinction. No new quality comparison or inference was run for this change.

## Editing and persistence

The pencil opens an editor for the selected swap type. The current default and original author prompt remain available. Every save or restore creates a version in the app data directory's `swap-prompts.json`. Restoration preserves intervening versions. An expected-version check rejects stale writes from another device. File replacement is atomic.

The editable text includes pose/style references. These follow the two existing source controls. Face assignments and optional extra instructions remain visible in a full-prompt preview. The preview uses the same compiler as submission. Each queued request stores the resolved prompt, editable template, version and seed. Later edits do not alter queued jobs. Loading a saved Swap restores its template snapshot and seed. Saving that snapshot again creates a normal shared version.

Seed is now visible in Swap. Random seed can be disabled. A manual seed survives browser reload. Loading a previous result disables randomization so the displayed seed is used for the next comparison.

## Mention contracts

| Screen | Display | Model input |
|---|---|---|
| Qwen images, Swap extra instructions and Swap template | `@Image1` | `<image1>` |
| Other image architectures | `@Image1` | `Image 1` |
| Image-to-video keyframes | `@FirstFrame`, `@LastFrame` | Natural-language first/last-frame references |
| MiniMax H3 reference-to-video | `@Image1` | `<Picture 1>` |
| LTX MSR | `@Image1` | `Image 1` |
| Video-to-video | `@Person1`, `@Ref1`, `@Source1`, `@SourceVideo` | Existing group-aware `<Picture N>` and `<Video 1>` compiler |

The shared picker shows selected inputs and available People with thumbnails and names. Selecting an unselected person adds one photo to the next available input, then inserts its mention. Full inputs exclude additional people. The negative prompt only offers existing inputs. Training captions and training sample text do not describe a multi-image input request, so they retain normal text editing.

Native textarea editing preserves mobile keyboard and text-selection behavior. Arrow keys, Enter, Tab and Escape control the picker. Touch selection keeps the menu open through Safari's blur event until the click arrives. Mention chips below the field show the selected image/person. Missing image references fail validation before entering the queue.

## Primary sources checked

- [BFS Head V1/V1.1 author guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md): source order, trigger, pose/MLP behavior, 1.0 starting strength and limitations.
- [BFS Body Swap author guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1-body.md): body-reference format and scene/reference sizing. Existing sizing remains unchanged.
- [Qwen-Image-2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1): multi-image editing.
- [ComfyUI MiniMax H3 reference node](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ReferenceToVideo/en.md): native media labels.
- [Licon MSR model card](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference): natural-language `Image N` labels.

## Validation

CPU HTTP tests exercise the real validation, preview and queue request handlers with scheduling disabled. They cover stable seeds, immutable prompt snapshots, custom templates, stale writes, restore, missing images, image/negative prompts, keyframes and reference-video notation. Browser validation uses the remote page with controlled media and a separate CPU-only prompt server. No test requests enter the production generation queue.

Artifacts and browser results: `E:/AI/qwen-image-studio/validation/swap-prompts/`.
Thirty CPU tests and three mention tests pass. Chromium and iPhone 13 WebKit pass remote interaction checks, including 320 px and landscape overflow checks. Final live checks use the actual Railway page and actual prompt endpoints. Saving identical text leaves the version unchanged. Restoring an existing result loads its original seed and turns off randomization. Both browsers report zero HTTP, console and page errors in these final checks. Screenshots were visually reviewed.

The canonical launcher restarted the server only after the active job and download finished. New backend PID: 54260. Frontend bundle: `index-YcUSiqGO.js` / `index-soevMpVm.css`. Build and setup fingerprint pass. Vite retains its existing large-chunk notice. Tests retain the existing Starlette TestClient deprecation warning. No GPU generation or model download was initiated for this work. Physical iPhone hardware and new Swap output quality remain outside this validation.

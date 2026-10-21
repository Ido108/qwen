# Swap, image destinations and saved defaults: 2026-10-21

## Requested changes

1. Restore the full Swap prompt.
2. Show the selected image style.
3. Send an output image to another input workflow with Use as.
4. Save parameter values as defaults for the selected model, including Swap.

## Prompt and style

The new `scene-v4` base restores explicit head angle, rotation, gaze, expression, micro-expressions, contrast, depth, occlusion and scene blending. It retains source framing, composition, perspective, body pose, background and text. Head swaps retain source clothing and body. Body swaps transfer the selected person's clothing and proportions. The replacement identity includes hair, eye color and nose structure.

The prompt uses positive instructions. It derives visual treatment from the selected image. It does not add generic sharpness or 4K instructions. Non-realistic mode appends its detailed shape, line, surface and shading instructions to the complete base. It does not replace the base. A face assignment and extra instructions remain additive.

All three live saved prompts were updated through the versioned API. The prior saved text and original versions remain intact. Existing jobs retain their immutable prompt snapshots. User edits are still available through the pencil and version history.

The selected style uses the radio control's `aria-checked` value for its fill, border, weight and color. Previously only `aria-selected` activated the CSS.

## Image destinations

The shared native select appears under the output, in its full-screen preview, in Gallery, in the desktop Recent rail and in mobile thumbnails. It offers image editing, Swap source, Swap replacement, video reference, video background, first frame, last frame, Orbit, video replacement person and training.

The existing reuse endpoint supplies the image. The app sets the chosen input and opens the destination. It closes image previews. It preserves other draft text and inputs. Append destinations reject a full list instead of discarding existing inputs. Sending an image starts no generation or training.

## Saved model defaults

`Set as default` saves the field through `/api/model-defaults/{model_id}`. Values live in `model-defaults.json` in the configured data directory. They are separate from imported author metadata. Each model and workflow has separate values. Swap-specific choices have a scope for each Swap type. Image dimensions, seed and sampling defaults apply to that model in both Create and Swap.

Reset restores the saved value. The adjacent menu can remove that saved value and restore the recommended setting. Selecting the model applies its saved settings. Existing drafts and restored result snapshots keep their values. Coupled controls, such as seed plus random seed or output size plus dimensions, save together. Model-default writes do not modify queued requests.

Controls cover image and Swap sampling, dimensions, seed, decoder and text settings, plus exposed video, Orbit, upscale and training settings. Saving defaults requires no model load.

## Verification

- 29 Python tests pass. They cover compiled prompts in both style modes, history, API snapshots, durable defaults, model/workflow separation, coupled fields and invalid input.
- TypeScript and the production build pass. Vite still reports the existing large-bundle notice.
- Local Chromium at 1440 px and WebKit at 390/320 px pass. All ten image destinations preserve the expected inputs. Prompt drafts remain. Result restoration overrides a different model's saved settings. Style buttons visibly differ. Default removal works. No horizontal overflow, page errors or console errors occur.
- The browser tests use neutral image fixtures and isolated default files. They route save payloads through the real persistence module. Generation, training and downloads are blocked. HTTP API tests use isolated data and authentication directories.
- The launcher restart followed a fresh idle queue and download check. Local and authenticated Moise asset bytes match the build. The real default API and all three saved prompts respond on both URLs.
- Evidence: `E:/AI/qwen-image-studio/validation/swap-output-defaults/` and `E:/AI/qwen-image-studio/validation/check_swap_output_defaults.py`.

Authenticated remote Chromium 1440 px and WebKit 390/320 px interaction checks also pass. Each browser case checks all ten destinations and 52 save/remove actions. Final page and console error lists are empty. Live assets match `index-Azi0eNI6.js` and `index-Bok462yL.css`. The setup fingerprint matches. The temporary Vite test server is stopped. Physical iPhone testing and image-quality comparisons are not part of these checks. Queue item 96 still requires the pending permission for the proposed image comparison.

# Result LoRA metadata

Date: 2026-10-19

## Behavior

Each completed image with saved LoRA data shows `LoRAs (strength)` under the result. The same saved text appears in Run details and Copy details. Gallery uses the shared component. The full-image viewer has a Details button on desktop and mobile.

The image engines already record the complete applied stack in `applied_lora`. Qwen includes the tool LoRA before the user's LoRAs. Krea includes its automatic style-reference LoRA when it uses input images. Z-Image and FLUX.3 record their applied user stacks. All four save names and actual strengths after generation. This change reads that existing field. It does not alter inference, file formats or model loading.

The UI keeps names intact, including commas, parentheses and Unicode. It does not split the saved text or look up current catalog names. Form changes, catalog changes and deleted catalog entries cannot change this saved fact. Disabled selections are excluded when the request is built. A recorded null or empty stack displays `None`. Missing metadata displays `Not saved for this result` in image details. A selected request alone is not proof that a LoRA ran.

Long names wrap. The details value uses the full dialog width. Arrow keys inside Run details do not navigate the image viewer or Gallery behind it. Closing Run details returns to the same image.

## Verification

- Six Node checks pass for saved LoRA values, missing data, bypass behavior, zero strength, model-family filtering and saved CFG.
- TypeScript and production build pass. Vite retains its existing large-bundle notice.
- Remote Chromium and iPhone 13 WebKit checks read three real saved results: one LoRA at 1.15, the same LoRA at 0.5, and no applied LoRAs. The result bar, details, copied details and downloaded JSON match each run.
- Additional mocked metadata covers multiple LoRAs, a tool LoRA, long names, punctuation, Unicode, zero strength, missing data and unrelated current selections. Checks cover result switching, Gallery, nested dialogs and a 320 px mobile viewport.
- Test image bytes are synthetic. Saved metadata and the remote app assets are real. No new inference or model download runs. A physical iPhone is not tested.

Evidence: `E:/AI/qwen-image-studio/validation/lora-result-metadata/`. Browser checks save `ui.json` and screenshots. The server remains running; no worker unload or server restart is needed for this frontend change.

Final remote checks report zero page errors and zero HTTP errors. The final JavaScript asset is `index-C6dV4Jl3.js`; CSS is `index-CgWzt0ye.css`. The setup fingerprint matches. Server PID 37092 remains unchanged.

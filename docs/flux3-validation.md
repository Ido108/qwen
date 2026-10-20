# FLUX.3 delivery checks — 2026-10-18

## Model contract

- Inference: FLUX.3 Image Klein distilled, INT8 ConvRot, 16 steps, CFG 1, Euler/simple. This uses FLUX.3 flow-map target-time conditioning, not the FLUX.2 pipeline.
- Training: the separate Klein Base INT8 ConvRot checkpoint. The trainer receives explicit Base, encoder, decoder and configuration paths. It cannot select a cached distilled file by its filename.
- Shared files: Qwen3-VL 4B INT8, FLUX.2 VAE. Inference also uses the local Qwen3.5 9B W4A8 prompt model. Input images stay local. Up to ten reference images are supported; each keeps its own aspect ratio.
- Base inference defaults follow the model card: 50 steps and CFG 3.5. The supplied ComfyUI Base workflow uses CFG 4 instead. The app retains the model-card default and allows an override. Both workflows use Euler/simple and shift ln(16). Training samples use the trainer's 28-step, CFG 3.5 defaults.
- Civitai import reads the actual tensor layout to distinguish Base from distilled. Future checkpoint IDs need no allowlist. LoRAs share the Base/distilled architecture. Arbitrary files from another architecture are not compatible merely because they share a display name.

## Sources and isolated runtime

- [Official model card](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- [Official ComfyUI package and workflows](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy), revision `feef60b0aaf63a4773ace96636ebbcf43b881312`.
- [Official Base configuration](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86/config.json).
- [Official prompt instructions](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json).
- ComfyUI `6607f36b53c7b695b160ac5b52d595e9e3752a2d`, comfy-kitchen 0.2.38 and comfy-aimdo 0.5.6 install under the FLUX.3 runtime directory. Existing Qwen/video packages stay separate.
- ai-toolkit remains pinned to `dcebc7442df70e5239c68167919fd2909c615769`. A FLUX-only entry script copies quantization metadata when its loader duplicates a tied INT8 embedding into the output head. The fix preserves the weights, scales and rotation. Other training architectures use the unchanged runner.

## Real GPU checks

| Check | Job | Result |
|---|---|---|
| Text to image, 2048 square, 16 steps | `936eca7062914cf19f971fedf954700d` | 56.6 s; 12.7 s sampling; valid PNG |
| Two-reference edit, 1024 square | `8718ffd253c94109a68a2d529250e7f1` | 23.7 s; kept the teapot and used the second image's green/yellow pattern |
| Base LoRA training, 50 steps, rank 4, 512 square | `9ead379127d34f3a9d218847cc7a1c47` | 96.2 s; LoRA saved and registered |
| Trained LoRA in distilled inference | `17e0764fc35c49688037c502be2f2d0f` | 18.4 s; all LoRA keys loaded |
| Training with baseline and final samples | `1988ed863a6141729c0b05d4eeb894b7` | 163.4 s; both JPEG samples generated and inspected |
| Trigger handling in inference | `38fd8ab5be734d6c9bb193d8be5bd0b2` | 38.9 s cold; trigger preserved without turning it into a printed wordmark |
| Repeated prompt with retained weights/conditioning | `8d2a95be380c4392a39745e041d7e14f` | 2.9 s; retained model and conditioning |

The repeat timing predates the final trigger masking change; it verifies cache reuse, not a matched benchmark against the cold trigger check. Fifty-step training checks execution, output and reuse. It does not establish trained-subject quality. Reference input limits, model dependencies, future Civitai imports, misleading tags, training file selection and shared-library behavior also have unit coverage.

## Browser and delivery checks

Railway Chromium and iPhone 13 WebKit verify model selection, 16-step/CFG-1 display, saved resolution, reload persistence, Base-only training selection and page width. Live Civitai model 3003490 supplies both Base and distilled versions. Training a local LoRA exposed an older shared-library bug in Civitai search; its version list now skips local training entries. The FLUX.3 architecture filter is accepted by the API.

The live Civitai page also has inconsistent tags: its non-distilled INT6 version carries the generic distilled tag, and both six-bit files say INT8 in the metadata. The app follows the author's explicit Base/SD filename explanation and the file precision suffix. Final live API results: INT8 Base 50, INT8 SD 16, INT6 Base 50, INT6 SD 16. Downloaded tensor inspection remains authoritative for registration. Image search indexes were refreshed for the added architecture and corrected variants.

The full Python suite passes 247 tests. After the final naming/precision correction, all 45 affected tests pass, including one new regression case. Nine JavaScript session/upload tests, TypeScript and the production build pass. The live build uses `index-B0RALBlC.js` and `index-CnUffR7r.css`. The setup fingerprint matches. The queue and downloads are idle; FLUX.3 remains loaded. Scheduled-task XML confirms no OpenClaw automatic trigger; Qwen and MoiseGpuServer retain their login triggers. No Windows reboot was used as a check.

Validation scripts, job snapshots, images and browser captures live at `E:/AI/qwen-image-studio/validation/flux3`. The named Windows Terminal Activity and Server tabs run separately from the server. The final Activity text capture shows stage durations, completed steps and output path, without successful HTTP lines.

Physical iPhone testing is unavailable. Native INT8 inference, multi-image editing and Base training are verified. Other quantized Civitai files, including future GGUF exports, require a real-file check before claiming equivalent output quality. The importer and loader validate their architecture; they do not guarantee every future export.

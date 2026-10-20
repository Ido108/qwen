# Image input audit — 2026-10-19

This audit separates model capability from the mode implemented in Qwen Studio. It uses published source and the current local engines. It does not claim a new GPU quality test.

## Z-Image

- **Image-to-image exists.** Diffusers documents `ZImageImg2ImgPipeline`, including a Turbo example with an input image and a change-strength value. It also documents masked inpainting.
- The app already implements one-image redraw in `backend/zimage_engine.py`: encode image 1 with the VAE, add noise, sample with the requested denoise strength, then decode. `backend/models.py` permits one input image for Z-Image. The Images page exposes change strength.
- This redraw path uses the source image as the starting canvas. It is different from a model trained to interpret image-edit instructions while preserving the rest of the scene. Lower strength retains more source information; it does not guarantee identity or exact detail preservation.
- Tongyi's current official model table still labels **Z-Image-Edit** and **Z-Image-Omni-Base** as **To be released**. The opening description lists them as family members, which alone does not establish downloadable weights. Do not infer availability from that paragraph.
- The app's current Z-Image mode does not expose the Diffusers mask-based inpainting route. This audit does not add one.

Sources: [Diffusers image-to-image and inpainting](https://huggingface.co/docs/diffusers/api/pipelines/z_image), [Tongyi model table](https://github.com/Tongyi-MAI/Z-Image#-model-zoo).

## Krea 2

- **Image input is possible.** Krea documents style references: colors, texture, lighting and other visual qualities influence a new image.
- The app's current mode is style reference. `backend/krea_engine.py` encodes up to three input images, attaches reference latents with `index_timestep_zero`, and automatically loads its configured style-reference LoRA. It starts the target from an empty latent rather than using the source as an img2img starting canvas.
- This mode does not target exact identity retention or general instruction-based photo editing. That is a statement about the current mode, not an absolute limit of the Krea architecture.
- The community `ostris/Krea2OstrisEdit` pipeline supports image-conditioned tasks according to the LoRA used: style, editing and subject references. It requires its reference path, not a generic text-to-image LoRA call.
- The `yijunwang2/krea2-reid` adapter specifically targets identity reference with Krea 2 Turbo. Its author requires both visual-language and clean VAE reference conditioning plus reference K/V caching. The card explicitly rejects ordinary style-LoRA workflows and the automatic Hugging Face example. It is not a general photo-reconstruction/edit model.
- Neither the Ostris edit integration nor the ReID-specific conditioning/cache contract exists in the app's current engine. Importing such a LoRA as a normal style LoRA does not establish correct support. No additional weights were downloaded for this audit.

Sources: [Krea style references](https://www.krea.ai/blog/style-references-krea-2), [Ostris reference pipeline](https://huggingface.co/ostris/Krea2OstrisEdit), [ReID adapter and required workflow](https://huggingface.co/yijunwang2/krea2-reid).

## Product wording and validation

- Replaced the blanket statements “Z-Image has no edit model” and “Krea 2 … does not keep the person or edit the photo.” The UI now describes the selected mode and its practical limits.
- The app no longer promises that redraw preserves the layout, pose and colors exactly.
- 22 existing Z-Image/Krea catalog, dependency, input-count, engine-routing, denoise and Civitai tests pass with a separate test data directory. These checks do not prove generation quality.
- No engine code, GPU cache policy, downloaded model or running job changed during this audit.

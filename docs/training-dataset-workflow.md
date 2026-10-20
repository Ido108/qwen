# Training datasets and captions

The Train page accepts image files with TXT/CAPTION sidecars, complete folders, and ZIP files. Nested folder names separate matching filenames. Existing captions keep their text. Duplicate image pixels with the same caption appear once; different captions for the same image remain separate training examples. Files use the existing shared upload store. ZIPs are read without extracting paths. Staging parts are removed after success, failure during import, or cancellation. Source images remain available to the library and saved jobs.

Limits: 300 images, 1 GB per import, 32 MB per image, 64 megapixels per decoded image, 1,500 files and 8,000 characters per caption. The remote upload uses 4 MB parts with retry checks. A dataset over the image limit is rejected instead of silently truncated. Symlinks, encrypted entries, absolute paths and parent paths are rejected.

Missing captions can run as a separate queue job or before training. Existing manual/file captions are preserved. Captions use the local Qwen3.5 vision-language model already configured for FLUX.3 prompt processing, with deterministic sampling and thinking disabled. It downloads only on the user's first caption job if absent. It does not use Antigravity or send dataset photos to a cloud caption service. Each completed caption is saved immediately. The reuse key includes image bytes, caption instructions, subject/style choice, trigger, dataset name and exact model revision. The UI permits edits and a retry of missing captions. Training cards can restore their dataset and completed captions for review.

Character captions describe varying clothes, poses, actions, framing and setting while the trigger represents the identity. Style captions describe image content while the trigger represents the style. These are explicit app choices; no claim of universally best caption length or caption model is made. A human can edit every caption.

## Model configuration and starting values

Architecture configuration follows the pinned [ai-toolkit UI/model implementation](https://github.com/ostris/ai-toolkit/blob/dcebc7442df70e5239c68167919fd2909c615769/extensions_built_in/diffusion_models/ui.tsx). The toolkit [dataset reader](https://github.com/ostris/ai-toolkit/blob/dcebc7442df70e5239c68167919fd2909c615769/toolkit/dataloader_mixins.py) pairs image and caption basenames. Its [UI documentation](https://github.com/ostris/ai-toolkit/blob/dcebc7442df70e5239c68167919fd2909c615769/ui/src/docs.tsx) describes trigger injection and caption dropout. Its example configurations give a 500–4,000-step starting range; this is not a dataset-specific optimum.

| Architecture | Training path | Sample steps / CFG |
|---|---|---|
| Qwen 2.1 Standard | Standard weights, shift timestep, convrot8 | 25 / 3 |
| Qwen 2.1 Turbo | Turbo weights plus ostris training adapter | 8 / 1 |
| Krea 2 Raw | Raw weights, linear timestep | 25 / 4 |
| Krea 2 Turbo | Turbo weights plus ostris training adapter | 9 / 1 |
| Z-Image Base | Base weights, weighted timestep | 30 / 4 |
| Z-Image Turbo | Turbo weights plus ostris training adapter v2 | 9 / 1 |
| FLUX.3 Klein | Explicit Klein Base weights; never Distilled for training | 28 / 3.5 |

The app's suggested training values use image count, subject/style and actual image dimensions. Steps stay between 500 and 4,000; Turbo uses a conservative 2,000-step cap. Rank is 16 or 32. Fewer than ten images use a lower learning rate. Resolution buckets follow the median source size, up to 1024 by default. These heuristics are labeled starting values. They are not attributed to an author as optimal values. Custom settings remain available. Fixed sample seeds make training snapshots easier to compare. Turbo training adapters are separate from the learned LoRA and its inference selection.

Additional primary sources: [Krea training details](https://ostris.com/docs/ai-toolkit/models/krea2), [Qwen Turbo training adapter](https://huggingface.co/ostris/qwen_image_2_turbo_training_adapter), [Krea Turbo training adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter), [Z-Image Turbo training adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter), [FLUX.3 Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base).

Validation uses synthetic images, CPU contract tests and intercepted queue submissions. No new caption inference, LoRA training or model download is allowed for this verification. Therefore caption quality and learned-LoRA quality remain unmeasured. Earlier real FLUX training evidence remains in `flux3-validation.md` and does not validate this new caption path.

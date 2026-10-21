# Installed LoRA author audit — 2026-10-19

## Fresh verification and remaining limit, 2026-10-20

The fresh audit covers 27 installed entries: 24 public pages respond, two entries are local training outputs, and Civitai version 3395802 returns HTTP 404. Earlier evidence for that page stays available below, but it is not described as a successful current fetch. Raw responses are in `E:/AI/qwen-image-studio/validation/lora-author-pages/2026-10-20/raw.json`.

The five reported Qwen style files (NiceGirls, Lenovo, GrainScape, 2000s Analog and Casual Snapshot) each contain 192 adapter pairs: 128 attention and 64 MLP pairs. Their up/B matrices contain finite, nonzero values. Their alpha/rank scale is 1. This excludes the specific all-zero MLP-up training fault described in [ai-toolkit issue 1075](https://github.com/ostris/ai-toolkit/issues/1075) for these installed files. It does not test visual style. The two short local FLUX training checks contain some zero up matrices; they are not the reported Qwen style adapters. Evidence: `all-weight-values.json` in the parent author-audit directory.

The source comparison includes the [Viggle runtime implementation](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/comfyui/viggle_turbo.py), [Comfy fused-MLP issue 16944](https://github.com/Comfy-Org/ComfyUI/issues/16944), [Qwen 2.1 Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage21), and the [Diffusers Qwen 2.1 training guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_qwenimage21.md). Every local image architecture runs through Comfy. Diffusers-format adapter files are converted for Comfy; a separate Diffusers inference pipeline is not used.

The checked loader now reads PEFT `lora_adapter_metadata`, preserves `alpha/rank`, handles rsLoRA's `alpha/sqrt(rank)`, and gives explicit alpha tensors priority. This is a format correction for files that supply those values. The reported style files have no missing metadata scale, so this is not claimed as their root cause. Existing native/Kohya/fused/GGUF and separate runtime-branch corrections remain intact. Z-Image and FLUX.3 now use the same checked loading path.

The author descriptions retain architecture boundaries. Lenovo's Krea Turbo strength advice does not become a Qwen recommendation. The Analog author's Krea Turbo warning is not a blanket Qwen Turbo restriction. The [High Resolution author](https://civitai.red/models/2436859?modelVersionId=3396843) explicitly describes Qwen 2.1 Turbo use. The checked Analog author profile remains 0.7, CFG 3.5, res_2m/beta on Standard. No undocumented activation sentence is injected.

Confirmed application faults in this pass: the UI and request validator excluded Turbo sampling controls; Qwen Turbo always used Euler; Z-Image and FLUX.3 ignored explicit sampler/scheduler values. These paths now honor the selected values. A sampler-only Qwen Turbo override retains the trained sigma schedule. An explicit schedule replaces it; resetting restores it. These controls allow author setups to be selected but do not prove quality for arbitrary combinations.

The new runtime report samples up to 1024 values from the first output row for each adapter/operation group. It records the base, adapter branch and applied output change. Sampling happens before flattening, to avoid copying full noncontiguous fused output tensors. Reductions remain on the GPU until the final report. The measurements are narrow diagnostic samples, not a percentage of image style.

A real user run after delivery, `804e2b60f0114625b86f94eb04e80e76`, completed with 224/224 adapter parts executed and 20 model calls. At strength 0.7, its sampled applied RMS change was 0.004630 in attention, 0.002631 in the MLP input, and 0.004748 in the MLP output. The corresponding base RMS values were 1.570471, 0.212322 and 0.407195. This proves a nonzero numerical contribution in that run. It does not prove that its visible style matches the author. The user initiated the run; no QA inference or user-image viewing was used.

The remaining style comparison requires permission under the user's earlier no-test-generation instruction. Four exact 1024-square teapot requests are prepared: Standard and Turbo, each with and without 2000s Analog at 0.7, with one prompt and seed. No model download is required. The request is pending, and these images were not generated. Do not mark the visible-style issue resolved from patch counts, changed pixels or the CPU math checks.

## Runtime correction

The existing merged-weight path rounds each LoRA update into the base weight format. A CPU check on a full 4096 × 4096 FP8 attention layer from the selected RedCraft model found quantization error larger than several intended small adapter updates. The [Viggle author's implementation](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/comfyui/viggle_turbo.py) describes the same limitation and uses a separate runtime branch.

`backend/lora_runtime.py` now applies ordinary Qwen linear adapters as `Wx + scale × B(Ax)`. It supports split/fused MLPs, native/Kohya/diffusers mappings, rank scaling, stacked adapters and output slices. Fused MLP output uses the final gate/up activation because the fused kernel can bypass linear forward hooks. Adapter banks register with Comfy memory management and unload on clone cleanup. The base remains unchanged. Unsupported advanced adapter types retain the checked merged path. This is selected from actual tensor structure, not LoRA names.

Twenty installed Qwen adapters validate on each of four real header layouts: Turbo BF16, the selected RedCraft FP8, standard Q8 GGUF and Turbo Q8 GGUF. Nine installed adapters also pass actual-weight CPU calculations on the full FP8 layer. Runtime output equals the native quantized layer output plus the expected adapter branch (maximum absolute difference 0 in these checks). Unit checks cover strengths, stacking, split/fused MLP, Q8 GGUF, repeated application, exceptions and hook removal. No GPU generation ran. These checks prove numerical application, not output quality or runtime speed.

## Exact installed versions

All 24 public entries were read. The two local training outputs have no public author page. API words, explicit author declarations and exact version guidance take priority over shared multi-model descriptions. Example prompts alone do not create required words. An explicit older shared declaration can apply only with corroboration in at least two exact-version examples.

| Entry | Author source | Finding |
|---|---|---|
| Outfit Swap | [ausboss](https://huggingface.co/ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA) | Ordered person/outfit references; strength 1; editing instructions; 25 steps, CFG 1, Euler/simple, 2 MP. |
| Restore | [ausboss](https://huggingface.co/ausboss/Qwen-Image-2.1-Restore-LoRA) | Strength 1 or 0.5 according to blur; 2 MP; Base 25 or Viggle 8; restoration task. |
| Outpaint | [ausboss](https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA) | Gray canvas, exact task instruction and paste-back; 25 steps, CFG 1. |
| Multiple Angles | [akhaliq](https://huggingface.co/akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA) | `<mva>` and trained angle labels; strength 0.8–1; object-view training. |
| Object Remover | [prithivMLmods](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Object-Remover-Bbox-turbo) | Red highlighted region and removal instruction; specialized editing. |
| Character Swap | [akhaliq](https://huggingface.co/akhaliq/Qwen-Image-2.1-Turbo-Character-Swap-LoRA) | Two ordered references; 8-step Turbo; character replacement instruction. |
| BFS Head | [Qwen 2.1 guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md) | Scene 1 / identity 2; `head_swap`; strength 1. |
| BFS Head Expression | [Qwen 2.1 guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md) | Alternative MLP scaling; correct fused MLP application matters. |
| BFS Body | [Body guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1-body.md) | Scene 1 / identity 2; `body_swap`; ordered image roles. |
| Krea Style Reference | [ostris](https://huggingface.co/ostris/krea2_turbo_style_reference) | Style-reference pipeline; 1–2 images; no trigger requirement. |
| H3 360 Orbit | [pablodawson](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA) | Author profile: same first/last image, 768 square, 73 frames, 24 fps, 28 steps, strength 1, no audio. App Turbo experiments are separate evidence. |
| LTX MSR V2 | [LiconStudio](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference) | Dedicated reference slots and guide; not first-frame conditioning. |
| Image edit adapter 3395802 | [Exact Civitai version](https://civitai.red/models/2994337?modelVersionId=3395802) | Specialized editing task; no universal style change or documented trigger. |
| Krea Snapshot 3371723 | [Exact version](https://civitai.red/models/2268008?modelVersionId=3371723) | Strength 1.5 is the stated best value; 2 is the maximum. Corrected. |
| NiceGirls Qwen 3352079 | [Exact version](https://civitai.red/models/1862761?modelVersionId=3352079) | The shared description's strength belongs to Krea. The later connected-graph audit below independently confirms Qwen example strength 0.6, with a second LoRA. No declared required word. |
| Lenovo Qwen 3349565 | [Exact version](https://civitai.red/models/1662740?modelVersionId=3349565) | Shared strength 0.8 belongs to Krea. Qwen guidance describes CFG 3–4 or fallback 1. No declared Qwen 2.1 trigger. |
| Casual Snapshot 3357747 | [Exact version](https://civitai.red/models/2790570?modelVersionId=3357747) | `candid_snap`; exact examples use 50 steps/CFG 4. |
| GrainScape 3352222 | [Exact version](https://civitai.red/models/1332651?modelVersionId=3352222) | Strength 0.7, corrected from 0.85. Examples use standard-model schedules. |
| 2000s Analog 3378081 | [Exact version](https://civitai.red/models/1134895?modelVersionId=3378081) | Strength 0.7, corrected from 0.9. Explicit `v8s` declaration also appears in exact-version examples. Added. |
| Radiance 3351819 | [Exact version](https://civitai.red/models/2323765?modelVersionId=3351819) | Strength 1, Euler/simple; examples 25 steps/CFG 1. An example style word is not declared mandatory. |
| Anatomy adapter 3354330 | [Exact version](https://civitai.red/models/2960871?modelVersionId=3354330) | Author supplies no useful sampling/strength guide. No settings invented. |
| Viggle accelerator 3351025 | [Exact version](https://civitai.red/models/1063735?modelVersionId=3351025) | Installed file is the old 4-step rank-64 adapter, despite the Civitai version label. Current HF v0.3 is 6-step rank-128. Current guidance cannot safely replace this old file's schedule. |
| High Resolution 3396843 | [Exact version](https://civitai.red/models/2436859?modelVersionId=3396843) | Restoration adapter, strength 0.3–0.9, Qwen Turbo; required `high quality` was missed due to the author's `tigger` typo. Added. |
| Unlocked 3392016 | [Exact version](https://civitai.red/models/2063193?modelVersionId=3392016) | Retains explicit author activation metadata and strength 0.5. |
| FLUX3 pipeline check | Local training 9ead379127d3 | Short local validation output; not an author-trained general style adapter. Preserved. |
| FLUX3 sample check | Local training 1988ed863a61 | Short local validation output; preserved with its saved training data. |

No author's undocumented strength is labeled as an author recommendation. The app uses its ordinary starting strength where unknown. Existing selected strengths and historical jobs remain unchanged. Twelve installed metadata records now retain exact-version author profiles. Backup: `E:/AI/qwen-image-studio/validation/lora-author-pages/user-models-before-20261019-162642.json`.

The imported legacy Viggle accelerator still uses the selected base model's sampling path. Its exact old custom schedule remains unverified; the newer 6-step profile was not applied to the wrong file. Adding any accelerator to an arbitrary base does not prove equivalent distillation or author-quality output.

## Evidence and delivery

47 CPU tests pass for loading, metadata, triggers, import and Swap compilation. `numeric-results.json` and four `runtime-audit-*.json` reports are in `E:/AI/qwen-image-studio/validation/lora-author-pages/`. The guarded server restart completed with no active job or download. No QA inference, training or model download occurred. New Qwen jobs save `application: runtime` when the supported branch attaches. Visual impact remains untested on newly generated images.

Swap now has separate editable negative drafts for Realistic and Non-realistic modes. Non-realistic mode explicitly adapts identity to source-image rendering, shapes, lines, shading, texture and integration. It keeps image 1 as the style source. CFG 1 shows that negative conditioning is ignored. Model CFG is never changed silently. Saved requests and settings restore retain the mode and negative text.

## Connected author workflows, 2026-10-19 follow-up

The earlier page review missed the embedded `images[].meta.comfy.prompt` graphs. This follow-up follows their model and conditioning connections to a saved image. It excludes disabled LoRAs and disconnected nodes. It does not execute downloaded nodes. Thirteen installed Civitai versions were read. Eight contain usable API graphs, with 47 inspected sampler paths. Five expose no embedded API graph. One High Resolution example also has other generation/upscale branches; its single recognized KSampler path is not a complete reproduction profile.

| Exact version | Connected example setup |
|---|---|
| NiceGirls 3352079 | Three examples. Standard Qwen 2.1 INT8 convrot; 50 steps, CFG 3, res_2m/beta; 2 MP. All use two model-only LoRAs: NiceGirls 0.6 plus Lenovo 0.7 (two), or GrainScape 0.7 (one). |
| Lenovo 3349565 | Eight examples. Standard INT8 convrot; 50 steps; CFG 3 or 4; res_2m, res_2s or res_multistep; beta; 2 MP. Lenovo is 0.7. Six use a second adapter at 0.6; two use Lenovo alone. Some older companion filenames differ from the currently installed NiceGirls v2 file. |
| GrainScape 3352222 | Five examples. Standard INT8 convrot; 50 steps; beta; 2 MP. Four use CFG 3/res_2m; one uses CFG 4/res_multistep. GrainScape is 0.7. Three also use NiceGirls (0.6 or 1.0); two use GrainScape alone. |
| 2000s Analog 3378081 | Six examples. Standard INT8 convrot; 50 steps, CFG 3.5, res_2m/beta, 2 MP. Adapter strength 0.7. Two also use NiceGirls 0.6. Graph filename is `hi8_v2_qwen21_000003000.safetensors`; published download is `analogcore_qwen21.safetensors`. The graph alone does not prove those two files are identical. |
| Casual Snapshot 3357747 | Ten examples. Standard Qwen 2.1 BF16; 50 steps, CFG 4, res_2s/beta, 1 MP. One LoRA at 1.0, through rgthree's loader. `candid_snap` occurs in both declared words and examples. Graphs use several filename variants. |
| Editing adapter 3395802 | Four examples, each with one input image. Two use Turbo, 8 steps, CFG 1, Euler/simple and V3 strength 0.8. Two use standard BF16, 40 steps, CFG 3 and older checkpoint filenames at 1.0. These are different configurations, not one universal default. |
| Lenovo Krea 3075606 | Ten examples. Raw INT8 mixed; CFG 4; 30 or 50 steps; custom Clownshark res_2s with beta and rescaled sigmas. Seven also use NiceGirls 0.8. Lenovo is 0.8 in nine examples and 1.0 in one. The Turbo acceleration adapter is disabled in these graphs. |

The first five Qwen style workflows are text-to-image. They have no reference images and no preservation instruction. Their negative prompt is empty. The standard Qwen VAE is selected. The app's automatic texture-fix VAE is another possible difference. Their positive prompts include concrete visual style details; only explicit author-declared words become automatic activation words. Two Lenovo Qwen and three Lenovo Krea summaries disagree with the attached graph's positive text. Both sources remain in the audit evidence; neither mismatch is silently treated as a mandatory trigger.

The six checked installed Qwen LoRA files match the SHA-256 of their exact published download: Lenovo, NiceGirls, GrainScape, Analog, Casual Snapshot and Editing V3. All their adapter keys target the image model. Casual Snapshot connects the rgthree loader's CLIP output, but its file has no text-encoder adapter weights. A missing text-encoder LoRA connection therefore does not explain the reported style issue for these files.

The app's standard Qwen path also applies model-only adapters. Current runtime branches keep low-strength changes outside quantized base weights. Each job now records which runtime branches actually executed. This checks execution, not style fidelity. No hidden instruction to preserve the source is added in normal Qwen generation; the Qwen 2.1 encoder's template says to comprehend and analyze the prompt. Swap has its own source-preservation instructions. FLUX.3 uses its official prompt expansion path, which is separate.

The recent 6–10-step Turbo runs do not reproduce these 50-step standard-model examples. This is a supported difference, not proof that Turbo cannot use the LoRA. The user reports some style effect at strength 1.5 with related wording. No new visual A/B test was run. No model weights were downloaded.

At the initial audit, the installed Comfy runtime only supplied `res_multistep`. The follow-up now installs and integrates the original `res_2m`/`res_2s` algorithms and their ODE variants; see [sampler runtime and checks](res4lyf-runtime.md). It retains exact published names. It does not rename an alternative, silently switch to 50 steps, or enable companion LoRAs. Full author-image reproduction is not claimed: custom Clownshark options, independently rescaled sigmas, unspecified source revisions, and older example LoRA files can still differ.

Raw exact-version responses and connected graphs: `E:/AI/qwen-image-studio/validation/lora-author-pages/version-*.json` and `connected-workflows.json`. The graph parser applies to future imports, without a model-ID allowlist. Sources: the exact Civitai links above; [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF); [Viggle runtime implementation](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/comfyui/viggle_turbo.py).

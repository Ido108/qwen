# LoRA dataset captioning (research, 2026-10-21)

What the app does with training captions, and why. The caption instructions live in
`reference/caption-prompts.json` and can be edited in Settings > Training captions.

## 1. Caption style for LLM-encoder models

Qwen-Image-2.1, Krea 2, Z-Image, FLUX.3 Klein and MOSS Image all read the prompt with an LLM text
encoder. They are trained on natural sentences, so captions are natural sentences, not tag lists.

- Prose for Z-Image, Krea and FLUX models; tags only for booru-style SDXL checkpoints
  ([perfectgf/lora-dataset-studio DATASET_GUIDE](https://github.com/perfectgf/lora-dataset-studio/blob/v1/docs/DATASET_GUIDE.md)).
- MOSS Serenity (the published MOSS people finetunes) uses one natural sentence per image:
  who, expression, framing, pose, clothing, background, light
  ([malcolmrey/moss-serenity docs/training.md](https://huggingface.co/malcolmrey/moss-serenity/blob/main/docs/training.md)).
- Krea 2 cuts the caption at 512 tokens, so very long captions lose their end
  ([ostris AI Toolkit Krea 2 docs](https://ostris.com/docs/ai-toolkit/models/krea2)).

App change: the caption length target went from 60 to 120 words to 40 to 90 words. MOSS keeps its one-sentence form.

## 2. Trigger word and class word

- Use a short invented trigger word, never a real name
  ([chengyansen-ai/krea2-lora-training](https://github.com/chengyansen-ai/krea2-lora-training)).
- Start every caption with the trigger word
  ([DATASET_GUIDE](https://github.com/perfectgf/lora-dataset-studio/blob/v1/docs/DATASET_GUIDE.md)).
- Put the class word after it: "photo of feliciaday, a woman with ..." (Serenity), "[trigger], single person, ..." (krea2-lora-training).
  The class word is also what Differential Output Preservation uses (class "person", "man" or "woman").
- ai-toolkit adds the trigger word to a caption that lacks it, and replaces `[trigger]` in captions
  ([ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)).

App change: character captions now start "<trigger>, a man ..." or "<trigger>, a woman ...". A quoted trigger ("idoperson") is now written bare.

## 3. What to describe and what to leave out (character LoRA)

The rule all sources share: what the caption describes stays changeable; what it never describes is
absorbed by the trigger word.

- Describe only what changes per image, so the face is not learned as a variable
  ([krea2-lora-training](https://github.com/chengyansen-ai/krea2-lora-training)).
- Leave out face, eyes, skin, age and ethnicity; describe scene, outfit, pose, light, framing and camera angle; vary captions across images
  ([DATASET_GUIDE](https://github.com/perfectgf/lora-dataset-studio/blob/v1/docs/DATASET_GUIDE.md)).
- Caption glasses, hats and a hairstyle you want to change later; leave fixed traits such as eye color out
  ([How to train a LoRA, Ostris AI Toolkit](https://medium.com/diffusion-doodles/how-to-train-a-lora-ostris-ai-toolkit-44216331056e)).
- Serenity captions include the hair ("a woman with reddish-brown wavy hair").

App change: the character instruction now asks for hair, expression, pose, clothes, accessories, framing,
camera angle, background and light, and forbids face shape, eyes, nose, skin tone and body build.

Style LoRA: keep the style itself out of the caption and describe the content, so the trigger word carries the style
(unchanged, it already did this).

Sources disagree on one point: one Qwen-Image tutorial uses the trigger word alone with no description
([FurkanGozukara wiki](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/Qwen-Image-Models-Training-0-to-Hero-Level-Tutorial-LoRA-and-Fine-Tuning-Base-and-Edit-Model)),
while most guides use descriptive captions. The app keeps descriptive captions: without them the clothes and
backgrounds of the photos bind to the trigger word and cannot be changed in prompts.

## 4. Caption dropout

0.05 in the ostris Krea 2 example and in krea2-lora-training ("robustness without losing concept").
The app already uses 0.05.

## 5. Caption model

| Model | Fit |
| --- | --- |
| Qwen-Image PE I2I, Heretic Q8_0 (Qwen3.5-VL 9B, uncensored), llama.cpp | Chosen. Already installed for Enhance. About 2 s per caption on the RTX 5090, 15 s for 3 photos with loading. Follows the instruction. Refuses nothing. |
| FLUX.3 prompt model in ComfyUI (previous captioner) | About 10 s per caption (ComfyUI TextGenerate is about 11 tokens/s against about 120 in llama.cpp). |
| JoyCaption Beta One (8B, Llama 3.1 + SigLIP2) | Built for diffusion training captions, equal SFW and NSFW coverage ([model card](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)). Another 16 GB download and a transformers runtime; not needed while Qwen PE works. |
| Qwen3-VL | Krea training guides use it ([krea2-lora-training](https://github.com/chengyansen-ai/krea2-lora-training)); a 2B build is the default of [lora-captioner](https://github.com/jmpijll/lora-captioner). The app's Qwen PE is a 9B model of the same family. |

## 6. Dataset rules (Train page "Dataset check")

| Check | Rule | Source |
| --- | --- | --- |
| Count | 15 to 40 varied photos of a person (about 20 to 25 is enough) | krea2-lora-training, DATASET_GUIDE |
| Framing | About 40 % close-up, 30 % half-body, 30 % full-body; only close-ups fail on full-body prompts | krea2-lora-training, DATASET_GUIDE |
| Size | Train at 512, 768 and 1024 buckets; photos under 768 px on the short side lack detail | ostris Krea 2 docs, [earngenix](https://www.earngenix.com/tutorials/fix-lora-dataset-before-training) |
| Blur | No motion blur or heavy compression | DATASET_GUIDE |
| Near copies | Keep one of each near-duplicate pair; bursts count as one photo | DATASET_GUIDE, [creatorlinkstudio](https://www.creatorlinkstudio.com/blog/lora-training-dataset-tips) |

How the app measures them (`backend/training_profiles.py`):
- Blur: the Laplacian variance of the sharpest 64 px tile of a 512 px gray copy. A photo is flagged when it is below 35 % of the dataset median, so a portrait with a soft background is not flagged.
- Near copies: a 64-bit difference hash; 4 or fewer different bits count as a near copy.
- Framing cannot be measured without a face detector, so the page states the mix as advice.

## Not done

- Automatic face crop and framing classification: needs a face detector model; the dataset check advises instead.
- JoyCaption as a second captioner: not added, see section 5.

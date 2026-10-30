# Ref2VA face swap and GPU use: 2026-10-30

## GPU and memory, user job `cb316e9c`

Turbo INT8 (PulpCut), FaceSwap LoRA, 576x1024, 362 frames (15 s), 8 steps.

- Weights 19.53 GB; ComfyUI working-memory estimate 22.35 GB. 4.3 GB of weights stay on the GPU, 15.3 GB in RAM. Measured peak allocation 23.4 GB.
- Sampling: 44.6 to 46 s per step, GPU 87 to 100% busy at 586 to 598 W of 600 W.
- RAM to GPU copy speed (unpinned, the app's path): 12 to 14 GB/s. Moving 15.3 GB per step costs at most 1.2 s of 44.6 s (2.6%).
- Async offload plus pinned memory (benchmark with the same job): "Pin error" from Windows, then CUDA out of memory in `get_cast_buffer` at the first step. The engines keep `disable_async_offload` and `disable_pinned_memory`. This closes master task S6.
- `fast_disk=False paths=[]` has no effect here: `fast_disk` is read only by `ModelPatcherDynamic` (dynamic VRAM), which the engines turn off, and `read_weights` drops the file link on purpose.
- Triton: every comfy-kitchen 0.2.35 Triton kernel also exists in its CUDA backend, which has priority (cuda, triton, eager). Nothing changes for H3 without Triton.
- Slow first reference-image encode (13 to 18 s): the encode itself is 1.6 s warm. The rest is a cold read of model files while the 20 GB model prefetch runs, plus 3.1 s of per-frame CPU lanczos for 362 frames.
- Step time grows with the square of the length: 107 frames 5.8 s per step, 362 frames 44.6 s per step.
- The singer-studio ACE-Step worker can hold 24 to 29 GB of VRAM while idle. A Qwen video job started then gets about 2 GB.

## Why the swap failed

User job `cb316e9c`: a six-shot source (cuts at 4.875, 8.167, 10.5, 12.708, 14.375 s). The result had other cuts (6.5, 8.9, 11.4 s), the shots in another order, the reference photo as a shot of its own, and an unswapped first second.

The FaceSwap LoRA author's native workflow (`UntMods/FaceSwap_MiniMaxH3_REF2VA`, `Workflow_Native_FaceSwap.zip`): one continuous 5.9 s clip, prompt `Faceswap` only, a tight face crop as reference, official Ref2VA INT8 plus the Ref2V 8-step adapter, res_multistep, 8 steps, CFG 1.

Tests with a made-up face (bench script with the real engine, seed 1589646294):

| Test | Source | Prompt | Result |
|---|---|---|---|
| C | one shot, 107 frames | app text (about 200 words) | swapped for 2.5 s, then the source face returns |
| D | one shot, 107 frames | `Faceswap` | swapped for the whole shot |
| B | six shots, 362 frames | `Faceswap` | shots reordered; no reference shot inserted |

App job `7bb19d56` (shots split, `Faceswap`, People photos): cuts and kept shots exact; shot 2 swapped; shot 1 replaced by the first reference photo, a full scene on a sofa.

## Changes

- `backend/reference_shots.py`: cuts from frame change (a change of at least 0.1 and three times the three frames before it; matches ffmpeg on the user's source, no cut on two continuous clips with fast hand motion). Shots with a face (app face detector) are swapped alone, padded to the 17k+5 grid and trimmed back; other shots keep the source frames, resized to the output size. Audio follows the same cuts.
- `backend/reference_prompt.py`: one source person and one reference identity send `Faceswap` plus the user's own text. Several people, Keep choices or unused reference pictures keep the assignment text.
- `backend/reference_people.py` `crop_to_face`: references for a swap are cropped to head and shoulders, framed like the author's reference.
- `src/VideoProgress.tsx`: a "Shot 1 of 2" badge, like the upscaler's clip badge.

App job `19bbe1da` (all three changes, same request as `7bb19d56`): every frame of both shots swapped, pose, shirt, microphone and room kept, cuts exact, 181 s against about 8 minutes for the original job.

Not covered: CrossView's camera pass still runs on the whole clip. The multi-person prompt path is unchanged and untested on a real source. The model comparison with the author's base model plus the Ref2V adapter was not run.

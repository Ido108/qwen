# Swap source and output checks

Date: 2026-10-18.

## Meaning of the modes

The [BFS author guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md) defines picture 1 as the scene and picture 2 as the replacement identity. The alternative V1.1 LoRA is intended to transfer the expression from picture 1 more strongly. The author reports a small loss of head-angle accuracy on a limited set of examples. This is not a guarantee for each input.

The app uses the correct input order. The worker saves picture 1 as `input-1.png` and passes it to the encoder as `image_1`. Picture 2 follows the same path as `image_2`. The LoRA identifiers point to the regular and alternative files respectively. The word "stronger" does not mean stronger preservation of every scene detail.

## Delivered controls

- Detect faces in both pictures. Map each scene face to a replacement face or Keep person. A one-face pair selects automatically. The server validates picture revisions and the selected face IDs before it queues the job.
- Select picture 1 or picture 2 separately for head pose/expression and color/style. Both default to picture 1. Identity still comes from the selected face in picture 2. The server builds the prompt from these choices.
- Select Auto aspect ratio at 1 MP or 2 MP, or custom dimensions. The result uses the requested dimensions. Steps and CFG remain available under Advanced image settings.
- Save and restore the input pictures, assignments, source choices, output size and extra instructions.
- Explain the Standard/Turbo difference. Remove unconditional claims that the scene stays unchanged.

## Actual input and browser checks

The user's close-up scene was not detected at its original 500 by 500 scale. A second pass at 320 pixels detects the large cropped face at the same confidence threshold. The live local and Railway endpoints now find one face in each original input. The detector test verifies the normalized box after the scale change.

Chromium and iPhone WebKit checks pass through Railway. They cover these original inputs, face selection, source choices, output size, reload persistence and horizontal overflow. Separate CPU browser tests cover two scene faces and four replacement faces, crossed assignments, rescan, stale choices, and a no-face input. A physical iPhone was not tested.

## Real generation checks

The tests use the original two inputs, seed 1768185292, alternative BFS LoRA at strength 1, picture 1 for both preservation choices and a 1408 by 1408 output. The tests keep the same generated prompt.

| Model | Job | Steps | Time | Observed result |
| --- | --- | --- | --- | --- |
| Turbo Q8 | `625b051e46cb414d9a0245d75084c17a` | 8 | 33.4 s | Replaces the identity but changes the blue face color and angle. |
| Standard Q8 variant | `bee46a6e9b034479a6b630055f40aa93` | 25 | 113.9 s | Also changes the blue face color and angle. |
| Original BF16 | `30f4fb2868be4918be424da0b75ba1b5` | 25 | 590.7 s | Also changes the blue face color and angle. |

A direct mode comparison then used regular Head swap with the same Turbo Q8 model, seed, input pictures, source choices and generated prompt. Job `ada3a614fa6f499eb23c74d043060eda` completed eight steps in 55.1 seconds, including the change back from the BF16 worker. Regular Head swap retained the scene's microphone; the alternative-mode result omitted it. Neither retained the original blue face color. This supports the user's concern for this pair, but does not establish a universal ranking of the two LoRAs. The label now distinguishes expression transfer from overall scene preservation.

The Standard Q8 file is the installed abenzerps variant. The original BF16 test also uses the BF16 encoder, while the Q8 tests use the INT8 encoder. All tests use the app's texture-fix VAE. These are comparisons of installed configurations, not isolated precision benchmarks or exact reproductions of the author's complete workflow.

The BF16 run reached about 30.4 GiB of dedicated GPU memory and took about 20 to 22 seconds per later step. It produced no clear preservation benefit for this input. Do not recommend BF16 as a proven fix for this case.

The source choices are functional prompt controls. They are not a pixel mask or a deterministic pose lock. These tests do not support a claim that Turbo alone caused the drift, or that selecting Standard guarantees the requested style. No input or existing result was replaced.

Final state: the queue is idle and Turbo Q8 with the INT8 encoder is loaded again. The application does not require a new model download for the next Turbo image.

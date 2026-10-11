<div align="center">

<img src="public/branding/icon-128.png" width="96" alt="Qwen icon">

# Qwen - Images & Video

**Create and edit images and videos on your own computer.**
One double-click to install. No API keys, no subscription, no cloud required.

<img src=".github/assets/images.png" alt="Qwen creating an image from a prompt" width="100%">

<img src=".github/assets/samples.jpg" alt="Three images made with Qwen: a watercolor fox, a perfume bottle, a cafe sign" width="100%">
<sub>Made locally with the Q8_0 model, prompt enhancement off.</sub>

</div>

## What it does

- **Create images** from a prompt with Qwen-Image-2.1. It writes clear text inside images, too.
- **Edit images**: add up to 10 images, then describe the change.
- **Turbo models**: the official 8-step Qwen-Image-2.1 Turbo, about 3 times faster.
- **Tools**: outfit swap, photo restore, expand image, new camera angle, remove object and character swap.
- **Create videos** from a prompt or from a first and last frame.
- **Change videos**: replace people, change the camera angle, or build a new scene from reference images.
- **Upscale videos** to a higher resolution.
- **Queue, recent results and gallery**: add many jobs, keep working, and find every result with its settings later.

Everything runs on your NVIDIA graphics card. Your images and videos stay on your computer.

## Requirements

- **Windows** 10 or 11
- **NVIDIA graphics card** with a current driver (tested on an RTX 4090 and an RTX 5090)
- **40 GB of free disk space** for images, more for video models

## Install

1. Download the app:
   ```
   git clone https://github.com/Ido108/qwen.git
   ```
   No Git? Click **Code → Download ZIP** on this page and unzip it.
2. Double-click **`Start Qwen.bat`**.
3. Choose where Qwen keeps its files. Press **Enter** for drive C, or type another drive letter, for example `D`.

The first start installs everything automatically and takes about **20 minutes**. Most of this time is the download of the graphics libraries (several GB). You do not need to install Python or anything else, and you do not need administrator rights.

When the installation is done, Qwen opens in its own window. To open it later, double-click **`Start Qwen.bat`** again, or the **Qwen** shortcut next to the app folder. Later starts take a few seconds.

## Create your first image

1. Write a prompt, for example: `A small coffee shop with a wooden sign that reads "QWEN CAFE"`.
2. Turn **Enhance prompt** off, or connect a Google account to use it (see below).
3. Click **Generate image**.

The first image also downloads the model (about 18 GB). The progress shows in the **Recent** panel and in **Queue**. After that, a 1K image takes about 20 seconds with a standard model and about 8 seconds with Turbo (RTX 5090, model already loaded).

To **edit** an image, add it to **Input images** and describe the change, for example `Make it a rainy night with neon reflections`. Ways to add an image:

- Drag it in, or click **Select files**.
- Press **Ctrl + V** anywhere on the page to paste a copied image.
- Click **Recent** to pick an image you used before, or one of your results.
- Click **Use as input** on a result in the **Recent** panel, or **Use for editing** under the result.

Turn on **Compare** above the result to slide between the input and the result.

> **Tip:** For a transparent background, open **How to use** (the **?** icon) and click **Use transparent image prompt**.

## Model, tool, size and steps

All image settings are in one row under the prompt.

<img src=".github/assets/tools.png" alt="Model, Tool, Output size and Steps under the prompt" width="100%">

**Model.** Choose a standard model or a Turbo model. A model that is not downloaded yet shows its size; it downloads when you generate.

| Image model | Size | Steps | Choose it when |
| --- | --- | --- | --- |
| **Standard Q8_0** | 7.6 GB | 25 | You want the best quality. Recommended. |
| Standard Q6_K / Q5_K_M | 5.9 / 5.2 GB | 25 | You want a smaller download. |
| Standard Q4_K_M / Q4_0 | 4.6 / 4.1 GB | 25 | Your graphics card has less memory. |
| Standard BF16 | 14.2 GB | 25 | You want the original full-precision model. |
| **Turbo Q8_0** | 7.6 GB | 8 | You want speed. Recommended Turbo. |
| Turbo Q6_K / Q5_K_M / Q4_K_M | 5.9 / 4.9 / 4.1 GB | 8 | You want a smaller Turbo. |
| Turbo BF16 | 14.2 GB | 8 | You want the original full-precision Turbo. |

All image models share one text encoder and VAE (about 10 GB, downloaded once). Turbo always uses exactly 8 steps with the schedule it was trained on, so its **Steps** field is locked. Turbo does not use the negative prompt.

**Tool.** A tool adds a small add-on model (LoRA, 80–320 MB) and writes the right prompt for you. The input chips show which picture goes where.

| Tool | Inputs | What it does |
| --- | --- | --- |
| **Outfit swap** | 1 person, 2 outfit | Dresses the person in the outfit. Face, pose and background stay the same. |
| **Restore photo** | 1 photo | Removes blur, noise, compression artifacts or old-print damage, and enlarges it. |
| **Expand image** | 1 image | Extends the picture to the sides you choose. The original part stays unchanged. |
| **New camera angle** | 1 subject | Shows the subject from another side or height. |
| **Remove object** (experimental) | 1 image | Click **Mark object**, draw a box around the object, then generate. |
| **Character swap** (experimental, Turbo only) | 1 scene, 2 character | Puts the character from picture 2 into picture 1. Results vary. |

Every tool has an **Effect strength** slider. Prompt enhancement turns off while a tool is selected, because each tool needs its own exact wording.

**Output size.** One list: **Auto 1K**, **Auto 2K**, the fixed sizes, and **Custom…** for any width and height.

**Advanced settings** has the seed, the negative prompt, the output folder, and **Image decoder (VAE)**. Choose **Texture fix** there for fewer checkerboard artifacts (0.7 GB download).

## Settings

Open **Settings** (the gear icon). It has three tabs:

- **Models**: download or remove any model, tool or video model. Removing a model does not delete your images or videos.
- **Prompt enhancement**: connect your Google account.
- **Processing**: run on this computer or on a RunPod cloud GPU.

<img src=".github/assets/models.png" alt="Image models in Settings" width="70%">

## Prompt enhancement (optional)

Prompt enhancement rewrites your short prompt into a detailed one, which usually gives better images. It uses Gemini through your Google account, so you do not need an API key.

1. Open **Settings**.
2. Click **Connect Google Account** and sign in.

When enhancement is on, your prompt and input images go to Google. When it is off, nothing leaves your computer.

## Video

Open the **Video** tab. It has four workflows:

<table>
<tr>
<td width="50%"><b>Create video</b>: write a prompt, or add a first and last frame. Choose the duration and size, then click <b>Generate video</b>.</td>
<td width="50%"><b>Video to video</b>: replace people in a clip, change the camera view, or edit it with reference images.</td>
</tr>
<tr>
<td><img src=".github/assets/video.png" alt="Create video"></td>
<td><img src=".github/assets/video-to-video.png" alt="Video to video"></td>
</tr>
<tr>
<td><b>Reference images</b>: create a new scene from up to four photos of people or objects and an optional background.</td>
<td><b>Upscale</b>: increase the resolution of a video and keep its sound.</td>
</tr>
</table>

Every video result has **Send to**, which opens it in another workflow. For example, create a video and then upscale it.

Video models are large (4–20 GB each). Each one downloads the first time you use it.

> **Reference images** uses LTX 2.5. Its files require access to the [Lightricks model](https://huggingface.co/Lightricks/LTX-2.5) on Hugging Face. Accept the terms there, then set the `HF_TOKEN` environment variable to a token of that account.

## Queue and gallery

<img src=".github/assets/gallery.png" alt="Gallery" width="100%">

- Click **Generate** as many times as you want. Each click adds a job to the **Queue**, and you can keep working.
- In **Queue**, you can reorder, pause or cancel jobs. Waiting jobs stay saved when you close the app.
- **Recent** (the panel on the right of the Images page) shows your latest results and the running jobs. Hover a result to use it as input or to load its settings. Click **>>** to hide the panel.
- **Gallery** shows all your images and videos. Open one, then press **←** and **→** to browse. Turn on **Compare with input** to slide between the input and the result. **Use settings** loads the prompt and settings of a result. **Run again** makes a new version.

## Other tools

These are in the `scripts` folder:

| File | What it does |
| --- | --- |
| `Restart Qwen.bat` | Closes Qwen and opens it again. |
| `Show Qwen Logs.bat` | Shows what Qwen is doing. Use it when something goes wrong. |
| `Download Models.bat` | Downloads the recommended image models without opening the app. |

To close Qwen, close its window.

## Troubleshooting

| Problem | Solution |
| --- | --- |
| The installation stopped | Double-click `Start Qwen.bat` again. It continues from where it stopped. |
| "NVIDIA CUDA is unavailable" | Update your NVIDIA driver, restart Windows, then start Qwen again. |
| Generation is very slow | Close other programs that use the graphics card, or choose a smaller model (Q4_K_M). |
| Out of memory | Choose **Auto 1K** or a smaller model. |
| "Connect the drive containing ..." | Connect the drive you chose during installation. |

## Uninstall

1. Delete the app folder.
2. Delete the data folder that you chose during installation (by default `C:\AI\qwen-image-studio`).

## Mac

On a Mac, open `quickStart.command`. Generation runs on a RunPod cloud GPU instead of your computer. See [RUNPOD.md](RUNPOD.md).

## License

The Qwen models and code are under the [Qwen Research License](reference/Qwen-Research-LICENSE.txt). Other models have their own licenses on their Hugging Face pages. Third-party code licenses are in the `vendor` folder.

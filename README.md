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
- **Create videos** from a prompt or from a first and last frame.
- **Change videos**: replace people, change the camera angle, or build a new scene from reference images.
- **Upscale videos** to a higher resolution.
- **Queue and gallery**: add many jobs, close the app, and find every result with its settings later.

Everything runs on your NVIDIA graphics card. Your images and videos stay on your computer.

## Requirements

- **Windows** 10 or 11
- **NVIDIA graphics card** with a current driver (tested on an RTX 4090 with 24 GB)
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

The first image also downloads the model (about 18 GB). The progress shows in **Queue**. After that, a 1K image takes about 1–2 minutes on an RTX 4090.

To **edit** an image, drop it into **Input images** and describe the change, for example `Make it a rainy night with neon reflections`. You can also click **Use for editing** under any result.

> **Tip:** For a transparent background, open **How to use** (the **?** icon) and click **Use transparent image prompt**.

## Choose a model

Open **Settings** (the gear icon). Click **Select** on any model. If it is not downloaded yet, it downloads when you start your next job. You can also click **Download** to get it now.

<img src=".github/assets/models.png" alt="Image models in Settings" width="70%">

| Image model | Size | Choose it when |
| --- | --- | --- |
| **Q8_0** | 7.6 GB | You want the best quality. Recommended. |
| Q6_K / Q5_K_M | 5.9 / 5.2 GB | You want a smaller download. |
| Q4_K_M / Q4_0 | 4.6 / 4.1 GB | Your graphics card has less memory. |
| BF16 | 14.2 GB | You want the original full-precision model. |

All image models also use a shared text encoder and VAE (about 10 GB, downloaded once).

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
- **Gallery** shows all your images and videos. **Use settings** loads the prompt and settings of a result. **Run again** makes a new version.

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

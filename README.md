# Qwen - Images & Video

Create and edit images and videos on your own computer with Qwen-Image-2.1, MiniMax H3, FastH3, LTX 2.5 and SeedVR2/FlashVSR upscaling.

## Requirements

- Windows 10 or 11
- An NVIDIA graphics card with a current driver
- At least 40 GB of free disk space (more for video models)

## Install and start

1. Clone the repository:
   ```
   git clone https://github.com/Ido108/qwen.git
   ```
2. Double-click **Start Qwen.bat**.
3. On the first start, choose where Qwen stores its files. Press Enter for drive C, or type another drive letter (for example `D`) or a folder (for example `D:\Qwen`).
4. Wait for the installation to finish. The app then opens in its own window.

The first start installs everything automatically: Python, Node.js (if missing) and all packages. You do not need administrator rights. Later starts open the app immediately. Setup also creates a **Qwen** shortcut next to the repository folder.

Models are not part of the installation. Choose a model in **Settings** and download it, or just start a job: missing model files download automatically.

## Create images

1. Write a prompt. To edit or combine images, drag, select or paste up to 10 images.
2. Keep **Enhance prompt** on, or turn it off to use your text exactly as written.
3. Choose **Auto 1K** (faster) or **Auto 2K** (more detail). Both choose the aspect ratio automatically. **Custom size** gives an exact size.
4. Click **Generate image**. The result is saved as a PNG.

You can write prompts in any language. **Advanced settings** has the seed, a negative prompt, the step count and the output folder. **Examples** has 37 ready-made prompts with their input images.

For a transparent background, use the prompt from **How to use**:
`This is an RGBA image with transparency. [Description]. The image has alpha channel and the background is transparent.`

## Choose models

Click the model button in the header or the gear icon to open **Settings**. Each model has **Download**, **Select** and **Remove**. A download includes all files that the model needs. Removing a model does not delete your images or videos.

| Image model | Size | Notes |
| --- | --- | --- |
| Q8_0 | 7.59 GB | Recommended |
| Q6_K | 5.88 GB | |
| Q5_K_M | 5.22 GB | |
| Q4_K_M | 4.60 GB | Less memory |
| Q4_0 | 4.05 GB | Least memory |
| Official BF16 | 14.23 GB | Full precision |

The text encoder and VAE are shared by all image models and download once (about 10 GB; BF16 uses a 17.5 GB encoder).

## Prompt enhancement

Prompt enhancement uses Gemini through your Google account. No API key is necessary.

1. Open **Settings**.
2. Click **Connect Google Account** and sign in.

When enhancement is on, the prompt and input images go to Google. When it is off, nothing leaves your computer. Image and video generation always runs locally.

## Video

Open the **Video** tab.

- **Text to video / Image to video**: write a prompt, choose duration and size, and click **Generate video**. Image mode accepts a first frame and an optional last frame.
- **Video to video**: replace people in a clip, change the camera view, or edit with reference images.
- **Reference images**: create a new scene from up to four subject images and an optional background (LTX 2.5).
- **Upscale**: increase the resolution of a video with SeedVR2, FlashVSR or SwiftVR.

Every result has **Send to**, which opens it as the input of another video workflow.

Some LTX files need access to the gated [Lightricks model](https://huggingface.co/Lightricks/LTX-2.5). Accept its terms on Hugging Face, then set the `HF_TOKEN` environment variable to an access token of that account.

## Queue and gallery

- You can add jobs while other jobs run. **Queue** shows the active job and the waiting jobs. You can move, cancel or pause them.
- Waiting jobs stay saved when you close the app.
- **Gallery** shows all images and videos. **Use settings** loads a result's settings into the form. **Run again** adds the same job to the queue.

## Other tools

These are in the `scripts` folder:

- **Restart Qwen.bat**: closes Qwen and opens it again.
- **Show Qwen Logs.bat**: opens a log viewer.
- **Download Models.bat**: downloads the recommended Q8 and Q4_K_M image models without opening the app.

Closing the Qwen window or its activity terminal stops Qwen.

## Mac

Open `quickStart.command`. On a Mac, generation runs on RunPod Serverless. See [RUNPOD.md](RUNPOD.md).

## License

The Qwen models and code are under the Qwen Research License (`reference/Qwen-Research-LICENSE.txt`). Third-party code licenses are in the `vendor` folders.

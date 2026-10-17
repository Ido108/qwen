# Qwen on macOS with RunPod Serverless

The Mac app keeps its interface, queue, prompts, uploads, gallery and finished files on the Mac. Image and video inference runs on dedicated RunPod Serverless endpoints. The worker uses the same pinned Qwen model catalog and image/video engines as the Windows app; it downloads only the model and supporting files required by the selected job to the attached network volume. Prompt enhancement stays in the desktop app through its Antigravity OAuth connection.

## Open on a Mac

```bash
git clone https://github.com/Ido108/qwen.git
cd qwen
./quickStart.command
```

Open `quickStart.command` from Finder on the first run. It also creates `Qwen.app` with the project icon in the cloned repository; open that app or drag it to the Dock for later runs. Its Python environment is `qwen/.venv`, inside the cloned repository. Results and connection settings default to `~/Library/Application Support/Qwen`; `QWEN_DATA_DIR` can override that path. Python 3.12 and Node 22+ are installed with Homebrew when needed. If Homebrew is missing, the launcher stops with a clear installation link rather than changing system software unexpectedly. The browser opens the local interface at `http://127.0.0.1:47821`.

## Create the RunPod resources

1. In RunPod, create a **network volume** in a [datacenter with S3 API support](https://docs.runpod.io/storage/s3-api). The volume and endpoints must be in the same datacenter. The worker sees it at `/runpod-volume`; the desktop uses its S3 API to transfer full-size images and videos. For Q8 image plus video models, start with ample free space and increase the volume based on selected models and Hugging Face cache use.
2. Create a RunPod API key and a **separate S3 API key** in RunPod Settings. Keep both private. The desktop saves them in macOS Keychain or Windows DPAPI; the worker never receives either key. RunPod's S3 API uses the S3 access key, secret, volume ID and datacenter ID.
3. Create two **queue-based Serverless endpoints**, one for images and one for video, both from this private GitHub repository's default `main` branch using `runpod/Dockerfile`. Attach the same network volume to both. The exact handler is `runpod/handler.py`; the Docker image is Linux `amd64` and installs CUDA 13 PyTorch wheels. A GPU with 80 GB VRAM is a conservative starting choice for the video endpoint. The image endpoint can start with 48 GB or 80 GB, depending on the model and budget. Actual speed and memory use need measurement on the selected worker.
4. Set **active workers 0**, **max workers 1** for each endpoint. An idle timeout around 120 seconds retains the loaded image model between nearby jobs but incurs a short idle charge. Each request sets a two-hour execution limit and a three-hour total TTL to allow the first model download; the RunPod default execution limit is only 10 minutes. Choose a CUDA-13-compatible host driver.
5. In Qwen **Settings → Processing location**, choose **RunPod GPU**. Enter the image endpoint ID, video endpoint ID, network volume ID, datacenter ID, RunPod API key, and S3 access/secret key. Select **Save settings**, then **Test connection**. Configure **Prompt enhancement** separately if desired.

The image endpoint uses `Qwen-Image-2.1` Q8/Q4/etc. with the matching encoder and VAE; the video endpoint uses FastH3 for text-to-video or MiniMax H3 for text and image-to-video according to the selected model. The network volume retains verified checkpoints across worker restarts. Generated PNG/MP4 files are copied back to the desktop's output folder and appear in its gallery. If the app closes while a cloud job is running, reopening it resumes polling by RunPod job ID; output files remain on the network volume.

The same video endpoint also handles **Replace subjects**, **Camera view**, the optional two-stage subject replacement followed by camera view, **Reference images** (LTX 2.5 MSR V2), and **Upscale**. The Mac uploads the selected videos, image references and optional voice references to the network volume for that job. The worker downloads the exact model and companions on demand; the final MP4, poster and any first-stage subject-replacement MP4 return to the Mac gallery.

For LTX, accept access to the [Lightricks base model](https://huggingface.co/Lightricks/LTX-2.5) and add that account's read token as `HF_TOKEN` in the video endpoint's environment. Both INT8 and GGUF require the matching gated Gemma encoder and video/audio VAEs. The worker preflights access before downloading other large files. The MSR V2 LoRA and optional spatial upscaler are selected automatically. Download progress and the actual generation steps use the same interface as local Windows processing. Rebuild the endpoint image from the updated `main` branch to activate this handler; editing the desktop alone does not update deployed workers.

RunPod [GitHub deployment](https://docs.runpod.io/serverless/workers/github-integration), [endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations), [async jobs](https://docs.runpod.io/serverless/endpoints/operation-reference), and [S3 path mapping](https://docs.runpod.io/storage/s3-api) are the relevant platform references. A real first-run Docker build, GPU generation and S3 transfer require your endpoint and keys; local tests cannot certify those remote steps or physical Mac behavior.

The worker runs the code that is in its image. After an update that adds engines (for example Krea 2 and Z-Image images, LTX 2.5 in Create video, MiniMax H3 Ref2VA in Reference images, or Civitai models), rebuild and redeploy both endpoint images before you use them on RunPod. `tests/test_runpod_handler.py` runs the real handler with the network and GPU replaced, to check that each of these paths reaches its engine with the right model files.

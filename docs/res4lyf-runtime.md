# RES4LYF runtime

## Installed support

The app now supports `res_2m`, `res_2s`, `res_2m_ode`, and `res_2s_ode`.
These names call the original RES4LYF beta algorithms. They are not aliases
for ComfyUI's `res_multistep`.

Source: [RES4LYF at 56e44755d866c87ff9cb4b6608c8e42155db6ecc](https://github.com/ClownsharkBatwing/RES4LYF/tree/56e44755d866c87ff9cb4b6608c8e42155db6ecc).
The upstream `beta/__init__.py` defines the wrapper defaults. The two ordinary
names use its SDE defaults, including eta 0.5. The ODE names set eta and
eta_substep to zero. The standard wrappers derive SDE noise from Torch's
initial seed; Comfy's `prepare_noise` sets that seed for each request.

The source list contains 82 files (3,231,186 bytes). Every file has a pinned
SHA-256 in `reference/res4lyf-source.json`. Setup downloads only these source,
configuration, and license files. It does not download example assets or model
weights. The source directory lives below `QWEN_RUNTIME_DIR/samplers`.

`PyWavelets==1.10.0` is added to the runtime lock. Existing NumPy, Matplotlib,
SciPy, and OpenCV headless provide the other required numerical libraries.
OpenCV headless provides the `cv2` module; the headless sampler path does not
need OpenCV desktop windows. Torch and CUDA versions stay unchanged.

## Engine and settings

- Standard Qwen and Krea image controls expose the four exact names.
- Civitai description and example parsing accepts these names for future
  imports. Author workflow records keep the original sampler name.
- Qwen, Krea, Z-Image, FLUX.3, and H3 engines register a requested RES sampler
  before they load model weights. LTX's official fixed Euler workflow stays
  unchanged. No new unsupported LTX sampling choice is exposed.
- Missing or modified source files fail with a repair instruction. Setup
  restores the verified source. Built-in samplers do not import RES4LYF.
- The integration bypasses the extension's web server and unrelated node
  startup. A small adapter supplies its logging and display settings. Its
  sampler math and numerical helpers stay unchanged.
- RES4LYF sends a final preview after its step loop. The app omits that duplicate
  progress event, so it cannot report six completed steps for a five-step job.
- Fixed Turbo sigma schedules keep their existing validation and behavior.

## Verification

`scripts/check_res4lyf.py` executes all four algorithms through the real Comfy
`KSAMPLER` wrapper with a small CPU-only denoiser. It checks finite 4D and 5D
results, seed repeatability, SDE/ODE noise behavior, distinct algorithms, and
five progress callbacks for five schedule steps. Both the main Comfy runtime
and the isolated FLUX.3 runtime pass. These checks load no model weights.

52 focused tests cover source integrity, missing dependencies, import names,
model settings, LoRA metadata, and unchanged fixed schedules. The dependency
check reports compatible installed packages. The locked-package install
reports no further package changes.

These checks establish that the sampler code runs and reaches the supported
engine path. They do not establish visual style quality or pixel equality with
an author's image. Some published Krea workflows use custom Clownshark options
and separately rescaled sigmas. Selecting `res_2s` alone does not reproduce all
those settings. Published graphs do not identify an exact RES4LYF revision.

## Test isolation incident

An initial test command inherited the real data directory. An existing Civitai
test wrote and deleted its sign-in file. No saved copy was found. The user
reconnected Civitai. The subsequent live status check reports connected=true
with account authentication.
`CivitaiTests.setUp` now redirects authentication and model-list writes to an
isolated temporary directory even when launcher environment variables exist.
All subsequent tests use separate data and authentication directories.

## Rendered interface and delivery

Update, 2026-10-20: Turbo now exposes the same supported sampler and schedule controls. The prior verification below that Turbo clears manual overrides describes the earlier UI. All image families now preserve and submit an explicit choice. Qwen's sampler reset restores its model sampler; its schedule reset restores the trained sigma list when the model supplies one. Z-Image and FLUX.3 engines now honor request values. Current browser and delivery evidence is in `validation/sampling-resets` and the active delivery queue. Custom combinations are not claimed to match the quality of the model's recommended combination.

The frontend-testing-debugging skill supplied the validation checklist. The
Browser plugin was absent and the computer tool listed no enabled browser.
Checks used the installed Python Playwright browsers.

Environment: authenticated `https://moise.up.railway.app/qwen-studio/`;
Chromium 1440x1000; WebKit 390x844 and 320x740. The first viewport and sampler
controls were captured. The screenshots show bounded controls and no horizontal
overflow. No physical iPhone check was possible.

| Check | Result |
|---|---|
| URL, title and meaningful page content | Pass |
| Blank page or framework error overlay | None |
| Page, console or HTTP errors in completed UI checks | None |
| Four exact RES selections reach submitted requests | Pass on all three viewports |
| Steps, CFG and seed retain their values | Pass |
| Krea exposes RES choices; Turbo clears manual overrides | Pass |
| Screenshot review and horizontal overflow check | Pass |
| Local and Moise asset content matches the new build | Pass |

Interaction: Images > Advanced settings > select each RES sampler and beta >
Generate image > inspect intercepted request. Switch to Krea and select res_2s;
switch to Turbo and verify cleared sampler overrides. Generation requests stay
intercepted. This does not run inference.

Evidence and runnable script:
`E:/AI/qwen-image-studio/validation/res4lyf/check_ui.py`, `ui-results.json`,
`sampler-chromium-1440.png`, `sampler-webkit-390.png`, and
`sampler-webkit-320.png`. The delivery script is under
`validation/lora-training-controls/check_delivery.py`.

Setup completed through `scripts/setup.ps1`, including the source install,
dependency check, CPU sampler execution, frontend build, and existing runtime
check. The new JavaScript asset is `index-1neiYMMN.js`. The launcher restarted
only after its job/download guards passed. Moise returned one transient 503
while reconnecting; the next asset and queue check passed. The existing large
bundle warning remains. Civitai sign-in recovery is now confirmed by the live
status endpoint.

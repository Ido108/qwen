# Repeated image inputs

Date: 2026-10-19.

## Change

The image worker previously wrote `input-N.png` for every input on every job. Image tools also wrote `prepared-N.png`. No runtime reader uses these files. Comparison, restored settings and image reuse use the durable upload IDs. The worker now uses those originals directly and does not create the duplicate PNGs. Result PNGs, their metadata, job settings, process records and viewing thumbnails retain their existing behavior.

`backend/image_inputs.py` keeps decoded RGBA inputs and deterministic tool preparation in one RAM cache. The cache holds at most 24 entries and 256 MiB of retained image pixels. It uses least-recently-used eviction. Large images above the limit still work without retention. Returned images are copies, so a tool cannot mutate a later job's cached input. File identity, size and timestamps invalidate changed files. Missing files still fail validation instead of using stale data. Preparation keys include ordered pixels, tool configuration, dimensions, output budget and tool options. Prompt and seed changes do not invalidate the image-only preparation.

This cache is independent of the existing model-specific prompt/image conditioning cache. A new prompt still needs the model's text/vision encoding. An unchanged consecutive prompt/input combination can use the existing conditioning cache. That cache retains one combination. Model/encoder changes, worker exits and memory-cache eviction still require later work. No GPU memory-placement policy or model-weight format changed.

Status text now distinguishes text processing, prompt/image processing and cached conditioning. Run details show how many input images were reused, whether tool preparation was reused, and input-read time.

## Old files

`scripts/cleanup_image_inputs.py` defaults to inspection. `--apply` removes only strictly named input/prepared PNGs in finished image-job folders. It validates resolved paths, preserves active/queued jobs and unrelated files, requires source uploads to remain present, and compares the pixels of each input copy with the original. It rechecks each candidate before deletion. Missing or different originals leave the candidate intact.

The live cleanup removed 25 verified copies totaling 45,869,451 bytes (43.74 MiB). SHA-256 checks before and after confirm that all 21 distinct originals remained unchanged. No eligible copies remained. Uploads, people, outputs, job records, model files and the general temporary folder were not cleanup targets.

## Verification and delivery

- Eighteen CPU tests pass: repeated source reads decode once, returned images cannot mutate retained pixels, replacements/deletions invalidate the cache, memory limits evict old entries, tool settings invalidate preparation, prompt/seed changes reuse image preparation, text-only jobs use no inputs, and cleanup preserves unsafe candidates. Tests execute the real image worker flow with a CPU image-engine stand-in; they verify outputs/settings and absence of input/prepared PNGs. Existing conditioning-cache, image-tool and resident-worker tests pass.
- The production TypeScript/Vite build passes with the existing chunk-size notice. Bundle: `index-CD1L8H7n.js`. Setup fingerprint matches.
- The real Railway page is checked in desktop Chromium and iPhone 13 WebKit for all three progress labels. The mobile check opens the Queue dialog; the desktop activity banner is hidden by the existing mobile layout. Desktop result details show reuse and input-read time. Job progress/results are mocked. No generation or model download runs. Browser plugin is not available; installed Playwright supplies the browser checks. One earlier navigation returned HTTP 503 and recovered automatically. That transient remote failure is separate from the input-cache changes; this work does not claim to repair it.
- The queue and downloads were idle. Only the old resident image worker was released through the guarded app endpoint. The server remained running and returned HTTP 200. The next real image job imports the updated worker and loads its model once. Later jobs can reuse the loaded model and image cache.
- Evidence: `E:/AI/qwen-image-studio/validation/image-input-reuse/`, including `cleanup.json`, `delivery.json`, `ui.json` and screenshots. New GPU execution time and physical iPhone behavior are not measured.

Final browser run: both desktop and mobile pass visible progress-label checks with zero HTTP/page errors and no horizontal overflow. Final live state: server running, queue empty, old worker released. No QA generation was submitted.

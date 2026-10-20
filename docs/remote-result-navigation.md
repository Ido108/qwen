# Remote result navigation — 2026-10-19

## Cause measured on the live Railway path

The click handler updates the selected result in 4–13 ms. The old image element then waits for the full PNG to cross the PC → Moise → browser relay. The API middleware overwrites every response's Cache-Control with no-store, including endpoints that already set a cache policy. Reopening a completed image therefore downloads it again.

The six baseline selections used three existing 896 × 1152 images, then revisited the same three images. No inference ran. PNG payloads were 1,284,350 / 1,331,456 / 1,368,025 bytes. All six remote selections transferred their PNG again. First-view latency was 419–3,097 ms; repeat-view latency was 439–886 ms. Local clicks took 18–66 ms including image loading. The delay occurs after the UI selection, mainly in the remote request/transfer path.

## Delivered behavior

- Completed PNGs, thumbnails and new display copies use `private, max-age=604800, immutable`. Job IDs are immutable; retry creates a new ID. Mutable JSON, sessions, live previews and errors remain no-store. Existing upload/avatar cache headers now take effect and explicitly remain private.
- `/api/jobs/{id}/display` creates a WebP viewing copy once. It uses quality 90, a maximum 1600-pixel side, no enlargement, preserved alpha/aspect/ICC information, bounded locks and atomic replacement. The PNG output is untouched. The existing thumbnail appears immediately while the sharper viewing copy loads.
- The workspace, full-screen viewer and gallery share a decoded-image cache with a 12-image bound. Only adjacent viewing copies preload; Save-Data/2G disables that preload. Shared in-flight requests survive one viewer closing. Unneeded requests cancel when selection changes. A late response cannot replace the current choice.
- Zoom requests the original PNG. Comparison also upgrades to the original. Download URLs continue to return the original bytes. Failed image reads keep the selected thumbnail and expose Retry.

## Live before/after

Same three result IDs, fresh Chromium browser contexts, authenticated Railway path, caching enabled. The test does not intercept requests. Times are browser samples, not a network guarantee.

| Action | Before | After |
| --- | --- | --- |
| First selection: any image visible | 419–3,097 ms | 7–11 ms (existing thumbnail) |
| First selection: sharp viewing copy | 419–3,097 ms (PNG) | 488–628 ms (WebP) |
| Revisit: sharp image visible | 439–886 ms | 6–7 ms |
| Bytes per first selected image | 1.28–1.37 MB | 152–166 KB |
| Reload: selected viewing copy | Network transfer | Browser cache, 0 transferred bytes |

The viewing copies are about 88% smaller in this sample. The first uncached image still needs a remote round trip. Preloading cannot remove that latency for an arbitrary image that the user did not view before. The immediate thumbnail prevents that wait from blocking selection feedback.

## Verification

- Six image-serving tests cover original bytes, download disposition, alpha (RGBA and LA), ratio, size bounds, no enlargement, cache reuse, concurrent first requests, mutable API cache policy and invalid/unfinished outputs. These pass. The broader API/relay run passed 36 tests before the additional LA case; the final six image tests pass after that case.
- Three Node tests cover shared request cancellation, bounded decoded reuse and remote URL preservation.
- Production TypeScript/build passes. The pre-existing large-chunk notice remains. `git diff --check` passes for changed tracked files.
- Chromium desktop and iPhone WebKit emulation pass locally and through Railway: selection, original-resolution zoom, next/previous, zoom reset, gallery, original comparison, reload cache reuse and 320-pixel/landscape overflow checks. The final remote runs report no page, console or HTTP errors. The reloaded queue returns HTTP 200 in both browsers.
- Controlled CPU-only checks verify recovery after failed display/original requests and navigation while an older image request is delayed. These checks do not claim live network failure reproduction.
- Earlier QA failures: the desktop-only Recent rail is closed on narrow screens, so the mobile fixture now uses the real mobile result strip. A broad close-button selector matched a hidden Civitai dialog; it now targets the open dialog. The first post-restart timing run did not find loaded results; the later live timing run passed. One earlier WebKit run reported an old-document queue cancellation at reload. The final run has none. Do not interpret those failed fixtures as performance measurements.
- The guarded launcher restarted only after active jobs, pending jobs and downloads were idle. Production runtime now records PID 7944 / owner 49232. The CPU-only validation server (PID 33552) stopped after identity verification. No GPU generation, model-weight download or user output deletion occurred.
- Delivered assets: `index-Cvabd9IZ.js` / `index-DkyYmZWx.css`. Setup fingerprint matches. Physical iPhone hardware is not available.

Evidence: `E:/AI/qwen-image-studio/validation/result-latency/`: before/after JSON and screenshots, live browser results, controlled failure results, scripts and isolated frontend bundle. Total retained validation folder: 206,166,962 bytes at the final check. The bundle includes existing public assets. No cleanup was requested.

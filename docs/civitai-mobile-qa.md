# Civitai mobile delivery — 2026-10-19

## Delivered behavior

- Two catalog columns on narrow phones, three on wide touch screens. Preview area is reserved before media arrives. Long titles and creator names stay within the card. Filters retain their values and backend semantics.
- Stable modal height. A fixed heading and close button, scrollable detail body, and fixed action area remain usable on small screens. Mobile has one outer content scroll area instead of competing media/info grid rows.
- Version changes select the new file, reset the preview and model-type choice together. Empty preview lists show a readable fallback. Missing or failed preview media cannot throw an exception.
- Only the most recent model-opening request can change the modal. Closing invalidates outstanding requests. A click inside empty dialog space does not count as a backdrop click. Escape and actual backdrop clicks close it.
- Catalog video cards stay paused. Only the active video preview plays. Selected model/version and dialog state survive background model polling.
- Image-input help now describes Z-Image redraw and the current Krea style-reference mode. It avoids blanket claims about an architecture's possible capabilities. See `image-input-capabilities.md`.

## Root-cause evidence

1. Before the change, iPhone WebKit at 390 px showed a single 366 px wide card, 569 px high. The grid used a 210 px minimum and therefore could not fit two columns.
2. Controlled version metadata reproduced a real render failure in the old build: select preview 3, then a version with only one preview. WebKit raised `undefined is not an object (evaluating 'r.type')`. The old render accessed the previous preview index before its effect reset it.
3. The previous modal sized its rows from remote images and nested scrolling content. The new media viewport and modal size do not depend on image dimensions.
4. The old request path accepted late responses without checking request identity. The new test delays the first response, opens a second model, and confirms that the first response can neither replace it nor reopen a closed modal.

The initial real card-opening check did not reproduce spontaneous closure by itself. The controlled preview boundary did reproduce a page-breaking error. The delivery fixes that error plus layout and request-order defects; it does not claim a single explanation for every earlier screenshot.

## Validation

Browser plugin is unavailable. Tests use installed Playwright Chromium and WebKit.

| Flow | Result |
| --- | --- |
| Authenticated Railway image catalog and actual model details | Pass in iPhone 13 WebKit emulation and desktop Chromium, with real API metadata and real preview files |
| Mobile widths 320 and 390; landscape 844 × 390 | No horizontal page overflow; two/three columns; dialog, close button and action remain within the viewport |
| Desktop 1440 × 1000 | Catalog and two-column modal fit; details scroll; actions and close remain visible |
| Image versions, file selection, scrolling, close/reopen, polling | Pass; no uncaught page errors or Qwen HTTP errors in the final live runs |
| Remote MiniMax H3 video catalog | Pass; DaSiWa has nine versions; first/last selection retains stable layout; catalog videos pause and active video has controls |
| Preview boundary and no-preview cases | Pass with controlled metadata; the previously failing case no longer throws |
| Delayed responses and rapid distinct card clicks | Newest click wins; close prevents delayed reopening; Escape/backdrop work |
| Existing Z-Image and Krea tests | 22 pass using separate test data; no GPU inference or model downloads |
| TypeScript and production build | Pass; existing Vite warning for a main chunk above 500 kB remains |
| `git diff --check` | Pass; Git reports existing CRLF conversion notices |

Final assets served through Railway: `index-vv8V2YQd.js`, `index-DOPAwwje.css`. Startup setup fingerprint matches. The server PID remains **32972** before and after. A user generation was active during the tests; no job, queue, download, model cache or server process was stopped or changed.

## Evidence and limits

Scripts, screenshots and JSON results: `E:\AI\qwen-image-studio\validation\civitai-mobile\`.

- `check_live.py`, `live-results.json`, `iphone-grid.png`, `iphone-details.png`, `desktop-grid.png`, `desktop-details.png`, `mobile-320-details.png`, `mobile-844-details.png`.
- `check_regressions.py`, `regression-before.json`, `regression-after.json`.
- `check_races.py`, `race-results.json`; `check_video.py`, `video-results.json`.

Real install requests are blocked in the browser tests to preserve user data and prevent large downloads. Existing installation logic is unchanged. Physical iPhone hardware is not available. New community edit/ReID workflows are researched, not installed or tested for image quality. This delivery contains frontend changes only and requires no server restart.

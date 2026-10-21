# Idle model memory release

Date: 2026-10-20.

## Behavior

- The server closes its image worker after 600 seconds without an active or waiting job.
- The deadline starts when a job finishes, including failure or cancellation. A long job does not consume the next idle interval.
- The timer checks once per second. Process shutdown follows the deadline. The existing worker shutdown waits up to 30 seconds before ending a stuck owned process tree.
- The same loaded model remains available before the deadline. After release, the next image job starts a new worker and loads existing files. No extra model download is required.
- Closing the worker releases its CPU model weights, decoded/prepared image cache, prompt conditioning cache and CUDA context. Merely emptying a CUDA allocator cache would leave CPU weights in RAM.
- UI polling and empty queue controls do not extend the deadline. A paused queue with waiting jobs prevents release. A model download does not prevent idle image-worker release: downloads belong to the server and continue.
- Server startup does not preload a model from old job history. Manual and automatic release suspend background warm-up until the next local image job.
- The timer uses the scheduler lock. New job submission and expiry cannot use or stop the same worker at the same time. The timer stops when the server shuts down.

## Terminal behavior

The user changed terminal behavior after the initial idle-memory delivery. Closing either Qwen terminal tab now stops the server and its workers, including active jobs and downloads. Closing the browser leaves Qwen running. See [terminal shutdown](terminal-shutdown.md).

`Stop Qwen.bat` calls the guarded launcher. It refuses to stop while jobs or downloads remain. The Windows owner controls its server and worker descendants. Other applications are outside that process tree.

The launcher and terminal now state that closing a Qwen terminal tab stops Qwen. The 10-minute idle-memory rule remains active while the terminal stays open.

## Verification scope

Tests isolate `QWEN_DATA_DIR` and `QWEN_AUTH_DIR`. They disable CUDA, remote relay and automatic model warm-up, except where a mocked warm-up proves the startup guard.

The idle tests advance a controlled clock to the 600-second boundary. They cover retention before expiry, release once, active/waiting/shutdown guards, unchanged timers under polling, no stale reload, scheduler locking, timer startup/shutdown and idle-time reset after actual scheduler completion.

A real lightweight subprocess verifies worker reuse, process exit on expiry and a new process for the next job. These checks do not load AI weights or measure generation quality. They do not wait ten wall-clock minutes.

The Windows lifecycle test covers owner cleanup, launcher exit, server reuse, restart and stop guards. The optional Electron close test is separate. A missing optional Electron install does not skip the normal server lifecycle checks.

## Delivery

Sixty tests pass with no failures or errors. One optional Electron test is skipped because its binary is absent. Normal Windows lifecycle tests pass, including restart and owner-process cleanup. `git diff --check` passes.

The guarded `scripts/desktop.py --restart --no-browser --no-console` owner restarted the production server after two idle checks confirmed no active/waiting job or download. This is the same stop/start controller called by `scripts/launch.ps1`. It avoided an unrelated full setup: the existing setup fingerprint differs only for four already-built frontend source files. No frontend source, dependency or setup stamp was changed for this task.

Local and authenticated Moise requests both report a 600-second idle policy, no resident model, no active/waiting job, no running download and a connected Civitai account. Both serve the existing `index-BXlQ034p.js` with the expected local/relay hashes. No Railway code change was needed.

The final Windows process snapshot shows the new owner (53960), server launcher (58084) and server interpreter (43568), with no image worker. Free physical RAM was 36.53 GB. The local GPU snapshot reported 29.27 GB free. These are whole-machine snapshots, not memory savings attributed to this change.

Evidence: `E:/AI/qwen-image-studio/validation/idle-memory/tests.json`, `tests.txt`, `delivery.json` and `processes.json`.

At the first current-state check, the old worker PID 59588 was already absent. Qwen reported no loaded image model. Windows reported about 36 GB of free physical RAM. This change did not cause that earlier release.

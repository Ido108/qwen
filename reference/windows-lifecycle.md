# Windows app lifetime and reconnect behavior

`Start Qwen.bat` -> `scripts/launch.ps1` -> `scripts/desktop.py` is the production
entrypoint. `Qwen.lnk` opens that BAT through Explorer, with the existing Qwen icon
and a normal visible terminal. The installer recreates and reads back the shortcut.

The desktop supervisor owns a named mutex per repository/data directory. A second
launch focuses the current app; Restart signals that owner, waits for its mutex,
then starts the replacement. The server and dedicated Edge/Chrome app window are
created suspended, assigned to a Windows Job Object, then resumed. Descendants
(including Python venv shims and local inference workers) inherit membership.
Closing the owner handle stops the entire tree, including when Windows closes the
terminal without executing Python cleanup. Closing the browser makes the owner
return normally, so the launching PowerShell/BAT also exits. No other browser
profile is attached to that job.

This uses Microsoft's documented [Job Object lifetime semantics](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
and [process assignment rules](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject).
The one-time migration checks the old server's app/data paths, live listening PID,
command line, and idle queue before stopping its verified process tree. It closes
the former app-mode window only for the production runtime file. It never stops
general Edge windows or another application's server.

Local active work stops on app close/restart. Interrupted work is marked failed on
reopening and is available to retry. Waiting jobs stay on disk. RunPod work is
remote and can continue independently. `Show Qwen Logs.bat` is explicitly a
separate read-only viewer, not the owning terminal.

## HTTP and authentication

The Windows HTTP server explicitly uses `SelectorEventLoop` via Uvicorn's supported
[custom event-loop entrypoint](https://uvicorn.dev/concepts/event-loop/). All local
worker processes use synchronous `subprocess.Popen`; the app does not need asyncio
subprocess transports. This avoids the Python 3.12 Proactor disconnect callback
path that raised WinError 10054 before socket cleanup. No exception handler or
log filter suppresses unrelated application failures. This is for the local HTTP
server; it does not change model execution or GPU sampling.

Every frontend mutation waits for a local session. On a specific middleware
`session_expired` rejection it renews the shared token and retries once. Concurrent
requests share the refresh. This covers automatic person scans and XHR video
uploads. Origin failures, arbitrary 403s, and network failures are not retried;
there is no retry after an uncertain generation submission. The guard rejects
expired tokens before a route can perform side effects. Drafts are not reset.

The dedicated browser profile is new on first upgrade from the old shared Edge
profile. Previously saved generations and uploads remain in the existing data
folder. Browser-only drafts from the old shared profile are not automatically
copied. Later managed restarts retain the dedicated profile and its drafts.

## Verification

- `python -m unittest discover -s tests -q`: API guard, local-only checks, queue,
  worker behavior, suspended-process ownership, grandchildren, verified legacy
  shutdown, duplicate launch, restart, abrupt owner exit, and socket resets.
- `node --test tests/session.test.mjs`: startup scan ordering, server-token
  rotation, concurrent refresh, non-retryable errors, bounded retries and XHR
  upload recovery without duplicate accepted requests.
- `python scripts/verify_desktop.py`: actual Start/Restart BATs with isolated
  data/ports and real app-mode browser windows; duplicate reuse, old-window
  closure, app-close shutdown and forced-owner shutdown.
- `scripts/verify_reference_workbench.py`: CloakBrowser clicks the actual person
  scan after a simulated session rotation; all mutation requests are intercepted.
  Reference selection, queueing, preview controls, trimming and responsive UI are
  also checked. It does not submit media generation.
- `npm run build`, shortcut property readback, ICO frames from 16 through 256 px,
  and `git diff --check`.

These checks do not benchmark generation speed and do not run model inference.

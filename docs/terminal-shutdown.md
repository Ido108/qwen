# Terminal shutdown

Date: 2026-10-20.

## Requested behavior

Closing either Qwen terminal tab, or their Windows Terminal window, stops Qwen. This ends active generation, downloads and Qwen's worker processes. The other Qwen tab also exits. Saved results, queued-job files and resumable download parts remain.

Closing a browser tab or the optional desktop client leaves Qwen running. The 600-second idle model release remains active while the terminal stays open. Normal launch and the existing Windows login task open the connected terminal tabs. The explicit `--no-console` option remains available for maintenance and isolated checks.

## Implementation

`scripts/desktop.py` still uses the Windows Job Object to own the server and its descendants. Each server launch creates a fresh terminal session. `scripts/terminal_lifetime.py` watches Activity and Server mutexes for that session.

Each viewer holds its mutex and signals readiness. A closed viewer releases or abandons the mutex. The owner detects this and closes the Job Object, even if a job is active. This works when the window closes without running Python cleanup. Unopened tabs do not count as closed tabs. Old sessions cannot stop a new owner.

Viewers also hold a read-only process handle for their owner. They exit when the owner stops or crashes. The browser is outside the server Job Object. Existing programmatic Stop and Restart retain their active-job and download guards.

The Stop/Restart guard uses `/api/activity`, which includes current jobs and downloads without scanning the full model list. It rejects incomplete status. A 404 from an older server uses the previous queue/model checks. Other request failures do not bypass the guard. This removes the cold model-list timeout found during lifecycle checks.

Microsoft documents [abandoned mutex behavior](https://learn.microsoft.com/en-us/windows/win32/sync/mutex-objects) and [process-tree cleanup when a Job Object closes](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects). The implementation uses these OS lifetime rules instead of depending on a terminal's Python exit handler.

## Checks

`tests/test_terminal_lifetime.py` starts isolated real Windows processes. The HTTP test server reports an active job and owns a child that holds 8 MiB of RAM. No AI weights load.

The tests cover:

- Closing either viewer process stops the server, worker and other viewer.
- The guarded stop command refuses an active job; explicit terminal closure still stops it.
- A separate process remains running.
- Reopening the terminal reuses both tabs.
- Stale terminal sessions and unopened tabs do not stop a new owner.
- Owner loss ends its workers and viewers.
- A real Windows Terminal window receives `WM_CLOSE`. Its test server and worker exit.

The real-window check uses a unique test title and closes only that observed test window. The full regression suite also covers idle memory, queue handling, manual release, server ownership, API behavior and log rendering.

## Delivery

The final suite runs 75 tests: 74 pass, zero fail, and the optional Electron test is skipped because its binary is absent. The real Windows Terminal close test passes. `git diff --check` passes. Test data and account paths are isolated.

The production owner restarted through `desktop.py --restart --no-browser` only after a fresh check confirmed no active/waiting jobs or downloads. Owner PID 46420 controls server launcher PID 23292. Both Activity and Server tabs hold their current session mutex. The visible window is titled `Qwen - Activity`.

Local and authenticated Moise routes return current activity and the 600-second memory policy. Both serve the unchanged `index-BXlQ034p.js` with the expected hashes. No image worker is loaded at the final snapshot. No test worker remains. The existing login trigger opens the terminal and has no automatic failure restart.

Evidence: `E:/AI/qwen-image-studio/validation/terminal-shutdown/tests.json`, `tests.txt`, and `delivery.json`. This delivery replaces the earlier independent-viewer behavior.

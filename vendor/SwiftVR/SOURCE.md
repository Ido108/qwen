# SwiftVR source

Upstream: https://github.com/H-oliday/SwiftVR
Commit: dbec2f993ab5b0abf9a2ae2eae67117d5cadb8a7
License: Apache-2.0 (LICENSE retained).

Python package and license vendored; weights are downloaded through Qwen's model manager.
Local change: optional decord/imageio imports in swiftvr/io.py moved into their
offline reader/writer functions. Qwen uses PyAV and the bounded H.264 writer.
The neural model and streaming implementation are unchanged.

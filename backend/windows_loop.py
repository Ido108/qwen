"""Socket event loop for the local Windows HTTP server.

Workers use subprocess.Popen, not asyncio subprocess transports. Selector avoids
the Python 3.12 Proactor socket-shutdown failure on browser disconnect/restart.
No application exceptions are filtered or suppressed.
"""
import asyncio


def new_event_loop():
    return asyncio.SelectorEventLoop()

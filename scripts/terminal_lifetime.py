"""Tie Qwen to its terminal clients, including abrupt Windows Terminal closure.

The viewer owns a named mutex. Windows abandons it even when no Python finally
block runs. A ready event prevents an unopened tab from counting as a closure.
Each server launch gets a new session, so an old tab cannot stop a new server.
"""
from windows_processes import Instance, set_event, wait_handle
from watch_logs import viewer_name


class TerminalLifetime:
    def __init__(self, data, session):
        self.controls = {channel: Instance(viewer_name(data, channel, session))
                         for channel in ('activity', 'server')}
        self.connected = set()
        self.closed = None

    def closed_channel(self):
        if self.closed:
            return self.closed
        for channel, control in self.controls.items():
            if wait_handle(control.focus, 0) == 0:
                self.connected.add(channel)
            if channel in self.connected and control.acquire():
                self.closed = channel
                return channel
        return None

    def close(self):
        # Stop only this server session's viewers. Their process handles also
        # detect owner loss if Windows ends the owner before this code runs.
        for control in self.controls.values():
            set_event(control.stop)
            control.close()
        self.controls.clear()

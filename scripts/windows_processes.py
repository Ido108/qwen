"""Small Windows lifetime primitives; no GPU/runtime dependencies.

Children are created suspended, assigned to a kill-on-close Job Object, then
resumed. This also covers venv shims and grandchildren without a spawn race.
"""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import os
import subprocess

k = c.WinDLL('kernel32', use_last_error=True)
u = c.WinDLL('user32', use_last_error=True)


def api(dll, name, result, *args):
    func = getattr(dll, name)
    func.restype, func.argtypes = result, args
    return func


close_handle = api(k, 'CloseHandle', w.BOOL, w.HANDLE)
wait_handle = api(k, 'WaitForSingleObject', w.DWORD, w.HANDLE, w.DWORD)
create_mutex = api(k, 'CreateMutexW', w.HANDLE, c.c_void_p, w.BOOL, w.LPCWSTR)
release_mutex = api(k, 'ReleaseMutex', w.BOOL, w.HANDLE)
create_event = api(k, 'CreateEventW', w.HANDLE, c.c_void_p, w.BOOL, w.BOOL, w.LPCWSTR)
set_event = api(k, 'SetEvent', w.BOOL, w.HANDLE)
reset_event = api(k, 'ResetEvent', w.BOOL, w.HANDLE)
get_exit_code = api(k, 'GetExitCodeProcess', w.BOOL, w.HANDLE, c.POINTER(w.DWORD))
open_process = api(k, 'OpenProcess', w.HANDLE, w.DWORD, w.BOOL, w.DWORD)


def checked(value):
    if not value:
        raise c.WinError(c.get_last_error())
    return value


class BasicLimits(c.Structure):
    _fields_ = [('per_process', c.c_int64), ('per_job', c.c_int64),
                ('flags', w.DWORD), ('min_working_set', c.c_size_t),
                ('max_working_set', c.c_size_t), ('active_limit', w.DWORD),
                ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]


class ExtendedLimits(c.Structure):
    _fields_ = [('basic', BasicLimits), ('io', c.c_uint64 * 6),
                ('process_memory', c.c_size_t), ('job_memory', c.c_size_t),
                ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]


class StartupInfo(c.Structure):
    _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR),
                ('title', w.LPWSTR), ('x', w.DWORD), ('y', w.DWORD),
                ('width', w.DWORD), ('height', w.DWORD), ('chars_x', w.DWORD),
                ('chars_y', w.DWORD), ('fill', w.DWORD), ('flags', w.DWORD),
                ('show', w.WORD), ('reserved_size', w.WORD), ('reserved2', c.c_void_p),
                ('stdin', w.HANDLE), ('stdout', w.HANDLE), ('stderr', w.HANDLE)]


class ProcessInfo(c.Structure):
    _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]


create_job = api(k, 'CreateJobObjectW', w.HANDLE, c.c_void_p, w.LPCWSTR)
set_job = api(k, 'SetInformationJobObject', w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
assign_job = api(k, 'AssignProcessToJobObject', w.BOOL, w.HANDLE, w.HANDLE)
in_job = api(k, 'IsProcessInJob', w.BOOL, w.HANDLE, w.HANDLE, c.POINTER(w.BOOL))
create_process = api(k, 'CreateProcessW', w.BOOL, w.LPCWSTR, w.LPWSTR, c.c_void_p,
                     c.c_void_p, w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR,
                     c.POINTER(StartupInfo), c.POINTER(ProcessInfo))
resume_thread = api(k, 'ResumeThread', w.DWORD, w.HANDLE)
terminate_process = api(k, 'TerminateProcess', w.BOOL, w.HANDLE, w.UINT)


class Process:
    def __init__(self, handle, pid):
        self.handle, self.pid = handle, pid

    def poll(self):
        if wait_handle(self.handle, 0) == 258:
            return None
        code = w.DWORD()
        checked(get_exit_code(self.handle, c.byref(code)))
        return code.value

    def close(self):
        if self.handle:
            close_handle(self.handle)
            self.handle = None


class Job:
    def __init__(self):
        self.handle = checked(create_job(None, None))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        checked(set_job(self.handle, 9, c.byref(limits), c.sizeof(limits)))
        self.processes = []

    def spawn(self, command, *, cwd, env=None, stdout=None, stderr=None, hidden=True):
        import msvcrt
        info, start = ProcessInfo(), StartupInfo()
        start.cb = c.sizeof(start)
        handles = []
        null = None
        if stdout is not None:
            null = open(os.devnull, 'rb')
            handles = [msvcrt.get_osfhandle(f.fileno()) for f in (null, stdout, stderr or stdout)]
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            start.flags |= 0x100  # STARTF_USESTDHANDLES
            start.stdin, start.stdout, start.stderr = handles
        block = c.create_unicode_buffer('\0'.join(f'{key}={value}' for key, value in sorted((env or os.environ).items())) + '\0')
        try:
            checked(create_process(str(command[0]), c.create_unicode_buffer(subprocess.list2cmdline(list(map(str, command)))),
                                   None, None, bool(handles), 0x4 | 0x400 | (0x08000000 if hidden else 0),
                                   block, str(cwd), c.byref(start), c.byref(info)))
            try:
                checked(assign_job(self.handle, info.process))
                if resume_thread(info.thread) == 0xffffffff:
                    raise c.WinError(c.get_last_error())
            except BaseException:
                terminate_process(info.process, 1)
                close_handle(info.process)
                raise
            finally:
                close_handle(info.thread)
        finally:
            for handle in handles:
                os.set_handle_inheritable(handle, False)
            if null:
                null.close()
        process = Process(info.process, info.pid)
        self.processes.append(process)
        return process

    def contains(self, pid):
        handle = open_process(0x1000, False, pid)
        if not handle:
            return False
        try:
            result = w.BOOL()
            return bool(in_job(handle, self.handle, c.byref(result)) and result.value)
        finally:
            close_handle(handle)

    def close(self):
        if self.handle:
            close_handle(self.handle)
            self.handle = None
            for process in self.processes:
                wait_handle(process.handle, 5000)
                process.close()


class Instance:
    def __init__(self, name):
        self.mutex = checked(create_mutex(None, False, name + '-lock'))
        self.stop = checked(create_event(None, True, False, name + '-stop'))
        self.focus = checked(create_event(None, False, False, name + '-focus'))
        self.owned = False

    def acquire(self, timeout=0):
        status = wait_handle(self.mutex, timeout)
        if status in (0, 0x80):  # includes an abandoned owner after console close
            self.owned = True
            reset_event(self.stop)
            return True
        if status != 258:
            raise c.WinError(c.get_last_error())
        return False

    def stopped(self):
        return wait_handle(self.stop, 0) == 0

    def close(self):
        if self.owned:
            release_mutex(self.mutex)
        for handle in (self.mutex, self.stop, self.focus):
            close_handle(handle)


WNDPROC = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
enum_windows = api(u, 'EnumWindows', w.BOOL, WNDPROC, w.LPARAM)
window_pid = api(u, 'GetWindowThreadProcessId', w.DWORD, w.HWND, c.POINTER(w.DWORD))
window_visible = api(u, 'IsWindowVisible', w.BOOL, w.HWND)
window_text = api(u, 'GetWindowTextW', c.c_int, w.HWND, w.LPWSTR, c.c_int)
show_window = api(u, 'ShowWindow', w.BOOL, w.HWND, c.c_int)
foreground = api(u, 'SetForegroundWindow', w.BOOL, w.HWND)
post_message = api(u, 'PostMessageW', w.BOOL, w.HWND, w.UINT, w.WPARAM, w.LPARAM)


def windows():
    result = []
    @WNDPROC
    def visit(hwnd, _):
        if window_visible(hwnd):
            pid, title = w.DWORD(), c.create_unicode_buffer(1024)
            window_pid(hwnd, c.byref(pid))
            window_text(hwnd, title, len(title))
            result.append((hwnd, pid.value, title.value))
        return True
    enum_windows(visit, 0)
    return result


def focus_job(job):
    for hwnd, pid, _ in windows():
        if job.contains(pid):
            show_window(hwnd, 9)
            foreground(hwnd)
            return

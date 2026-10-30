"""Own a worker and its descendants, including Windows training loaders."""
import os
import signal


class ProcessTree:
    def __init__(self, process):
        self.process = process
        self.handle = None
        if os.name != 'nt': return
        import ctypes as c
        from ctypes import wintypes as w
        class Limits(c.Structure):
            _fields_ = [('process_time', c.c_longlong), ('job_time', c.c_longlong), ('flags', w.DWORD),
                        ('min_working', c.c_size_t), ('max_working', c.c_size_t), ('active', w.DWORD),
                        ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class Counters(c.Structure):
            _fields_ = [(name, c.c_ulonglong) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
        class Extended(c.Structure):
            _fields_ = [('basic', Limits), ('io', Counters), ('process_memory', c.c_size_t),
                        ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]
        kernel = c.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]; kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        limits = Extended(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not handle: raise c.WinError(c.get_last_error())
        if not kernel.SetInformationJobObject(handle, 9, c.byref(limits), c.sizeof(limits)):
            code = c.get_last_error(); kernel.CloseHandle(handle); raise c.WinError(code)
        if not kernel.AssignProcessToJobObject(handle, int(process._handle)):
            code = c.get_last_error(); kernel.CloseHandle(handle)
            if process.poll() is None: raise c.WinError(code)
            return
        self.handle, self.kernel = handle, kernel

    def stop(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)
        elif os.name != 'nt':
            try: os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        elif self.process.poll() is None: self.process.terminate()

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle); self.handle = None

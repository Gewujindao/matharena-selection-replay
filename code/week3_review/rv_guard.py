# Process guard for the reviewer: BelowNormal priority, 2 GiB job-object memory cap,
# and a gate on the Windows counter \Memory\Available MBytes (>= 1500) before heavy steps.
import ctypes
import ctypes.wintypes as wt
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
pdh = ctypes.WinDLL("pdh")

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
k32.GetCurrentProcess.restype = wt.HANDLE
k32.GetCurrentProcess.argtypes = []
k32.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
k32.CreateJobObjectW.restype = wt.HANDLE
k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]


def set_below_normal():
    k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wt.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wt.DWORD),
                ("SchedulingClass", wt.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


_job = None


def cap_memory(limit_bytes=2 * 1024 ** 3):
    global _job
    job = k32.CreateJobObjectW(None, None)
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00000100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = limit_bytes
    ok1 = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    ok2 = k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
    _job = job
    return bool(ok1 and ok2)


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("pad", wt.DWORD), ("doubleValue", ctypes.c_double)]


def available_mb():
    q = wt.HANDLE()
    c = wt.HANDLE()
    pdh.PdhOpenQueryW(None, None, ctypes.byref(q))
    pdh.PdhAddEnglishCounterW(q, "\\Memory\\Available MBytes", None, ctypes.byref(c))
    pdh.PdhCollectQueryData(q)
    v = PDH_FMT_COUNTERVALUE()
    pdh.PdhGetFormattedCounterValue(c, 0x00000200, None, ctypes.byref(v))  # PDH_FMT_DOUBLE
    pdh.PdhCloseQuery(q)
    return v.doubleValue


def gate(need=1500, log=print):
    while True:
        a = available_mb()
        if a >= need:
            log("available MB %.0f >= %d" % (a, need))
            return a
        log("available MB %.0f < %d, waiting" % (a, need))
        time.sleep(20)


def peak_ws_mib():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
    m = PMC()
    m.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
    return m.PeakWorkingSetSize / 2 ** 20


def setup(log=print):
    set_below_normal()
    ok = cap_memory()
    log("below-normal set; job memory cap 2 GiB assigned=%s" % ok)

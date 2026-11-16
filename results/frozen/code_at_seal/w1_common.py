"""MathArena wrong-answer screening replay (frozen preregistration, 2026-10-23).

Pure logic lives here (config matching, masked precheck, namespace-hash roles, exact metrics, seed manifest and
literal scan) so the unit tests can exercise it on synthetic rows.  Rules are the preregistration's; every place
where the draft leaves an operational detail open is marked DECIDED-BEFORE-UNMASK and is fixed in this file before
the correctness column is ever read.
"""
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARD = HERE.parent
DOWNLOADS = CARD / "downloads"
DATA_DIR = DOWNLOADS / "hf_aime_2025_outputs_16e93d3"
PARQUET = DATA_DIR / "train-00000-of-00001.parquet"
META_WEB = DOWNLOADS / "meta_web"
RESULTS = CARD / "results"
FROZEN = RESULTS / "frozen"
TABLES = RESULTS / "tables"
PREREG = CARD / "预注册_第一周_冻结_2026-10-23.md"
CARDS_ROOT = CARD.parent                                  # parent directory (scanned for registered seeds)

REPO = "MathArena/aime_2025_outputs"
REV = "16e93d3bb9b0a25ea9422893464d46b3c5ff0ae9"
LFS_SHA256 = "c7980ee5518384624cb690d08b03bfff5892d19141b7ca1de666a2e863b0e589"
PARQUET_PATH_IN_REPO = "data/train-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{PARQUET_PATH_IN_REPO}"
LICENSE = "cc-by-nc-sa-4.0"
NS = "line160/v1"
CFG_A = "anthropic/claude-35"
CFG_B = "anthropic/claude-37"
IDX = (0, 1, 2, 3)
TARGET_N = 30
PREFIX, SUFFIX = "configs/models/", ".yaml"
MIN_FREE_MB = 1500
MEM_CAP = 2 * 1024 ** 3


class StopGate(Exception):
    pass


# ----------------------------------------------------------------------------- resources


def apply_limits():
    """BelowNormal priority and a 2 GiB process-memory cap (job object) on this process only."""
    rep = {}
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        rep["below_normal"] = bool(k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000))

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IOC(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("a", "b", "c", "d", "e", "f")]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.Basic.LimitFlags = 0x100                                  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = MEM_CAP
        ok1 = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        ok2 = k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
        rep["memory_cap_job"] = bool(ok1 and ok2)
        globals()["_JOB"] = job
    except Exception as e:                                             # noqa: BLE001
        rep["error"] = repr(e)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    return rep


def available_mb():
    """Windows counter \\Memory\\Available MBytes (typeperf), as the brief requires."""
    try:
        raw = subprocess.run(["typeperf", r"\Memory\Available MBytes", "-sc", "1"], capture_output=True,
                             timeout=60).stdout
        out = raw.decode("mbcs", errors="replace")
        rows = [ln for ln in out.splitlines() if ln.startswith('"') and re.match(r'^"\d', ln)]
        return float(rows[-1].split(",")[1].strip().strip('"'))
    except Exception:                                                  # noqa: BLE001
        return None


def wait_ram(log, label, min_mb=MIN_FREE_MB, max_wait_s=6 * 3600):
    t0 = time.time()
    while True:
        mb = available_mb()
        if mb is not None and mb >= min_mb:
            log(f"RAM ok before {label}: {mb:.0f} MB available")
            return mb
        if time.time() - t0 > max_wait_s:
            raise StopGate(f"RAM below {min_mb} MB for {max_wait_s}s before {label}")
        log(f"waiting for RAM before {label}: {mb} MB available")
        time.sleep(60)


def peak_rss_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 2 ** 20, 1), round(c.PeakPagefileUsage / 2 ** 20, 1)
    except Exception:                                                  # noqa: BLE001
        return None, None


# ----------------------------------------------------------------------------- small io


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def write_json(path, obj, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and frozen:
        raise StopGate(f"refusing to overwrite sealed file {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False, default=str), encoding="utf8")
    os.replace(tmp, path)
    if frozen:
        os.chmod(path, stat.S_IREAD)


def write_csv(path, header, rows, frozen=False):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and frozen:
        raise StopGate(f"refusing to overwrite sealed file {path}")
    with open(path, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    if frozen:
        os.chmod(path, stat.S_IREAD)


def is_read_only(path):
    return Path(path).is_file() and not os.access(path, os.W_OK)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def now_local():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------- config identity


def normalize_config(path):
    """Only the fixed prefix configs/models/ and the suffix .yaml may be removed; nothing else is matched."""
    s = path
    if s.startswith(PREFIX):
        s = s[len(PREFIX):]
    if s.endswith(SUFFIX):
        s = s[:-len(SUFFIX)]
    return s


# ----------------------------------------------------------------------------- masked precheck

# DECIDED-BEFORE-UNMASK (operational definitions the draft leaves open):
#  * candidate problems = all distinct (source, problem_idx) pairs present anywhere in the pinned revision;
#  * "unique and complete idx=0,1,2,3": each of 0..3 exactly once after collapsing fully identical duplicate rows;
#    a duplicate key whose rows differ in any column is a conflict; any idx outside 0..3 is a conflict;
#  * explicit technical failure / no valid response = answer null or whitespace-only AND output_tokens null or 0;
#    an empty visible answer with output_tokens > 0 is a saved truncation and stays, scored by the locked correct;
#  * grading field missing = correct is null (a null parsed_answer with non-null correct is a scored attempt);
#  * pairing: one SHA256(problem) value and one gold_answer value across all kept A and B rows of the problem.

ANSWER_OK, ANSWER_EMPTY_TRUNC, ANSWER_FAIL = "ok", "empty_visible_answer_tokens>0", "no_valid_response"


def answer_status(answer, output_tokens):
    if answer is not None and answer.strip():
        return ANSWER_OK
    if output_tokens is not None and output_tokens > 0:
        return ANSWER_EMPTY_TRUNC
    return ANSWER_FAIL


def precheck(rows, candidates, cfg_a=CFG_A, cfg_b=CFG_B):
    """rows: dicts with row, cfg (normalized), source, problem_idx, idx, problem_sha, gold, answer_status,
    correct_is_null, row_digest.  Returns (F, exclusions, kept, slot_status) without touching correctness values."""
    by = {}
    for r in rows:
        if r["cfg"] in (cfg_a, cfg_b):
            by.setdefault((r["cfg"], r["source"], r["problem_idx"]), []).append(r)
    F, exclusions, kept, slot_status = [], {}, {}, []
    for q in candidates:
        reasons = []
        keep_q = {}
        for cfg, side in ((cfg_a, "A"), (cfg_b, "B")):
            rs = by.get((cfg, q[0], q[1]), [])
            groups = {}
            for r in rs:
                groups.setdefault(r["idx"], []).append(r)
            for k in sorted(groups, key=lambda x: (x is None, x)):
                g = groups[k]
                if k not in IDX:
                    reasons.append(f"{side}: unexpected idx {k}")
                    for r in g:
                        slot_status.append((side, q, k, r["row"], "conflict_unexpected_idx"))
                    continue
                if len(g) > 1 and len({r["row_digest"] for r in g}) > 1:
                    reasons.append(f"{side}: conflicting duplicate rows for idx {k}")
                    for r in g:
                        slot_status.append((side, q, k, r["row"], "conflict_duplicate"))
                    continue
                first = min(g, key=lambda r: r["row"])
                for r in g:
                    if r is not first:
                        slot_status.append((side, q, k, r["row"], f"identical_duplicate_collapsed_into_row_{first['row']}"))
                keep_q[(side, k)] = first
            for k in IDX:
                if k not in groups:
                    reasons.append(f"{side}: missing idx {k}")
                    slot_status.append((side, q, k, None, "missing"))
            for k in IDX:
                r = keep_q.get((side, k))
                if r is None:
                    continue
                st = []
                if r["answer_status"] == ANSWER_FAIL:
                    reasons.append(f"{side}: idx {k} no valid response / technical failure")
                    st.append("no_valid_response")
                if r["correct_is_null"]:
                    reasons.append(f"{side}: idx {k} grading field (correct) missing")
                    st.append("grade_missing")
                if r["answer_status"] == ANSWER_EMPTY_TRUNC:
                    st.append("saved_truncation_empty_visible_answer_kept")
                slot_status.append((side, q, k, r["row"], "+".join(st) if st else "ok"))
        texts = {r["problem_sha"] for r in keep_q.values()}
        golds = {r["gold"] for r in keep_q.values()}
        if len(texts) > 1:
            reasons.append("problem text hash differs across kept rows")
        if len(golds) > 1:
            reasons.append("gold_answer differs across kept rows")
        if reasons:
            exclusions[q] = reasons
        else:
            F.append(q)
            kept[q] = {key: r["row"] for key, r in keep_q.items()}
    return F, exclusions, kept, slot_status


# ----------------------------------------------------------------------------- namespace-hash roles


def ns_string(source, problem_idx, idx, rev=REV):
    return f"{NS}|{rev}|{source}|{problem_idx}|{idx}"


def ns_digest(source, problem_idx, idx, rev=REV):
    return hashlib.sha256(ns_string(source, problem_idx, idx, rev).encode("utf8")).digest()


def role_order(q, digest_fn=None):
    """p0..p3: the four idx sorted by the SHA256 digest bytes, ties by idx ascending."""
    digest_fn = digest_fn or (lambda k: ns_digest(q[0], q[1], k))
    return sorted(IDX, key=lambda k: (digest_fn(k), k))


def sha63(digest):
    """Project seed convention (other lines): first 8 bytes little-endian, top bit cleared."""
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def u64be(digest):
    return int.from_bytes(digest[:8], "big")


def seed_manifest(candidates):
    ents = []
    for q in candidates:
        order = role_order(q)
        for k in IDX:
            d = ns_digest(q[0], q[1], k)
            ents.append(dict(source=q[0], problem_idx=q[1], idx=k, string=ns_string(q[0], q[1], k),
                             sha256=d.hex(), sha63=sha63(d), u64be=u64be(d), position=order.index(k)))
    return ents


# ----------------------------------------------------------------------------- registered-seed scan

TEXT_EXT = {".py", ".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".cfg", ".ini",
            ".log", ".out", ".sh", ".ps1", ".r", ".ipynb", ".err", ".status"}
LIT = re.compile(rb"(?<![0-9])([0-9]{10,20})(?![0-9])")
HEX64 = re.compile(rb"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])")
MAX_SCAN_BYTES = 20 * 1024 * 1024


def read_shared(path, max_bytes=MAX_SCAN_BYTES):
    """Read with FILE_SHARE_READ|WRITE|DELETE so no other job's write/rename/delete is ever blocked."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if h is None or h == wintypes.HANDLE(-1).value:
        return None
    try:
        size = ctypes.c_int64()
        if not k32.GetFileSizeEx(h, ctypes.byref(size)) or size.value > max_bytes:
            return None
        buf = ctypes.create_string_buffer(max(size.value, 1))
        got = wintypes.DWORD()
        if size.value and not k32.ReadFile(h, buf, size.value, ctypes.byref(got), None):
            return None
        return buf.raw[:got.value]
    finally:
        k32.CloseHandle(h)


def scan_tree(root, skip_dirs, wanted_dec, wanted_hex, ns_token=b"line160/"):
    """Walk `root`; for text files <= 20 MB find 10-20 digit literals in wanted_dec, 64-hex literals in wanted_hex
    and the namespace token.  Lock files and *lock* folders are skipped; nothing is opened for writing."""
    skip_dirs = [os.path.normcase(str(Path(s).resolve())) for s in skip_dirs]
    hits_dec, hits_hex, ns_files = {}, {}, []
    n_files = n_big = n_unreadable = n_literals = 0
    seed_files = []
    for dp, dns, fns in os.walk(root):
        ndp = os.path.normcase(os.path.abspath(dp))
        if any(ndp == s or ndp.startswith(s + os.sep) for s in skip_dirs):
            dns[:] = []
            continue
        dns[:] = [d for d in dns if "lock" not in d.lower()]
        for fn in fns:
            if "lock" in fn.lower() or os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) > MAX_SCAN_BYTES:
                    n_big += 1
                    continue
            except OSError:
                n_unreadable += 1
                continue
            b = read_shared(p)
            if b is None:
                n_unreadable += 1
                continue
            n_files += 1
            if "seed" in fn.lower():
                seed_files.append(os.path.relpath(p, root))
            for m in LIT.finditer(b):
                n_literals += 1
                v = int(m.group(1))
                if v in wanted_dec:
                    hits_dec.setdefault(v, set()).add(os.path.relpath(p, root))
            for m in HEX64.finditer(b):
                v = m.group(1).decode().lower()
                if v in wanted_hex:
                    hits_hex.setdefault(v, set()).add(os.path.relpath(p, root))
            if ns_token in b:
                ns_files.append(os.path.relpath(p, root))
    return dict(n_text_files=n_files, n_skipped_over_20MB=n_big, n_unreadable=n_unreadable,
                n_literal_occurrences=n_literals, seed_named_files=sorted(seed_files),
                hits_dec={str(k): sorted(v) for k, v in hits_dec.items()},
                hits_hex={k: sorted(v) for k, v in hits_hex.items()}, ns_token_files=sorted(ns_files))


def registered_ranges(root):
    """(name, lo, hi) ranges recorded in seed_check.json files anywhere under root."""
    out = []
    for p in Path(root).rglob("seed_check.json"):
        b = read_shared(p)
        if b is None:
            continue
        try:
            d = json.loads(b.decode("utf8"))
        except ValueError:
            continue
        for x in d.get("registered_ranges", []) or []:
            if isinstance(x, (list, tuple)) and len(x) == 3:
                out.append((f"{p.relative_to(root)}:{x[0]}", int(x[1]), int(x[2])))
        for x in d.get("ranges", []) or []:
            if isinstance(x, dict) and "lo" in x and "hi" in x:
                out.append((f"{p.relative_to(root)}:{x.get('name')}", int(x["lo"]), int(x["hi"])))
    return out


# ----------------------------------------------------------------------------- exact metrics


def mean(xs):
    xs = list(xs)
    return None if not xs else sum(xs, Fraction(0)) / len(xs)


def replay_metrics(F, order, corr, j=0):
    """corr[(side, q, idx)] in {0,1}.  Rotation j: screening slot = order[q][j], rescoring = the other three."""
    per = []
    for q in F:
        o = order[q]
        scr = o[j]
        rest = [k for k in o if k != scr]
        a0 = Fraction(corr[("A", q, scr)])
        a1 = mean(Fraction(corr[("A", q, k)]) for k in rest)
        b1 = mean(Fraction(corr[("B", q, k)]) for k in rest)
        per.append(dict(q=q, screen_idx=scr, rescore_idx=rest, A0=a0, A1=a1, B1=b1, in_S=(a0 == 0)))
    S = [p for p in per if p["in_S"]]
    K = len(S)
    out = dict(rotation=j, K=K, F=len(F), S=[p["q"] for p in S])
    if K:
        out.update(A1_S=mean(p["A1"] for p in S), B1_S=mean(p["B1"] for p in S),
                   G_naive=mean(p["B1"] - p["A0"] for p in S), R=mean(p["A1"] - p["A0"] for p in S),
                   D=mean(p["B1"] - p["A1"] for p in S))
        assert out["G_naive"] - out["R"] == out["D"]
    else:
        out.update(A1_S=None, B1_S=None, G_naive=None, R=None, D=None)
    out["D_F"] = mean(p["B1"] - p["A1"] for p in per)
    out["D_minus_D_F"] = None if out["D"] is None or out["D_F"] is None else out["D"] - out["D_F"]
    return out, per


def fr(x):
    """Exact fraction and decimal for JSON."""
    if x is None:
        return None
    return dict(exact=f"{x.numerator}/{x.denominator}", value=round(float(x), 6))

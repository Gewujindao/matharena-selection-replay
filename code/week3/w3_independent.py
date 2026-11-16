"""Week 3 independent recomputation (frozen Part 1 item 7: 另写代码，不导入运行者的代码).

Separate code path: its own parquet decoding written from the format spec (no import of pq_reader, w1_*, w2_* or
w3_*), its own reading of the tier / base-endpoint / adjacency rule, its own per-cell contract (complete frame,
idx 0-3 both sides, one problem text and gold, identical full prompt over the 8 slots, stored correct non-null,
response status), its own exact arithmetic for every cell (primary slot idx0 and the four-slot cycle) and for the
frozen aggregates (all cells, excluding seen cells), then a comparison with week3/frame/frame.json and
week3/summary.json.  Written by the same executor; code path separate, person not.

Run:  python -X utf8 -I -B w3_independent.py
"""
import ctypes
import json
import os
import re
import struct
import sys
import time
import zlib
from ctypes import wintypes
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "downloads"
FILES = {"AIME": DL / "w3_hf_aime_2025_outputs_16e93d3" / "train-00000-of-00001.parquet",
         "HMMT": DL / "hf_hmmt_feb_2025_outputs_e3fb6c7" / "train-00000-of-00001.parquet",
         "BRUMO": DL / "w3_hf_brumo_2025_outputs_3b234ef" / "train-00000-of-00001.parquet",
         "CMIMC": DL / "w3_hf_cmimc_2025_outputs_2782618" / "train-00000-of-00001.parquet"}
ORDER = ["AIME", "HMMT", "BRUMO", "CMIMC"]
OUT = ROOT / "week3" / "independent_recompute.json"
TIERS = ("minimal", "low", "medium", "high", "xhigh")
SEEN_PAIRS = {("HMMT", "openai/o3-mini--medium", "openai/o3-mini")}


# ================================================================ resources (own implementation)
def limit_self():
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    me = k.GetCurrentProcess()
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    prio = bool(k.SetPriorityClass(me, 0x00004000))

    class LIM(ctypes.Structure):
        _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("flags", wintypes.DWORD),
                    ("mn", ctypes.c_size_t), ("mx", ctypes.c_size_t), ("apl", wintypes.DWORD),
                    ("aff", ctypes.c_size_t), ("pc", wintypes.DWORD), ("sc", wintypes.DWORD),
                    ("io", ctypes.c_ulonglong * 6), ("proc_mem", ctypes.c_size_t), ("job_mem", ctypes.c_size_t),
                    ("peak_p", ctypes.c_size_t), ("peak_j", ctypes.c_size_t)]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k.CreateJobObjectW(None, None)
    lim = LIM()
    lim.flags = 0x00000100
    lim.proc_mem = 2 * 1024 ** 3
    ok = bool(k.SetInformationJobObject(job, 9, ctypes.byref(lim), ctypes.sizeof(lim))
              and k.AssignProcessToJobObject(job, me))
    globals()["_job_handle"] = job
    return dict(below_normal=prio, job_memory_cap_2GiB=ok)


def avail_mb():
    pdh = ctypes.windll.pdh
    query, counter = ctypes.c_void_p(), ctypes.c_void_p()
    pdh.PdhOpenQueryW(None, None, ctypes.byref(query))
    pdh.PdhAddEnglishCounterW(query, "\\Memory\\Available MBytes", None, ctypes.byref(counter))
    pdh.PdhCollectQueryData(query)
    buf = (ctypes.c_double * 2)()                 # PDH_FMT_COUNTERVALUE: DWORD status (+pad), double value
    pdh.PdhGetFormattedCounterValue(counter, 0x200, None, buf)
    pdh.PdhCloseQuery(query)
    return buf[1]


def gate_ram(label):
    t = time.time()
    while True:
        v = avail_mb()
        if v >= 1500:
            print(f"RAM {v:.0f} MB before {label}", flush=True)
            return v
        if time.time() - t > 3600:
            raise SystemExit(f"RAM below 1500 MB for an hour before {label}")
        time.sleep(30)


def peak_ws_mib():
    class C(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("pf", wintypes.DWORD), ("peak", ctypes.c_size_t)] + \
                   [(f"x{i}", ctypes.c_size_t) for i in range(7)]
    c = C()
    c.cb = ctypes.sizeof(c)
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.peak / 2 ** 20, 1)


# ================================================================ parquet, from the spec
class Buf:
    __slots__ = ("b", "i")

    def __init__(self, b, i=0):
        self.b, self.i = b, i

    def uleb(self):
        r = sh = 0
        while True:
            x = self.b[self.i]
            self.i += 1
            r |= (x & 0x7F) << sh
            if x < 0x80:
                return r
            sh += 7

    def sint(self):
        u = self.uleb()
        return -(u >> 1) - 1 if u & 1 else u >> 1


def th_read(buf, t):
    if t == 1:
        return True
    if t == 2:
        return False
    if t == 3:
        v = buf.b[buf.i]
        buf.i += 1
        return v - 256 if v >= 128 else v
    if t in (4, 5, 6):
        return buf.sint()
    if t == 7:
        v = struct.unpack_from("<d", buf.b, buf.i)[0]
        buf.i += 8
        return v
    if t == 8:
        n = buf.uleb()
        v = bytes(buf.b[buf.i:buf.i + n])
        buf.i += n
        return v
    if t in (9, 10):
        h = buf.b[buf.i]
        buf.i += 1
        n = h >> 4
        et = h & 15
        if n == 15:
            n = buf.uleb()
        if et in (1, 2):
            vals = []
            for _ in range(n):
                vals.append(buf.b[buf.i] == 1)
                buf.i += 1
            return vals
        return [th_read(buf, et) for _ in range(n)]
    if t == 11:
        n = buf.uleb()
        if not n:
            return {}
        kv = buf.b[buf.i]
        buf.i += 1
        return {th_read(buf, kv >> 4): th_read(buf, kv & 15) for _ in range(n)}
    if t == 12:
        return th_struct(buf)
    raise ValueError(f"thrift type {t}")


def th_struct(buf):
    d, fid = {}, 0
    while True:
        h = buf.b[buf.i]
        buf.i += 1
        if h == 0:
            return d
        delta, t = h >> 4, h & 15
        fid = fid + delta if delta else buf.sint()
        d[fid] = th_read(buf, t)


def snappy(data):
    s = Buf(data)
    total = s.uleb()
    i, out = s.i, bytearray()
    n = len(data)
    while i < n:
        tag = data[i]
        i += 1
        kind = tag & 3
        if kind == 0:
            ln = (tag >> 2) + 1
            if ln > 60:
                extra = ln - 60
                ln = int.from_bytes(data[i:i + extra], "little") + 1
                i += extra
            out += data[i:i + ln]
            i += ln
        else:
            if kind == 1:
                ln = 4 + ((tag >> 2) & 7)
                off = ((tag & 0xE0) << 3) | data[i]
                i += 1
            elif kind == 2:
                ln = 1 + (tag >> 2)
                off = data[i] | data[i + 1] << 8
                i += 2
            else:
                ln = 1 + (tag >> 2)
                off = int.from_bytes(data[i:i + 4], "little")
                i += 4
            p = len(out) - off
            if off <= 0 or p < 0:
                raise ValueError("snappy offset")
            while ln > 0:                               # overlap: the copied span doubles on each pass
                take = min(ln, len(out) - p)
                out += out[p:p + take]
                ln -= take
    if len(out) != total:
        raise ValueError("snappy size")
    return bytes(out)


def inflate(codec, data):
    if codec == 0:
        return bytes(data)
    if codec == 1:
        return snappy(bytes(data))
    if codec == 2:
        return zlib.decompress(bytes(data), 31)
    raise ValueError(f"codec {codec}")


def hybrid(b, i, end, width, count):
    vals = []
    nb = (width + 7) // 8
    while len(vals) < count and i < end:
        s = Buf(b, i)
        head = s.uleb()
        i = s.i
        if head & 1:
            groups = head >> 1
            nbytes = groups * width
            bits = int.from_bytes(b[i:i + nbytes], "little")
            i += nbytes
            m = (1 << width) - 1
            vals += [(bits >> (width * j)) & m for j in range(groups * 8)]
        else:
            rep = head >> 1
            v = int.from_bytes(b[i:i + nb], "little") if nb else 0
            i += nb
            vals += [v] * rep
    if len(vals) < count:
        raise ValueError("hybrid short")
    return vals[:count]


def plain_vals(b, ptype, n):
    if ptype == 6:
        out, i = [], 0
        for _ in range(n):
            ln = struct.unpack_from("<I", b, i)[0]
            out.append(bytes(b[i + 4:i + 4 + ln]))
            i += 4 + ln
        return out
    if ptype == 0:
        return [(b[j >> 3] >> (j & 7)) & 1 == 1 for j in range(n)]
    fmt = {1: "i", 2: "q", 5: "d"}[ptype]
    return list(struct.unpack_from(f"<{n}{fmt}", b, 0))


class PQ:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            f.seek(-8, 2)
            tail = f.read(8)
            assert tail[4:] == b"PAR1"
            ln = struct.unpack("<I", tail[:4])[0]
            f.seek(-8 - ln, 2)
            self.fm = th_struct(Buf(f.read(ln)))
        self.leaf = {}
        for el in self.fm[2][1:]:
            self.leaf[el[4].decode()] = (el.get(1), el.get(3, 0))
        self.n = self.fm[3]

    def values(self, col, wanted=None, keep=lambda v: v):
        """{row: keep(value)} for rows in `wanted` (all rows if None)."""
        ptype, rep = self.leaf[col]
        nullable = rep == 1
        want = None if wanted is None else set(wanted)
        res = {}
        row0 = 0
        with open(self.path, "rb") as f:
            for rg in self.fm[4]:
                nrows = rg[3]
                if want is not None and not any(row0 <= r < row0 + nrows for r in want):
                    row0 += nrows
                    continue
                md = [c for c in rg[1] if b".".join(c[3][3]).decode() == col][0][3]
                off = md[9] if 11 not in md else min(md[9], md[11])
                f.seek(off)
                blob = f.read(md[7])
                s = Buf(blob)
                dic, seen = None, 0
                while seen < md[5]:
                    hd = th_struct(s)
                    body = blob[s.i:s.i + hd[3]]
                    s.i += hd[3]
                    if hd[1] == 2:
                        dic = plain_vals(inflate(md[4], body), ptype, hd[7][1])
                        continue
                    if hd[1] == 0:
                        nv = hd[5][1]
                        enc = hd[5][2]
                        r0 = row0 + seen
                        if want is not None and not any(r0 <= r < r0 + nv for r in want):
                            seen += nv
                            continue
                        raw = inflate(md[4], body)
                        p = 0
                        if nullable:
                            dl = struct.unpack_from("<I", raw, 0)[0]
                            defs = hybrid(raw, 4, 4 + dl, 1, nv)
                            p = 4 + dl
                        else:
                            defs = [1] * nv
                        vb = raw[p:]
                    elif hd[1] == 3:
                        h2 = hd[8]
                        nv, enc, dlen, rlen = h2[1], h2[4], h2[5], h2[6]
                        r0 = row0 + seen
                        if want is not None and not any(r0 <= r < r0 + nv for r in want):
                            seen += nv
                            continue
                        defs = hybrid(body, rlen, rlen + dlen, 1, nv) if nullable else [1] * nv
                        rest = body[rlen + dlen:]
                        vb = inflate(md[4], rest) if h2.get(7, True) else bytes(rest)
                    else:
                        raise ValueError("page type")
                    k = sum(defs)
                    if enc == 0:
                        vals = plain_vals(vb, ptype, k)
                    elif enc in (2, 8):
                        vals = [dic[j] for j in hybrid(vb, 1, len(vb), vb[0], k)]
                    elif enc == 3:
                        ln = struct.unpack_from("<I", vb, 0)[0]
                        vals = [x == 1 for x in hybrid(vb, 4, 4 + ln, 1, k)]
                    else:
                        raise ValueError(f"encoding {enc}")
                    it = iter(vals)
                    for j, d in enumerate(defs):
                        v = next(it) if d else None
                        r = r0 + j
                        if want is None or r in want:
                            if isinstance(v, bytes) and ptype == 6:
                                v = v.decode("utf-8")
                            res[r] = keep(v)
                    seen += nv
                row0 += nrows
        return res


# ================================================================ rules (independent reading of the frozen text)
def tier_info(cfg, name):
    m = re.fullmatch(r"(.*?)\s*\(([^()]*)\)", name)
    tag_n = m.group(2).lower() if m and m.group(2).lower() in TIERS else None
    disp = m.group(1) if tag_n else name
    path = cfg
    if path.startswith("configs/models/"):
        path = path[len("configs/models/"):]
    if path.endswith(".yaml"):
        path = path[:-5]
    tag_c, cbase = None, path
    last = path.split("/")[-1]
    if "--" in last:
        suf = last.split("--")[-1]
        if suf.lower() in TIERS:
            tag_c, cbase = suf.lower(), path[:len(path) - len(suf) - 2]
    if tag_n and tag_c and tag_n != tag_c:
        return None, None, "conflict"
    t = tag_n or tag_c
    return t, ((disp, cbase) if t else None), ("ok" if t else "untiered")


def pairs_of(cfg_to_names):
    groups, notes = {}, {}
    for cfg, names in cfg_to_names.items():
        if len(names) != 1:
            notes[cfg] = "multi-name"
            continue
        t, base, st = tier_info(cfg, next(iter(names)))
        notes[cfg] = st
        if t:
            groups.setdefault(base, {}).setdefault(t, []).append(cfg)
    out = []
    for base, bt in groups.items():
        if any(len(v) > 1 for v in bt.values()):
            continue
        for i in range(len(TIERS) - 1):
            lo, hi = TIERS[i], TIERS[i + 1]
            if lo in bt and hi in bt:
                out.append((base, lo, hi, bt[lo][0], bt[hi][0]))
    return sorted(out), notes


def prompt_key(user_message, all_messages):
    if not user_message or not user_message.strip():
        return None
    try:
        msgs = json.loads(all_messages)
    except Exception:                               # noqa: BLE001
        return None
    if not isinstance(msgs, list):
        return None
    pre = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "assistant")]
    if not any(isinstance(m, dict) and m.get("role") == "user" for m in pre):
        return None
    return json.dumps({"u": user_message, "m": pre}, sort_keys=True, ensure_ascii=False)


def avg(xs):
    xs = list(xs)
    return None if not xs else sum(xs, Q(0)) / len(xs)


def metrics(probs, a, b, s):
    """a[q][k], b[q][k] in {0,1}; screening slot s."""
    rest = [k for k in range(4) if k != s]
    A0 = {q: Q(a[q][s]) for q in probs}
    A1 = {q: Q(sum(a[q][k] for k in rest), 3) for q in probs}
    B1 = {q: Q(sum(b[q][k] for k in rest), 3) for q in probs}
    S = [q for q in probs if A0[q] == 0]
    dF = avg(B1[q] - A1[q] for q in probs)
    if not S:
        return dict(K=0, G=None, R=None, D=None, D_F=dF, C=None, GmDF=None, flag=None)
    G = avg(B1[q] - A0[q] for q in S)
    R = avg(A1[q] - A0[q] for q in S)
    D = avg(B1[q] - A1[q] for q in S)
    return dict(K=len(S), G=G, R=R, D=D, D_F=dF, C=D - dF, GmDF=G - dF, flag=int(D * dF < 0))


def macro(cells, key, use):
    per_comp = []
    comp_vals = {}
    for comp in ORDER:
        eps = {}
        for c in cells:
            if c["comp"] == comp and use(c) and c["valid"] and c["m"][key] is not None:
                eps.setdefault(c["ep"], []).append(c["m"][key])
        ev = [avg(v) for v in eps.values()]
        cv = avg(ev)
        comp_vals[comp] = cv
        if cv is not None:
            per_comp.append(cv)
    return avg(per_comp), comp_vals


def s(x):
    return None if x is None else f"{x.numerator}/{x.denominator}"


def main():
    t0 = time.time()
    lim = limit_self()
    print("independent start", os.getpid(), lim, flush=True)
    cells = []
    for comp in ORDER:
        gate_ram(comp)
        pq = PQ(FILES[comp])
        meta = {c: pq.values(c) for c in ("model_config", "model_name", "problem_idx", "idx_answer", "output_tokens")}
        names = {}
        for r in range(pq.n):
            names.setdefault(meta["model_config"][r], set()).add(meta["model_name"][r])
        frame_q = sorted(set(meta["problem_idx"].values()), key=lambda x: int(x))
        prs, _ = pairs_of(names)
        used = {x for p in prs for x in p[3:]}
        rows = [r for r in range(pq.n) if meta["model_config"][r] in used]
        txt = {c: pq.values(c, rows) for c in ("problem", "gold_answer", "user_message")}
        am = pq.values("all_messages", rows, keep=lambda v: v)
        pk = {r: prompt_key(txt["user_message"][r], am[r]) for r in rows}
        del am
        ans = pq.values("answer", rows, keep=lambda v: bool(v and v.strip()))
        cor = pq.values("correct", rows)
        for base, lo, hi, ca, cb in prs:
            slot = {}
            dup = False
            for r in rows:
                cf = meta["model_config"][r]
                if cf not in (ca, cb):
                    continue
                key = ("A" if cf == ca else "B", meta["problem_idx"][r], meta["idx_answer"][r])
                if key in slot:
                    dup = True
                slot[key] = r
            bad = {}
            for q in frame_q:
                why = []
                rs = []
                for side in "AB":
                    for k in range(4):
                        r = slot.get((side, q, k))
                        if r is None:
                            why.append("missing")
                            continue
                        rs.append(r)
                        tok = meta["output_tokens"][r]
                        if not ans[r] and not (tok is not None and tok > 0):
                            why.append("unknown status")
                        if cor[r] is None:
                            why.append("correct null")
                        if pk[r] is None:
                            why.append("prompt incomplete")
                        if not (txt["gold_answer"][r] or "").strip():
                            why.append("gold empty")
                if any(qq == q and k not in range(4) for (sd, qq, k) in slot):
                    why.append("idx outside 0-3")
                if rs:
                    if len({txt["problem"][r] for r in rs}) > 1:
                        why.append("problem text")
                    if len({txt["gold_answer"][r] for r in rs}) > 1:
                        why.append("gold")
                    if len({pk[r] for r in rs}) > 1:
                        why.append("prompt differs")
                if why:
                    bad[q] = sorted(set(why))
            extra_idx = [k for (sd, qq, k) in slot if k not in range(4)]
            valid = not bad and not dup and not extra_idx
            cid = f"{comp}|{base[0]}|{lo}->{hi}"
            c = dict(cid=cid, comp=comp, ep=f"{base[0]} | {base[1]}", A=ca, B=cb, valid=valid,
                     seen=(comp, ca, cb) in SEEN_PAIRS, n_bad=len(bad),
                     bad_reasons=sorted({x for v in bad.values() for x in v}), m=None, cyc=None)
            if valid:
                a = {q: [int(cor[slot[("A", q, k)]]) for k in range(4)] for q in frame_q}
                b = {q: [int(cor[slot[("B", q, k)]]) for k in range(4)] for q in frame_q}
                c["m"] = metrics(frame_q, a, b, 0)
                c["cyc"] = [metrics(frame_q, a, b, j) for j in range(4)]
            cells.append(c)
            print(cid, valid, c["m"] and {k: (s(v) if isinstance(v, Q) else v) for k, v in c["m"].items()}, flush=True)
        del txt, ans, cor, pk, meta

    keys = ["G", "R", "D", "D_F", "C", "GmDF"]
    agg = {}
    for lab, use in (("all", lambda c: True), ("excluding_seen", lambda c: not c["seen"])):
        agg[lab] = {}
        for k in keys:
            m, cv = macro(cells, k, use)
            agg[lab][k] = dict(macro=s(m), competitions={x: s(v) for x, v in cv.items()})
        den = [c for c in cells if use(c) and c["valid"] and c["m"]["D"] is not None]
        agg[lab]["reversal"] = dict(n=sum(c["m"]["flag"] for c in den), den=len(den))

    # ------------------------------------------------------------- compare with the runner's files
    frame = json.loads((ROOT / "week3" / "frame" / "frame.json").read_text(encoding="utf8"))
    summ = json.loads((ROOT / "week3" / "summary.json").read_text(encoding="utf8"))
    mism = []
    fcells = {c["cell_id"]: c for c in frame["cells"]}
    scells = {c["cell_id"]: c for c in summ["cells"]}
    if set(fcells) != {c["cid"] for c in cells}:
        mism.append(dict(what="cell list", runner=sorted(fcells), independent=sorted(c["cid"] for c in cells)))
    m2 = dict(G="G", R="R", D="D", D_F="D_F", C="C", GmDF="G_minus_D_F")
    for c in cells:
        fc, sc = fcells.get(c["cid"]), scells.get(c["cid"])
        if fc is None:
            continue
        if fc["contract_valid"] != c["valid"]:
            mism.append(dict(cell=c["cid"], what="contract", runner=fc["contract_valid"], independent=c["valid"]))
        if fc["n_problems_breaking"] != c["n_bad"]:
            mism.append(dict(cell=c["cid"], what="n problems breaking", runner=fc["n_problems_breaking"],
                             independent=c["n_bad"]))
        if fc["seen"] != c["seen"]:
            mism.append(dict(cell=c["cid"], what="seen flag"))
        if c["valid"]:
            for j, (mi, ro) in enumerate(zip(c["cyc"], sc["cycle"])):
                if mi["K"] != ro["K"]:
                    mism.append(dict(cell=c["cid"], slot=j, what="K", runner=ro["K"], independent=mi["K"]))
                for k, rk in m2.items():
                    rv = ro[rk]["exact"] if ro[rk] else None
                    if s(mi[k]) != rv and not (mi[k] is not None and rv is not None and Q(rv) == mi[k]):
                        mism.append(dict(cell=c["cid"], slot=j, what=k, runner=rv, independent=s(mi[k])))
                if (mi["flag"] or 0) != (ro["sign_reversal"] or 0):
                    mism.append(dict(cell=c["cid"], slot=j, what="flag"))
    for lab, rlab in (("all", "aggregates_all"), ("excluding_seen", "aggregates_excluding_seen")):
        for k, rk in m2.items():
            r = summ[rlab][rk]
            rv = r["macro"]["exact"] if r["macro"] else None
            if (rv is None) != (agg[lab][k]["macro"] is None) or (rv and Q(rv) != Q(agg[lab][k]["macro"])):
                mism.append(dict(set=lab, what=f"macro {k}", runner=rv, independent=agg[lab][k]["macro"]))
            for comp in ORDER:
                cvr = r["competitions"][comp]["value"]
                cvr = cvr["exact"] if cvr else None
                cvi = agg[lab][k]["competitions"][comp]
                if (cvr is None) != (cvi is None) or (cvr and Q(cvr) != Q(cvi)):
                    mism.append(dict(set=lab, what=f"{comp} {k}", runner=cvr, independent=cvi))
        rr = summ["sign_reversal_all" if lab == "all" else "sign_reversal_excluding_seen"]
        if (rr["n_sign_reversal"], rr["denominator_cells_with_D_and_D_F_defined"]) != \
                (agg[lab]["reversal"]["n"], agg[lab]["reversal"]["den"]):
            mism.append(dict(set=lab, what="sign reversal count"))
    res = dict(
        note="独立重算：自写 parquet 解码（Thrift compact、Snappy、RLE/位打包、字典编码、v1/v2 数据页），不导入 pq_reader、"
             "w1_*、w2_*、w3_* 代码；自做档位/端点/相邻规则、逐 cell 合同、精确分数运算与冻结汇总；与 frame.json、summary.json 对比。"
             "由同一执行者写成，代码路径独立，人员不独立。",
        cells=[dict(cell=c["cid"], valid=c["valid"], seen=c["seen"], n_problems_breaking=c["n_bad"],
                    breaking_reasons=c["bad_reasons"],
                    primary=None if c["m"] is None else {k: (s(v) if isinstance(v, Q) else v) for k, v in c["m"].items()})
               for c in cells],
        aggregates=agg, mismatches=mism, all_agree=not mism, seconds=round(time.time() - t0, 2),
        peak_working_set_mib=peak_ws_mib(), limits=lim, pid=os.getpid())
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print("mismatches:", len(mism), "all_agree:", not mism, "seconds:", res["seconds"], "peak MiB:",
          res["peak_working_set_mib"], flush=True)


if __name__ == "__main__":
    main()

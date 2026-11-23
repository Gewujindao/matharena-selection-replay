"""HMMT o3-mini run: independent recomputation of the primary quantities.

Separate code path: its own parquet decoding written from the format spec (no import of pq_reader or of any other module in code/),
its own config selection (model_name equals the official display names), its own completeness check and its own
arithmetic.  It reads only model_name, model_config, problem_idx, idx_answer and correct, then compares with
results/hmmt_o3mini_pair/summary.json.

Run:  python -X utf8 -I -B independent_recompute.py
"""
import json
import os
import struct
import sys
import zlib
from fractions import Fraction as Q
from pathlib import Path

CARD = Path(__file__).resolve().parents[2]
PQ = CARD / "downloads" / "hf_hmmt_feb_2025_outputs_e3fb6c7" / "train-00000-of-00001.parquet"
OUT = CARD / "results/hmmt_o3mini_pair" / "independent_recompute.json"
NAMES = {"A": "o3-mini (medium)", "B": "o3-mini (high)"}


# ---------------------------------------------------------------- thrift compact (generic)
def uvar(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 127) << s
        s += 7
        if c < 128:
            return v, i


def zz(n):
    return (n >> 1) ^ -(n & 1)


def tval(b, i, t):
    if t in (1, 2):
        return t == 1, i
    if t == 3:
        return struct.unpack_from("b", b, i)[0], i + 1
    if t in (4, 5, 6):
        v, i = uvar(b, i)
        return zz(v), i
    if t == 7:
        return struct.unpack_from("<d", b, i)[0], i + 8
    if t == 8:
        n, i = uvar(b, i)
        return bytes(b[i:i + n]), i + n
    if t in (9, 10):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 15
        if n == 15:
            n, i = uvar(b, i)
        out = []
        for _ in range(n):
            if et in (1, 2):
                out.append(b[i] == 1)
                i += 1
            else:
                x, i = tval(b, i, et)
                out.append(x)
        return out, i
    if t == 11:
        n, i = uvar(b, i)
        if not n:
            return {}, i
        kt, vt = b[i] >> 4, b[i] & 15
        i += 1
        d = {}
        for _ in range(n):
            k, i = tval(b, i, kt)
            d[k], i = tval(b, i, vt)
        return d, i
    if t == 12:
        return tstruct(b, i)
    raise ValueError(f"thrift type {t}")


def tstruct(b, i):
    d, fid = {}, 0
    while True:
        h = b[i]
        i += 1
        if h == 0:
            return d, i
        t = h & 15
        if h >> 4:
            fid += h >> 4
        else:
            v, i = uvar(b, i)
            fid = zz(v)
        d[fid], i = tval(b, i, t)


# ---------------------------------------------------------------- snappy (raw block)
def unsnappy(src):
    n, i = uvar(src, 0)
    out = bytearray()
    while i < len(src):
        tag = src[i]
        i += 1
        k = tag & 3
        if k == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(src[i:i + nb], "little")
                i += nb
            ln += 1
            out += src[i:i + ln]
            i += ln
            continue
        if k == 1:
            ln, off = 4 + ((tag >> 2) & 7), ((tag >> 5) << 8) | src[i]
            i += 1
        elif k == 2:
            ln, off = 1 + (tag >> 2), int.from_bytes(src[i:i + 2], "little")
            i += 2
        else:
            ln, off = 1 + (tag >> 2), int.from_bytes(src[i:i + 4], "little")
            i += 4
        st = len(out) - off
        if st < 0 or off == 0:
            raise ValueError("bad snappy offset")
        for j in range(ln):                       # byte loop handles overlap directly
            out.append(out[st + j])
    if len(out) != n:
        raise ValueError("snappy size")
    return bytes(out)


def inflate(codec, data):
    if codec == 0:
        return bytes(data)
    if codec == 1:
        return unsnappy(data)
    if codec == 2:
        return zlib.decompress(bytes(data), 47)
    raise ValueError(f"codec {codec}")


# ---------------------------------------------------------------- encodings
def hybrid(b, i, end, w, n):
    vals = []
    if w == 0:
        return [0] * n
    while len(vals) < n and i < end:
        h, i = uvar(b, i)
        if h & 1:
            ng = h >> 1
            blob = int.from_bytes(b[i:i + ng * w], "little")
            i += ng * w
            vals += [(blob >> (w * j)) & ((1 << w) - 1) for j in range(ng * 8)]
        else:
            nb = (w + 7) // 8
            vals += [int.from_bytes(b[i:i + nb], "little")] * (h >> 1)
            i += nb
    if len(vals) < n:
        raise ValueError("hybrid short")
    return vals[:n]


def plain(b, pt, n):
    if pt == 0:
        return [bool((b[j >> 3] >> (j & 7)) & 1) for j in range(n)]
    if pt == 2:
        return list(struct.unpack_from(f"<{n}q", b))
    if pt == 1:
        return list(struct.unpack_from(f"<{n}i", b))
    if pt == 6:
        out, i = [], 0
        for _ in range(n):
            ln = struct.unpack_from("<I", b, i)[0]
            out.append(bytes(b[i + 4:i + 4 + ln]))
            i += 4 + ln
        return out
    raise ValueError(f"ptype {pt}")


_RAW = {}


def column(path, name):
    if path not in _RAW:
        with open(path, "rb") as f:
            _RAW[path] = f.read()
    raw = _RAW[path]
    flen = struct.unpack_from("<I", raw, len(raw) - 8)[0]
    if raw[-4:] != b"PAR1" or raw[:4] != b"PAR1":
        raise ValueError("magic")
    meta, _ = tstruct(raw, len(raw) - 8 - flen)
    leaf = next(e for e in meta[2][1:] if e[4].decode() == name)
    pt, optional = leaf[1], leaf.get(3) == 1
    out = []
    for rg in meta[4]:
        cm = next(c[3] for c in rg[1] if b".".join(c[3][3]).decode() == name)
        codec, total = cm[4], cm[5]
        pos = min(x for x in (cm.get(9), cm.get(11)) if x is not None)
        dic, got = None, 0
        while got < total:
            ph, pos = tstruct(raw, pos)
            body = raw[pos:pos + ph[3]]
            pos += ph[3]
            if ph[1] == 2:
                dic = plain(inflate(codec, body), pt, ph[7][1])
                continue
            if ph[1] == 0:
                nv, enc = ph[5][1], ph[5][2]
                data = inflate(codec, body)
                j = 0
                if optional:
                    dl = struct.unpack_from("<I", data, 0)[0]
                    defs = hybrid(data, 4, 4 + dl, 1, nv)
                    j = 4 + dl
                else:
                    defs = [1] * nv
                vb = data[j:]
            elif ph[1] == 3:
                h2 = ph[8]
                nv, enc, dlen, rlen = h2[1], h2[4], h2[5], h2[6]
                defs = hybrid(body, rlen, rlen + dlen, 1, nv) if optional else [1] * nv
                rest = body[rlen + dlen:]
                vb = inflate(codec, rest) if h2.get(7, True) else bytes(rest)
            else:
                raise ValueError(f"page type {ph[1]}")
            k = sum(defs)
            if enc == 0:
                vals = plain(vb, pt, k)
            elif enc in (2, 8):
                vals = [dic[x] for x in hybrid(vb, 1, len(vb), vb[0], k)]
            elif enc == 3 and pt == 0:
                ln = struct.unpack_from("<I", vb, 0)[0]
                vals = [bool(x) for x in hybrid(vb, 4, 4 + ln, 1, k)]
            else:
                raise ValueError(f"encoding {enc}")
            it = iter(vals)
            out += [next(it) if d else None for d in defs]
            got += nv
    if pt == 6:
        out = [x.decode("utf8") if x is not None else None for x in out]
    if len(out) != meta[3]:
        raise ValueError("row count")
    return out


# ---------------------------------------------------------------- recompute
def main():
    cols = {c: column(PQ, c) for c in ("model_name", "model_config", "problem_idx", "idx_answer", "correct")}
    n = len(cols["correct"])
    side_cfg, cell = {}, {}
    for r in range(n):
        for s, nm in NAMES.items():
            if cols["model_name"][r] == nm:
                side_cfg.setdefault(s, set()).add(cols["model_config"][r])
                key = (s, cols["problem_idx"][r], cols["idx_answer"][r])
                cell.setdefault(key, []).append(cols["correct"][r])
    problems = sorted({p for p in cols["problem_idx"]}, key=int)
    complete = all(len(cell.get((s, p, k), [])) == 1 and cell[(s, p, k)][0] is not None
                   for s in "AB" for p in problems for k in range(4))
    extra_idx = sorted({k for (s, p, k) in cell if k not in range(4)})
    y = {key: int(v[0]) for key, v in cell.items()}
    # primary: screen on A idx0; rescore idx1-3
    S = [p for p in problems if y[("A", p, 0)] == 0]
    a1 = {p: Q(sum(y[("A", p, k)] for k in (1, 2, 3)), 3) for p in problems}
    b1 = {p: Q(sum(y[("B", p, k)] for k in (1, 2, 3)), 3) for p in problems}
    K = len(S)
    G = sum((b1[p] - y[("A", p, 0)] for p in S), Q(0)) / K if K else None
    R = sum((a1[p] - y[("A", p, 0)] for p in S), Q(0)) / K if K else None
    D = sum((b1[p] - a1[p] for p in S), Q(0)) / K if K else None
    DF = sum((b1[p] - a1[p] for p in problems), Q(0)) / len(problems)
    mine = dict(K=K, G=str(G), R=str(R), D=str(D), D_F=str(DF), D_minus_D_F=str(D - DF if D is not None else None),
                S=S)
    summ = json.loads((CARD / "results/hmmt_o3mini_pair" / "summary.json").read_text(encoding="utf8"))["primary"]
    norm = lambda s: str(Q(s)) if s not in (None, "None") else None  # noqa: E731
    theirs = dict(K=summ["K"], G=norm(summ["G"]["exact"]), R=norm(summ["R"]["exact"]), D=norm(summ["D"]["exact"]),
                  D_F=norm(summ["D_F"]["exact"]), D_minus_D_F=norm(summ["D_minus_D_F"]["exact"]), S=summ["S"])
    agree = {k: mine[k] == theirs[k] for k in mine}
    res = dict(method="Separate code path: own parquet decoding (Thrift compact, Snappy, RLE/bit-packing, dictionary), A/B selected by model_name equal to the official display name, "
                      "own completeness check and exact fraction arithmetic",
               n_rows=n, configs_by_display_name={s: sorted(v) for s, v in side_cfg.items()},
               n_problems=len(problems), complete_unique_idx0_3_nonnull=complete, idx_outside_0_3=extra_idx,
               A_all4_hits=sum(y[("A", p, k)] for p in problems for k in range(4)),
               B_all4_hits=sum(y[("B", p, k)] for p in problems for k in range(4)),
               independent=mine, main_pipeline=theirs, agree=agree, all_agree=all(agree.values()),
               identity_G_minus_R_equals_D=(G - R == D) if K else None)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({k: res[k] for k in ("configs_by_display_name", "complete_unique_idx0_3_nonnull", "independent",
                                          "all_agree")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

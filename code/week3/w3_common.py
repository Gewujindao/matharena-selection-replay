"""Week 3 (frozen design 预注册_第三周_冻结_2026-10-23.md, Part 1 overrides Part 2): finite-ledger
audit of every adjacent reasoning-tier pair in the four 2025 final-answer competitions AIME / HMMT Feb / BRUMO /
CMIMC.

Pure rules live here so they can be exercised on synthetic rows (test_w3.py).  Every operational detail the frozen
text leaves open is marked DECIDED-BEFORE-CORRECT and is fixed in this file before any `correct` value is decoded.
Reuses week-1 helpers (resource limits, exact mean, json/csv writers) from code/w1_common.py and the primitives of the
pure-Python parquet reader code/pq_reader.py (extended below with a row-selective streaming scan that can also read
only the null mask of a column without decoding its values).
"""
import bisect
import ctypes
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
CARD = CODE.parent
sys.path.insert(0, str(CODE))
import w1_common as C1  # noqa: E402
import pq_reader as P  # noqa: E402

W3 = CARD / "week3"
FRAME_DIR = W3 / "frame"
TABLES = W3 / "tables"
DL = CARD / "downloads"
META = DL / "w3_meta_web"
PREREG = CARD / "预注册_第三周_冻结_2026-10-23.md"
LICENSE = "cc-by-nc-sa-4.0"
PQ_NAME = "train-00000-of-00001.parquet"

COMPS = {
    "AIME": dict(repo="MathArena/aime_2025_outputs", rev="16e93d3bb9b0a25ea9422893464d46b3c5ff0ae9",
                 parquet=DL / "w3_hf_aime_2025_outputs_16e93d3" / PQ_NAME, new_download=True),
    "HMMT": dict(repo="MathArena/hmmt_feb_2025_outputs", rev="e3fb6c74f24eabf83413eb9e62b9816c1f3f7d2e",
                 parquet=DL / "hf_hmmt_feb_2025_outputs_e3fb6c7" / PQ_NAME, new_download=False),
    "BRUMO": dict(repo="MathArena/brumo_2025_outputs", rev="3b234efea64d03f80ec57a85e29879fc41b04864",
                  parquet=DL / "w3_hf_brumo_2025_outputs_3b234ef" / PQ_NAME, new_download=True),
    "CMIMC": dict(repo="MathArena/cmimc_2025_outputs", rev="27826187da75fb124dca803cc0274d3c5694df5c",
                  parquet=DL / "w3_hf_cmimc_2025_outputs_2782618" / PQ_NAME, new_download=True),
}
COMP_ORDER = ["AIME", "HMMT", "BRUMO", "CMIMC"]
TRANSFER_CAP = 450 * 1000 ** 2
WALL_CAP_S = 2 * 3600
LADDER = ["minimal", "low", "medium", "high", "xhigh"]
IDX = (0, 1, 2, 3)
P0 = 0
PREFIX, SUFFIX = "configs/models/", ".yaml"

# Seen cells (development material).  Week 2: HMMT, o3-mini medium -> high (A = openai/o3-mini--medium,
# B = openai/o3-mini).  Week 1 (AIME, anthropic/claude-35 -> anthropic/claude-37) is not a reasoning-tier pair; it is
# matched here by raw config values so that it would be flagged if the tier rule ever produced it.
SEEN = {("HMMT", "openai/o3-mini--medium", "openai/o3-mini"): "第二周（HMMT o3-mini medium→high）",
        ("AIME", "anthropic/claude-35", "anthropic/claude-37"): "第一周（AIME Claude-3.5→3.7 Think）"}


class StopNA(Exception):
    pass


# ----------------------------------------------------------------------------- resources

def available_mb_pdh():
    """Windows counter \\Memory\\Available MBytes read through PDH in-process (typeperf timed out in week 2);
    falls back to the week-1 typeperf reader."""
    try:
        pdh = ctypes.windll.pdh
        q = ctypes.c_void_p()
        c = ctypes.c_void_p()
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
            raise OSError("PdhOpenQuery")
        try:
            if pdh.PdhAddEnglishCounterW(q, "\\Memory\\Available MBytes", None, ctypes.byref(c)) != 0:
                raise OSError("PdhAddEnglishCounter")
            if pdh.PdhCollectQueryData(q) != 0:
                raise OSError("PdhCollectQueryData")

            class FMT(ctypes.Structure):
                _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]
            v = FMT()
            if pdh.PdhGetFormattedCounterValue(c, 0x00000200, None, ctypes.byref(v)) != 0:
                raise OSError("PdhGetFormattedCounterValue")
            return float(v.doubleValue)
        finally:
            pdh.PdhCloseQuery(q)
    except Exception:                                                  # noqa: BLE001
        return C1.available_mb()


def wait_ram(log, label, min_mb=1500, max_wait_s=3600):
    import time
    t0 = time.time()
    while True:
        mb = available_mb_pdh()
        if mb is not None and mb >= min_mb:
            log(f"RAM ok before {label}: {mb:.0f} MB available")
            return mb
        if time.time() - t0 > max_wait_s:
            raise StopNA(f"RAM below {min_mb} MB for {max_wait_s}s before {label}")
        log(f"waiting for RAM before {label}: {mb} MB available")
        time.sleep(30)


# ----------------------------------------------------------------------------- streaming parquet scan

def scan_column(pf, name, rows, fn, values=True):
    """{row: fn(value)} for the global row numbers in `rows`.  Pages holding none of these rows are not
    decompressed.  values=False: only definition levels are decoded and fn receives is_null (bool); the value bytes
    of the page are never decoded (used for the null check of `correct` before the frame is sealed)."""
    want_sorted = sorted(set(rows))
    out = {}
    ptype = pf.ptype[name]
    max_def = 1 if pf.rep[name] == 1 else 0
    base = 0
    with open(pf.path, "rb") as f:
        for g in pf.row_groups:
            nrg = g[3]
            i0 = bisect.bisect_left(want_sorted, base)
            if i0 >= len(want_sorted) or want_sorted[i0] >= base + nrg:
                base += nrg
                continue
            cc = next(c for c in g[1] if b".".join(c[3][3]).decode() == name)
            md = cc[3]
            codec = md[4]
            start = md[9] if md.get(11) is None else min(md[9], md[11])
            f.seek(start)
            chunk = f.read(md[7])
            pos, dictionary, got = 0, None, 0
            while got < md[5]:
                tr = P.ThriftReader(chunk, pos)
                ph = tr.struct()
                pos = tr.pos
                ptyp, usize, csize = ph[1], ph[2], ph[3]
                body = chunk[pos:pos + csize]
                pos += csize
                if ptyp == 2:
                    if values:
                        dictionary = P.plain(P.decompress(codec, body, usize), ptype, ph[7][1])
                    continue
                if ptyp == 0:
                    nval = ph[5][1]
                elif ptyp == 3:
                    nval = ph[8][1]
                else:
                    raise NotImplementedError(f"page type {ptyp}")
                lo, hi = base + got, base + got + nval
                a = bisect.bisect_left(want_sorted, lo)
                b = bisect.bisect_left(want_sorted, hi)
                if a == b:
                    got += nval
                    continue
                if ptyp == 0:
                    dh = ph[5]
                    raw = P.decompress(codec, body, usize)
                    p = 0
                    if max_def:
                        ln = int.from_bytes(raw[p:p + 4], "little")
                        p += 4
                        defs = P.rle_hybrid(raw, p, p + ln, 1, nval)
                        p += ln
                    else:
                        defs = [1] * nval
                    vals_buf = raw[p:] if values else None
                    enc = dh[2]
                else:
                    dh = ph[8]
                    dlen, rlen = dh[5], dh[6]
                    if rlen:
                        raise NotImplementedError("repetition levels present")
                    defs = P.rle_hybrid(body, 0, dlen, 1, nval) if max_def else [1] * nval
                    vals_buf = None
                    if values:
                        rest = body[rlen + dlen:]
                        vals_buf = (P.decompress(codec, rest, usize - rlen - dlen) if dh.get(7, True)
                                    else bytes(rest))
                    enc = dh[4]
                if not values:
                    for r in want_sorted[a:b]:
                        out[r] = fn(defs[r - lo] == 0)
                    got += nval
                    continue
                nnon = sum(defs)
                if enc == 0:
                    vals = P.plain(vals_buf, ptype, nnon)
                elif enc in (2, 8):
                    if dictionary is None:
                        raise ValueError("dictionary-encoded page without dictionary")
                    w = vals_buf[0]
                    vals = [dictionary[i] for i in P.rle_hybrid(vals_buf, 1, len(vals_buf), w, nnon)]
                elif enc == 3 and ptype == 0:
                    ln = int.from_bytes(vals_buf[:4], "little")
                    vals = [bool(v) for v in P.rle_hybrid(vals_buf, 4, 4 + ln, 1, nnon)]
                else:
                    raise NotImplementedError(f"encoding {enc} for type {ptype}")
                # position of each row's value among the non-null values of the page
                need = {r - lo for r in want_sorted[a:b]}
                k = 0
                for j, d in enumerate(defs):
                    if j in need:
                        v = vals[k] if d else None
                        if ptype == 6 and isinstance(v, bytes):
                            v = v.decode("utf8")
                        out[lo + j] = fn(v)
                    if d:
                        k += 1
                del vals, vals_buf
                got += nval
            del chunk
            base += nrg
    return out


# ----------------------------------------------------------------------------- tiers, base endpoints, pairs

# DECIDED-BEFORE-CORRECT (frozen Part 1 item 3, operational reading):
#  * display-name tag = content of a parenthesis at the very end of model_name; it counts as a tier tag only if it is
#    exactly one ladder word (case-insensitive, no trimming inside the parenthesis).  Any other trailing parenthesis
#    ("Think", "05-06", "preview", ...) is not a tier tag and stays part of the display base.
#  * config suffix = the text after the last "--" in the last path component of model_config (after removing only
#    the fixed prefix configs/models/ and suffix .yaml); it counts only if it is exactly a ladder word
#    (case-insensitive).  A single hyphen ("-low", "-high") is NOT the frozen "--<tier>" suffix.
#  * tier = name tag if present, else config suffix; both present and different -> config NA.
#  * base endpoint = (display name without the trailing tier parenthesis and the whitespace before it,
#    model_config without its "--<tier>" suffix); two configs share a base only if both strings are equal.
#  * a config value carrying more than one model_name -> config NA (identity ambiguous).
#  * one base with two configs on the same tier -> that base endpoint is NA in that competition.
#  * eligible pair: same base, tiers adjacent on the ladder, A = lower tier.  Non-adjacent tiers are never joined.

TAIL_PAREN = re.compile(r"\s*\(([^()]*)\)$")


def normalize_config(c):
    s = c or ""
    if s.startswith(PREFIX):
        s = s[len(PREFIX):]
    if s.endswith(SUFFIX):
        s = s[:-len(SUFFIX)]
    return s


def name_tier(model_name):
    m = TAIL_PAREN.search(model_name or "")
    if m and m.group(1).lower() in LADDER:
        return m.group(1).lower(), (model_name[:m.start()])
    return None, model_name


def config_tier(model_config):
    s = normalize_config(model_config)
    head, _, last = s.rpartition("/")
    if "--" in last:
        stem, _, suf = last.rpartition("--")
        if suf.lower() in LADDER:
            return suf.lower(), (head + "/" if head else "") + stem
    return None, s


def classify_config(model_config, model_names):
    """Returns dict(tier, base, status, reason, name_tag, cfg_tag)."""
    names = sorted(model_names)
    if len(names) != 1:
        return dict(tier=None, base=None, status="NA", reason="该配置带不止一个 model_name，身份不明",
                    name_tag=None, cfg_tag=None, model_name=names)
    nm = names[0]
    nt, nbase = name_tier(nm)
    ct, cbase = config_tier(model_config)
    rec = dict(model_name=nm, name_tag=nt, cfg_tag=ct, display_base=nbase, config_base=cbase)
    if nt and ct and nt != ct:
        rec.update(tier=None, base=None, status="NA", reason=f"显示名标签 {nt} 与路径后缀 {ct} 不一致")
        return rec
    tier = nt or ct
    if tier is None:
        rec.update(tier=None, base=None, status="no_tier", reason="没有阶梯档位标签（显示名末尾括号与路径 --后缀均无）")
        return rec
    rec.update(tier=tier, base=(nbase, cbase), status="tiered",
               reason=f"档位 {tier}（来自{'显示名' if nt else '路径后缀'}{'，两处一致' if nt and ct else ''}）")
    return rec


def enumerate_pairs(cfg_names):
    """cfg_names {raw model_config: set(model_name)} -> (configs {raw: rec}, endpoints {base: rec}, pairs list)."""
    configs = {c: classify_config(c, ns) for c, ns in cfg_names.items()}
    bases = {}
    for c, r in configs.items():
        if r["status"] == "tiered":
            bases.setdefault(r["base"], []).append(c)
    endpoints, pairs = {}, []
    for base in sorted(bases):
        cs = sorted(bases[base], key=lambda c: (LADDER.index(configs[c]["tier"]), c))
        by_tier = {}
        for c in cs:
            by_tier.setdefault(configs[c]["tier"], []).append(c)
        dup = {t: v for t, v in by_tier.items() if len(v) > 1}
        ep = dict(display_base=base[0], config_base=base[1], configs=cs,
                  tiers=[configs[c]["tier"] for c in cs], status="ok", pairs=[], reason="")
        if dup:
            ep.update(status="NA", reason="同一基础端点同一档位有不止一个配置：" + "; ".join(
                f"{t}: {v}" for t, v in sorted(dup.items())))
            for c in cs:
                configs[c]["pair_status"] = "端点 NA（同档位重复配置）"
        else:
            levels = sorted(by_tier, key=LADDER.index)
            for lo_t, hi_t in zip(levels, levels[1:]):
                if LADDER.index(hi_t) - LADDER.index(lo_t) == 1:
                    a, b = by_tier[lo_t][0], by_tier[hi_t][0]
                    pr = dict(display_base=base[0], config_base=base[1], tier_A=lo_t, tier_B=hi_t, A=a, B=b,
                              A_name=configs[a]["model_name"], B_name=configs[b]["model_name"])
                    pairs.append(pr)
                    ep["pairs"].append(f"{lo_t}->{hi_t}")
            in_pair = {p["A"] for p in pairs if (p["display_base"], p["config_base"]) == base} | \
                      {p["B"] for p in pairs if (p["display_base"], p["config_base"]) == base}
            for c in cs:
                if c in in_pair:
                    configs[c]["pair_status"] = "进入合格 pair"
                elif len(levels) == 1:
                    configs[c]["pair_status"] = "该基础端点在本赛只有这一个档位，没有相邻档位可配"
                else:
                    configs[c]["pair_status"] = ("同一基础端点的其他档位都不相邻（缺中间档，不跳接）："
                                                 + ",".join(levels))
            if not ep["pairs"]:
                ep["reason"] = "没有相邻档位" if len(levels) > 1 else "只有一个档位"
        endpoints[base] = ep
    for c, r in configs.items():
        if r["status"] == "no_tier":
            r["pair_status"] = "不进入配对（无阶梯档位）"
        elif r["status"] == "NA":
            r["pair_status"] = "配置 NA：" + r["reason"]
    return configs, endpoints, pairs


# ----------------------------------------------------------------------------- per-cell contract

ST_OK, ST_TRUNC, ST_UNKNOWN = "ok", "saved_truncation_empty_answer_tokens>0", "unknown_response_status"


def response_status(answer_nonempty, output_tokens):
    """Week-2 rule: non-empty answer -> ok (kept even with 0 tokens); empty answer with output_tokens > 0 -> saved
    truncation, kept and scored by the stored correct; empty answer with output_tokens null/0 -> unknown status."""
    if answer_nonempty:
        return ST_OK
    if output_tokens is not None and output_tokens > 0:
        return ST_TRUNC
    return ST_UNKNOWN


# DECIDED-BEFORE-CORRECT: "full prompt" of a slot = (user_message, the ordered list of every all_messages entry whose
# role is not "assistant", each entry as canonical JSON with sorted keys).  It must parse as a JSON list with at least
# one user turn and a non-empty user_message.  "identical full prompt across the eight slots" = this pair of values
# is byte-identical for the 8 kept slots (A idx0-3, B idx0-3) of the problem.  No normalisation beyond canonical JSON.

def prompt_part(all_messages):
    """all_messages -> (ok, why, canonical JSON of the non-assistant entries or None)."""
    try:
        msgs = json.loads(all_messages) if all_messages is not None else None
    except (TypeError, ValueError):
        return False, "all_messages 不是 JSON", None
    if not isinstance(msgs, list):
        return False, "all_messages 不是列表", None
    non_asst = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "assistant")]
    if not any(isinstance(m, dict) and m.get("role") == "user" for m in non_asst):
        return False, "all_messages 没有 user 轮", None
    return True, "ok", json.dumps(non_asst, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def full_prompt_from_part(user_message, part):
    """(user_message, prompt_part(...)) -> (ok, why, canonical full-prompt string or None)."""
    if not (user_message and user_message.strip()):
        return False, "user_message 为空", None
    ok, why, canon = part
    if not ok:
        return False, why, None
    return True, "ok", json.dumps([user_message, canon], ensure_ascii=False, separators=(",", ":"))


def full_prompt(user_message, all_messages):
    """Returns (ok, why, canonical string or None)."""
    return full_prompt_from_part(user_message, prompt_part(all_messages))


def cell_contract(rows, problems, a_cfg, b_cfg):
    """rows: dicts row, cfg, problem_idx, idx, problem, gold, status, prompt_ok, prompt_why, prompt, correct_is_null,
    digest.  problems: the competition's complete problem frame.  Returns (valid, breaks {q: [reasons]},
    kept {q: {"A0": row, ...}}, slots list)."""
    side_of = {a_cfg: "A", b_cfg: "B"}
    by = {}
    for r in rows:
        if r["cfg"] in side_of:
            by.setdefault((side_of[r["cfg"]], r["problem_idx"]), []).append(r)
    breaks, kept, slots = {}, {}, []
    for q in problems:
        reasons, keep = [], {}
        for side in ("A", "B"):
            groups = {}
            for r in by.get((side, q), []):
                groups.setdefault(r["idx"], []).append(r)
            for k in sorted(groups, key=lambda x: (x is None, x if x is not None else 0)):
                g = groups[k]
                if k not in IDX:
                    reasons.append(f"{side}: 出现 0–3 以外的 idx {k}")
                    slots += [(side, q, k, r["row"], "unexpected_idx") for r in g]
                    continue
                if len({r["digest"] for r in g}) > 1:
                    reasons.append(f"{side}: idx {k} 有内容不同的重复行")
                    slots += [(side, q, k, r["row"], "differing_duplicate") for r in g]
                    continue
                first = min(g, key=lambda r: r["row"])
                slots += [(side, q, k, r["row"], f"identical_duplicate_of_row_{first['row']}") for r in g
                          if r is not first]
                keep[(side, k)] = first
            for k in IDX:
                if k not in groups:
                    reasons.append(f"{side}: 缺 idx {k}")
                    slots.append((side, q, k, None, "missing"))
            for k in IDX:
                r = keep.get((side, k))
                if r is None:
                    continue
                st = []
                if r["status"] == ST_UNKNOWN:
                    reasons.append(f"{side}: idx {k} 响应状态未知（answer 空且 output_tokens 空/0）")
                    st.append(ST_UNKNOWN)
                elif r["status"] == ST_TRUNC:
                    st.append(ST_TRUNC)
                if not r["prompt_ok"]:
                    reasons.append(f"{side}: idx {k} 提示不完整（{r['prompt_why']}）")
                    st.append("prompt_incomplete")
                if r["correct_is_null"]:
                    reasons.append(f"{side}: idx {k} 已存 correct 缺失")
                    st.append("correct_missing")
                if not (r["gold"] and r["gold"].strip()):
                    reasons.append(f"{side}: idx {k} gold_answer 为空")
                    st.append("gold_empty")
                slots.append((side, q, k, r["row"], "+".join(st) if st else "ok"))
        if len({r["problem"] for r in keep.values()}) > 1:
            reasons.append("保留记录的题文不唯一")
        if len({r["gold"] for r in keep.values()}) > 1:
            reasons.append("保留记录的 gold_answer 不唯一")
        prompts = {r["prompt"] for r in keep.values() if r["prompt_ok"]}
        if len(prompts) > 1:
            reasons.append("八槽完整提示不相同")
        if reasons:
            breaks[q] = reasons
        else:
            kept[q] = {f"{s}{k}": r["row"] for (s, k), r in keep.items()}
    valid = not breaks and len(kept) == len(problems) and len(problems) > 0
    return valid, breaks, kept, slots


# ----------------------------------------------------------------------------- exact ledger

def ledger(F, corr, screen=P0):
    """corr[(side, q, idx)] in {0, 1}.  Screening slot `screen`; A1, B1 = mean of the other three idx.
    S = {A0 = 0}; on S: G = mean(B1 - A0), R = mean(A1 - A0), D = mean(B1 - A1); D_F on all of F; C = D - D_F."""
    per = []
    for q in F:
        rest = [k for k in IDX if k != screen]
        a0 = Fraction(corr[("A", q, screen)])
        a1 = sum((Fraction(corr[("A", q, k)]) for k in rest), Fraction(0)) / len(rest)
        b1 = sum((Fraction(corr[("B", q, k)]) for k in rest), Fraction(0)) / len(rest)
        per.append(dict(q=q, A0=a0, A1=a1, B1=b1, B0=Fraction(corr[("B", q, screen)]), in_S=(a0 == 0)))
    S = [p for p in per if p["in_S"]]
    K = len(S)
    out = dict(screen_idx=screen, K=K, n_F=len(F), S=[p["q"] for p in S])
    out["D_F"] = C1.mean(p["B1"] - p["A1"] for p in per)
    if K:
        out.update(A1_S=C1.mean(p["A1"] for p in S), B1_S=C1.mean(p["B1"] for p in S),
                   G=C1.mean(p["B1"] - p["A0"] for p in S), R=C1.mean(p["A1"] - p["A0"] for p in S),
                   D=C1.mean(p["B1"] - p["A1"] for p in S))
        assert out["G"] - out["R"] == out["D"]
        out["C"] = out["D"] - out["D_F"]
        out["G_minus_D_F"] = out["G"] - out["D_F"]
        assert out["G_minus_D_F"] == out["R"] + out["C"]
        out["sign_reversal"] = int(out["D"] * out["D_F"] < 0)
        out["D_zero"] = int(out["D"] == 0)
    else:
        out.update(A1_S=None, B1_S=None, G=None, R=None, D=None, C=None, G_minus_D_F=None, sign_reversal=None,
                   D_zero=None)
    out["D_F_zero"] = int(out["D_F"] == 0)
    return out, per


QUANTS = ["G", "R", "D", "D_F", "C", "G_minus_D_F"]


def aggregate(cells, quant, include=lambda c: True):
    """Frozen weights: within a competition, equal weight over the valid adjacent pairs of a base endpoint, then
    equal weight over endpoints; then equal-weight macro over valid competitions.  A cell enters a layer only if it
    is contract-valid and the quantity is defined (K = 0 leaves G/R/D/C/G-D_F undefined)."""
    comps = {}
    for c in cells:
        if not include(c):
            continue
        comps.setdefault(c["comp"], {}).setdefault(c["endpoint"], []).append(c)
    rep = dict(quantity=quant, competitions={})
    comp_vals = []
    for comp in COMP_ORDER:
        eps = comps.get(comp, {})
        ep_vals, ep_rep = [], {}
        n_cells = n_valid = n_na = n_undef = 0
        for ep, cs in sorted(eps.items()):
            vals = []
            na = undef = 0
            for c in cs:
                n_cells += 1
                if not c["valid"]:
                    na += 1
                elif c["primary"][quant] is None:
                    undef += 1
                else:
                    vals.append(c["primary"][quant])
            n_valid += len(vals)
            n_na += na
            n_undef += undef
            v = C1.mean(vals)
            ep_rep[ep] = dict(n_cells=len(cs), n_valid=len(vals), n_contract_NA=na, n_undefined_K0=undef,
                              value=v)
            if v is not None:
                ep_vals.append(v)
        cv = C1.mean(ep_vals)
        rep["competitions"][comp] = dict(n_endpoints=len(eps), n_endpoints_valid=len(ep_vals),
                                         n_endpoints_NA=len(eps) - len(ep_vals), n_cells=n_cells,
                                         n_cells_valid=n_valid, n_cells_contract_NA=n_na, n_cells_undefined_K0=n_undef,
                                         endpoints=ep_rep, value=cv)
        if cv is not None:
            comp_vals.append(cv)
    rep["n_competitions"] = len(COMP_ORDER)
    rep["n_competitions_valid"] = len(comp_vals)
    rep["n_competitions_NA"] = len(COMP_ORDER) - len(comp_vals)
    rep["macro"] = C1.mean(comp_vals)
    return rep


def fr(x):
    return C1.fr(x)


def frac(x):
    return "未定义" if x is None else f"{x.numerator}/{x.denominator}"


def pp(x):
    return "未定义" if x is None else f"{float(x) * 100:+.2f} 个百分点"


def pidx_key(q):
    return (0, int(q)) if str(q).isdigit() else (1, str(q))

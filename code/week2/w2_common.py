"""Week 2 (frozen design 预注册_第二周_冻结_2026-10-23.md): HMMT Feb 2025, A = o3-mini (medium),
B = o3-mini (high), fixed screening slot p0 = idx0.

Pure rules live here so they can be exercised on synthetic rows.  Every operational detail the frozen design leaves
open is marked DECIDED-BEFORE-CORRECT and is fixed in this file before any `correct` value is used.
Reuses week-1 helpers (resource limits, RAM gate, exact means) from code/w1_common.py and the pure-Python parquet
reader code/pq_reader.py.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
CARD = CODE.parent
sys.path.insert(0, str(CODE))
import w1_common as C1  # noqa: E402  (week-1 helpers: apply_limits, wait_ram, peak_rss_mb, mean, fr, write_*)
import pq_reader as P  # noqa: E402

W2 = CARD / "week2"
FRAME_DIR = W2 / "frame"
TABLES = W2 / "tables"
DATA_DIR = CARD / "downloads" / "hf_hmmt_feb_2025_outputs_e3fb6c7"
PARQUET = DATA_DIR / "train-00000-of-00001.parquet"
META = CARD / "downloads" / "w2_meta_web"
PREREG = CARD / "预注册_第二周_冻结_2026-10-23.md"

REPO = "MathArena/hmmt_feb_2025_outputs"
REV = "e3fb6c74f24eabf83413eb9e62b9816c1f3f7d2e"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/data/train-00000-of-00001.parquet"
LICENSE = "cc-by-nc-sa-4.0"
IDX = (0, 1, 2, 3)
TARGET_N = 30
P0 = 0                                  # fixed screening slot (design: 固定 p0 = idx0)
TIME_CAP_S = 3600                       # cumulative CPU-process budget (design: 累计 1h)
TRANSFER_CAP = 250 * 1000 ** 2          # design: 含重试传输上限 250MB

# The pair is locked by the design (official identifiers and display names quoted there).
SIDES = {"A": dict(identifier="o3-mini--medium", display="o3-mini (medium)"),
         "B": dict(identifier="o3-mini--high", display="o3-mini (high)")}
PREFIX, SUFFIX = "configs/models/", ".yaml"


class StopNA(Exception):
    """Frozen contract not met: this round is NA (no config switch, no problem deletion)."""


# ----------------------------------------------------------------------------- config correspondence

# DECIDED-BEFORE-CORRECT: a raw model_config value c belongs to side X if (i) c with only the fixed prefix
# "configs/models/" and suffix ".yaml" removed equals X's official identifier or ends with "/" + identifier, or
# (ii) the rows of c carry model_name exactly equal to X's official display name.  Correspondence is clear only if
# each side collects exactly one raw config value, the two values differ, and each value carries exactly one
# model_name.  Anything else (zero, several, overlap) -> NA.  No fuzzy matching, no choice by results.

def normalize_config(c):
    s = c or ""
    if s.startswith(PREFIX):
        s = s[len(PREFIX):]
    if s.endswith(SUFFIX):
        s = s[:-len(SUFFIX)]
    return s


def match_configs(cfg_names):
    """cfg_names: {raw model_config: set(model_name)}.  Returns (mapping {side: raw}, detail, clear: bool)."""
    detail, mapping = {}, {}
    for side, ref in SIDES.items():
        by_id = sorted(c for c in cfg_names if normalize_config(c) == ref["identifier"]
                       or normalize_config(c).endswith("/" + ref["identifier"]))
        by_name = sorted(c for c, ns in cfg_names.items() if ref["display"] in ns)
        cand = sorted(set(by_id) | set(by_name))
        detail[side] = dict(identifier=ref["identifier"], display=ref["display"], matched_by_identifier=by_id,
                            matched_by_display_name=by_name, candidates=cand,
                            model_names_of_candidates={c: sorted(cfg_names[c]) for c in cand})
        if len(cand) == 1:
            mapping[side] = cand[0]
    clear = (len(mapping) == 2 and mapping["A"] != mapping["B"]
             and all(len(cfg_names[mapping[s]]) == 1 for s in mapping))
    return mapping, detail, clear


# ----------------------------------------------------------------------------- 30-problem contract

# DECIDED-BEFORE-CORRECT (operational reading of the design's F contract):
#  * problems are keyed by problem_idx; candidates = distinct problem_idx in the whole pinned file; must be 30;
#  * each side has each idx 0..3 exactly once (fully identical duplicate rows collapse into one record; duplicates
#    that differ in any column, or any idx outside 0..3, break the contract);
#  * "共同题文/gold": one problem string and one gold_answer string across the 8 kept rows of the problem;
#  * "完整提示": every kept row has a non-empty user_message and an all_messages JSON list with >= 1 user turn;
#    whether the user prompt is identical across A/B is recorded as a call-condition attribute (a difference would
#    be part of the A/B condition difference, as in week 1), not used to drop anything;
#  * "correct 无缺失 / 可解释的最终答案判分": correct is a stored boolean (not null) and gold_answer is non-empty;
#  * response status: non-empty answer -> ok (kept even when token counts are 0; usage read at its own bookkeeping
#    level); empty answer with output_tokens > 0 -> saved truncation, kept and scored by the stored correct;
#    empty answer with output_tokens null/0 -> unknown response status, which is NOT counted as wrong and breaks the
#    contract;
#  * any break anywhere -> the whole round is NA (no problem is deleted, no config is swapped).

ST_OK, ST_TRUNC, ST_UNKNOWN = "ok", "saved_truncation_empty_answer_tokens>0", "unknown_response_status"


def response_status(answer, output_tokens):
    if answer is not None and answer.strip():
        return ST_OK
    if output_tokens is not None and output_tokens > 0:
        return ST_TRUNC
    return ST_UNKNOWN


def prompt_complete(user_message, all_messages):
    if not (user_message and user_message.strip()):
        return False, "user_message empty"
    try:
        msgs = json.loads(all_messages) if all_messages is not None else None
    except (TypeError, ValueError):
        return False, "all_messages not JSON"
    if not isinstance(msgs, list):
        return False, "all_messages not a list"
    if not any(isinstance(m, dict) and m.get("role") == "user" for m in msgs):
        return False, "no user turn in all_messages"
    return True, "ok"


def contract(rows, candidates, mapping):
    """rows: dicts with row, cfg, problem_idx, idx, problem, gold, status, prompt_ok, prompt_why, correct_is_null,
    correct_is_bool, digest.  Returns (F, breaks {q: [reasons]}, kept {q: {(side, idx): row}}, slot_status list)."""
    inv = {raw: side for side, raw in mapping.items()}
    by = {}
    for r in rows:
        if r["cfg"] in inv:
            by.setdefault((inv[r["cfg"]], r["problem_idx"]), []).append(r)
    F, breaks, kept, slots = [], {}, {}, []
    for q in candidates:
        reasons, keep = [], {}
        for side in ("A", "B"):
            groups = {}
            for r in by.get((side, q), []):
                groups.setdefault(r["idx"], []).append(r)
            for k in sorted(groups, key=lambda x: (x is None, x)):
                g = groups[k]
                if k not in IDX:
                    reasons.append(f"{side}: unexpected idx {k}")
                    slots += [(side, q, k, r["row"], "unexpected_idx") for r in g]
                    continue
                if len({r["digest"] for r in g}) > 1:
                    reasons.append(f"{side}: idx {k} has differing duplicate rows")
                    slots += [(side, q, k, r["row"], "differing_duplicate") for r in g]
                    continue
                first = min(g, key=lambda r: r["row"])
                slots += [(side, q, k, r["row"], f"identical_duplicate_of_row_{first['row']}") for r in g
                          if r is not first]
                keep[(side, k)] = first
            for k in IDX:
                if k not in groups:
                    reasons.append(f"{side}: idx {k} missing")
                    slots.append((side, q, k, None, "missing"))
            for k in IDX:
                r = keep.get((side, k))
                if r is None:
                    continue
                st = []
                if r["status"] == ST_UNKNOWN:
                    reasons.append(f"{side}: idx {k} unknown response status")
                    st.append(ST_UNKNOWN)
                elif r["status"] == ST_TRUNC:
                    st.append(ST_TRUNC)
                if not r["prompt_ok"]:
                    reasons.append(f"{side}: idx {k} prompt incomplete ({r['prompt_why']})")
                    st.append("prompt_incomplete")
                if r["correct_is_null"] or not r["correct_is_bool"]:
                    reasons.append(f"{side}: idx {k} correct missing or not boolean")
                    st.append("grade_missing")
                if not (r["gold"] and r["gold"].strip()):
                    reasons.append(f"{side}: idx {k} gold_answer empty")
                    st.append("gold_empty")
                slots.append((side, q, k, r["row"], "+".join(st) if st else "ok"))
        if len({r["problem"] for r in keep.values()}) > 1:
            reasons.append("problem text differs across kept rows")
        if len({r["gold"] for r in keep.values()}) > 1:
            reasons.append("gold_answer differs across kept rows")
        if reasons:
            breaks[q] = reasons
        else:
            F.append(q)
            kept[q] = {key: r["row"] for key, r in keep.items()}
    return F, breaks, kept, slots


# ----------------------------------------------------------------------------- exact ledger quantities

def ledger(F, corr, screen=P0):
    """corr[(side, q, idx)] in {0, 1}.  Screening slot `screen`; A1, B1 = mean of the other three idx.
    S = {A0 = 0}; on S: G = mean(B1 - A0), R = mean(A1 - A0), D = mean(B1 - A1); D_F on all of F."""
    per = []
    for q in F:
        rest = [k for k in IDX if k != screen]
        a0 = Fraction(corr[("A", q, screen)])
        a1 = sum((Fraction(corr[("A", q, k)]) for k in rest), Fraction(0)) / len(rest)
        b1 = sum((Fraction(corr[("B", q, k)]) for k in rest), Fraction(0)) / len(rest)
        per.append(dict(q=q, screen_idx=screen, rescore_idx=rest, A0=a0, A1=a1, B1=b1,
                        B0=Fraction(corr[("B", q, screen)]), in_S=(a0 == 0)))
    S = [p for p in per if p["in_S"]]
    K = len(S)
    out = dict(screen_idx=screen, K=K, n_F=len(F), S=[p["q"] for p in S])
    if K:
        out.update(A0_S=C1.mean(p["A0"] for p in S), A1_S=C1.mean(p["A1"] for p in S),
                   B1_S=C1.mean(p["B1"] for p in S),
                   G=C1.mean(p["B1"] - p["A0"] for p in S), R=C1.mean(p["A1"] - p["A0"] for p in S),
                   D=C1.mean(p["B1"] - p["A1"] for p in S))
        assert out["G"] - out["R"] == out["D"]
    else:
        out.update(A0_S=None, A1_S=None, B1_S=None, G=None, R=None, D=None)
    out["D_F"] = C1.mean(p["B1"] - p["A1"] for p in per)
    out["D_minus_D_F"] = None if out["D"] is None else out["D"] - out["D_F"]
    return out, per


def fr(x):
    return C1.fr(x)


def pp(x):
    return "未定义" if x is None else f"{float(x) * 100:+.2f} pp"


def frac(x):
    return "未定义" if x is None else f"{x.numerator}/{x.denominator}"

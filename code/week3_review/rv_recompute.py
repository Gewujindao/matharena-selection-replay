# Reviewer's independent recomputation for week 3 (own decoder rv_pq; no runner code imported).
# (a) tier / base endpoint / adjacent pair mapping from the frozen rule (metadata only)
# (b) per-cell contract + K, G, R, D, D_F, C, sign flag, NA status; four-slot cycle
# (c) frozen-weight macro aggregates (all cells, excluding seen)
import json
import os
import re
import sys
import time
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_guard
import rv_pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..") + "/"
FILES = {
    "AIME": ROOT + "downloads/w3_hf_aime_2025_outputs_16e93d3/train-00000-of-00001.parquet",
    "HMMT": ROOT + "downloads/hf_hmmt_feb_2025_outputs_e3fb6c7/train-00000-of-00001.parquet",
    "BRUMO": ROOT + "downloads/w3_hf_brumo_2025_outputs_3b234ef/train-00000-of-00001.parquet",
    "CMIMC": ROOT + "downloads/w3_hf_cmimc_2025_outputs_2782618/train-00000-of-00001.parquet",
}
OUT = ROOT + "week3/review_recompute.json"
LADDER = ["minimal", "low", "medium", "high", "xhigh"]
RANK = {t: i for i, t in enumerate(LADDER)}
# seen cells (competition, A config, B config): week 2 HMMT o3-mini (medium) -> o3-mini (high)
SEEN = {("HMMT", "openai/o3-mini--medium", "openai/o3-mini")}

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def s(b):
    return None if b is None else b.decode("utf-8")


def norm_path(p):
    if p.startswith("configs/models/"):
        p = p[len("configs/models/"):]
    if p.endswith(".yaml"):
        p = p[:-5]
    return p


NAME_TAG = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
PATH_TAG = re.compile(r"^(.*)--([^/\-][^/]*)$")


def name_tier(name):
    m = NAME_TAG.match(name)
    if m and m.group(2).strip().lower() in RANK:
        return m.group(2).strip().lower(), m.group(1)
    return None, name


def path_tier(path):
    m = PATH_TAG.match(path)
    if m and m.group(2).lower() in RANK:
        return m.group(2).lower(), m.group(1)
    return None, path


def mapping(comp, names, configs):
    """names/configs: per-row lists. Returns configs dict, endpoints dict, pairs list."""
    cfg = {}
    for n, c in zip(names, configs):
        c = norm_path(c)
        d = cfg.setdefault(c, {"names": set(), "rows": 0})
        d["names"].add(n)
        d["rows"] += 1
    out_cfg = {}
    for c, d in sorted(cfg.items()):
        rec = {"names": sorted(d["names"]), "rows": d["rows"]}
        if len(d["names"]) != 1:
            rec.update(status="NA", reason="multiple model_name values")
            out_cfg[c] = rec
            continue
        n = rec["names"][0]
        nt, nbase = name_tier(n)
        pt, pbase = path_tier(c)
        rec.update(name_tier=nt, path_tier=pt, display_base=nbase if nt else n, config_base=pbase if pt else c)
        if nt and pt and nt != pt:
            rec.update(status="NA", reason="name/path tier conflict", tier=None)
        elif nt or pt:
            rec.update(status="tiered", tier=nt or pt)
        else:
            rec.update(status="no_tier", tier=None)
        out_cfg[c] = rec
    eps = {}
    for c, r in out_cfg.items():
        if r.get("status") != "tiered":
            continue
        key = (r["display_base"], r["config_base"])
        eps.setdefault(key, {}).setdefault(r["tier"], []).append(c)
    out_eps = {}
    pairs = []
    for key, tiers in sorted(eps.items()):
        dup = any(len(v) > 1 for v in tiers.values())
        e = {"display_base": key[0], "config_base": key[1],
             "tiers": {t: v for t, v in sorted(tiers.items(), key=lambda kv: RANK[kv[0]])},
             "status": "NA_duplicate_tier" if dup else "ok", "pairs": []}
        if not dup:
            present = sorted(tiers, key=RANK.get)
            for a, b in zip(present, present[1:]):
                if RANK[b] == RANK[a] + 1:
                    A, B = tiers[a][0], tiers[b][0]
                    e["pairs"].append([a, b])
                    pairs.append({"competition": comp, "endpoint": key[0] + " | " + key[1], "tier_A": a, "tier_B": b,
                                  "A": A, "B": B, "seen": (comp, A, B) in SEEN})
        out_eps[key[0] + " | " + key[1]] = e
    return out_cfg, out_eps, pairs


def canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prompt_info(v):
    """From all_messages JSON text: (canonical non-assistant entries, canonical entries before first assistant, roles)."""
    if v is None:
        return None
    try:
        msgs = json.loads(v.decode("utf-8"))
    except Exception:
        return ("BADJSON", "BADJSON", None)
    if not isinstance(msgs, list):
        return ("NOTLIST", "NOTLIST", None)
    roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
    nonasst = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "assistant")]
    pre = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            break
        pre.append(m)
    return (canon(nonasst), canon(pre), tuple(roles))


def mean(xs):
    xs = list(xs)
    return sum(xs, Fr(0)) / len(xs) if xs else None


def cell_stats(A, B, probs, j):
    """A,B: {prob: [c0..c3]} ints. j = screening slot. Returns dict of exact Fractions."""
    others = [k for k in range(4) if k != j]
    a0 = {p: A[p][j] for p in probs}
    a1 = {p: Fr(sum(A[p][k] for k in others), 3) for p in probs}
    b1 = {p: Fr(sum(B[p][k] for k in others), 3) for p in probs}
    S = [p for p in probs if a0[p] == 0]
    K = len(S)
    DF = mean(b1[p] - a1[p] for p in probs)
    if K == 0:
        return {"K": 0, "G": None, "R": None, "D": None, "D_F": DF, "C": None, "G-D_F": None, "flag": None}
    G = mean(b1[p] - a0[p] for p in S)
    R = mean(a1[p] - a0[p] for p in S)
    D = mean(b1[p] - a1[p] for p in S)
    assert G - R == D
    return {"K": K, "G": G, "R": R, "D": D, "D_F": DF, "C": D - DF, "G-D_F": G - DF,
            "flag": int(D * DF < 0), "D_zero": int(D == 0), "D_F_zero": int(DF == 0)}


def frs(x):
    return None if x is None else "%d/%d" % (x.numerator, x.denominator)


def main():
    rv_guard.setup(log)
    result = {"mapping": {}, "cells": [], "aggregates": {}, "four_slot": [], "notes": {}}
    all_pairs = []
    data = {}
    for comp, path in FILES.items():
        rv_guard.gate(1500, log)
        pf = rv_pq.PFile(path)
        names = [s(x) for x in pf.read_flat("model_name")]
        configs = [s(x) for x in pf.read_flat("model_config")]
        pidx = [s(x) for x in pf.read_flat("problem_idx")]
        cfgs, eps, pairs = mapping(comp, names, configs)
        frame_probs = sorted(set(pidx), key=lambda x: (len(x), x))
        result["mapping"][comp] = {"n_rows": pf.num_rows, "n_configs": len(cfgs), "n_problems": len(frame_probs),
                                   "configs": cfgs, "endpoints": eps, "pairs": pairs,
                                   "null_problem_idx": sum(1 for x in pidx if x is None)}
        log(comp, "rows", pf.num_rows, "configs", len(cfgs), "problems", len(frame_probs), "pairs", len(pairs))
        all_pairs += pairs
        # rows needed for cells
        need_cfg = set()
        for pr in pairs:
            need_cfg.add(pr["A"])
            need_cfg.add(pr["B"])
        rows = [i for i, c in enumerate(configs) if norm_path(c) in need_cfg]
        wanted = set(rows)
        log(comp, "decoding", len(rows), "rows for", len(need_cfg), "configs")
        col = {}
        for name in ["idx_answer", "problem", "gold_answer", "user_message", "answer", "output_tokens", "correct",
                     "source", "parsed_answer"]:
            col[name] = rv_pq.read_rows(pf, name, wanted, lambda v: v)
        col["all_messages"] = rv_pq.read_rows(pf, "all_messages", wanted, prompt_info)
        recs = {}
        for i in rows:
            c = norm_path(configs[i])
            recs.setdefault(c, []).append({
                "row": i, "prob": pidx[i], "idx": col["idx_answer"][i], "problem": col["problem"][i],
                "gold": col["gold_answer"][i], "user": col["user_message"][i], "answer": col["answer"][i],
                "otok": col["output_tokens"][i], "correct": col["correct"][i], "msgs": col["all_messages"][i],
                "source": col["source"][i], "parsed": col["parsed_answer"][i]})
        data[comp] = (frame_probs, recs)
        log(comp, "peak WS MiB %.1f" % rv_guard.peak_ws_mib())

    # ---------- per-cell contract and stats ----------
    retained = 0
    notes = {"otok0_idx123": {}, "otok_64000": {}, "empty_answer": 0, "null_correct": 0, "dup_rows": 0,
             "source_values": {}, "prompt_variants": {}}
    for pr in all_pairs:
        comp = pr["competition"]
        frame_probs, recs = data[comp]
        cell = dict(pr)
        cell["id"] = "%s|%s|%s->%s" % (comp, pr["endpoint"].split(" | ")[0], pr["tier_A"], pr["tier_B"])
        sides = {}
        breaking = {}
        variant_break = {"user_only": 0, "nonasst": 0, "pre_asst": 0, "full_runner_style": 0}
        for side, c in (("A", pr["A"]), ("B", pr["B"])):
            byp = {}
            for r in recs[c]:
                byp.setdefault(r["prob"], {}).setdefault(r["idx"], []).append(r)
            sides[side] = byp
        A = {}
        B = {}
        for p in frame_probs:
            why = []
            slots = []
            for side in ("A", "B"):
                d = sides[side].get(p, {})
                if set(d) - {0, 1, 2, 3}:
                    why.append("idx outside 0..3")
                for k in range(4):
                    lst = d.get(k, [])
                    if len(lst) == 0:
                        why.append("missing %s idx%d" % (side, k))
                    elif len(lst) > 1:
                        notes["dup_rows"] += len(lst) - 1
                        why.append("duplicate %s idx%d" % (side, k))
                    else:
                        slots.append((side, k, lst[0]))
            if len(slots) == 8:
                if len({r["problem"] for _, _, r in slots}) != 1:
                    why.append("problem text differs")
                golds = {r["gold"] for _, _, r in slots}
                if len(golds) != 1 or None in golds or b"" in golds:
                    why.append("gold differs/empty")
                for _, _, r in slots:
                    if r["correct"] is None:
                        notes["null_correct"] += 1
                        why.append("correct missing")
                    if r["answer"] is None or r["answer"] == b"":
                        notes["empty_answer"] += 1
                        if not r["otok"]:
                            why.append("unknown response state")
                    if r["msgs"] is None or r["user"] in (None, b""):
                        why.append("prompt missing")
                u = {r["user"] for _, _, r in slots}
                na = {r["msgs"][0] if r["msgs"] else None for _, _, r in slots}
                pa = {r["msgs"][1] if r["msgs"] else None for _, _, r in slots}
                full = {(r["user"], r["msgs"][0] if r["msgs"] else None) for _, _, r in slots}
                variant_break["user_only"] += len(u) != 1
                variant_break["nonasst"] += len(na) != 1
                variant_break["pre_asst"] += len(pa) != 1
                variant_break["full_runner_style"] += len(full) != 1
                if len(full) != 1 or len(pa) != 1:
                    why.append("prompt differs across 8 slots")
                for side, k, r in slots:
                    if comp not in notes["source_values"]:
                        notes["source_values"][comp] = set()
                    notes["source_values"][comp].add(s(r["source"]))
            if why:
                breaking[p] = sorted(set(why))
            else:
                A[p] = [0] * 4
                B[p] = [0] * 4
                for side, k, r in slots:
                    (A if side == "A" else B)[p][k] = int(bool(r["correct"]))
            retained += len(slots)
        cell["n_frame"] = len(frame_probs)
        cell["n_complete"] = len(frame_probs) - len(breaking)
        cell["n_breaking"] = len(breaking)
        reasons = sorted({w for v in breaking.values() for w in v})
        cell["na_reasons"] = reasons
        cell["prompt_variant_breaking_problems"] = variant_break
        # A/B prompt diff vs within-side diff
        ab_diff = 0
        within = {"A": 0, "B": 0}
        for p in frame_probs:
            us = {}
            for side in ("A", "B"):
                d = sides[side].get(p, {})
                us[side] = {(r["user"], r["msgs"][0] if r["msgs"] else None) for k in range(4) for r in d.get(k, [])}
                within[side] += len(us[side]) > 1
            ab_diff += us["A"] != us["B"]
        cell["prompt_ab_diff_problems"] = ab_diff
        cell["prompt_within_side_diff"] = within
        if breaking:
            cell["contract"] = "NA"
        else:
            cell["contract"] = "valid"
            probs = frame_probs
            st = cell_stats(A, B, probs, 0)
            cell["primary"] = {k: (frs(v) if isinstance(v, Fr) else v) for k, v in st.items()}
            cell["_st"] = st
            accA = Fr(sum(sum(A[p]) for p in probs), 4 * len(probs))
            accB = Fr(sum(sum(B[p]) for p in probs), 4 * len(probs))
            cell["acc_A_4slot"] = frs(accA)
            cell["acc_B_4slot"] = frs(accB)
            cell["stratum_30_70"] = int(Fr(3, 10) <= accA <= Fr(7, 10))
            cyc = []
            for j in range(4):
                stj = cell_stats(A, B, probs, j)
                cyc.append({k: (frs(v) if isinstance(v, Fr) else v) for k, v in stj.items()})
            cell["four_slot"] = cyc
            # token notes
            for side, c in (("A", pr["A"]), ("B", pr["B"])):
                z = sum(1 for p in probs if all(sides[side][p][k][0]["otok"] == 0 for k in (1, 2, 3)))
                notes["otok0_idx123"]["%s|%s" % (comp, c)] = z
                m = sum(1 for p in probs for k in range(4) if sides[side][p][k][0]["otok"] == 64000)
                notes["otok_64000"]["%s|%s" % (comp, c)] = m
        result["cells"].append(cell)
    notes["retained_slots"] = retained
    notes["source_values"] = {k: sorted(map(str, v)) for k, v in notes["source_values"].items()}

    # ---------- aggregates ----------
    QS = ["G", "R", "D", "D_F", "C", "G-D_F"]

    def agg(exclude_seen):
        res = {}
        for q in QS:
            comp_vals = {}
            layers = {}
            for comp in FILES:
                eps = {}
                for c in result["cells"]:
                    if c["competition"] != comp:
                        continue
                    if exclude_seen and c["seen"]:
                        continue
                    eps.setdefault(c["endpoint"], [])
                    if c["contract"] == "valid" and c["_st"][q] is not None:
                        eps[c["endpoint"]].append(c["_st"][q])
                ep_means = [mean(v) for v in eps.values() if v]
                layers[comp] = {"endpoints_valid": len(ep_means), "endpoints_NA": len(eps) - len(ep_means)}
                if ep_means:
                    comp_vals[comp] = mean(ep_means)
            macro = mean(comp_vals.values()) if comp_vals else None
            res[q] = {"macro": frs(macro), "macro_pp": None if macro is None else round(float(macro) * 100, 4),
                      "per_comp": {k: frs(v) for k, v in comp_vals.items()},
                      "per_comp_pp": {k: round(float(v) * 100, 4) for k, v in comp_vals.items()},
                      "layers": layers, "n_comp_valid": len(comp_vals)}
        return res

    result["aggregates"]["all"] = agg(False)
    result["aggregates"]["excluding_seen"] = agg(True)

    # unweighted sign-reversal counts
    for lab, ex in (("all", False), ("excluding_seen", True)):
        vc = [c for c in result["cells"] if c["contract"] == "valid" and not (ex and c["seen"])]
        den = [c for c in vc if c["_st"]["D"] is not None and c["_st"]["D_F"] is not None]
        result["aggregates"]["sign_reversal_" + lab] = "%d/%d" % (sum(c["_st"]["flag"] for c in den), len(den))
        for q in ["R", "C"]:
            vals = sorted(c["_st"][q] for c in den)
            n = len(vals)
            med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            result["aggregates"]["%s_dist_%s" % (q, lab)] = {
                "min": frs(vals[0]), "median": frs(med), "max": frs(vals[-1]),
                "pos": sum(v > 0 for v in vals), "zero": sum(v == 0 for v in vals), "neg": sum(v < 0 for v in vals)}
    for c in result["cells"]:
        c.pop("_st", None)
    result["notes"] = notes
    result["wall_s"] = round(time.time() - T0, 1)
    result["peak_ws_mib"] = round(rv_guard.peak_ws_mib(), 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    log("wrote", OUT, "peak WS MiB", result["peak_ws_mib"])


if __name__ == "__main__":
    main()

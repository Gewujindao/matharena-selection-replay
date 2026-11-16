# Compare the reviewer's recompute (week3/review_recompute.json) with the runner's frame/ and tables/.
# Prints disagreements only (plus counts of items compared).
import csv
import json
import os
from fractions import Fraction as Fr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "week3") + "/"
me = json.load(open(ROOT + "review_recompute.json", encoding="utf-8"))
diffs = []
n = {"configs": 0, "endpoints": 0, "cells": 0, "four_slot": 0, "aggregates": 0}


def rd(p):
    with open(ROOT + p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def F(x):
    if x in (None, "", "NA", "未定义"):
        return None
    return Fr(x)


# ---- configs ----
rows = rd("frame/configs.csv")
seen_keys = set()
for r in rows:
    comp, c = r["competition"], r["model_config"]
    seen_keys.add((comp, c))
    m = me["mapping"][comp]["configs"].get(c)
    n["configs"] += 1
    if m is None:
        diffs.append(("config missing in reviewer", comp, c))
        continue
    mine = {"model_name": m["names"][0] if len(m["names"]) == 1 else "|".join(m["names"]), "n_rows": str(m["rows"]),
            "tier": m.get("tier") or "", "display_base": m.get("display_base", ""), "config_base": m.get("config_base", "")}
    for k, v in mine.items():
        if r[k] != v:
            diffs.append(("config field", comp, c, k, r[k], v))
    st_r = r["status"]
    st_m = m["status"]
    if (st_m == "no_tier") != (st_r == "no_tier") or (st_m == "NA") != (st_r.startswith("NA")):
        diffs.append(("config status", comp, c, st_r, st_m))
for comp, mm in me["mapping"].items():
    for c in mm["configs"]:
        if (comp, c) not in seen_keys:
            diffs.append(("config missing in runner", comp, c))

# ---- endpoints ----
rows = rd("frame/endpoints.csv")
rk = set()
for r in rows:
    comp = r["competition"]
    key = r["display_base"] + " | " + r["config_base"]
    rk.add((comp, key))
    n["endpoints"] += 1
    e = me["mapping"][comp]["endpoints"].get(key)
    if e is None:
        diffs.append(("endpoint missing in reviewer", comp, key))
        continue
    my_tiers = list(e["tiers"].keys())
    r_tiers = [t for t in r["tiers"].replace(";", ",").replace("|", ",").replace(" ", "").split(",") if t]
    if my_tiers != r_tiers:
        diffs.append(("endpoint tiers", comp, key, r["tiers"], my_tiers))
    my_pairs = ["%s->%s" % (a, b) for a, b in e["pairs"]]
    r_pairs = [x for x in r["adjacent_pairs"].replace(";", ",").replace("|", ",").replace(" ", "").replace("→", "->").split(",") if x]
    if sorted(my_pairs) != sorted(r_pairs):
        diffs.append(("endpoint pairs", comp, key, r["adjacent_pairs"], my_pairs))
    if (e["status"] == "ok") != (r["status"] == "ok"):
        diffs.append(("endpoint status", comp, key, r["status"], e["status"]))
for comp, mm in me["mapping"].items():
    for key in mm["endpoints"]:
        if (comp, key) not in rk:
            diffs.append(("endpoint missing in runner", comp, key))

# ---- cells ----
mycells = {c["id"]: c for c in me["cells"]}
rows = rd("tables/cells_all.csv")
rids = set()
QMAP = {"K": "K", "G": "G", "R": "R", "D": "D", "D_F": "D_F", "C=D-D_F": "C", "G-D_F": "G-D_F",
        "sign_reversal_1[D*D_F<0]": "flag", "D_zero": "D_zero", "D_F_zero": "D_F_zero"}
for r in rows:
    cid = r["cell_id"]
    rids.add(cid)
    n["cells"] += 1
    c = mycells.get(cid)
    if c is None:
        diffs.append(("cell missing in reviewer", cid))
        continue
    for k_r, k_m in (("A_config", "A"), ("B_config", "B"), ("tier_A", "tier_A"), ("tier_B", "tier_B")):
        if r[k_r] != c[k_m]:
            diffs.append(("cell field", cid, k_r, r[k_r], c[k_m]))
    if int(r["seen"]) != int(c["seen"]):
        diffs.append(("cell seen", cid, r["seen"], c["seen"]))
    if r["contract"] != c["contract"]:
        diffs.append(("cell contract", cid, r["contract"], c["contract"]))
    if int(r["n_problems_complete"]) != c["n_complete"] or int(r["n_problems_frame"]) != c["n_frame"]:
        diffs.append(("cell coverage", cid, r["n_problems_frame"], r["n_problems_complete"], c["n_frame"], c["n_complete"]))
    if c["contract"] == "valid":
        for k_r, k_m in QMAP.items():
            a, b = F(r[k_r]), F(c["primary"].get(k_m))
            if a != b:
                diffs.append(("cell value", cid, k_r, r[k_r], c["primary"].get(k_m)))
        for k in ("acc_A_4slot", "acc_B_4slot"):
            if F(r[k]) != F(c[k]):
                diffs.append(("cell acc", cid, k, r[k], c[k]))
        if int(r["stratum_A_acc_30_70"]) != c["stratum_30_70"]:
            diffs.append(("cell stratum", cid, r["stratum_A_acc_30_70"], c["stratum_30_70"]))
for cid in mycells:
    if cid not in rids:
        diffs.append(("cell missing in runner", cid))
# frame cells.csv breaking counts
for r in rd("frame/cells.csv"):
    c = mycells.get(r["cell_id"])
    if c and int(r["n_problems_breaking"]) != c["n_breaking"]:
        diffs.append(("frame cell breaking", r["cell_id"], r["n_problems_breaking"], c["n_breaking"]))

# ---- four-slot ----
for r in rd("tables/four_slot_cycle.csv"):
    c = mycells.get(r["cell_id"])
    n["four_slot"] += 1
    j = int(r["screen_idx"])
    m = c["four_slot"][j]
    for k_r, k_m in (("K", "K"), ("G", "G"), ("R", "R"), ("D", "D"), ("D_F", "D_F"), ("C", "C"), ("G-D_F", "G-D_F"),
                     ("sign_reversal", "flag")):
        if F(r[k_r]) != F(m.get(k_m)):
            diffs.append(("four-slot", r["cell_id"], j, k_r, r[k_r], m.get(k_m)))

# ---- aggregates (macro, competition, endpoint layers), recomputed here from reviewer cell values ----
QS = {"G": "G", "R": "R", "D": "D", "D_F": "D_F", "C": "C", "C=D-D_F": "C", "G-D_F": "G-D_F", "G_minus_D_F": "G-D_F"}


def mean(xs):
    xs = list(xs)
    return sum(xs, Fr(0)) / len(xs) if xs else None


mine_layers = {}
for lab, ex in (("all_cells", False), ("excluding_seen", True)):
    for q in set(QS.values()):
        compv = {}
        for comp in ["AIME", "HMMT", "BRUMO", "CMIMC"]:
            eps = {}
            for c in me["cells"]:
                if c["competition"] != comp or (ex and c["seen"]):
                    continue
                eps.setdefault(c["endpoint"], [])
                if c["contract"] == "valid" and c["primary"][q] is not None:
                    eps[c["endpoint"]].append(Fr(c["primary"][q]))
            for ep, v in eps.items():
                mine_layers[(lab, q, "endpoint", comp + " | " + ep)] = mean(v)
            em = [mean(v) for v in eps.values() if v]
            compv[comp] = mean(em)
            mine_layers[(lab, q, "competition", comp)] = compv[comp]
        mine_layers[(lab, q, "macro", "")] = mean(v for v in compv.values() if v is not None)
for r in rd("tables/aggregates.csv"):
    q = QS.get(r["quantity"], r["quantity"])
    key = (r["set"], q, r["layer"], r["unit"])
    n["aggregates"] += 1
    if key not in mine_layers:
        diffs.append(("aggregate key unknown", key))
        continue
    if F(r["exact"]) != mine_layers[key]:
        diffs.append(("aggregate", key, r["exact"], mine_layers[key]))
# cross-check with the reviewer's own json macro values
for lab, lab2 in (("all", "all_cells"), ("excluding_seen", "excluding_seen")):
    for q, v in me["aggregates"][lab].items():
        if F(v["macro"]) != mine_layers[(lab2, q, "macro", "")]:
            diffs.append(("internal macro mismatch", lab, q))

print("compared:", n)
print("disagreements:", len(diffs))
for d in diffs:
    print("  ", d)

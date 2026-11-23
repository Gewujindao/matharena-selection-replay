"""All-pairs run, step 2: decode the stored `correct` for the kept slots of the complete (valid) cells of the frame
and compute, per cell, K, G, R, D, D_F, C = D - D_F, G - D_F and the sign-reversal flag 1[D * D_F < 0] (zeros kept and
listed) as exact fractions; the four-slot cycle as sensitivity only; the fixed-weight aggregates (all cells and excluding
seen cells) and the unweighted cell distribution.  Writes results/all_pairs/summary.json and results/all_pairs/tables/.

Run (after build_frame.py):  python -X utf8 -I -B compute_quantities.py
"""
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_pairs_rules as rules  # noqa: E402

helpers, P = rules.helpers, rules.P
LOG = rules.RESULTS / "run_log.txt"


def log(msg):
    line = f"[{helpers.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def jsq(o):
    keys = ["screen_idx", "K", "n_F", "S", "A1_S", "B1_S", "G", "R", "D", "D_F", "C", "G_minus_D_F", "sign_reversal",
            "D_zero", "D_F_zero"]
    return {k: (rules.fr(o[k]) if isinstance(o.get(k), Fraction) else o.get(k)) for k in keys}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def distribution(cells, quant):
    vals = [(c["cell_id"], Fraction(c["primary"][quant])) for c in cells
            if c["valid"] and c["primary"][quant] is not None]
    xs = [v for _, v in vals]
    return dict(quantity=quant, n_cells=len(cells), n_defined=len(xs),
                n_contract_NA=sum(not c["valid"] for c in cells),
                n_undefined_K0=sum(1 for c in cells if c["valid"] and c["primary"][quant] is None),
                min=rules.fr(min(xs)) if xs else None, median=rules.fr(median(xs)), max=rules.fr(max(xs)) if xs else None,
                n_positive=sum(x > 0 for x in xs), n_zero=sum(x == 0 for x in xs), n_negative=sum(x < 0 for x in xs),
                values=[dict(cell_id=i, value=rules.fr(v)) for i, v in sorted(vals, key=lambda t: t[1])])


def reversal(cells):
    den = [c for c in cells if c["valid"] and c["primary"]["D"] is not None and c["primary"]["D_F"] is not None]
    rev = [c["cell_id"] for c in den if c["primary"]["sign_reversal"]]
    return dict(denominator_cells_with_D_and_D_F_defined=len(den), n_sign_reversal=len(rev),
                share=rules.fr(Fraction(len(rev), len(den))) if den else None, reversal_cells=rev,
                cells_with_D_zero=[c["cell_id"] for c in den if c["primary"]["D"] == 0],
                cells_with_D_F_zero=[c["cell_id"] for c in den if c["primary"]["D_F"] == 0],
                n_cells_total=len(cells), n_contract_NA=sum(not c["valid"] for c in cells),
                n_valid_K0=sum(1 for c in cells if c["valid"] and c["primary"]["D"] is None))


def agg_json(rep):
    out = dict(quantity=rep["quantity"], macro=rules.fr(rep["macro"]), n_competitions=rep["n_competitions"],
               n_competitions_valid=rep["n_competitions_valid"], n_competitions_NA=rep["n_competitions_NA"],
               competitions={})
    for comp, d in rep["competitions"].items():
        out["competitions"][comp] = dict(value=rules.fr(d["value"]), n_endpoints=d["n_endpoints"],
                                         n_endpoints_valid=d["n_endpoints_valid"], n_endpoints_NA=d["n_endpoints_NA"],
                                         n_cells=d["n_cells"], n_cells_valid=d["n_cells_valid"],
                                         n_cells_contract_NA=d["n_cells_contract_NA"],
                                         n_cells_undefined_K0=d["n_cells_undefined_K0"],
                                         endpoints={e: dict(v, value=rules.fr(v["value"])) for e, v in
                                                    d["endpoints"].items()})
    return out


def main():
    t0 = time.time()
    log("step 2 start")
    frame = helpers.read_json(rules.FRAME_DIR / "frame.json")
    cells = []
    slot_rows, per_rows, cyc_rows = [], [], []
    for comp in rules.COMP_ORDER:
        comp_cells = [c for c in frame["cells"] if c["comp"] == comp]
        valid_cells = [c for c in comp_cells if c["contract_valid"]]
        corr_all = {}
        if valid_cells:
            pf = P.ParquetFile(str(rules.COMPS[comp]["parquet"]))
            need = {r for c in valid_cells for q in c["kept"].values() for r in q.values()}
            vals = rules.scan_column(pf, "correct", need, lambda v: v)
            otok = rules.scan_column(pf, "output_tokens", need, lambda v: v)
            for r in need:
                if vals.get(r) is None or not isinstance(vals[r], bool):
                    raise rules.StopNA(f"{comp}: correct missing / not boolean at kept row {r}")
                corr_all[r] = int(vals[r])
            log(f"{comp}: read correct for {len(need)} kept rows")
        for c in comp_cells:
            rec = dict(cell_id=c["cell_id"], comp=comp, endpoint=c["endpoint"], tier_A=c["tier_A"], tier_B=c["tier_B"],
                       A=c["A"], B=c["B"], A_name=c["A_name"], B_name=c["B_name"], seen=c["seen"],
                       seen_label=c["seen_label"], valid=c["contract_valid"], na_reason=c["na_reason"],
                       n_problems_frame=c["n_problems_frame"], n_problems_complete=c["n_problems_complete"],
                       primary=None, cycle=None)
            if c["contract_valid"]:
                F = sorted(c["kept"], key=rules.pidx_key)
                corr = {(key[0], q, int(key[1:])): corr_all[r] for q, v in c["kept"].items() for key, r in v.items()}
                rots = [rules.ledger(F, corr, screen=j) for j in rules.IDX]
                prim, per0 = rots[0]
                rec["primary"] = prim
                rec["cycle"] = [o for o, _ in rots]
                accA = Fraction(sum(corr[("A", q, k)] for q in F for k in rules.IDX), 4 * len(F))
                accB = Fraction(sum(corr[("B", q, k)] for q in F for k in rules.IDX), 4 * len(F))
                rec["acc_A_4slot"] = accA
                rec["acc_B_4slot"] = accB
                rec["acc_A_idx0"] = Fraction(sum(corr[("A", q, 0)] for q in F), len(F))
                rec["stratum_30_70"] = Fraction(3, 10) <= accA <= Fraction(7, 10)
                rec["A_all4_wrong"] = sum(1 for q in F if all(corr[("A", q, k)] == 0 for k in rules.IDX))
                rec["A_all4_right"] = sum(1 for q in F if all(corr[("A", q, k)] == 1 for k in rules.IDX))
                for p in per0:
                    per_rows.append([c["cell_id"], p["q"], int(p["A0"]), rules.frac(p["A1"]), rules.frac(p["B1"]),
                                     int(p["B0"]), int(p["in_S"])])
                for o, _ in rots:
                    cyc_rows.append([c["cell_id"], int(c["seen"]), o["screen_idx"], o["K"], o["n_F"],
                                     rules.frac(o["G"]), rules.frac(o["R"]), rules.frac(o["D"]), rules.frac(o["D_F"]), rules.frac(o["C"]),
                                     rules.frac(o["G_minus_D_F"]), "" if o["sign_reversal"] is None else o["sign_reversal"]])
                for q in F:
                    for key, r in sorted(c["kept"][q].items()):
                        slot_rows.append([c["cell_id"], key[0], q, int(key[1:]), r, corr_all[r], otok[r]])
                log(f"{c['cell_id']}: K={prim['K']}/{prim['n_F']} G={rules.frac(prim['G'])} R={rules.frac(prim['R'])} "
                    f"D={rules.frac(prim['D'])} D_F={rules.frac(prim['D_F'])} C={rules.frac(prim['C'])} "
                    f"flag={prim['sign_reversal']}")
            cells.append(rec)
    t_read = round(time.time() - t0, 2)

    seen_out = lambda c: not c["seen"]  # noqa: E731
    aggs = {q: rules.aggregate(cells, q) for q in rules.QUANTS}
    aggs_unseen = {q: rules.aggregate(cells, q, include=seen_out) for q in rules.QUANTS}
    unseen = [c for c in cells if not c["seen"]]
    dist = {q: distribution(cells, q) for q in ["K"] + rules.QUANTS}
    dist_unseen = {q: distribution(unseen, q) for q in ["K"] + rules.QUANTS}
    strat = {}
    for lab, sel in (("A_acc_in_30_70", True), ("A_acc_outside_30_70", False)):
        cs = [c for c in cells if c["valid"] and c["stratum_30_70"] == sel]
        strat[lab] = dict(n_cells=len(cs), cells=[c["cell_id"] for c in cs],
                          R=distribution(cs, "R"), C=distribution(cs, "C"), reversal=reversal(cs))

    # ---------------------------------------------------------------- tables
    rules.TABLES.mkdir(parents=True, exist_ok=True)
    hdr = ["cell_id", "competition", "endpoint", "tier_A", "tier_B", "A_config", "B_config", "A_name", "B_name",
           "seen", "seen_label", "contract", "na_reason", "n_problems_frame", "n_problems_complete", "K", "G", "R", "D",
           "D_F", "C=D-D_F", "G-D_F", "sign_reversal_1[D*D_F<0]", "D_zero", "D_F_zero", "G_pp", "R_pp", "D_pp",
           "D_F_pp", "C_pp", "G-D_F_pp", "acc_A_4slot", "acc_B_4slot", "stratum_A_acc_30_70", "A_all4_wrong",
           "A_all4_right"]
    rows = []
    for c in cells:
        p = c["primary"]
        base = [c["cell_id"], c["comp"], c["endpoint"], c["tier_A"], c["tier_B"], c["A"], c["B"], c["A_name"],
                c["B_name"], int(c["seen"]), c["seen_label"], "valid" if c["valid"] else "NA", c["na_reason"],
                c["n_problems_frame"], c["n_problems_complete"]]
        if p is None:
            rows.append(base + ["NA"] * (len(hdr) - len(base)))
            continue
        f = lambda k: "undefined (K=0)" if p[k] is None else rules.frac(p[k])  # noqa: E731
        g = lambda k: "" if p[k] is None else round(float(p[k]) * 100, 4)  # noqa: E731
        rows.append(base + [p["K"], f("G"), f("R"), f("D"), f("D_F"), f("C"), f("G_minus_D_F"),
                            "" if p["sign_reversal"] is None else p["sign_reversal"],
                            "" if p["D_zero"] is None else p["D_zero"], p["D_F_zero"], g("G"), g("R"), g("D"),
                            g("D_F"), g("C"), g("G_minus_D_F"), rules.frac(c["acc_A_4slot"]), rules.frac(c["acc_B_4slot"]),
                            int(c["stratum_30_70"]), c["A_all4_wrong"], c["A_all4_right"]])
    helpers.write_csv(rules.TABLES / "pairs_all.csv", hdr, rows)
    helpers.write_csv(rules.TABLES / "per_problem_primary.csv",
                 ["cell_id", "problem_idx", "A0_idx0", "A1_mean_idx1-3", "B1_mean_idx1-3", "B0_idx0", "in_S"],
                 per_rows)
    helpers.write_csv(rules.TABLES / "four_slot_cycle.csv",
                 ["cell_id", "seen", "screen_idx", "K", "n_F", "G", "R", "D", "D_F", "C", "G-D_F", "sign_reversal"],
                 cyc_rows)
    helpers.write_csv(rules.TABLES / "slot_correct.csv",
                 ["cell_id", "side", "problem_idx", "idx", "parquet_row", "correct", "output_tokens"], slot_rows)
    arows = []
    for lab, A in (("all_cells", aggs), ("excluding_seen", aggs_unseen)):
        for q, rep in A.items():
            arows.append([lab, q, "macro", "", rules.frac(rep["macro"]),
                          "" if rep["macro"] is None else round(float(rep["macro"]) * 100, 4),
                          f"competitions valid {rep['n_competitions_valid']} / NA {rep['n_competitions_NA']}"])
            for comp, d in rep["competitions"].items():
                arows.append([lab, q, "competition", comp, rules.frac(d["value"]),
                              "" if d["value"] is None else round(float(d["value"]) * 100, 4),
                              f"endpoints valid {d['n_endpoints_valid']} / NA {d['n_endpoints_NA']}; cells valid "
                              f"{d['n_cells_valid']} / contract NA {d['n_cells_contract_NA']} / K=0 "
                              f"{d['n_cells_undefined_K0']}"])
                for e, v in d["endpoints"].items():
                    arows.append([lab, q, "endpoint", f"{comp} | {e}", rules.frac(v["value"]),
                                  "" if v["value"] is None else round(float(v["value"]) * 100, 4),
                                  f"cells {v['n_cells']}: valid {v['n_valid']} / contract NA {v['n_contract_NA']} / "
                                  f"K=0 {v['n_undefined_K0']}"])
    helpers.write_csv(rules.TABLES / "aggregates.csv", ["set", "quantity", "layer", "unit", "exact", "pp", "counts"], arows)

    summary = dict(
        frame=str(rules.FRAME_DIR / "frame.json"),
        written_local=helpers.now_local(), data=frame["data"], transfer=frame["transfer"],
        coverage={comp: dict(n_configs=frame["competitions"][comp]["n_configs"],
                             n_problems=frame["competitions"][comp]["n_problems"],
                             n_tiered=frame["competitions"][comp]["n_tiered"],
                             n_no_tier=frame["competitions"][comp]["n_no_tier"],
                             n_config_NA=frame["competitions"][comp]["n_config_NA"],
                             n_eligible_pairs=frame["competitions"][comp]["n_eligible_pairs"],
                             n_cells_valid=sum(1 for c in cells if c["comp"] == comp and c["valid"]),
                             n_cells_NA=sum(1 for c in cells if c["comp"] == comp and not c["valid"]),
                             na_cells={c["cell_id"]: c["na_reason"] for c in cells if c["comp"] == comp and not c["valid"]})
                  for comp in rules.COMP_ORDER},
        cells=[dict(cell_id=c["cell_id"], comp=c["comp"], endpoint=c["endpoint"], tier_A=c["tier_A"],
                    tier_B=c["tier_B"], A=c["A"], B=c["B"], seen=c["seen"], seen_label=c["seen_label"],
                    valid=c["valid"], na_reason=c["na_reason"], n_problems_frame=c["n_problems_frame"],
                    n_problems_complete=c["n_problems_complete"],
                    primary=None if c["primary"] is None else jsq(c["primary"]),
                    cycle=None if c["cycle"] is None else [jsq(o) for o in c["cycle"]],
                    acc_A_4slot=rules.fr(c.get("acc_A_4slot")), acc_B_4slot=rules.fr(c.get("acc_B_4slot")),
                    acc_A_idx0=rules.fr(c.get("acc_A_idx0")), stratum_A_acc_30_70=c.get("stratum_30_70"),
                    A_all4_wrong=c.get("A_all4_wrong"), A_all4_right=c.get("A_all4_right")) for c in cells],
        aggregates_all={q: agg_json(r) for q, r in aggs.items()},
        aggregates_excluding_seen={q: agg_json(r) for q, r in aggs_unseen.items()},
        distribution_unweighted_all=dist, distribution_unweighted_excluding_seen=dist_unseen,
        sign_reversal_all=reversal(cells), sign_reversal_excluding_seen=reversal(unseen),
        stratum_A_acc_30_70=strat,
        cycle_ranges={c["cell_id"]: {q: [rules.fr(min(o[q] for o in c["cycle"] if o[q] is not None)),
                                         rules.fr(max(o[q] for o in c["cycle"] if o[q] is not None))]
                                     if any(o[q] is not None for o in c["cycle"]) else None
                                     for q in rules.QUANTS} for c in cells if c["cycle"]},
        resources=dict(step1_s=frame["step1_seconds"],
                       step2_s=round(time.time() - t0, 2), step2_read_s=t_read))
    helpers.write_json(rules.RESULTS / "summary.json", summary)
    log(f"summary written ({summary['resources']['step2_s']} s)")


if __name__ == "__main__":
    try:
        main()
    except rules.StopNA as e:
        log(f"STOP: {e}")
        sys.exit(3)

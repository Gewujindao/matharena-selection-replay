"""HMMT o3-mini run, step 2: read the stored `correct` for the 240 kept slots of the frame and compute the
ledger quantities (primary: screening slot idx0; four-slot cycle as sensitivity only).

Run (after build_frame.py):  python -X utf8 -I -B compute_quantities.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmmt_pair_rules as rules  # noqa: E402

helpers, P = rules.helpers, rules.P
LOG = rules.RESULTS / "run_log.txt"
OFFICIAL_PCT = {"A": "53.33%", "B": "67.50%"}      # official leaderboard accuracy (development info only)


def log(msg):
    line = f"[{helpers.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def js(o):
    return dict(screen_idx=o["screen_idx"], K=o["K"], n_F=o["n_F"], S=o["S"], A0_S=rules.fr(o["A0_S"]),
                A1_S=rules.fr(o["A1_S"]), B1_S=rules.fr(o["B1_S"]), G=rules.fr(o["G"]), R=rules.fr(o["R"]), D=rules.fr(o["D"]),
                D_F=rules.fr(o["D_F"]), D_minus_D_F=rules.fr(o["D_minus_D_F"]))


def main():
    t0 = time.time()
    log("step 2 start")
    frame = helpers.read_json(rules.FRAME_DIR / "frame.json")
    if not frame["full_30_contract"]:
        raise rules.StopNA("frame does not hold the full 30-problem contract")
    F = frame["F"]
    rowmap = {}
    for q, v in frame["kept"].items():
        for key, r in v.items():
            rowmap[(key[0], q, int(key[1:]))] = r
    pf = P.ParquetFile(str(rules.PARQUET))
    need = set(rowmap.values())
    correct = pf.read_column("correct", rows=need)
    out_tok = pf.read_column("output_tokens")
    cfg_all = pf.read_column("model_config")
    names_all = pf.read_column("model_name")
    corr = {}
    for key, r in rowmap.items():
        v = correct[r]
        if v is None or v is P.SKIPPED:
            raise rules.StopNA(f"correct missing for kept slot {key} (row {r})")
        corr[key] = int(bool(v))
    t_read = round(time.time() - t0, 2)
    log(f"read correct for {len(corr)} kept slots ({len(F)} problems)")

    rots, pers = [], []
    for j in rules.IDX:
        o, per = rules.ledger(F, corr, screen=j)
        rots.append(o)
        pers.append(per)
    prim, per0 = rots[0], pers[0]
    defined = [o for o in rots if o["D"] is not None]

    def rng(key):
        return None if not defined else [rules.fr(min(o[key] for o in defined)), rules.fr(max(o[key] for o in defined))]

    acc = {s: helpers.mean(helpers.Fraction(corr[(s, q, k)]) for q in F for k in rules.IDX) for s in ("A", "B")}
    acc_n = {s: sum(corr[(s, q, k)] for q in F for k in rules.IDX) for s in ("A", "B")}
    b_rows = [rowmap[("B", q, k)] for q in F for k in rules.IDX]
    b_over = sum(1 for r in b_rows if (out_tok[r] or 0) > 32000)
    o3_configs = sorted({(c, nm) for c, nm in zip(cfg_all, names_all) if c and "o3-mini" in c})
    a0_counts = dict(A0_correct=sum(corr[("A", q, 0)] for q in F), B0_correct=sum(corr[("B", q, 0)] for q in F))
    s_counts = dict(A_rescore_hits_on_S=sum(corr[("A", q, k)] for q in prim["S"] for k in (1, 2, 3)),
                    B_rescore_hits_on_S=sum(corr[("B", q, k)] for q in prim["S"] for k in (1, 2, 3)),
                    rescore_records_per_side_on_S=3 * prim["K"])
    pattern = {}
    for p in per0:
        key = (int(p["A0"]), str(p["A1"]), str(p["B1"]))
        pattern[key] = pattern.get(key, 0) + 1

    summary = dict(
        evidence_level="Data first read and computed after the analysis plan was fixed; the data are retrospective public cached records. The pair was chosen by public leaderboard scores, so it is not a blind validation of the effect.",
        data=frame["data"], config_map=dict(A=dict(raw_model_config=frame["config_match"]["mapping"]["A"],
                                                   model_name="o3-mini (medium)", official_identifier="o3-mini--medium"),
                                            B=dict(raw_model_config=frame["config_match"]["mapping"]["B"],
                                                   model_name="o3-mini (high)", official_identifier="o3-mini--high"),
                                            rule_detail=frame["config_match"]["detail"],
                                            o3_mini_configs_in_file=[list(x) for x in o3_configs]),
        frame=dict(n_candidates=frame["n_candidates"], n_F=len(F), full_30_contract=frame["full_30_contract"],
                   breaks=frame["breaks"], decided_before_correct=frame["decided_before_correct"]),
        primary=js(prim), primary_counts=dict(**a0_counts, **s_counts),
        sensitivity_four_slot_cycle=[js(o) for o in rots],
        sensitivity_range_where_defined=dict(n_defined=len(defined), G=rng("G"), R=rng("R"), D=rng("D"),
                                             D_minus_D_F=rng("D_minus_D_F")),
        per_problem_patterns_p0=[dict(A0=k[0], A1=k[1], B1=k[2], n=v) for k, v in sorted(pattern.items())],
        descriptive_not_registered=dict(
            A_all4_on_F=rules.fr(acc["A"]), A_all4_hits=acc_n["A"], B_all4_on_F=rules.fr(acc["B"]), B_all4_hits=acc_n["B"],
            n_records_per_side=4 * len(F), official_page_pct_quoted_in_design=OFFICIAL_PCT,
            B_rows_output_tokens_over_32000=b_over, call_conditions=frame["call_condition_attributes"]),
        uncertainty="Exact description of a fixed 30-problem ledger; no confidence interval over a problem population or independent calls is given. The four-slot cycle shares the same cached records, "
                    "is a sensitivity analysis only, is not four independent repetitions, and the primary slot is not chosen by results.")
    rules.TABLES.mkdir(parents=True, exist_ok=True)
    with open(rules.TABLES / "per_problem_p0.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem_idx", "screen_idx", "rescore_idx", "A0", "A1", "B1", "B1_minus_A1", "in_S", "B0_not_used"])
        for p in per0:
            w.writerow([p["q"], p["screen_idx"], " ".join(map(str, p["rescore_idx"])), int(p["A0"]), rules.frac(p["A1"]),
                        rules.frac(p["B1"]), rules.frac(p["B1"] - p["A1"]), int(p["in_S"]), int(p["B0"])])
    with open(rules.TABLES / "four_slot_cycle.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["screen_idx", "role", "K", "n_F", "A1_S", "B1_S", "G", "R", "D", "D_F", "D_minus_D_F"])
        g = lambda x: "" if x is None else f"{rules.frac(x)} ({float(x):.4f})"  # noqa: E731
        for o in rots:
            w.writerow([o["screen_idx"], "primary" if o["screen_idx"] == 0 else "sensitivity", o["K"], o["n_F"],
                        g(o["A1_S"]), g(o["B1_S"]), g(o["G"]), g(o["R"]), g(o["D"]), g(o["D_F"]), g(o["D_minus_D_F"])])
    with open(rules.TABLES / "slot_correct.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "problem_idx", "idx", "parquet_row", "correct", "output_tokens"])
        for (s, q, k), r in sorted(rowmap.items(), key=lambda kv: (int(kv[0][1]), kv[0][0], kv[0][2])):
            w.writerow([s, q, k, r, corr[(s, q, k)], out_tok[r]])

    t2 = round(time.time() - t0, 2)
    s1 = frame["timings"]["step1_total_s"]
    summary["timings"] = dict(download_s=56.04, step1_total_s=s1, step2_read_s=t_read, step2_total_s=t2,
                              cumulative_cpu_process_s=round(s1 + t2, 2), cap_s=rules.TIME_CAP_S)
    if s1 + t2 > rules.TIME_CAP_S:
        summary["decision"] = "NA (cumulative time cap exceeded)"
    helpers.write_json(rules.RESULTS / "summary.json", summary)
    log(f"primary (idx0 screening): K={prim['K']}, G={rules.frac(prim['G'])}, R={rules.frac(prim['R'])}, "
        f"D={rules.frac(prim['D'])}, D_F={rules.frac(prim['D_F'])}, D-D_F={rules.frac(prim['D_minus_D_F'])}")


if __name__ == "__main__":
    try:
        main()
    except rules.StopNA as e:
        helpers.write_json(rules.RESULTS / "STOP_NA.json", dict(decision="NA", step="compute", reason=str(e),
                                                  time_local=helpers.now_local()))
        log(f"NA STOP at compute: {e}")
        sys.exit(3)

"""Week 2, stage 2: read the stored `correct` for the 240 kept slots of the read-only frame and compute the frozen
ledger quantities (primary: screening slot idx0; four-slot cycle as sensitivity only).

Run (after w2_extract.py):  python -X utf8 -I -B w2_compute.py
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w2_common as W  # noqa: E402

C1, P = W.C1, W.P
LOG = W.W2 / "run_log.txt"
TIMELINE = W.W2 / "timeline.jsonl"
OFFICIAL_PCT = {"A": "53.33%", "B": "67.50%"}      # quoted in the frozen design (development info only)


def log(msg):
    line = f"[{C1.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def event(name, **kw):
    with open(TIMELINE, "a", encoding="utf8") as f:
        f.write(json.dumps(dict(t_local=C1.now_local(), event=name, **kw), ensure_ascii=False) + "\n")


def js(o):
    return dict(screen_idx=o["screen_idx"], K=o["K"], n_F=o["n_F"], S=o["S"], A0_S=W.fr(o["A0_S"]),
                A1_S=W.fr(o["A1_S"]), B1_S=W.fr(o["B1_S"]), G=W.fr(o["G"]), R=W.fr(o["R"]), D=W.fr(o["D"]),
                D_F=W.fr(o["D_F"]), D_minus_D_F=W.fr(o["D_minus_D_F"]))


def main():
    if (W.W2 / "summary.json").exists():
        log("summary.json exists; stage 2 is not re-run")
        sys.exit(2)
    t0 = time.time()
    limits = C1.apply_limits()
    log(f"stage 2 start; pid {os.getpid()}; limits {limits}")
    frame = C1.read_json(W.FRAME_DIR / "frame.json")
    if not frame["full_30_contract"]:
        raise W.StopNA("frame does not hold the full 30-problem contract")
    F = frame["F"]
    rowmap = {}
    for q, v in frame["kept"].items():
        for key, r in v.items():
            rowmap[(key[0], q, int(key[1:]))] = r
    event("stage2_start", pid=os.getpid(), n_F=len(F))
    C1.wait_ram(log, "reading correct")
    pf = P.ParquetFile(str(W.PARQUET))
    need = set(rowmap.values())
    correct = pf.read_column("correct", rows=need)
    out_tok = pf.read_column("output_tokens")
    cfg_all = pf.read_column("model_config")
    names_all = pf.read_column("model_name")
    corr = {}
    for key, r in rowmap.items():
        v = correct[r]
        if v is None or v is P.SKIPPED:
            raise W.StopNA(f"correct missing for kept slot {key} (row {r})")
        corr[key] = int(bool(v))
    t_read = round(time.time() - t0, 2)
    log(f"read correct for {len(corr)} kept slots ({len(F)} problems)")

    rots, pers = [], []
    for j in W.IDX:
        o, per = W.ledger(F, corr, screen=j)
        rots.append(o)
        pers.append(per)
    prim, per0 = rots[0], pers[0]
    defined = [o for o in rots if o["D"] is not None]

    def rng(key):
        return None if not defined else [W.fr(min(o[key] for o in defined)), W.fr(max(o[key] for o in defined))]

    acc = {s: C1.mean(C1.Fraction(corr[(s, q, k)]) for q in F for k in W.IDX) for s in ("A", "B")}
    acc_n = {s: sum(corr[(s, q, k)] for q in F for k in W.IDX) for s in ("A", "B")}
    b_rows = [rowmap[("B", q, k)] for q in F for k in W.IDX]
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
        card="matharena-selection-replay", week=2, prereg=str(W.PREREG),
        evidence_level="冻结设计后首次取数与计算；数据是公开缓存的回顾性记录。配对按公开榜分选型，不算未见效果验证。",
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
            A_all4_on_F=W.fr(acc["A"]), A_all4_hits=acc_n["A"], B_all4_on_F=W.fr(acc["B"]), B_all4_hits=acc_n["B"],
            n_records_per_side=4 * len(F), official_page_pct_quoted_in_design=OFFICIAL_PCT,
            B_rows_output_tokens_over_32000=b_over, call_conditions=frame["call_condition_attributes"]),
        uncertainty="固定 30 题账本的精确描述；按冻结设计不给题目总体或独立调用置信区间。四槽循环共享同一批缓存记录，"
                    "只作敏感性，不是四次独立重复，不据结果选主槽。",
        limits=limits)
    W.TABLES.mkdir(parents=True, exist_ok=True)
    with open(W.TABLES / "per_problem_p0.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem_idx", "screen_idx", "rescore_idx", "A0", "A1", "B1", "B1_minus_A1", "in_S", "B0_not_used"])
        for p in per0:
            w.writerow([p["q"], p["screen_idx"], " ".join(map(str, p["rescore_idx"])), int(p["A0"]), W.frac(p["A1"]),
                        W.frac(p["B1"]), W.frac(p["B1"] - p["A1"]), int(p["in_S"]), int(p["B0"])])
    with open(W.TABLES / "four_slot_cycle.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["screen_idx", "role", "K", "n_F", "A1_S", "B1_S", "G", "R", "D", "D_F", "D_minus_D_F"])
        g = lambda x: "" if x is None else f"{W.frac(x)} ({float(x):.4f})"  # noqa: E731
        for o in rots:
            w.writerow([o["screen_idx"], "primary" if o["screen_idx"] == 0 else "sensitivity", o["K"], o["n_F"],
                        g(o["A1_S"]), g(o["B1_S"]), g(o["G"]), g(o["R"]), g(o["D"]), g(o["D_F"]), g(o["D_minus_D_F"])])
    with open(W.TABLES / "slot_correct.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "problem_idx", "idx", "parquet_row", "correct", "output_tokens"])
        for (s, q, k), r in sorted(rowmap.items(), key=lambda kv: (int(kv[0][1]), kv[0][0], kv[0][2])):
            w.writerow([s, q, k, r, corr[(s, q, k)], out_tok[r]])

    t2 = round(time.time() - t0, 2)
    s1 = frame["timings"]["stage1_total_s"]
    summary["timings"] = dict(download_s=56.04, stage1_total_s=s1, stage2_read_s=t_read, stage2_total_s=t2,
                              cumulative_cpu_process_s=round(s1 + t2, 2), cap_s=W.TIME_CAP_S)
    summary["peak_working_set_mib"] = dict(stage1=frame.get("peak_working_set_mib"), stage2=C1.peak_rss_mb()[0])
    if s1 + t2 > W.TIME_CAP_S:
        summary["decision"] = "NA (cumulative time cap exceeded)"
    C1.write_json(W.W2 / "summary.json", summary)
    event("summary_written", K=prim["K"], stage2_s=t2)
    log(f"primary (idx0 screening): K={prim['K']}, G={W.frac(prim['G'])}, R={W.frac(prim['R'])}, "
        f"D={W.frac(prim['D'])}, D_F={W.frac(prim['D_F'])}, D-D_F={W.frac(prim['D_minus_D_F'])}")


if __name__ == "__main__":
    try:
        main()
    except (W.StopNA, C1.StopGate) as e:
        C1.write_json(W.W2 / "STOP_NA.json", dict(decision="NA", stage="compute", reason=str(e),
                                                  time_local=C1.now_local()))
        log(f"NA STOP at compute: {e}")
        sys.exit(3)

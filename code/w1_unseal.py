"""Stage 2: unseal the stored `correct` column and compute the frozen quantities on the sealed frame and roles.

Run (only after stage 1 sealed results/frozen/):  python -X utf8 -I w1_unseal.py
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1_common as C  # noqa: E402
import pq_reader as P  # noqa: E402

LOG = C.RESULTS / "run_log.txt"
TIMELINE = C.RESULTS / "timeline.jsonl"
SEALED = ["manifest_frozen.json", "seed_manifest.json", "seed_check.json", "exclusions.csv", "roles.csv",
          "slots.csv", "gates.json"]


def log(msg):
    line = f"[{C.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def event(name, **kw):
    with open(TIMELINE, "a", encoding="utf8") as f:
        f.write(json.dumps(dict(t_local=C.now_local(), event=name, **kw), ensure_ascii=False) + "\n")


def load_sealed():
    for fn in SEALED:
        if not C.is_read_only(C.FROZEN / fn):
            raise C.StopGate(f"sealed file {fn} missing or writable")
    man = C.read_json(C.FROZEN / "manifest_frozen.json")
    gates = C.read_json(C.FROZEN / "gates.json")
    seeds = C.read_json(C.FROZEN / "seed_check.json")
    if not (gates["entry"] and gates["cost"] and gates["execution"] and gates["seed_collision_free"]
            and seeds["collision_free"]):
        raise C.StopGate("a sealed gate did not pass")
    with open(C.FROZEN / "roles.csv", encoding="utf8") as f:
        roles = list(csv.DictReader(f))
    F = [tuple(q) for q in man["F"]]
    order, rowmap = {}, {}
    for r in roles:
        q = (r["source"], r["problem_idx"])
        o = [int(r[f"p{j}"]) for j in range(4)]
        if o != C.role_order(q):
            raise C.StopGate(f"sealed role order for {q} differs from the namespace-hash rule")
        order[q] = o
        if r["in_F"] == "1":
            for j, k in enumerate(o):
                rowmap[("A", q, k)] = int(r[f"A_row_p{j}"])
                rowmap[("B", q, k)] = int(r[f"B_row_p{j}"])
    if sorted(q for q in order if (("A", q, 0) in rowmap)) != sorted(F):
        raise C.StopGate("sealed roles and sealed F disagree")
    return man, gates, seeds, F, order, rowmap


def build_summary(man, gates, F, order, corr, limits, timings):
    rots, pers = [], []
    for j in range(4):
        out, per = C.replay_metrics(F, order, corr, j=j)
        rots.append(out)
        pers.append(per)
    prim = rots[0]
    # descriptive background (not a registered quantity): all four slots on F
    acc = {s: C.mean(C.Fraction(corr[(s, q, k)]) for q in F for k in C.IDX) for s in ("A", "B")}

    def js(o):
        return dict(rotation=o["rotation"], K=o["K"], F=o["F"], S=[list(q) for q in o["S"]],
                    A1_S=C.fr(o["A1_S"]), B1_S=C.fr(o["B1_S"]), G_naive=C.fr(o["G_naive"]), R=C.fr(o["R"]),
                    D=C.fr(o["D"]), D_F=C.fr(o["D_F"]), D_minus_D_F=C.fr(o["D_minus_D_F"]))

    defined = [o for o in rots if o["D"] is not None]

    def rng(key):
        return None if not defined else [C.fr(min(o[key] for o in defined)), C.fr(max(o[key] for o in defined))]

    summary = dict(
        line=160, card="matharena-selection-replay", evidence_level="前瞻（预注册冻结后首次取数与计算）；数据为公开缓存的回顾性记录",
        data=dict(repo=C.REPO, revision=C.REV, file=C.PARQUET_PATH_IN_REPO, url=C.URL, license=C.LICENSE,
                  bytes=man["data"]["entry_gate"]["local_size"], sha256=C.LFS_SHA256,
                  snapshot_commit_date_utc=man["data"]["entry_gate"]["snapshot_commit_date_utc"]),
        gates=dict(entry=gates["entry"], cost=gates["cost"], execution=gates["execution"],
                   seed_collision_free=gates["seed_collision_free"]),
        frame=dict(n_candidates=len(man["candidates"]), n_F=len(F), target=C.TARGET_N, excluded=man["exclusions"]),
        primary=js(prim), sensitivity_rotations=[js(o) for o in rots],
        rotation_range_where_defined=dict(n_defined=len(defined), G_naive=rng("G_naive"), R=rng("R"), D=rng("D"),
                                          D_minus_D_F=rng("D_minus_D_F")),
        descriptive_background_not_registered=dict(A_all4_on_F=C.fr(acc["A"]), B_all4_on_F=C.fr(acc["B"])),
        uncertainty="不强加 95% CI（预注册）：主结果是 30 题有限记录的精确描述；四套 p0 轮换仅作角色协议敏感性，"
                    "共享同一缓存，不是四次独立实验，不除以 sqrt(4)。",
        timings=dict(timings), limits=limits)
    return summary, rots, pers


def main():
    if (C.RESULTS / "summary.json").exists():
        log("summary.json exists; stage 2 is not re-run")
        sys.exit(2)
    t0 = time.time()
    limits = C.apply_limits()
    log(f"stage 2 start; pid {os.getpid()}; limits {limits}")
    man, gates, seeds, F, order, rowmap = load_sealed()
    event("unseal_start", pid=os.getpid(), n_F=len(F))
    C.wait_ram(log, "reading correct")
    if C.sha256_file(C.PARQUET) != C.LFS_SHA256:
        raise C.StopGate("parquet bytes changed after sealing")
    pf = P.ParquetFile(str(C.PARQUET))
    need = set(rowmap.values())
    correct = pf.read_column("correct", rows=need)
    corr = {}
    for key, row in rowmap.items():
        v = correct[row]
        if v is None or v is P.SKIPPED:
            raise C.StopGate(f"correct missing for sealed slot {key} (row {row})")
        corr[key] = int(bool(v))
    t_read = round(time.time() - t0, 2)
    log(f"unsealed correct for {len(corr)} sealed slots ({len(F)} problems)")
    event("unsealed", n_slots=len(corr))

    summary, rots, pers = build_summary(man, gates, F, order, corr, limits,
                                        dict(stage1=man.get("timings"), stage2_read_s=t_read))
    prim, per0 = rots[0], pers[0]
    C.TABLES.mkdir(parents=True, exist_ok=True)
    with open(C.TABLES / "per_problem_rotation0.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "problem_idx", "screen_idx_p0", "rescore_idx", "A0", "A1", "B1", "B1_minus_A1", "in_S"])
        for p in per0:
            w.writerow([p["q"][0], p["q"][1], p["screen_idx"], " ".join(map(str, p["rescore_idx"])), int(p["A0"]),
                        f"{p['A1'].numerator}/{p['A1'].denominator}", f"{p['B1'].numerator}/{p['B1'].denominator}",
                        f"{(p['B1'] - p['A1']).numerator}/{(p['B1'] - p['A1']).denominator}", int(p["in_S"])])
    with open(C.TABLES / "rotations.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rotation", "screen_position", "K", "F", "A1_S", "B1_S", "G_naive", "R", "D", "D_F", "D_minus_D_F"])
        for o in rots:
            g = lambda x: "" if x is None else f"{x.numerator}/{x.denominator} ({float(x):.4f})"  # noqa: E731
            w.writerow([o["rotation"], f"p{o['rotation']}", o["K"], o["F"], g(o["A1_S"]), g(o["B1_S"]),
                        g(o["G_naive"]), g(o["R"]), g(o["D"]), g(o["D_F"]), g(o["D_minus_D_F"])])
    with open(C.TABLES / "slot_correct_unsealed.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "source", "problem_idx", "idx", "position", "parquet_row", "correct"])
        for (s, q, k), row in sorted(rowmap.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
            w.writerow([s, q[0], q[1], k, order[q].index(k), row, corr[(s, q, k)]])

    summary["timings"]["stage2_total_s"] = round(time.time() - t0, 2)
    summary["peak_working_set_mib"] = C.peak_rss_mb()[0]
    C.write_json(C.RESULTS / "summary.json", summary)
    event("summary_written")
    log(f"primary: K={prim['K']}, G_naive={prim['G_naive']}, R={prim['R']}, D={prim['D']}, D_F={prim['D_F']}")
    import w1_report
    w1_report.write_report(summary, man, seeds, gates)
    log(f"report written ({round(time.time() - t0, 2)} s stage 2)")


if __name__ == "__main__":
    try:
        main()
    except C.StopGate as e:
        rec = dict(decision="STOP", gate="unseal", reason=str(e), time_local=C.now_local())
        C.write_json(C.RESULTS / "STOP_unseal.json", rec)
        log(f"STOP at unseal: {e}")
        sys.exit(3)

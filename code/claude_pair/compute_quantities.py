"""Claude-pair run, step 2: read the stored `correct` column for the problem frame and screening roles written by step 1
(build_frame.py), compute G, R, D and D_F for the four screening-slot rotations and write the tables.

Run after build_frame.py:  python -X utf8 -I compute_quantities.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_pair_rules as C  # noqa: E402
import pq_reader as P  # noqa: E402

LOG = C.RESULTS / "run_log.txt"


def log(msg):
    line = f"[{C.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def load_step1():
    man = C.read_json(C.STEP1 / "manifest.json")
    gates = C.read_json(C.STEP1 / "gates.json")
    seeds = C.read_json(C.STEP1 / "seed_check.json")
    if not (gates["entry"] and gates["cost"] and gates["execution"] and gates["seed_collision_free"]
            and seeds["collision_free"]):
        raise C.StopGate("a step-1 gate did not pass")
    with open(C.STEP1 / "roles.csv", encoding="utf8") as f:
        roles = list(csv.DictReader(f))
    F = [tuple(q) for q in man["F"]]
    order, rowmap = {}, {}
    for r in roles:
        q = (r["source"], r["problem_idx"])
        o = [int(r[f"p{j}"]) for j in range(4)]
        if o != C.role_order(q):
            raise C.StopGate(f"step-1 role order for {q} differs from the namespace-hash rule")
        order[q] = o
        if r["in_F"] == "1":
            for j, k in enumerate(o):
                rowmap[("A", q, k)] = int(r[f"A_row_p{j}"])
                rowmap[("B", q, k)] = int(r[f"B_row_p{j}"])
    if sorted(q for q in order if (("A", q, 0) in rowmap)) != sorted(F):
        raise C.StopGate("step-1 roles and step-1 F disagree")
    return man, gates, seeds, F, order, rowmap


def build_summary(man, gates, F, order, corr, timings):
    rots, pers = [], []
    for j in range(4):
        out, per = C.replay_metrics(F, order, corr, j=j)
        rots.append(out)
        pers.append(per)
    prim = rots[0]
    # descriptive background (not a primary quantity): all four slots on F
    acc = {s: C.mean(C.Fraction(corr[(s, q, k)]) for q in F for k in C.IDX) for s in ("A", "B")}

    def js(o):
        return dict(rotation=o["rotation"], K=o["K"], F=o["F"], S=[list(q) for q in o["S"]],
                    A1_S=C.fr(o["A1_S"]), B1_S=C.fr(o["B1_S"]), G_naive=C.fr(o["G_naive"]), R=C.fr(o["R"]),
                    D=C.fr(o["D"]), D_F=C.fr(o["D_F"]), D_minus_D_F=C.fr(o["D_minus_D_F"]))

    defined = [o for o in rots if o["D"] is not None]

    def rng(key):
        return None if not defined else [C.fr(min(o[key] for o in defined)), C.fr(max(o[key] for o in defined))]

    summary = dict(
        evidence_level="Prospective (data first read and computed after the analysis plan was fixed); the data are retrospective public cached records",
        data=dict(repo=C.REPO, revision=C.REV, file=C.PARQUET_PATH_IN_REPO, url=C.URL, license=C.LICENSE,
                  bytes=man["data"]["entry_gate"]["local_size"],
                  snapshot_commit_date_utc=man["data"]["entry_gate"]["snapshot_commit_date_utc"]),
        gates=dict(entry=gates["entry"], cost=gates["cost"], execution=gates["execution"],
                   seed_collision_free=gates["seed_collision_free"]),
        frame=dict(n_candidates=len(man["candidates"]), n_F=len(F), target=C.TARGET_N, excluded=man["exclusions"]),
        primary=js(prim), sensitivity_rotations=[js(o) for o in rots],
        rotation_range_where_defined=dict(n_defined=len(defined), G_naive=rng("G_naive"), R=rng("R"), D=rng("D"),
                                          D_minus_D_F=rng("D_minus_D_F")),
        descriptive_background_not_registered=dict(A_all4_on_F=C.fr(acc["A"]), B_all4_on_F=C.fr(acc["B"])),
        uncertainty="No 95% CI is imposed: the primary result is an exact description of a finite record of 30 problems; the four p0 "
                    "rotations are only a sensitivity analysis of the role assignment, share one cache, are not four independent "
                    "experiments and are not divided by sqrt(4).",
        timings=dict(timings))
    return summary, rots, pers


def main():
    t0 = time.time()
    log("step 2 start")
    man, gates, seeds, F, order, rowmap = load_step1()
    pf = P.ParquetFile(str(C.PARQUET))
    need = set(rowmap.values())
    correct = pf.read_column("correct", rows=need)
    corr = {}
    for key, row in rowmap.items():
        v = correct[row]
        if v is None or v is P.SKIPPED:
            raise C.StopGate(f"correct missing for step-1 slot {key} (row {row})")
        corr[key] = int(bool(v))
    t_read = round(time.time() - t0, 2)
    log(f"read correct for {len(corr)} step-1 slots ({len(F)} problems)")

    summary, rots, pers = build_summary(man, gates, F, order, corr,
                                        dict(step1=man.get("timings"), step2_read_s=t_read))
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
    with open(C.TABLES / "slot_correct.csv", "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "source", "problem_idx", "idx", "position", "parquet_row", "correct"])
        for (s, q, k), row in sorted(rowmap.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
            w.writerow([s, q[0], q[1], k, order[q].index(k), row, corr[(s, q, k)]])

    summary["timings"]["step2_total_s"] = round(time.time() - t0, 2)
    C.write_json(C.RESULTS / "summary.json", summary)
    log(f"primary: K={prim['K']}, G_naive={prim['G_naive']}, R={prim['R']}, D={prim['D']}, D_F={prim['D_F']}")


if __name__ == "__main__":
    try:
        main()
    except C.StopGate as e:
        rec = dict(decision="STOP", gate="read_correct", reason=str(e), time_local=C.now_local())
        C.write_json(C.RESULTS / "STOP_read_correct.json", rec)
        log(f"STOP at read_correct: {e}")
        sys.exit(3)

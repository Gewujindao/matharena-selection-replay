"""All-pairs run diagnostic (run after build_frame.py; reads prompts only, never `correct`): for each NA cell, how the
8-slot full prompts differ (within side vs between sides, and where the first difference sits).
Writes results/all_pairs/diag_prompt_mismatch.json.  Run: python -X utf8 -I -B diag_prompt_mismatch.py"""
import json, os, sys, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_pairs_rules as rules
helpers, P = rules.helpers, rules.P
frame = helpers.read_json(rules.FRAME_DIR / "frame.json")
out = {}
for cell in frame["cells"]:
    if cell["contract_valid"]:
        continue
    comp = cell["comp"]
    pf = P.ParquetFile(str(rules.COMPS[comp]["parquet"]))
    cfg = pf.read_column("model_config"); pi = pf.read_column("problem_idx"); ia = pf.read_column("idx_answer")
    rows = [i for i in range(pf.num_rows) if cfg[i] in (cell["A"], cell["B"])]
    um = rules.scan_column(pf, "user_message", rows, lambda v: v)
    am = rules.scan_column(pf, "all_messages", rows, lambda v: rules.prompt_part(v)[2])
    per_q = {}
    for i in rows:
        side = "A" if cfg[i] == cell["A"] else "B"
        per_q.setdefault(pi[i], {}).setdefault(side, {})[ia[i]] = (um[i], am[i])
    rep = dict(n_problems=len(per_q), within_side_distinct={}, between_sides_differ=0, user_message_differs=0,
               examples=[])
    wd = {"A": 0, "B": 0}
    for q, d in sorted(per_q.items(), key=lambda t: rules.pidx_key(t[0])):
        for s in "AB":
            if len({json.dumps(v) for v in d.get(s, {}).values()}) > 1:
                wd[s] += 1
        a = {json.dumps(v) for v in d.get("A", {}).values()}
        b = {json.dumps(v) for v in d.get("B", {}).values()}
        if a != b:
            rep["between_sides_differ"] += 1
        ua = {v[0] for v in d.get("A", {}).values()}; ub = {v[0] for v in d.get("B", {}).values()}
        if ua != ub:
            rep["user_message_differs"] += 1
            if len(rep["examples"]) < 3 and len(ua) == 1 and len(ub) == 1:
                x, y = next(iter(ua)), next(iter(ub))
                sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
                ops = [(t, x[i1:i2][:160], y[j1:j2][:160]) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal"]
                rep["examples"].append(dict(problem=q, len_A=len(x), len_B=len(y), n_diff_ops=len(ops), ops=ops[:6]))
    rep["within_side_distinct"] = wd
    out[cell["cell_id"]] = rep
    print(cell["cell_id"], json.dumps({k: v for k, v in rep.items() if k != "examples"}, ensure_ascii=False))
    for e in rep["examples"]:
        print("   ", json.dumps(e, ensure_ascii=False)[:1500])
helpers.write_json(rules.RESULTS / "diag_prompt_mismatch.json", out)

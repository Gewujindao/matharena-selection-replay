"""Added diagnostic (not preregistered), run after sealing and BEFORE unsealing correct.
Record-mechanism facts only: zero-token slots per problem, A/B prompt differences, and whether answer texts repeat
across idx within a problem (compared by SHA256; texts are never printed).  It changes no frozen rule or quantity."""
import csv, difflib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import w1_common as C  # noqa: E402
import pq_reader as P  # noqa: E402

C.apply_limits()
assert not (C.RESULTS / "summary.json").exists(), "must run before unsealing"
roles = list(csv.DictReader(open(C.FROZEN / "roles.csv", encoding="utf8")))
rowmap = {}
for r in roles:
    for j in range(4):
        k = int(r[f"p{j}"])
        rowmap[("A", r["problem_idx"], k)] = int(r[f"A_row_p{j}"])
        rowmap[("B", r["problem_idx"], k)] = int(r[f"B_row_p{j}"])
need = set(rowmap.values())
pf = P.ParquetFile(str(C.PARQUET))
ans = pf.read_column("answer", rows=need)
um = pf.read_column("user_message")
ot = pf.read_column("output_tokens")
it = pf.read_column("input_tokens")
probs = sorted({k[1] for k in rowmap}, key=int)
out = dict(per_problem=[], note="answer texts compared only by SHA256; nothing printed")
for q in probs:
    rec = dict(problem_idx=q)
    for s in ("A", "B"):
        hs = [C.sha256_text(ans[rowmap[(s, q, k)]]) for k in C.IDX]
        rec[f"{s}_distinct_answer_texts"] = len(set(hs))
        rec[f"{s}_zero_output_token_idx"] = [k for k in C.IDX if ot[rowmap[(s, q, k)]] == 0]
        rec[f"{s}_zero_input_token_idx"] = [k for k in C.IDX if it[rowmap[(s, q, k)]] == 0]
    ua, ub = um[rowmap[("A", q, 0)]], um[rowmap[("B", q, 0)]]
    rec["user_message_same_A_B"] = ua == ub
    if ua != ub:
        sm = difflib.SequenceMatcher(a=ua, b=ub)
        rec["prompt_diff_ops"] = [(op, ua[i1:i2][:80], ub[j1:j2][:80]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"][:4]
    out["per_problem"].append(rec)
pp = out["per_problem"]
out["summary"] = dict(
    A_problems_with_all4_identical_answers=sum(r["A_distinct_answer_texts"] == 1 for r in pp),
    B_problems_with_all4_identical_answers=sum(r["B_distinct_answer_texts"] == 1 for r in pp),
    A_problems_with_any_repeat=sum(r["A_distinct_answer_texts"] < 4 for r in pp),
    B_problems_with_any_repeat=sum(r["B_distinct_answer_texts"] < 4 for r in pp),
    A_problems_with_zero_token_slots=[r["problem_idx"] for r in pp if r["A_zero_output_token_idx"]],
    problems_with_prompt_diff=[r["problem_idx"] for r in pp if not r["user_message_same_A_B"]])
C.write_json(C.RESULTS / "diag" / "d1_pre_unseal_record_mechanism.json", out, frozen=True)
print(json.dumps(out["summary"], ensure_ascii=False))
for r in pp[:2] + [x for x in pp if not x["user_message_same_A_B"]][:1]:
    print(json.dumps(r, ensure_ascii=False)[:900])

"""Synthetic checks of the HMMT o3-mini run rules (no real data).  Run: python -X utf8 -I -B test_hmmt_o3mini_pair.py"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmmt_pair_rules as rules  # noqa: E402


def test_match():
    m, _, ok = rules.match_configs({"openai/o3-mini--medium": {"o3-mini (medium)"}, "openai/o3-mini": {"o3-mini (high)"},
                                "x/y": {"other"}})
    assert ok and m == {"A": "openai/o3-mini--medium", "B": "openai/o3-mini"}
    m, _, ok = rules.match_configs({"configs/models/openai/o3-mini--high.yaml": {"o3-mini (high)"},
                                "openai/o3-mini-medium": {"o3-mini (medium)"}})
    assert ok and m["B"] == "configs/models/openai/o3-mini--high.yaml" and m["A"] == "openai/o3-mini-medium"
    _, _, ok = rules.match_configs({"a": {"o3-mini (high)"}, "b": {"o3-mini (high)"}, "c": {"o3-mini (medium)"}})
    assert not ok                                           # two configs share the display name
    _, _, ok = rules.match_configs({"openai/o3-mini--high": {"o3-mini (medium)"}})
    assert not ok                                           # one config claimed by both sides
    _, _, ok = rules.match_configs({"a": {"o3-mini (high)", "z"}, "c": {"o3-mini (medium)"}})
    assert not ok                                           # config with two model_names


def mk(rowid, cfg, q, k, status=rules.ST_OK, cnull=False, prob="p", gold="g", dig=None):
    return dict(row=rowid, cfg=cfg, problem_idx=q, idx=k, problem=prob, gold=gold, status=status, prompt_ok=True,
                prompt_why="ok", correct_is_null=cnull, correct_is_bool=not cnull, digest=dig or f"d{rowid}")


def test_contract():
    mp = {"A": "ca", "B": "cb"}
    rows, rid = [], 0
    for q in ("1", "2", "3"):
        for c in ("ca", "cb"):
            for k in rules.IDX:
                rows.append(mk(rid, c, q, k))
                rid += 1
    F, br, kept, _ = rules.contract(rows, ["1", "2", "3"], mp)
    assert F == ["1", "2", "3"] and not br
    r2 = [r for r in rows if not (r["problem_idx"] == "2" and r["cfg"] == "cb" and r["idx"] == 3)]
    F, br, _, _ = rules.contract(r2, ["1", "2", "3"], mp)
    assert F == ["1", "3"] and "2" in br
    r3 = [dict(r, status=rules.ST_UNKNOWN) if r["row"] == 0 else r for r in rows]
    F, br, _, _ = rules.contract(r3, ["1", "2", "3"], mp)
    assert "1" in br                                        # unknown status breaks, is not scored as wrong
    r4 = [dict(r, status=rules.ST_TRUNC) if r["row"] == 0 else r for r in rows]
    F, br, _, _ = rules.contract(r4, ["1", "2", "3"], mp)
    assert F == ["1", "2", "3"]                             # saved truncation is kept
    r5 = rows + [dict(rows[0], row=999)]                    # identical duplicate collapses
    F, br, kept, _ = rules.contract(r5, ["1", "2", "3"], mp)
    assert F == ["1", "2", "3"] and kept["1"][("A", 0)] == 0
    r6 = rows + [dict(rows[0], row=999, digest="other")]    # differing duplicate breaks
    F, br, _, _ = rules.contract(r6, ["1", "2", "3"], mp)
    assert "1" in br
    r7 = [dict(r, gold="h") if r["row"] == 5 else r for r in rows]
    F, br, _, _ = rules.contract(r7, ["1", "2", "3"], mp)
    assert "1" in br
    r8 = [dict(r, cnull=True, correct_is_null=True, correct_is_bool=False) if r["row"] == 7 else r for r in rows]
    F, br, _, _ = rules.contract(r8, ["1", "2", "3"], mp)
    assert "1" in br


def test_ledger():
    F = ["1", "2", "3"]
    c = {}
    pat = {"1": ([0, 1, 0, 0], [1, 1, 1, 0]), "2": ([1, 1, 1, 1], [1, 1, 1, 1]), "3": ([0, 0, 0, 0], [0, 1, 0, 0])}
    for q, (a, b) in pat.items():
        for k in rules.IDX:
            c[("A", q, k)] = a[k]
            c[("B", q, k)] = b[k]
    o, per = rules.ledger(F, c, screen=0)
    assert o["K"] == 2 and o["S"] == ["1", "3"]
    assert o["R"] == Fraction(1, 6) and o["G"] == (Fraction(2, 3) + Fraction(1, 3)) / 2
    assert o["D"] == o["G"] - o["R"]
    assert o["D_F"] == ((Fraction(2, 3) - Fraction(1, 3)) + 0 + Fraction(1, 3)) / 3
    allright = {k: 1 for k in c}
    o, _ = rules.ledger(F, allright, screen=0)
    assert o["K"] == 0 and o["G"] is None and o["R"] is None and o["D"] is None and o["D_F"] == 0


if __name__ == "__main__":
    for fn in (test_match, test_contract, test_ledger):
        fn()
        print("ok", fn.__name__)

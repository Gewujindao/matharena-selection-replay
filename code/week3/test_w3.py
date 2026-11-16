"""Checks of the week-3 rules on synthetic rows, plus a reader check of scan_column against pq_reader.read_column on
non-`correct` columns of the HMMT copy (no `correct` value is touched).  Run: python -X utf8 -I -B test_w3.py"""
import os
import sys
from fractions import Fraction as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_common as W  # noqa: E402


def test_tiers():
    cfgs = {"openai/o3-mini": {"o3-mini (high)"}, "openai/o3-mini--medium": {"o3-mini (medium)"},
            "openai/o3-mini--low": {"o3-mini (low)"}, "openai/gpt-52": {"GPT-5.2 (xhigh)"},
            "openai/gpt-52-high": {"GPT-5.2 (high)"}, "xai/g-low": {"G (low)"}, "xai/g-high": {"G (high)"},
            "a/claude": {"Claude (Think)"}, "b/m--low": {"M"}, "b/m--high": {"M"}, "c/x--low": {"X (high)"},
            "d/y": {"Y (low)", "Y2 (low)"}, "e/z--low": {"Z (low)"}, "e/z--high": {"Z (high)"},
            "f/u--LOW": {"U"}, "f/u--Medium": {"U (medium)"}, "g/v": {"V (low)"}, "g/v--low": {"V"}}
    configs, eps, pairs = W.enumerate_pairs(cfgs)
    got = {(p["display_base"], p["tier_A"], p["tier_B"], p["A"], p["B"]) for p in pairs}
    assert ("o3-mini", "low", "medium", "openai/o3-mini--low", "openai/o3-mini--medium") in got
    assert ("o3-mini", "medium", "high", "openai/o3-mini--medium", "openai/o3-mini") in got
    assert ("U", "low", "medium", "f/u--LOW", "f/u--Medium") in got
    assert not any(p["display_base"] == "GPT-5.2" for p in pairs)       # -high is not the frozen --high suffix
    assert not any(p["display_base"] == "G" for p in pairs)
    assert not any(p["display_base"] == "M" for p in pairs)              # low/high not adjacent
    assert not any(p["display_base"] == "Z" for p in pairs)
    assert configs["a/claude"]["status"] == "no_tier"
    assert configs["c/x--low"]["status"] == "NA"                         # name tag vs suffix disagree
    assert configs["d/y"]["status"] == "NA"                              # two model_names
    assert eps[("V", "g/v")]["status"] == "NA"                           # same base, same tier twice
    assert len(pairs) == 3, pairs


def mk(row, cfg, q, k, **kw):
    r = dict(row=row, cfg=cfg, problem_idx=q, idx=k, problem="p" + q, gold="g" + q, status=W.ST_OK, prompt_ok=True,
             prompt_why="ok", prompt="P" + q, correct_is_null=False, digest=f"d{row}")
    r.update(kw)
    return r


def full(qs=("1", "2")):
    rows, n = [], 0
    for q in qs:
        for c in ("a", "b"):
            for k in range(4):
                rows.append(mk(n, c, q, k))
                n += 1
    return rows


def test_contract():
    ok, br, kept, _ = W.cell_contract(full(), ["1", "2"], "a", "b")
    assert ok and not br and len(kept) == 2
    rows = full()
    rows[3]["prompt"] = "other"
    ok, br, _, _ = W.cell_contract(rows, ["1", "2"], "a", "b")
    assert not ok and "八槽完整提示不相同" in br["1"]
    rows = full()
    rows[9]["status"] = W.ST_UNKNOWN
    ok, br, _, _ = W.cell_contract(rows, ["1", "2"], "a", "b")
    assert not ok and any("响应状态未知" in x for x in br["2"])
    rows = full()
    rows[9]["status"] = W.ST_TRUNC
    assert W.cell_contract(rows, ["1", "2"], "a", "b")[0]
    rows = full()
    rows[0]["correct_is_null"] = True
    assert not W.cell_contract(rows, ["1", "2"], "a", "b")[0]
    rows = full()[:-1]
    ok, br, _, _ = W.cell_contract(rows, ["1", "2"], "a", "b")
    assert not ok and "B: 缺 idx 3" in br["2"]
    ok, br, _, _ = W.cell_contract(full(("1",)), ["1", "2"], "a", "b")     # problem 2 absent -> NA, no intersection
    assert not ok and "2" in br
    rows = full() + [dict(mk(99, "a", "1", 0), digest="d0")]
    assert W.cell_contract(rows, ["1", "2"], "a", "b")[0]                # identical duplicate collapses
    rows = full() + [mk(99, "a", "1", 0)]
    assert not W.cell_contract(rows, ["1", "2"], "a", "b")[0]            # differing duplicate breaks
    rows = full()
    rows[5]["gold"] = "zz"
    assert not W.cell_contract(rows, ["1", "2"], "a", "b")[0]
    assert W.response_status(True, 0) == W.ST_OK and W.response_status(False, 5) == W.ST_TRUNC
    assert W.response_status(False, None) == W.ST_UNKNOWN
    ok, _, c1 = W.full_prompt("hi", '[{"role":"user","content":"hi"},{"role":"assistant","content":"x"}]')
    ok2, _, c2 = W.full_prompt("hi", '[{"content":"hi","role":"user"},{"role":"assistant","content":"y"}]')
    assert ok and ok2 and c1 == c2
    assert not W.full_prompt("", '[{"role":"user","content":"hi"}]')[0]
    assert not W.full_prompt("hi", '[{"role":"assistant","content":"hi"}]')[0]


def test_ledger_and_aggregate():
    F = ["1", "2", "3"]
    corr = {}
    tab = {"1": ([0, 0, 0, 1], [1, 1, 1, 1]), "2": ([0, 1, 1, 1], [0, 0, 0, 0]), "3": ([1, 1, 1, 1], [1, 1, 1, 1])}
    for q, (a, b) in tab.items():
        for k in range(4):
            corr[("A", q, k)] = a[k]
            corr[("B", q, k)] = b[k]
    o, _ = W.ledger(F, corr)
    assert o["K"] == 2
    assert o["R"] == Q(1, 2) * (Q(1, 3) + 1) and o["G"] == Q(1, 2) and o["D"] == o["G"] - o["R"]
    assert o["D_F"] == Q(1, 3) * ((1 - Q(1, 3)) + (0 - 1) + 0)
    assert o["C"] == o["D"] - o["D_F"] and o["G_minus_D_F"] == o["R"] + o["C"]
    assert o["sign_reversal"] == int(o["D"] * o["D_F"] < 0)
    allone = {k: 1 for k in corr}
    o2, _ = W.ledger(F, allone)
    assert o2["K"] == 0 and o2["R"] is None and o2["D_F"] == 0 and o2["D_F_zero"] == 1
    cells = [dict(comp="AIME", endpoint="e1", valid=True, primary=dict(R=Q(1, 2))),
             dict(comp="AIME", endpoint="e1", valid=True, primary=dict(R=Q(1, 4))),
             dict(comp="AIME", endpoint="e2", valid=True, primary=dict(R=Q(1))),
             dict(comp="HMMT", endpoint="e1", valid=False, primary=None),
             dict(comp="BRUMO", endpoint="e1", valid=True, primary=dict(R=None)),
             dict(comp="CMIMC", endpoint="e1", valid=True, primary=dict(R=Q(0)))]
    rep = W.aggregate(cells, "R")
    assert rep["competitions"]["AIME"]["value"] == (Q(3, 8) + 1) / 2
    assert rep["competitions"]["HMMT"]["value"] is None and rep["competitions"]["HMMT"]["n_cells_contract_NA"] == 1
    assert rep["competitions"]["BRUMO"]["n_cells_undefined_K0"] == 1
    assert rep["n_competitions_valid"] == 2 and rep["macro"] == ((Q(3, 8) + 1) / 2 + 0) / 2


def test_reader():
    pf = W.P.ParquetFile(str(W.COMPS["HMMT"]["parquet"]))
    rows = list(range(0, pf.num_rows, 97)) + [pf.num_rows - 1, 1870, 1871]
    for name in ("problem_idx", "output_tokens", "model_config", "parsed_answer"):
        full_col = pf.read_column(name)
        got = W.scan_column(pf, name, rows, lambda v: v)
        assert all(got[r] == full_col[r] for r in rows), name
        nul = W.scan_column(pf, name, rows, lambda z: z, values=False)
        assert all(nul[r] == (full_col[r] is None) for r in rows), name


if __name__ == "__main__":
    W.C1.apply_limits()
    test_tiers()
    test_contract()
    test_ledger_and_aggregate()
    test_reader()
    print("all week-3 checks passed")

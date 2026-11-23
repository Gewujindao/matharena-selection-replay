"""All-pairs run probe (metadata/prompt structure only; `correct` is not read): eligible pairs by the tier rule and the
message-role structure of all_messages for the configs in those pairs.  Run: python -X utf8 -I -B probe_prompt_structure.py"""
import os, sys, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_pairs_rules as rules
P = rules.P
for comp in rules.COMP_ORDER:
    pf = P.ParquetFile(str(rules.COMPS[comp]["parquet"]))
    cfg = pf.read_column("model_config"); nm = pf.read_column("model_name")
    cfg_names = {}
    for i in range(pf.num_rows):
        cfg_names.setdefault(cfg[i], set()).add(nm[i])
    configs, eps, pairs = rules.enumerate_pairs(cfg_names)
    print("==", comp, "configs", len(configs), "tiered", sum(r["status"] == "tiered" for r in configs.values()), "pairs", len(pairs))
    for p in pairs:
        print("   pair", p["display_base"], "|", p["config_base"], p["tier_A"], "->", p["tier_B"], p["A"], p["B"])
    for b, e in eps.items():
        print("   endpoint", b, e["tiers"], e["status"], e["pairs"], e["reason"])
    elig = {p["A"] for p in pairs} | {p["B"] for p in pairs}
    rows = [i for i in range(pf.num_rows) if cfg[i] in elig]
    def struct(s):
        try:
            m = json.loads(s)
        except Exception:
            return ("BADJSON",)
        if not isinstance(m, list):
            return ("NOTLIST", type(m).__name__)
        return tuple((x.get("role") if isinstance(x, dict) else type(x).__name__,
                      type(x.get("content")).__name__ if isinstance(x, dict) else "", tuple(sorted(x.keys())) if isinstance(x, dict) else ()) for x in m)
    st = rules.scan_column(pf, "all_messages", rows, struct)
    um = rules.scan_column(pf, "user_message", rows, lambda v: v)
    def first_user(s):
        try:
            m = json.loads(s)
            u = [x for x in m if isinstance(x, dict) and x.get("role") == "user"]
            return u[0].get("content") if u else None
        except Exception:
            return None
    fu = rules.scan_column(pf, "all_messages", rows, first_user)
    for c in sorted(elig):
        rs = [i for i in rows if cfg[i] == c]
        print("   ", c, Counter(st[i] for i in rs).most_common(4))
        print("      user_message == first user content:", sum(um[i] == fu[i] for i in rs), "/", len(rs))

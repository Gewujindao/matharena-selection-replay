# Prompt-difference check for the three NA cells (reads user_message / all_messages / output_tokens /
# input cost columns only; never reads correct).
import difflib
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parquet_decoder
from recompute import FILES, norm_path


def side_prompts(pf, configs, pidx, cfg, col):
    rows = {i for i, c in enumerate(configs) if norm_path(c) == cfg}
    vals = parquet_decoder.read_rows(pf, col, rows, lambda v: v)
    idxs = parquet_decoder.read_rows(pf, "idx_answer", rows, lambda v: v)
    out = {}
    for i in rows:
        out.setdefault(pidx[i], {})[idxs[i]] = vals[i]
    return out


def segs(a, b):
    a = a.decode("utf-8")
    b = b.decode("utf-8")
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    res = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            res.append((op, a[i1:i2][:160], b[j1:j2][:160]))
    return res


def main():
    for comp, A, B in (("AIME", "openai/o3-mini--medium", "openai/o3-mini"),
                       ("CMIMC", "openai/o4-mini--low", "openai/o4-mini--medium"),
                       ("CMIMC", "openai/o4-mini--medium", "openai/o4-mini--high")):
        pf = parquet_decoder.PFile(FILES[comp])
        configs = [x.decode() for x in pf.read_flat("model_config")]
        pidx = [x.decode() for x in pf.read_flat("problem_idx")]
        pa = side_prompts(pf, configs, pidx, A, "user_message")
        pb = side_prompts(pf, configs, pidx, B, "user_message")
        diffp = sorted([p for p in pa if set(pa[p].values()) != set(pb[p].values())], key=lambda x: (len(x), x))
        print("==", comp, A, "->", B, "A/B differ on", len(diffp), "problems:", diffp)
        for side, d in (("A", pa), ("B", pb)):
            w = [p for p in d if len(set(d[p].values())) > 1]
            print("   within-side differ", side, w)
            for p in w:
                vs = list(d[p].items())
                for k, v in vs[1:]:
                    if v != vs[0][1]:
                        print("     ", side, p, "idx", vs[0][0], "vs", k, segs(vs[0][1], v)[:3])
        for p in diffp[:2]:
            print("   problem", p, "A idx0 vs B idx0:", segs(pa[p][0], pb[p][0])[:4])
        if comp == "AIME":
            # output_tokens zero on idx1..3 for the high side; per-token prices for all three o3-mini tiers
            for cfg in ("openai/o3-mini--low", "openai/o3-mini--medium", "openai/o3-mini"):
                ot = side_prompts(pf, configs, pidx, cfg, "output_tokens")
                z = sum(1 for p in ot if all(ot[p][k] == 0 for k in (1, 2, 3)))
                print("   otok0 idx1-3", cfg, z)
        for cfg in sorted({A, B}):
            ic = side_prompts(pf, configs, pidx, cfg, "input_cost_per_tokens")
            oc = side_prompts(pf, configs, pidx, cfg, "output_cost_per_tokens")
            print("   prices", cfg, sorted({v for d in ic.values() for v in d.values()}),
                  sorted({v for d in oc.values() for v in d.values()}))


if __name__ == "__main__":
    main()

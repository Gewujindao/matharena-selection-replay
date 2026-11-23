"""All-pairs run probe: schema / row groups / config names of the four pinned parquet files (metadata only; `correct` is
not read).  Run: python -X utf8 -I -B probe_parquet_schema.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pq_reader as P
CARD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {"AIME": "downloads/hf_aime_2025_outputs_16e93d3/train-00000-of-00001.parquet",
         "HMMT": "downloads/hf_hmmt_feb_2025_outputs_e3fb6c7/train-00000-of-00001.parquet",
         "BRUMO": "downloads/hf_brumo_2025_outputs_3b234ef/train-00000-of-00001.parquet",
         "CMIMC": "downloads/hf_cmimc_2025_outputs_2782618/train-00000-of-00001.parquet"}
for comp, rel in FILES.items():
    pf = P.ParquetFile(os.path.join(CARD, rel))
    d = pf.describe()
    print("==", comp, d["num_rows"], d["created_by"], "row_groups", len(d["row_groups"]))
    print(" ptype", d["ptype"]); print(" rep", d["repetition"])
    for g in d["row_groups"][:3]:
        print(" rg", g["num_rows"], {k: (v["codec"], v["encodings"], v["compressed"]) for k, v in g["columns"].items() if k in ("correct", "all_messages", "answer", "model_config")})
    cfg = pf.read_column("model_config"); nm = pf.read_column("model_name"); pi = pf.read_column("problem_idx"); src = pf.read_column("source"); ia = pf.read_column("idx_answer")
    pairs = {}
    for i in range(pf.num_rows):
        pairs.setdefault((cfg[i], nm[i]), 0); pairs[(cfg[i], nm[i])] += 1
    print(" n_configs", len({c for c in cfg}), "n_pairs", len(pairs), "problems", len(set(pi)), sorted(set(pi), key=lambda x: (len(str(x)), str(x)))[:50])
    print(" sources", sorted({str(s) for s in src})[:10], "n_sources", len({str(s) for s in src}), "idx values", sorted({x for x in ia if x is not None}), "idx null", sum(x is None for x in ia))
    for (c, n), k in sorted(pairs.items(), key=lambda t: str(t[0])):
        print("   ", repr(c), "|", repr(n), "|", k)

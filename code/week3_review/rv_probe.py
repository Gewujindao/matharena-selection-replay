# Metadata-only probe: schema leaves, codecs, encodings, row groups. Reads no column values.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv_guard
import rv_pq

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "downloads") + "/"
FILES = {
    "AIME": BASE + "w3_hf_aime_2025_outputs_16e93d3/train-00000-of-00001.parquet",
    "HMMT": BASE + "hf_hmmt_feb_2025_outputs_e3fb6c7/train-00000-of-00001.parquet",
    "BRUMO": BASE + "w3_hf_brumo_2025_outputs_3b234ef/train-00000-of-00001.parquet",
    "CMIMC": BASE + "w3_hf_cmimc_2025_outputs_2782618/train-00000-of-00001.parquet",
}

if __name__ == "__main__":
    rv_guard.setup()
    for comp, path in FILES.items():
        pf = rv_pq.PFile(path)
        print("==", comp, "rows", pf.num_rows, "rowgroups", len(pf.meta[4]))
        kv = pf.meta.get(5) or []
        for e in kv:
            k = e.get(1, b"").decode()
            print("  kv", k, len(e.get(2, b"")))
        for i, lf in enumerate(pf.leaves):
            codecs = set()
            encs = set()
            for nr, md in pf.chunks(i):
                codecs.add(md[4])
                encs.update(md[2])
            print("  leaf", i, ".".join(lf["path"]), "ptype", lf["ptype"], "def", lf["max_def"], "rep", lf["max_rep"],
                  "conv", lf["conv"], "codec", sorted(codecs), "enc", sorted(encs))

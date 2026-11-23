"""Claude-pair run reader check, run after compute_quantities.py: per-config accuracy over all four cached slots vs the
AIME 2025 accuracy printed on the official MathArena model pages (saved in downloads/meta_web_claude_pair)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import claude_pair_rules as C  # noqa: E402

s = C.read_json(C.RESULTS / "summary.json")
out = {}
for side, page in (("A", "anthropic_claude_35"), ("B", "anthropic_claude_37")):
    html = (C.META_WEB / f"matharena_model_{page}.html.txt").read_text(encoding="utf8")
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html))
    m = re.search(r"AIME 2025 \S* Final-Answer Comps Accuracy ([0-9.]+)% CI: ± ([0-9.]+)%", txt)
    cached = s["descriptive_background_not_registered"][f"{side}_all4_on_F"]
    out[side] = dict(official_page=f"https://matharena.ai/models/{page}", official_accuracy_pct=float(m.group(1)),
                     official_ci_pct=float(m.group(2)), cached_all4_exact=cached["exact"],
                     cached_all4_pct=round(cached["value"] * 100, 2),
                     match=abs(float(m.group(1)) - cached["value"] * 100) < 0.01)
C.write_json(C.RESULTS / "diag" / "d2_official_page_check.json", out)
print(json.dumps(out, ensure_ascii=False))

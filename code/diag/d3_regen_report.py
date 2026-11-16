"""Regenerate report.txt with the added diagnostics (numbers come unchanged from summary.json)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import w1_common as C  # noqa: E402
import w1_report as RP  # noqa: E402

summary = C.read_json(C.RESULTS / "summary.json")
man = C.read_json(C.FROZEN / "manifest_frozen.json")
seeds = C.read_json(C.FROZEN / "seed_check.json")
gates = C.read_json(C.FROZEN / "gates.json")
d1 = C.read_json(C.RESULTS / "diag" / "d1_pre_unseal_record_mechanism.json")["summary"]
d2 = C.read_json(C.RESULTS / "diag" / "d2_official_page_check.json")
p = summary["primary"]
pct = lambda x: f"{x['value'] * 100:.1f}"  # noqa: E731
rr = summary["rotation_range_where_defined"]
nA = round(d2["A"]["cached_all4_pct"] * 1.2)
conclusion = (
    f"结论：A 在本缓存中几乎总答错（F 上 120 条记录只对 {nA} 条），所以按 A 答错筛题几乎不改变人群"
    f"（K = {p['K']}/{p['F']}）。筛错后的缓存记录回升 R 只有 {pct(p['R'])} 个百分点，G_naive（{pct(p['G_naive'])} 个百分点）"
    f"与 D（{pct(p['D'])} 个百分点）几乎相同；D 与全框 D_F（{pct(p['D_F'])} 个百分点）只差 {pct(p['D_minus_D_F'])} 个百分点。"
    f"四套轮换下 R 在 {rr['R'][0]['value'] * 100:.1f} 到 {rr['R'][1]['value'] * 100:.1f} 个百分点之间，D − D_F 在 "
    f"{rr['D_minus_D_F'][0]['value'] * 100:+.1f} 到 {rr['D_minus_D_F'][1]['value'] * 100:+.1f} 个百分点之间。"
    "就这批冻结记录而言，按答错筛题造成的差异很小。但这是 A 几乎全错的地板情形（A 的总正确率 3.33% 在冻结前已公开），"
    "R 和 D − D_F 小主要由此决定，不能外推到旧配置正确率较高时的错筛。")
aI = ", ".join(d1["A_problems_with_zero_token_slots"])
section = [
    f"读取器核对：F 上四槽合计正确率 A = {d2['A']['cached_all4_pct']}%、B = {d2['B']['cached_all4_pct']}%，与 MathArena 官网模型页"
    f"所列 AIME 2025 正确率（{d2['A']['official_accuracy_pct']}%、{d2['B']['official_accuracy_pct']}%）一致，说明 correct 列的合计解码无误"
    "（逐槽取值另经复核者独立读取器核对，240/240 一致）；"
    "也提示官网榜分就来自这批缓存记录，并非独立评估，所以 R 不能因“旧榜分已独立”而免扣。",
    "以下三项在封存之后、解封 correct 之前查看（只看提示词、token 计数和答案文本是否重复，答案文本只以 SHA256 比较）：",
    f"  1）A 与 B 的提示词在第 1–15 题（AIME I）不同：A 为 “Output the final answer modulo 1000”，B 为 “The answer is an "
    "integer between 0 and 999 inclusive”；第 16–30 题相同。题文 hash 和 gold 一致，所以按冻结规则仍配对，未排除。",
    f"  2）A 在第 {aI.split(', ')[0]}–{aI.split(', ')[-1]} 题的 idx 1–3 记录的输入、输出 token 均为 0（答案非空）；B 全部有计数。"
    "（复核更正）idx 0 上的计数不是单条计数，而是同题四次调用的合计：这 15 题 A 的 idx 0 input_tokens 都恰为单次提示的 4 倍"
    "（B 的单次 input_tokens 减去 A 的 1/4，14 题为常数 33、1 题为 37），output_tokens 也与四条答案的合计长度相符。"
    "类似的 token 记为 0 的模式还见于该 revision 中另外 12 个早期配置的 AIME I 记录（多数为 idx 1–3）。所以这是记账方式（合计记在 idx 0），"
    "不能说明 idx 0 与 idx 1–3 来自不同收集路径；它只说明 A 的 AIME I 记录与其 AIME II 记录、与 B 的记录出自不同的收集/记账版本。",
    f"  3）A、B 在任何一题内四条答案文本都互不相同（A 有重复的题 {d1['A_problems_with_any_repeat']} 道，B {d1['B_problems_with_any_repeat']} 道）。",
    "该 revision 中 source 字段在所有行都是字符串 “None”；命名空间哈希按数据原值使用 “None”，配对实际靠 problem_idx 与题文 hash。",
]
limitations = [
    "增补诊断发现的记录差异：A 在 AIME I 上用了不同的提示词（“modulo 1000” 版），这属于 A/B “记录配置差”的一部分；"
    "A 在 AIME I 上把四次调用的 token/费用合计记在 idx 0、idx 1–3 记为 0，说明 A 的 AIME I 记录出自与 AIME II 不同的"
    "收集/记账版本（题间异质）。（复核更正：这不是题内四槽“并非同一收集路径”的证据，也不直接削弱题内可交换性；"
    "但“重复可交换、筛选与复评随机性独立”仍未核实，R 仍只能按缓存记录回升读。）",
]
summary["added_diagnostics_not_preregistered"] = dict(d1_pre_unseal=d1, d2_official_page=d2,
                                                      files=["results/diag/d1_pre_unseal_record_mechanism.json",
                                                             "results/diag/d2_official_page_check.json"])
summary["plain_conclusion"] = conclusion
C.write_json(C.RESULTS / "summary.json", summary)
RP.write_report(summary, man, seeds, gates, diag=dict(section=section, limitations=limitations), conclusion=conclusion)
print(conclusion)

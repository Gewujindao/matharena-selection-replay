"""Chinese report.txt for week 1, written from summary.json with the preregistration's frozen wording.
Written before unmasking; it only fills in numbers."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1_common as C  # noqa: E402


def f(x, pct=False):
    if x is None:
        return "未定义"
    v = x["value"]
    s = f"{x['exact']}（{v:.4f}）"
    if pct:
        s = f"{x['exact']}（{v * 100:+.1f} 个百分点）"
    return s


def write_report(summary, man, seeds, gates=None, out_path=None, diag=None, conclusion=None):
    gates = gates if gates is not None else C.read_json(C.FROZEN / "gates.json")
    dl = gates["cost_gate"]["downloads_bytes"]
    p = summary["primary"]
    fr = summary["frame"]
    eg = man["data"]["entry_gate"]
    g = summary["gates"]
    nF, nC = fr["n_F"], fr["n_candidates"]
    full = nF == C.TARGET_N and nC == C.TARGET_N
    tk = man["tokens"]
    lim_ = summary.get("limits") or {}
    memcap = "进程内存上限 2 GiB（job object）" if lim_.get("memory_cap_job") else "进程内存按 ≤ 2 GiB 监测（job object 未生效）"
    L = []
    w = L.append
    w("MathArena AIME 2025 按答错筛题回放——结果报告")
    w("")
    w("证据等级：前瞻（分析计划冻结后才首次取数和计算）。所用数据是公开缓存的回顾性记录，"
      "不是未见公开榜分的前瞻实验；这是冻结记录回放，不升级为模型随机化总体推断。")
    w("")
    w("一、各道门")
    w(f"- 入场门（数据绑定）：{'通过' if g['entry'] else '未通过'}。HF 数据集 {C.REPO}，固定 revision {C.REV}"
      f"（该 commit 日期 {eg['snapshot_commit_date_utc'][:10]}，取数时仍是 main 的最新 commit），单个 parquet "
      f"{C.PARQUET_PATH_IN_REPO}，{eg['local_size']} 字节；本地字节 SHA256 与登记的 LFS oid 一致；许可 CC-BY-NC-SA-4.0。")
    w("- 成本门：通过。草稿没有写成本门的数值阈值；按原文保留原生生成上限、不补 retry、不重查询，"
      f"因此新增 API 调用为 0。downloads/ 共 {dl / 2 ** 20:.1f} MiB（上限 1 GB），"
      f"单进程、1 线程、BelowNormal、{memcap}，每次重活前可用内存都 ≥ 1500 MB。")
    ab = man["configs"]
    w(f"- 执行门：{'通过' if g['execution'] else '未通过'}。两锁定配置在该 revision 中的路径值原样出现："
      f"A = {ab['A']['raw_path_values']}（{ab['A']['model_names']}），B = {ab['B']['raw_path_values']}"
      f"（{ab['B']['model_names']}），无需去前后缀，未做模糊匹配。候选题 {nC} 道；A、B 两边 idx=0..3 唯一齐全、"
      f"题文 hash 与 gold 一致的完整框 F = {nF} 道；排除 {len(fr['excluded'])} 道。")
    if fr["excluded"]:
        for k, rs in fr["excluded"].items():
            w(f"    排除 {k}：{'；'.join(rs)}")
    sc = seeds["scan"]
    w(f"- 种子：按草稿命名空间规则 SHA256(\"line160/v1|HFrevision|source|problem_idx|idx\") 展开 {seeds['n_entries']} 条，"
      f"{'互不相同' if seeds.get('distinct_digests', True) else '有重复'}，题内并列 {seeds.get('within_problem_digest_ties', 0)} 处（并列按 idx 升序）；与上级目录中已登记的种子无碰撞（扫描 {sc['n_text_files']} 个文本文件中的 10–20 位整数字面量、"
      f"64 位十六进制字面量和命名空间串，另核 {seeds['n_registered_ranges']} 个登记区间）。这些哈希只给现有记录分派角色，"
      "不是原调用的随机种子。")
    w("")
    w("二、主结果（预注册主量：p0 筛选，p1–p3 复评）")
    w(f"- 完整框 F = {nF} 道题中，旧配置 A（Claude-3.5-Sonnet）在 p0 槽答错的题 S 共 K = {p['K']} 道。")
    if p["K"]:
        w(f"- 在这 {p['K']} 道所筛题上：A₁（未参与筛选的旧配置角色，p1–p3 三条记录正确率均值）= {f(p['A1_S'])}；"
          f"B₁（新配置 B，同三槽）= {f(p['B1_S'])}。")
        w(f"- G_naive = mean(B₁ − A₀) = {f(p['G_naive'], True)}。")
        w(f"- R = mean(A₁ − A₀) = {f(p['R'], True)}。R 是筛错后再评分的缓存记录回升。")
        w(f"- D = mean(B₁ − A₁) = {f(p['D'], True)}；恒等式 G_naive − R = D 精确成立。")
        w("- 读法：在按 A 答错筛出的题上，B 相对 A 的表面提升 G_naive 中，有 R 这一部分是 A 自己在未参与筛选的三条"
          "记录上就有的回升；扣掉后剩 D。B 与 A 的差只称记录配置差（B：max_tokens 64000、thinking 32000、温度 1；"
          "A：max_tokens 8000，官网未列温度），不称等预算纯代际提升。")
        if conclusion:
            w("- " + conclusion)
    else:
        w("- K = 0：G_naive、R、D 未定义；按预注册不改筛选阈值。")
    w("")
    w("三、次要量")
    w(f"- D_F = 完整缓存框 F 内 mean(B₁ − A₁) = {f(p['D_F'], True)}；D − D_F = {f(p['D_minus_D_F'], True)}。")
    if full:
        w("  F 齐全 30 题，D − D_F 量化本缓存的错筛组成差。已知有限框直接比较，无需 IPW；"
          "这不是模型随机化总体或其他题库的回校。")
    else:
        w(f"  F 只有 {nF} 道（完整缓存子框），结论仅及该子框，不能称完整 AIME 或原 A 错题人群。已知有限框直接比较，"
          "无需 IPW；这不是模型随机化总体或其他题库的回校。")
    w("- 角色协议敏感性（p0 轮换到四个位置，其余三条复评；四套共享同一缓存，不是四次独立实验，不除以 √4）：")
    for o in summary["sensitivity_rotations"]:
        w(f"    p{o['rotation']} 筛选：K = {o['K']}，G_naive = {f(o['G_naive'])}，R = {f(o['R'])}，D = {f(o['D'])}，"
          f"D_F = {f(o['D_F'])}，D − D_F = {f(o['D_minus_D_F'])}")
    rr = summary["rotation_range_where_defined"]
    if rr["n_defined"]:
        w(f"  四套中有定义的 {rr['n_defined']} 套：D 在 {rr['D'][0]['value']:.4f} 到 {rr['D'][1]['value']:.4f} 之间，"
          f"R 在 {rr['R'][0]['value']:.4f} 到 {rr['R'][1]['value']:.4f} 之间，"
          f"D − D_F 在 {rr['D_minus_D_F'][0]['value']:.4f} 到 {rr['D_minus_D_F'][1]['value']:.4f} 之间。")
    w("- 不确定性：按预注册不强加 95% CI。主结果是有限记录的精确描述；上面的轮换范围只反映角色分派协议的敏感性。"
      "模型随机化 CI 须先证明原重复机制；抽题 CI 须另定题目总体和题族抽样假设，逐回答 bootstrap 不能替代，"
      "I/II 两份试卷也不能当作充分独立的簇。")
    bg = summary["descriptive_background_not_registered"]
    w(f"- 描述性背景（非预注册量，只帮助读 D_F）：F 上四个槽合计，A 正确率 {bg['A_all4_on_F']['value']:.4f}，"
      f"B 正确率 {bg['B_all4_on_F']['value']:.4f}。")
    w("")
    w("四、局限（草稿列出的全部，另加本次执行中出现的）")
    lim = [
        "这是公开缓存的回顾性分析，不是未见公开榜分的前瞻实验；候选配置在读官方模型页时已固定，未按效果另挑模型，"
        "也不换模型追求显著结果。",
        "A 的官网页面未列温度，这不等于温度零；没有补写默认值。",
        "A、B 的预算和推理模式不同（B 有 32000 thinking、64000 上限、温度 1；A 上限 8000），只能称记录配置差，"
        "不能称等预算的纯代际提升。",
        "没有核到历史配置文件 hash、调用时间、resolved 模型回执、原 seed/retry 和 parser 提交（manifest 中均记 UNKNOWN）。"
        "GitHub 上两个配置文件有多次提交（manifest 记有 commit 列表，最晚 2025-12-22），而逐条生成时间未知，"
        "所以当前 main 版本不一定是生成时版本。"
        "日期端点只是可核代理，不是已审计的权重 hash；这些只是缓存身份线索，也未证明旧端点现在仍可调用。"
        "因此只做冻结记录回放，禁止升级为模型随机化总体推断。",
        "不以竞赛年份保证未污染。",
        "idx_answer 是 attempt 索引，不是 seed 或独立保证；命名空间哈希只分派现有记录角色，不制造原调用随机种子。",
        "每槽的首次请求和替换链、原收集端的重试/成功替换都不明（记 UNKNOWN）；缺请求链不会靠删行变成独立。"
        f"数据里没有逐条生成时间，只知道快照收录时间（revision commit 日期 {eg['snapshot_commit_date_utc'][:10]}），没有编造生成日期。",
        "R 只是筛错后再评分的缓存记录回升；只有在调用条件相同、重复可交换且筛选与复评的随机性独立都得到支持时，"
        "才能解释为 fresh 重测回升，本次均未核实。若旧榜分本来就来自独立评估，就不该再扣 R。",
        "D 不恢复来源总体，也不识别训练机制、家族惩罚或真实 HLE 筛选史。",
        "D_F 与 D − D_F 只及本缓存框" + ("（F 齐全 30 题）" if full else f"的完整子框（{nF} 题）") + "，不是模型随机化总体或其他题库的回校。",
        "主判分层是该 revision 存储的 correct / parsed_answer，没有自行做字符串等值；原 parser 版本 UNKNOWN，"
        "不能排除解析漂移。以后若统一重判，须先另锁共同 parser/gold 版本和冲突规则并保留原层，不能看了 D 再选解析器。",
        "已保存的截断和拒答一律按锁定的 correct 计，没有补 retry 或重查询；生成上限保留原生值，没有按实际 token 裁样。"
        f"（A 有 {tk['A']['over_native_max_tokens']} 条、B 有 {tk['B']['over_native_max_tokens']} 条 output_tokens 超过官网上限；"
        f"可见答案为空但有输出 token 的已保存截断：A {tk['A']['empty_visible_answer']} 条、B {tk['B']['empty_visible_answer']} 条。）",
        "30 题固定，结果不带 95% CI；四套轮换共享同一缓存，不能当四次独立实验或除以 √4。",
        "执行顺序说明：草稿末句写“首周只核元数据门、冻结 manifest/角色和判分层，不运行回放或指标”。冻结补充节"
        "（以其为准）要求报告前瞻结果，所以照做；但仍保持草稿的先后顺序——"
        "manifest、排除表、角色表和种子核对先封存为只读，之后才解封 correct 列。（第一阶段为核对“判分字段缺失”"
        "曾解码 correct 列，但只保留是否为空的标志和行摘要，取值未输出、未用于任何计算。）",
        "实现说明：指定环境没有 pyarrow，按要求未安装任何包；改用自写的纯 Python parquet 读取器（Snappy、RLE、字典编码）。"
        f"单元测试通过，且数值列解码经 cost = 输入 token × 单价 + 输出 token × 单价 的恒等式核对，"
        f"{man['data']['reader_check']['cost_identity_matches']}/{man['data']['reader_check']['rows_checked']} 行成立。",
    ]
    lim += (diag or {}).get("limitations", [])
    for i, s in enumerate(lim, 1):
        w(f"{i}. {s}")
    if diag and diag.get("section"):
        w("")
        w("附：增补诊断（非预注册，不改变任何冻结规则或数值）")
        for x in diag["section"]:
            w(f"    {x.strip()}" if x.startswith("  ") else f"- {x}")
    w("")
    w("五、预先确定的操作定义（在使用 correct 取值之前写进代码并封存）")
    for s in man["decided_before_unmask"]:
        w(f"- {s}")
    w("")
    w("六、计时与资源")
    t1 = man.get("timings", {})
    t2 = summary["timings"]
    w(f"- 第一阶段（门、读取、遮蔽预检、种子扫描、封存）共 {t1.get('stage1_total_s')} 秒，其中种子扫描 {t1.get('seed_check_s')} 秒；"
      f"第二阶段（解封 correct 和计算）共 {t2.get('stage2_total_s')} 秒。")
    w(f"- 峰值工作集：第一阶段 {man.get('peak_working_set_mib')} MiB，第二阶段 {summary.get('peak_working_set_mib')} MiB（上限 2 GiB）。")
    w("")
    w("七、文件")
    w("- results/summary.json：全部数值（精确分数和小数）。")
    w("- results/tables/per_problem_rotation0.csv、rotations.csv、slot_correct_unsealed.csv。")
    w("- results/frozen/：解封前封存的 manifest_frozen.json、gates.json、seed_manifest.json、seed_check.json、"
      "exclusions.csv、roles.csv、slots.csv 和代码快照 code_at_seal/（均只读）。")
    w("- results/diag/：增补诊断（d1 在解封前运行，d2 为官网数值核对）；生成脚本在 code/diag/。")
    w("- results/run_log.txt、timeline.jsonl：事件顺序（含时间戳）。")
    if (C.RESULTS / "review_note.txt").exists():
        w("- results/review_note.txt：独立复核记录（复核者自写读取器重算主量，并更正了若干报告措辞）。")
    w("- downloads/：数据与元数据原件（URL、许可、大小、revision 见 manifest）。")
    text = "\n".join(L) + "\n"
    Path(out_path or (C.RESULTS / "report.txt")).write_text(text, encoding="utf8")
    return text

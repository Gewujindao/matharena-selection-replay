"""Week 3, stage 4: write week3/report.txt (Chinese, plain) from frame.json, summary.json,
independent_recompute.json and diag_prompt_mismatch.json, and add the independent result and run totals to
summary.json.  Run:  python -X utf8 -I -B w3_report.py  (optional args: total wall seconds, download seconds)"""
import json
import os
import sys
from fractions import Fraction as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_common as W  # noqa: E402

C1 = W.C1


def q(d):
    return None if d is None else Q(d["exact"])


def pp(d):
    return "未定义" if d is None else f"{float(q(d)) * 100:+.2f} pp"


def fx(d):
    return "未定义" if d is None else d["exact"]


def both(d):
    return "未定义" if d is None else f"{d['exact']}（{float(q(d)) * 100:+.2f} pp）"


NAMES = {"G": "G", "R": "R", "D": "D", "D_F": "D_F", "C": "C = D − D_F", "G_minus_D_F": "G − D_F"}


def short(cid):
    comp, base, tiers = cid.split("|")
    return f"{comp} {base} {tiers.replace('->', '→')}"


def main():
    W.C1.apply_limits()
    wall_total = float(sys.argv[1]) if len(sys.argv) > 1 else None
    dl_s = float(sys.argv[2]) if len(sys.argv) > 2 else None
    frame = C1.read_json(W.FRAME_DIR / "frame.json")
    S = C1.read_json(W.W3 / "summary.json")
    ind = C1.read_json(W.W3 / "independent_recompute.json")
    diag = C1.read_json(W.W3 / "diag_prompt_mismatch.json")
    cells = S["cells"]
    valid = [c for c in cells if c["valid"]]
    na = [c for c in cells if not c["valid"]]
    seen = [c for c in cells if c["seen"]]
    A = S["aggregates_all"]
    U = S["aggregates_excluding_seen"]
    DA = S["distribution_unweighted_all"]
    DU = S["distribution_unweighted_excluding_seen"]
    RV = S["sign_reversal_all"]
    RVU = S["sign_reversal_excluding_seen"]
    L = []
    add = L.append

    add("第三周：四场 2025 最终答案赛中相邻 reasoning 档位配置的有限账本审计——结果报告")
    add("")
    add("证据等级：按冻结设计（预注册_第三周_冻结_2026-10-23.md，18:24 冻结；第一部分的 8 条操作定义优先于第二部分草稿）"
        "执行一次，范围固定。数据是 MathArena 公开缓存的回顾性记录。本报告只是有限账本描述，范围是“这四场比赛中有完整记录的相邻档位配置”，"
        "不推广到所有数学题或所有 LLM；不做逐 cell 显著性筛选，不给独立 pair 区间或 bootstrap 区间。"
        f"已见 cell（开发材料）：{', '.join(short(c['cell_id']) for c in seen)}（第二周）；第一周的 AIME Claude-3.5→3.7 Think 不是档位 pair，不在本普查的 cell 中。"
        "全部配置、档位、合格 pair、不合格原因和逐 cell 合同在解码任何 correct 取值之前写入只读的 week3/frame/。")
    add("")
    rev_txt = ", ".join(short(x) for x in RV["reversal_cells"]) or "无"
    add(f"结论（概括）：四场比赛共 {sum(S['coverage'][c]['n_configs'] for c in W.COMP_ORDER)} 个配置，按冻结规则得到 "
        f"{len(cells)} 个相邻档位 cell（全部来自 o3-mini 与 o4-mini 两个基础端点）；其中 {len(valid)} 个合同有效，{len(na)} 个 NA，"
        f"NA 全部因为八槽完整提示不相同。CMIMC 的两个 cell 都 NA，所以加权汇总只覆盖 AIME、HMMT、BRUMO 三场比赛。"
        f"在 {len(valid)} 个有效 cell 上（主槽 idx0）：R 在 {pp(DA['R']['min'])} 到 {pp(DA['R']['max'])} 之间，"
        f"C = D − D_F 在 {pp(DA['C']['min'])} 到 {pp(DA['C']['max'])} 之间（{DA['C']['n_positive']} 正、{DA['C']['n_zero']} 零、"
        f"{DA['C']['n_negative']} 负）；D 与 D_F 严格异号 {RV['n_sign_reversal']}/{RV['denominator_cells_with_D_and_D_F_defined']}（{rev_txt}）。"
        f"冻结权重的三赛宏平均：G = {pp(A['G']['macro'])}，R = {pp(A['R']['macro'])}，D = {pp(A['D']['macro'])}，"
        f"D_F = {pp(A['D_F']['macro'])}，C = {pp(A['C']['macro'])}，G − D_F = {pp(A['G_minus_D_F']['macro'])}；"
        f"去掉已见 cell 后分别为 {pp(U['G']['macro'])}、{pp(U['R']['macro'])}、{pp(U['D']['macro'])}、{pp(U['D_F']['macro'])}、"
        f"{pp(U['C']['macro'])}、{pp(U['G_minus_D_F']['macro'])}。")
    add("")

    # ------------------------------------------------------------------ 1 data
    add("一、数据与传输")
    for comp in W.COMP_ORDER:
        d = S["data"][comp]
        add(f"- {comp}：{d['repo']}，revision {d['revision']}（取到：{'是' if d['revision_found'] else '否'}；该 revision 最后修改 "
            f"{d['revision_last_modified']}），单个 parquet {d['parquet_bytes']} 字节，{d['source']}，只读：{'是' if d['read_only'] else '否'}；"
            f"许可 {d['license']}。")
    tr = S["transfer"]
    add(f"- 第三周传输合计 {tr['bytes_transferred_week3']} 字节（三个 parquet 362638457 字节 + HF 元数据 JSON {tr['metadata_json_bytes']} 字节），"
        f"上限 450 MB（含重试），重试 {tr['retries']} 次；下载前 C 盘可用约 7.9 GiB（≥ 4 GB 门槛），下载后约 7.4 GiB（未触及 3 GB 停止线）；"
        "没有删除任何文件；新增模型 API 调用 0。")
    add("- AIME 第一周已有同一 revision 的副本（非只读）；冻结设计写的是“新下载三个 parquet”、只复用 HMMT，所以 AIME 也重新下载到新文件夹并设为只读，第一周副本未动。")
    add("")

    # ------------------------------------------------------------------ 2 configs
    add("二、配置枚举、档位与合格 pair（只用元数据，规则在读取 correct 之前固定）")
    for comp in W.COMP_ORDER:
        cv = S["coverage"][comp]
        add(f"- {comp}：配置 {cv['n_configs']} 个，题框 {cv['n_problems']} 题；有阶梯档位 {cv['n_tiered']} 个，无档位 {cv['n_no_tier']} 个，"
            f"配置 NA {cv['n_config_NA']} 个；合格相邻 pair {cv['n_eligible_pairs']} 个；有效 cell {cv['n_cells_valid']}，NA cell {cv['n_cells_NA']}。")
    add("- 合格 pair：o3-mini（low→medium、medium→high，只在 AIME、HMMT 有三档）与 o4-mini（low→medium、medium→high，四场都有三档）。"
        "o3-mini 的 high 档原始路径是 openai/o3-mini（无后缀），由显示名 “o3-mini (high)” 定档，与第二周相同。")
    add("- 不合格配置的原因只有两类：显示名末尾括号与路径 --后缀都没有阶梯词（例如 “(Think)”、“(05-06)”、“(preview)” 不算档位），"
        "或该基础端点在这场比赛只有一个档位（GPT-5、GPT-5-mini、GPT-5.1、o1、o3、GPT OSS 等）。没有出现档位冲突、一个配置多个显示名或同端点同档位重复。"
        "逐配置原因见 frame/configs.csv，逐端点见 frame/endpoints.csv。")
    add("- 规则边界上的配置（frame/near_miss_notes.csv，均按冻结规则不接）：GPT-5.2 (high) 路径 openai/gpt-52-high 与 GPT-5.2 (xhigh) 路径 openai/gpt-52 "
        "去掉 --<档位> 后不同（-high 是单连字符，不是冻结的 --<档位> 后缀），所以 AIME、HMMT 中 high→xhigh 这一对没有进入普查；"
        "若放宽到单连字符，它会是相邻的一对，本次按冻结规则不计算。Grok 3 Mini (low)/(high) 与 AIME 的 GPT-5-nano (low)/(high) 同样路径不同，而且即使合并也缺 medium，不相邻。")
    add("")

    # ------------------------------------------------------------------ 3 contract
    add("三、逐 cell 合同（完整题框、两边 idx 0–3、每题单一题文与 gold、八槽完整提示逐字节相同、已存 correct 非缺失、响应状态）")
    for c in cells:
        f = next(x for x in frame["cells"] if x["cell_id"] == c["cell_id"])
        tag = "【已见】" if c["seen"] else ""
        if c["valid"]:
            add(f"- {short(c['cell_id'])}{tag}：有效，{c['n_problems_complete']}/{c['n_problems_frame']} 题齐全。")
        else:
            dg = diag.get(c["cell_id"], {})
            add(f"- {short(c['cell_id'])}：NA，{f['n_problems_breaking']}/{c['n_problems_frame']} 题违约，原因全部是“八槽完整提示不相同”"
                f"（A、B 两侧 user 提示不同的题 {dg.get('user_message_differs')} 道；侧内四槽提示不一的题：A {dg.get('within_side_distinct', {}).get('A')}、"
                f"B {dg.get('within_side_distinct', {}).get('B')} 道）。")
    add("  NA 的具体差别（frame 写定之后另读提示文本，不读 correct；diag_prompt_mismatch.json）：AIME o3-mini medium→high 中，第 1–13 题两侧 user 提示不同"
        "（抽看的第 1–3 题：high 档要求“输出最终答案对 1000 取模”，medium 档说“答案是 0 到 999 之间的整数”，与第一周 AIME I 的提示差同一类），"
        "另有 1 道题 high 档四槽之间提示也不一致；CMIMC 中，抽看的示例题里 o4-mini medium 档提示比 low、high 档多一段最终答案格式说明"
        "（要求按固定标题格式给出 \\boxed{} 答案），low→medium 有 36/40 题、medium→high 有 39/40 题两侧提示不同。按冻结规则这些 cell 只在覆盖表出现，不取两边交集、不删题。")
    add("- 全部 12 个 cell 的 3040 个保留槽（含 NA cell；每个有效 cell 240 个）answer 都非空（已保存截断 0、未知响应状态 0），correct 无缺失，未出现重复行。"
        "AIME o3-mini 三档在 15 道题的 idx1–3 记录 output_tokens = 0（与第一周 AIME I 的记账方式相同：token 合计记在 idx0），按规则保留，不当作零调用。"
        "o4-mini high 档在 AIME 有 3 条、HMMT 有 6 条输出 token 恰为 64000，answer 非空，按原 correct 计。各配置单价均为输入 1.1、输出 4.4 美元/百万 token。")
    add("")

    # ------------------------------------------------------------------ 4 cell table
    add("四、全部 cell 表（主槽固定 idx0，idx1–3 平均为 A₁、B₁；S = {A₀ = 0}，K = |S|；精确分数，括号内为百分点）")
    for c in cells:
        p = c["primary"]
        tag = "【已见】" if c["seen"] else ""
        if p is None:
            add(f"- {short(c['cell_id'])}：NA（{c['na_reason']}）")
            continue
        add(f"- {short(c['cell_id'])}{tag}：K = {p['K']}/{p['n_F']}；G = {both(p['G'])}；R = {both(p['R'])}；D = {both(p['D'])}；"
            f"D_F = {both(p['D_F'])}；C = {both(p['C'])}；G − D_F = {both(p['G_minus_D_F'])}；异号标志 1[D·D_F<0] = {p['sign_reversal']}；"
            f"A 四槽正确率 {fx(c['acc_A_4slot'])}，B 四槽正确率 {fx(c['acc_B_4slot'])}。")
    add("  每个有效 cell 都满足 G − R = D 与 G − D_F = R + C（代码内精确断言）。逐题明细见 tables/per_problem_primary.csv，逐槽 correct 见 tables/slot_correct.csv。")
    add("")

    # ------------------------------------------------------------------ 5 distribution
    add("五、未加权 cell 分布（这组有限配置的分布；同一配置会同时出现在一个端点的两个 cell 中，cell 不是独立样本）")
    for lab, D, RR in (("全部有效 cell", DA, RV), ("去掉已见 cell", DU, RVU)):
        add(f"- {lab}：有效 {D['R']['n_defined']} 个（合同 NA {D['R']['n_contract_NA']} 个，K = 0 的 {D['R']['n_undefined_K0']} 个）。")
        for k in ("K", "G", "R", "D", "D_F", "C", "G_minus_D_F"):
            d = D[k]
            vals = "、".join(f"{x['value']['exact']}" for x in d["values"])
            if k == "K":
                kv = "、".join(str(int(x["value"]["value"])) for x in d["values"])
                add(f"    K：最小 {int(d['min']['value'])}，中位 {d['median']['value']:g}，最大 {int(d['max']['value'])}（从小到大：{kv}）")
            else:
                add(f"    {NAMES[k]}：最小 {pp(d['min'])}，中位 {pp(d['median'])}，最大 {pp(d['max'])}；正 {d['n_positive']}、零 {d['n_zero']}、负 {d['n_negative']}"
                    f"（从小到大：{vals}）")
        add(f"    D 与 D_F 严格异号：{RR['n_sign_reversal']}/{RR['denominator_cells_with_D_and_D_F_defined']}（"
            f"{', '.join(short(x) for x in RR['reversal_cells']) or '无'}）；分母是 D、D_F 都有定义的全部 cell，D 为零的 cell "
            f"{len(RR['cells_with_D_zero'])} 个，D_F 为零的 cell {len(RR['cells_with_D_F_zero'])} 个（零值留在分母内）。")
    zc = [x["cell_id"] for x in DA["C"]["values"] if q(x["value"]) == 0]
    zr = [x["cell_id"] for x in DA["R"]["values"] if q(x["value"]) == 0]
    add(f"- 零值另列：C = 0 的 cell：{', '.join(short(x) for x in zc) or '无'}（D = D_F）；R = 0 的 cell：{', '.join(short(x) for x in zr) or '无'}"
        "（即第二周已见 cell，A 在被筛出题的 idx1–3 也全错）。R ≥ 0 在布尔判分加 A₀ = 0 的筛选下由定义保证，R 为正本身不是机制证据。")
    add("- 唯一异号 cell（BRUMO o4-mini medium→high）：K = 4，在被筛出的 4 题上 D = −1/12，全框 D_F = +1/90；R = 1/2 而 G = 5/12，即 medium 档在这 4 题其余三槽的回升大于 high 档。"
        "K 只有 4，这个异号只描述这本账，不推断 high 档在这类题上更差。")
    add("")

    # ------------------------------------------------------------------ 6 aggregates
    add("六、冻结权重汇总（每场比赛先对一个基础端点的有效相邻 pair 等权，再对端点等权；有效比赛再等权宏平均）")
    for lab, AG in (("全部 cell", A), ("去掉已见 cell", U)):
        add(f"- {lab}：")
        for k in ("G", "R", "D", "D_F", "C", "G_minus_D_F"):
            r = AG[k]
            comps = "；".join(
                f"{comp} {pp(r['competitions'][comp]['value'])}（端点 有效 {r['competitions'][comp]['n_endpoints_valid']}/NA {r['competitions'][comp]['n_endpoints_NA']}，"
                f"cell 有效 {r['competitions'][comp]['n_cells_valid']}/合同NA {r['competitions'][comp]['n_cells_contract_NA']}/K=0 {r['competitions'][comp]['n_cells_undefined_K0']}）"
                for comp in W.COMP_ORDER)
            add(f"    {NAMES[k]}：宏平均 {both(r['macro'])}，有效比赛 {r['n_competitions_valid']}/{r['n_competitions']}（NA {r['n_competitions_NA']}）；{comps}")
    add("  端点层：AIME 的 o3-mini 端点只剩 low→medium 一个有效 cell（medium→high 为 NA），所以该端点的值就是这一 cell；"
        "去掉已见 cell 后 HMMT 的 o3-mini 端点也只剩 low→medium。CMIMC 没有有效端点，宏平均是三场比赛的平均，不是四场结论。逐层数值见 tables/aggregates.csv。")
    add("")

    # ------------------------------------------------------------------ 7 strata
    st = S["stratum_A_acc_30_70"]
    add("七、预定描述分层（A 在全题框四槽合计缓存正确率是否落在 30%–70%；只作描述，不用于入选）")
    for lab, nm in (("A_acc_in_30_70", "30%–70% 内"), ("A_acc_outside_30_70", "30%–70% 外")):
        s = st[lab]
        add(f"- {nm}：{s['n_cells']} 个 cell（{', '.join(short(x) for x in s['cells'])}）；R {pp(s['R']['min'])} 到 {pp(s['R']['max'])}，"
            f"C {pp(s['C']['min'])} 到 {pp(s['C']['max'])}，异号 {s['reversal']['n_sign_reversal']}/{s['reversal']['denominator_cells_with_D_and_D_F_defined']}。")
    add("  层内 cell 太少且共享记录，这只是描述，不说明分层与效果的关系。")
    add("")

    # ------------------------------------------------------------------ 8 cycle
    add("八、四槽循环（只作敏感性；筛选槽依次为 idx0–3，其余三槽复评；四套共享同一批缓存记录，不是四次独立重复；主槽 idx0 冻结时已固定）")
    n_c_sign_change, n_flag_any = 0, 0
    for c in valid:
        cyc = c["cycle"]
        cs = [q(o["C"]) for o in cyc if o["C"] is not None]
        flags = [o["sign_reversal"] for o in cyc if o["sign_reversal"] is not None]
        if cs and (min(cs) < 0 < max(cs) or (min(cs) <= 0 < max(cs) and min(cs) < 0)):
            n_c_sign_change += 1
        if any(flags):
            n_flag_any += 1
        rg = S["cycle_ranges"][c["cell_id"]]
        add(f"- {short(c['cell_id'])}：K {', '.join(str(o['K']) for o in cyc)}；R {pp(rg['R'][0])} 到 {pp(rg['R'][1])}；"
            f"C {pp(rg['C'][0])} 到 {pp(rg['C'][1])}；异号标志 {', '.join(str(f) for f in flags)}。")
    add(f"  四槽中 C 同时出现正值和负值的 cell 有 {n_c_sign_change} 个；至少一个槽出现 D、D_F 异号的 cell 有 {n_flag_any} 个。"
        "不按循环结果改主槽。逐槽明细见 tables/four_slot_cycle.csv。")
    add("")

    # ------------------------------------------------------------------ 9 independent
    add("九、独立重算")
    add(f"- 另写代码 code/week3/w3_independent.py：自写 parquet 解码（Thrift compact、Snappy、RLE/位打包、字典编码、v1/v2 数据页），不导入 pq_reader、w1_*、w2_*、w3_* 代码；"
        "自己实现档位/基础端点/相邻规则、逐 cell 合同（含八槽提示比较）、主槽与四槽循环的精确分数运算、冻结权重汇总和异号计数，从三个新 parquet 与 HMMT 副本重新解码，"
        "再与 frame.json、summary.json 逐项比较。")
    add(f"- 结果：不一致 {len(ind['mismatches'])} 项（{'全部一致' if ind['all_agree'] else '有不一致，见 independent_recompute.json'}）；"
        f"墙钟 {ind['seconds']} 秒，峰值工作集 {ind['peak_working_set_mib']} MiB。")
    add("- 局限：由同一执行者写成，代码路径独立，人员不独立。")
    add("")

    # ------------------------------------------------------------------ 10 limits
    add("十、局限")
    add("1. 有限账本描述：范围只是这四场比赛中有完整记录的相邻档位配置，实际只涉及 OpenAI 的 o3-mini 与 o4-mini 两个基础端点、9 个有效 cell；"
        "不推广到所有数学题、所有 LLM 或所有 reasoning 档位。CMIMC 缺失，汇总是三场比赛的描述。")
    add("2. cell 不是独立样本：同一基础端点的两个 cell 共用中间档（medium）的记录，同题、同配置和四槽也共享记录；不做显著性筛选，不给独立 pair 区间或把逐行 bootstrap 当复现。")
    add("3. 档位比较是同一基础端点不同 reasoning 档位的缓存比较，包含预算和调用条件差（例如 o4-mini high 有输出 token 恰为 64000 的记录），不识别纯能力效应；"
        "命名相同不证明权重或服务历史恒定。历史请求参数、重试/替换链、真实随机种子、判分版本与 usage 归属均为 UNKNOWN，可交换性与独立性 UNKNOWN；账本恒等式不需要这些假设，把槽均值解释为 iid 成功概率则需要。")
    add("4. 判分层是各 revision 存储的 correct，没有重判、没有另写宽松解析；原 parser 版本 UNKNOWN。")
    add("5. 已见材料：HMMT o3-mini medium→high 是第二周开发材料，已单列并另给去掉它的汇总；HMMT o3-mini low→medium 的 B 侧（medium 档）记录也在第二周解码过，但按冻结规则（比赛×pair）不算已见 cell。"
        "第二周及更早的工作看过这些配置的公开榜分，配对范围本身不是未见验证。")
    add("6. 覆盖受冻结规则限制：严格读取 “--<档位>” 后缀使 GPT-5.2 high→xhigh（AIME、HMMT）未进入普查；八槽提示须逐字节相同使 3 个 cell 成为 NA。这些是覆盖缺口，不是模型机制被否定。")
    add("7. 30%–70% 分层只是预定描述层，非效果保证；K 小（最小 4）的 cell 中单题就能明显改变 R、D、C。")
    add("")

    # ------------------------------------------------------------------ 11 resources
    R = S["resources"]
    add("十一、计时与资源")
    add(f"- 下载（含 HF 元数据请求）约 {dl_s if dl_s is not None else '—'} 秒；第一阶段（元数据、档位、合同、写只读 frame）{R['stage1_s']} 秒，峰值工作集 {R['stage1_peak_working_set_mib']} MiB；"
        f"第二阶段（解码 correct、计算、写表）{R['stage2_s']} 秒，峰值工作集 {R['stage2_peak_working_set_mib']} MiB（此前一次尝试在算完 cell 后因分布函数的类型错误退出，约 10 秒，未写任何输出）；"
        f"独立重算 {ind['seconds']} 秒，峰值工作集 {ind['peak_working_set_mib']} MiB。"
        + (f"从第一次联网请求到报告写成的总墙钟约 {wall_total / 60:.1f} 分钟（含编写代码的时间），低于 2 小时上限。" if wall_total else ""))
    add(f"- 每个 Python 进程单进程、单线程、BelowNormal，job object 进程内存上限 2 GiB；重步骤前用 PDH 读 \\Memory\\Available MBytes，均 ≥ 1500 MB 才放行"
        f"（读 correct 前读数：{', '.join(f'{k} {v:.0f} MB' for k, v in R['ram_before_correct_mb'].items())}）。没有动任何其他进程或其他项目的文件。")
    add("")

    # ------------------------------------------------------------------ 12 files
    add("十二、文件")
    add("- week3/frame/（只读，解码 correct 之前写定）：frame.json（全部配置、档位、端点、合格 pair、逐 cell 合同与操作定义）、configs.csv、endpoints.csv、cells.csv、slots.csv、near_miss_notes.csv。")
    add("- week3/summary.json：全部数值（精确分数与小数）、覆盖、汇总、分布、分层、四槽范围、独立重算结果、计时。")
    add("- week3/tables/：cells_all.csv（全部 12 个 cell，含 NA 及原因）、per_problem_primary.csv、four_slot_cycle.csv、aggregates.csv、slot_correct.csv。")
    add("- week3/independent_recompute.json、diag_prompt_mismatch.json、run_log.txt、timeline.jsonl、stage*.stdout.txt、independent.stdout.txt。")
    add("- 代码：code/week3/（w3_common.py、w3_frame.py、w3_compute.py、w3_independent.py、w3_report.py、w3_diag_prompts.py、test_w3.py、两个只读元数据探查脚本 w3_probe_*.py）。")
    add("- 数据：downloads/w3_hf_aime_2025_outputs_16e93d3/、w3_hf_brumo_2025_outputs_3b234ef/、w3_hf_cmimc_2025_outputs_2782618/（只读）；复用 downloads/hf_hmmt_feb_2025_outputs_e3fb6c7/；元数据与下载记录 downloads/w3_meta_web/。")
    add("")

    # ------------------------------------------------------------------ 13 decisions
    add("十三、执行中自行决定的操作细节（均在解码 correct 之前写进代码与 frame.json）")
    for x in frame["decided_before_correct"]:
        add(f"- {x}")
    add("- 合同里的 correct 非缺失只读 parquet 定义层（空值掩码），取值字节在第一阶段从未解码。")
    add("- 汇总中某个量在某 cell 无定义（K = 0）时，该 cell 在该量的这一层计为无定义；本次所有有效 cell 的 K ≥ 4，未出现 K = 0。")
    add("- 冻结第一部分第 7 条要求的独立重算照做。")
    add("- 写 frame 之前跑过两个只读探查（w3_probe_schema.py：列结构与配置名；w3_probe_prompts.py：按规则得到的合格配置的消息角色结构），都不读 correct；"
        "完整提示的定义在探查之前已写进代码，探查只确认 all_messages 的结构是 [user, assistant]。")

    txt = "\n".join(L) + "\n"
    (W.W3 / "report.txt").write_text(txt, encoding="utf8")
    S["independent_recompute"] = dict(all_agree=ind["all_agree"], n_mismatches=len(ind["mismatches"]),
                                      seconds=ind["seconds"], peak_working_set_mib=ind["peak_working_set_mib"],
                                      file=str(W.W3 / "independent_recompute.json"))
    S["resources"]["download_s_approx"] = dl_s
    S["resources"]["total_wall_s_first_request_to_report"] = wall_total
    C1.write_json(W.W3 / "summary.json", S)
    print("report written", len(txt), "chars")


if __name__ == "__main__":
    main()

"""Week 2: write week2/report.txt (Chinese) from summary.json, frame/frame.json and independent_recompute.json,
and add the independent-recompute result and total timings to summary.json.  No data is read here.

Run:  python -X utf8 -I -B w2_report.py
"""
import json
import os
import sys
from fractions import Fraction as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w2_common as W  # noqa: E402

C1 = W.C1
FIRST_INDEP_ATTEMPT_S = 60.0        # aborted at the RAM counter (typeperf timeout) before any data was read


def q(d):
    return None if d is None else Q(d["exact"])


def pp(x):
    return "未定义" if x is None else f"{float(x) * 100:+.2f} 个百分点"


def fx(x):
    return "未定义" if x is None else f"{x.numerator}/{x.denominator}"


def main():
    s = C1.read_json(W.W2 / "summary.json")
    fr = C1.read_json(W.FRAME_DIR / "frame.json")
    ind = C1.read_json(W.W2 / "independent_recompute.json")
    p = s["primary"]
    K, nF = p["K"], p["n_F"]
    G, R, D, DF, DD = (q(p[k]) for k in ("G", "R", "D", "D_F", "D_minus_D_F"))
    A1S, B1S = q(p["A1_S"]), q(p["B1_S"])
    pc = s["primary_counts"]
    de = s["descriptive_not_registered"]
    cc = de["call_conditions"]
    d = s["data"]
    cm = s["config_map"]
    t = s["timings"]
    total_s = round(t["stage1_total_s"] + t["stage2_total_s"] + ind["seconds"] + FIRST_INDEP_ATTEMPT_S, 2)
    s["independent_recompute"] = dict(all_agree=ind["all_agree"], agree=ind["agree"], values=ind["independent"],
                                      method=ind["method"], seconds=ind["seconds"], limits=ind["limits"])
    s["timings"].update(independent_s=ind["seconds"], independent_first_attempt_aborted_s=FIRST_INDEP_ATTEMPT_S,
                        cumulative_cpu_process_s=total_s)
    s["decision"] = "完成（30 题合同满足；未触发 NA）" if total_s <= W.TIME_CAP_S else "NA（累计时间超限）"
    C1.write_json(W.W2 / "summary.json", s)

    rots = s["sensitivity_four_slot_cycle"]
    rot_lines = []
    for o in rots:
        rot_lines.append(
            f"    筛选槽 idx{o['screen_idx']}{'（主）' if o['screen_idx'] == 0 else ''}：K = {o['K']}，"
            f"G = {fx(q(o['G']))}（{pp(q(o['G']))}），R = {fx(q(o['R']))}（{pp(q(o['R']))}），"
            f"D = {fx(q(o['D']))}（{pp(q(o['D']))}），D_F = {fx(q(o['D_F']))}（{pp(q(o['D_F']))}），"
            f"D − D_F = {fx(q(o['D_minus_D_F']))}（{pp(q(o['D_minus_D_F']))}）")
    rg = s["sensitivity_range_where_defined"]
    rr = lambda k: f"{pp(q(rg[k][0]))} 到 {pp(q(rg[k][1]))}"  # noqa: E731
    pats = s["per_problem_patterns_p0"]
    a_all_wrong = sum(x["n"] for x in pats if x["A0"] == 0 and x["A1"] == "0")
    a_all_right = sum(x["n"] for x in pats if x["A0"] == 1 and x["A1"] == "1")
    b_zero_on_S = sum(x["n"] for x in pats if x["A0"] == 0 and x["B1"] == "0")
    S_list = "、".join(p["S"])
    ok_ind = "全部一致" if ind["all_agree"] else "存在不一致（见 independent_recompute.json）"

    txt = f"""第二周：HMMT February 2025 按答错筛题回放（A = o3-mini (medium)，B = o3-mini (high)）——结果报告

证据等级：按冻结设计（预注册_第二周_冻结_2026-10-23.md，冻结于 17:08）首次取数和计算；数据是公开缓存的回顾性记录。配对是按冻结前可见的官网公开榜分选型的（A 53.33%、B 67.50%），配对选择不算未见效果验证。只执行了冻结设计写明的这一个设计；设计列为“不做”的都没有做（不用总榜 Expected Performance、不看新 D 换配置或删题、不补调用、不重试、不另写宽松判分、不联系 MathArena、不给题目总体或独立调用置信区间、不按结果挑主槽）。

结论（一句话）：K = {K}/{nF} 道被 A 的 idx0 答错筛出；在这 {K} 题上 G = {pp(G)}，R = {pp(R)}（A 在 {pc['rescore_records_per_side_on_S']} 条复评记录中命中 {pc['A_rescore_hits_on_S']} 条），D = {pp(D)}；全框 D_F = {pp(DF)}，D − D_F = {pp(DD)}。在这本固定 30 题账本里，筛选后没有缓存记录回升，但筛选集上的配置差比全框大 {float(DD) * 100:.2f} 个百分点。

一、数据、配置对应与 30 题合同
- 数据：HF 数据集 {d['repo']}，固定 revision {d['revision']}（commit 日期 {d['revision_commit_date_utc']}，取数时是 main 的最新 commit），单个 parquet，{d['parquet_bytes']} 字节，{fr['n_rows_file']} 行、{fr['n_configs_file']} 个配置；许可 {d['license_in_card']}。下载后设为只读副本，全程只读。传输合计 {d['bytes_transferred_total']} 字节（上限 250 MB，含 HF 元数据和数据卡），无重试，新增 API 调用 0。
- 配置对应（规则在使用 correct 之前写进代码）：A = {cm['A']['raw_model_config']}（model_name “o3-mini (medium)”），路径和显示名两条规则都指向它；B = {cm['B']['raw_model_config']}（model_name “o3-mini (high)”），只由显示名规则命中——原始路径没有 “--high” 后缀，与设计所引 B 官方模型页地址 openai_o3_mini 相符。每侧恰好一个原始值、两侧不同、各只带一个 model_name，按规则判为明确。文件中另有 openai/o3-mini--low（o3-mini (low)），未使用。
- 30 题合同：候选题 {fr['n_candidates']} 道，F = {fr['n_F']} 道，违约 {len(fr['breaks'])} 道，未触发 NA。两配置各 idx 0–3 唯一齐全；每题 8 条记录题文与 gold 各只有一个取值；8 条的 user 提示在全部 30 题上逐字相同（{cc['prompts']['problems_with_identical_user_message_across_all_8_slots']}/30），都没有 system/developer 轮（消息结构为 user/assistant）；correct 全部是已存布尔值，无缺失；240 条答案都非空（已保存截断 0 条、未知响应状态 0 条），parsed_answer 无空值；输入、输出 token 在每个槽都非零（没有第一周 AIME I 那种把四次调用合计记在 idx0 的记账）。
- 可见调用条件（描述，不据此删题）：两配置单价相同（输入 1.1、输出 4.4 美元/百万 token）。A 输出 token {cc['A']['output_tokens']['min']}–{cc['A']['output_tokens']['max']}，没有超过 A 官网所列 max_tokens 32000 的；B 输出 token {cc['B']['output_tokens']['min']}–{cc['B']['output_tokens']['max']}，120 条中有 {de['B_rows_output_tokens_over_32000']} 条超过 32000，所以 B 的预算上限不同于 A 所列的 32000（设计已写明不推定与 A 相同）。

二、主结果（冻结定义：固定 p0 = idx0 筛题，idx1–3 平均为 A₁、B₁；S = {{A₀ = 0}}，K = |S|）
- K = {K}（30 题中 A 的 idx0 答错 {K} 题：第 {S_list} 题）。
- 在 S 上：A₁ = {fx(A1S)}（{pc['rescore_records_per_side_on_S']} 条 A 复评记录命中 {pc['A_rescore_hits_on_S']} 条），B₁ = {fx(B1S)}（{pc['rescore_records_per_side_on_S']} 条 B 复评记录命中 {pc['B_rescore_hits_on_S']} 条）。
- G = mean(B₁ − A₀) = {fx(G)}（{pp(G)}）
- R = mean(A₁ − A₀) = {fx(R)}（{pp(R)}）
- D = mean(B₁ − A₁) = {fx(D)}（{pp(D)}）；G − R = D 精确成立。
- 读法（冻结措辞）：R 是缓存记录回升，D 是同一筛选集的配置差，D − D_F 才是该账本的组成差，三者不能互相代替。本次 R 精确为 0：A 在被筛出的 {K} 题上，idx1–3 的 {pc['rescore_records_per_side_on_S']} 条记录也全错，所以 G 与 D 相等。这是零结果，照实报告，不称“无效”，也不称“等效”或“没有回归效应”。

三、次要量
- D_F = 全框 F 上 mean(B₁ − A₁) = {fx(DF)}（{pp(DF)}）；D − D_F = {fx(DD)}（{pp(DD)}）。即在这本账本里，按 A 的 idx0 答错筛出的题上，B 与 A 的配置差比全 30 题大 {float(DD) * 100:.2f} 个百分点。R 与 D − D_F 分开报告。
- 四槽循环筛选（只作敏感性；四套共享同一批缓存记录，不是四次独立重复；主槽 idx0 在冻结时已固定，不按结果选择）：
{chr(10).join(rot_lines)}
  四套都有定义：R 在 {rr('R')} 之间，D 在 {rr('D')} 之间，D − D_F 在 {rr('D_minus_D_F')} 之间（四套都为正）。主槽的 R = 0 是四套中最小的；换筛选槽后 R 可到 {float(q(rg['R'][1])) * 100:.2f} 个百分点，所以“R = 0”只属于 idx0 这一账本，不能说成这对配置一般没有回升。D_F 随筛选槽变，是因为 A₁、B₁ 是其余三槽的均值。
- 不确定性：主量是固定 30 题账本的精确描述；按冻结设计不给题目总体或独立调用置信区间。

四、描述性背景（非预注册量）
- F 上四槽合计：A {de['A_all4_hits']}/120（{float(q(de['A_all4_on_F'])) * 100:.2f}%），B {de['B_all4_hits']}/120（{float(q(de['B_all4_on_F'])) * 100:.2f}%），与设计所引官网实际比赛正确率 53.33%、67.50% 数值相同；A 的 idx0 单槽 {pc['A0_correct']}/30。这提示官网数字由这批缓存记录算出，但不证明逐条对应，也不证明历史调用一致。
- 逐题形态（idx0 筛选）：A 在 {a_all_wrong} 题上四槽全错（正好就是 S 的全部 {K} 题），这是 R = 0 的直接来由；在 {a_all_right} 题上四槽全对，其余 {nF - a_all_wrong - a_all_right} 题混合；B 在 S 中 {b_zero_on_S} 题的 idx1–3 也全错（逐题明细见 tables/per_problem_p0.csv）。

五、独立重算
- 另写代码重算主量：自写的 parquet 解码（Thrift compact、Snappy、RLE/位打包、字典编码；不导入 pq_reader 或 w1_*/w2_* 代码），按 model_name 等于官网显示名选 A/B（得到同样两个原始配置），自做 idx 0–3 唯一齐全和 correct 非空检查，自做精确分数运算。K、S、G、R、D、D_F、D − D_F {ok_ind}；四槽合计 A {ind['A_all4_hits']}/120、B {ind['B_all4_hits']}/120 也一致。
- 局限：重算由同一执行者写成，代码路径独立，但不是另一位复核者。

六、局限（冻结设计列出的，另加本次执行中出现的）
1. 公开缓存的回顾性分析；配对按冻结前可见的公开榜分选型，配对选择不算未见效果验证。
2. 这是同一基础模型不同 reasoning 档位的缓存比较，包含预算/调用条件差（B 有 {de['B_rows_output_tokens_over_32000']} 条输出超过 32000 token），不识别跨代或纯能力效应；A 只称较低档旧配置，不称旧一代模型。
3. 当前 API/batch 一致不证明历史调用一致，更不证明 iid。调用时间、resolved endpoint、完整提示之外的请求参数、尝试/重试与替换链、真实随机种子、判分版本和 usage 归属都不明（UNKNOWN），可交换性与独立性仍 UNKNOWN。账本恒等式无需这些假设；把槽均值解释为 iid 的题目成功概率则需要。
4. 30–70% 不保证偏差大：仅在条件 iid 总体模型下，μ = E p_A < 1 时 R_pop = μ − Var(p_A)/(1 − μ)；μ = .5 且题目 p_A 各为 0 或 1 时 R_pop = 0。这不是本次有限账本 R 的代数恒等式。总体组成差还取决于 p_B − p_A 与筛选概率的协方差，方向不定。本次账本 R = 0、A 在 {a_all_wrong + a_all_right}/30 题上四槽全对或全错，只是描述，不据此推断 p_A。
5. 判分层用该 revision 存储的 correct；没有重判，没有另写宽松判分；已保存的拒答、截断按原 correct 计（本框内没有截断或空答案）。原 parser 版本 UNKNOWN。“可解释的最终答案判分”按“correct 为已存布尔值且 gold 非空”执行，没有逐条审读解析。
6. 若需概率机制解释，可向 MathArena 作者请求原始调用时间、resolved endpoint、完整提示、尝试/重试与替换链、判分版本及 usage 归属；本次不联系、不补调用。缺这些材料限制解释范围，不否定有限账本研究。
7. 合同里“共同题文/gold与完整提示”有两种读法（提示须相同，或只须完整）；本次 8 槽提示在 30 题上逐字相同，两种读法结果一样。
8. 预先确定的操作定义（在使用 correct 取值之前写进代码，frame/frame.json 记录，只读）：
{chr(10).join('   - ' + x for x in fr['decided_before_correct'])}
   第一阶段为核对“correct 无缺失、为布尔值”解码过 correct 列，但只保留是否为空、是否布尔两个标志（另有仅在内存中用于区分重复行的行摘要），取值未输出、未用于任何计算。
9. 资源执行说明：独立重算那一步，运行前的内存检查循环按代码逻辑只有读到 ≥ 1500 MB 才放行（读数未记下）；随后记录起始读数的第二次计数器调用连续超时，日志记为 0.0；之后才读数据。运行期间系统可用内存因其他进程降到约 459 MB（本进程工作集约 172 MiB）；没有动任何其他进程。这一步墙钟 {ind['seconds']} 秒，远长于计算所需；中途观察时该进程 CPU 时间约 1.7 秒、5 秒内不增长，说明大部分时间在等待（计数器调用超时重试，或内存紧张时换页），未逐项核实。此前还有一次尝试，在内存计数器处超时退出（约 {FIRST_INDEP_ATTEMPT_S:.0f} 秒），当时还没读数据。

七、计时与资源
- 下载 {t['download_s']} 秒（163.5 MB，一次完成）；第一阶段（读取、配置对应、合同、写只读 frame）{t['stage1_total_s']} 秒；第二阶段（读 correct、计算、写表）{t['stage2_total_s']} 秒；独立重算 {ind['seconds']} 秒（另有一次约 {FIRST_INDEP_ATTEMPT_S:.0f} 秒的中止尝试）。以上都是墙钟时间；进程累计约 {total_s} 秒（约 {total_s / 60:.1f} 分钟，含中止尝试、不含下载；含下载约 {total_s + t['download_s']:.2f} 秒、{(total_s + t['download_s']) / 60:.1f} 分钟），低于 1 小时上限。
- 单进程、单线程、BelowNormal、进程内存上限 2 GiB（job object）；峰值工作集：第一阶段 {s['peak_working_set_mib']['stage1']} MiB，第二阶段 {s['peak_working_set_mib']['stage2']} MiB，独立重算约 172 MiB。第一、二阶段读数据前内存计数器读数分别为 2839/2707 MB 和 2754 MB。

八、文件
- week2/summary.json：全部数值（精确分数和小数）、配置对应细节、调用条件描述、独立重算结果、计时。
- week2/tables/per_problem_p0.csv（主槽逐题）、four_slot_cycle.csv（四槽循环）、slot_correct.csv（240 个槽的 correct 和输出 token）。
- week2/frame/frame.json、slots.csv：使用 correct 之前写定的配置对应、30 题合同和操作定义（只读）。
- week2/independent_recompute.json：独立重算。
- week2/run_log.txt、timeline.jsonl、stage*.stdout.txt：运行记录。
- 代码：code/week2/（w2_common.py、w2_extract.py、w2_compute.py、w2_independent.py、w2_report.py、test_w2.py；复用第一周的 code/pq_reader.py 和 code/w1_common.py 中的资源与精确均值函数）。
- 数据：downloads/hf_hmmt_feb_2025_outputs_e3fb6c7/（只读）；元数据：downloads/w2_meta_web/。
"""
    (W.W2 / "report.txt").write_text(txt, encoding="utf8")
    print(f"report written; cumulative {total_s} s; independent all_agree={ind['all_agree']}")


if __name__ == "__main__":
    main()

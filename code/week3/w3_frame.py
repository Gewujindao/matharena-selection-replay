"""Week 3, stage 1: from metadata only, enumerate every config of the four competitions, assign tiers / base endpoints
by the frozen rule, list the eligible adjacent pairs and every ineligible config with its reason, run the per-cell
contract and write it all read-only to week3/frame/ BEFORE any `correct` value is decoded.  For `correct` only the
parquet definition levels (null mask) are read; its value bytes are never decoded here.

Run:  python -X utf8 -I -B w3_frame.py
"""
import hashlib
import json
import os
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w3_common as W  # noqa: E402

C1, P = W.C1, W.P
LOG = W.W3 / "run_log.txt"
TIMELINE = W.W3 / "timeline.jsonl"


def log(msg):
    line = f"[{C1.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def event(name, **kw):
    with open(TIMELINE, "a", encoding="utf8") as f:
        f.write(json.dumps(dict(t_local=C1.now_local(), event=name, **kw), ensure_ascii=False) + "\n")


def ro(path):
    os.chmod(path, stat.S_IREAD)


def h16(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()[:16] if s is not None else "NULL"   # in-run equality only


DECIDED = [
    "档位阶梯 minimal < low < medium < high < xhigh；显示名末尾括号内容恰为阶梯词（不分大小写）才算档位标签，其他末尾括号（Think、05-06、preview 等）不算，留在基础名里",
    "路径后缀只认最后一段路径里最后一个 \"--\" 之后、恰为阶梯词的部分；单个连字符（-low、-high）不是冻结的 --<档位> 后缀；路径只去掉固定前缀 configs/models/ 与后缀 .yaml",
    "档位优先取显示名标签，没有时取路径后缀；两处都有但不一致 → 该配置 NA；一个配置值带不止一个 model_name → 该配置 NA",
    "基础端点 = (显示名去掉末尾档位括号及其前空白, 路径去掉 --<档位> 后缀)，两者都相同才算同一端点；同端点同档位多于一个配置 → 该端点本赛 NA",
    "合格 pair：同一端点、阶梯上相邻的两档，A 为较低档；不相邻不接",
    "题框 = 该比赛冻结副本全部行中出现的不同 problem_idx（source 列全为空，不参与键）",
    "每个 cell 每题：A、B 各 idx 0..3 恰一条（除 correct 外各列完全相同的重复行折叠，第二阶段再核 correct 是否也相同）；内容不同的重复或 0..3 以外的 idx 即违约；缺任何一条即违约",
    "题文/gold：每题 8 条保留记录的 problem 与 gold_answer 各只有一个取值，gold 非空",
    "完整提示 = (user_message, all_messages 中所有非 assistant 条目按原顺序、键排序的规范 JSON)；须 user_message 非空、all_messages 为含 user 轮的 JSON 列表；八槽须逐字节相同",
    "已存 correct 非缺失：只读 parquet 定义层（空值掩码），不解码取值；物理类型须为 BOOLEAN",
    "响应状态（沿用第二周）：answer 非空即保留（token 为 0 也保留，usage 按原记账层级读）；answer 空且 output_tokens>0 为已保存截断，按原 correct 计；answer 空且 output_tokens 空/0 为未知响应状态 → 违约",
    "任一题违约 → 该 cell NA，不取两边交集、不删题、不换配置；NA cell 留在覆盖分母",
    "主槽固定 idx0 筛选、idx1–3 复评；四槽循环只作敏感性",
    "已见 cell：第二周 HMMT openai/o3-mini--medium → openai/o3-mini；第一周 AIME claude-35 → claude-37 不是档位 pair，不会出现在合格 cell 中",
    "预定描述分层（第二部分“30–70%只作预定描述分层”）：A（较低档）在全题框四槽合计的缓存正确率落在闭区间 [30%, 70%]；只作描述，不用于入选",
]


def main():
    W.W3.mkdir(parents=True, exist_ok=True)
    if (W.FRAME_DIR / "frame.json").exists():
        log("frame.json exists; stage 1 is not re-run")
        sys.exit(2)
    t0 = time.time()
    limits = C1.apply_limits()
    log(f"stage 1 start; pid {os.getpid()}; limits {limits}")
    event("stage1_start", pid=os.getpid(), limits=limits)

    # ------------------------------------------------------------------ data gate (revisions, transfer)
    data = {}
    transferred = 0
    for comp in W.COMP_ORDER:
        spec = W.COMPS[comp]
        pq = spec["parquet"]
        info = {}
        if spec["new_download"]:
            name = spec["repo"].split("/")[1]
            info = C1.read_json(W.META / f"hf_{name}_revision_info.json")
            tree = C1.read_json(W.META / f"hf_{name}_tree_data.json")
            size_api = tree[0]["size"] if tree else None
            transferred += os.path.getsize(pq)
        else:
            info = C1.read_json(W.DL / "w2_meta_web" / "hf_dataset_info.json")
            size_api = None
        data[comp] = dict(repo=spec["repo"], revision=spec["rev"], revision_found=info.get("sha") == spec["rev"],
                          revision_last_modified=info.get("lastModified"),
                          license=(info.get("cardData") or {}).get("license"), parquet=str(pq),
                          parquet_bytes=os.path.getsize(pq), size_listed_at_revision=size_api,
                          read_only=not os.access(pq, os.W_OK),
                          source="新下载（第三周）" if spec["new_download"] else "复用第二周只读副本")
    meta_bytes = sum(os.path.getsize(W.META / f) for f in os.listdir(W.META) if f.endswith(".json"))
    transferred += meta_bytes
    gate = dict(bytes_transferred_week3=transferred, metadata_json_bytes=meta_bytes, transfer_cap=W.TRANSFER_CAP,
                retries=0, new_api_calls_to_models=0)
    log(f"data gate: {json.dumps({c: (d['revision'][:7], d['revision_found'], d['parquet_bytes'], d['read_only']) for c, d in data.items()})}; transferred {transferred}")
    if transferred > W.TRANSFER_CAP:
        raise W.StopNA("transfer cap exceeded")

    comps_out, cells_out, cfg_rows, cell_rows, slot_rows, endpoint_rows, notes = {}, [], [], [], [], [], []
    for comp in W.COMP_ORDER:
        spec = W.COMPS[comp]
        if not data[comp]["revision_found"]:
            comps_out[comp] = dict(status="NA", reason="取不到冻结 revision")
            continue
        W.wait_ram(log, f"{comp} metadata")
        tc = time.time()
        pf = P.ParquetFile(str(spec["parquet"]))
        meta_cols = ["model_config", "model_name", "problem_idx", "idx_answer", "source", "input_tokens",
                     "output_tokens", "cost"]
        col = {c: pf.read_column(c) for c in meta_cols}
        n = pf.num_rows
        cfg_names = {}
        for i in range(n):
            cfg_names.setdefault(col["model_config"][i], set()).add(col["model_name"][i])
        configs, endpoints, pairs = W.enumerate_pairs(cfg_names)
        problems = sorted({col["problem_idx"][i] for i in range(n)}, key=W.pidx_key)
        rows_per_cfg = {}
        for i in range(n):
            rows_per_cfg[col["model_config"][i]] = rows_per_cfg.get(col["model_config"][i], 0) + 1
        log(f"{comp}: {n} rows, {len(cfg_names)} configs, {len(problems)} problems, "
            f"{sum(r['status'] == 'tiered' for r in configs.values())} tiered, {len(pairs)} eligible pairs")

        # near-miss notes: same display base, different config base
        by_disp = {}
        for c, r in configs.items():
            if r["status"] == "tiered":
                by_disp.setdefault(r["display_base"], set()).add(r["config_base"])
        for disp, cbs in sorted(by_disp.items()):
            if len(cbs) > 1:
                members = sorted((c, configs[c]["tier"]) for c in configs
                                 if configs[c]["status"] == "tiered" and configs[c]["display_base"] == disp)
                tiers = sorted({t for _, t in members}, key=W.LADDER.index)
                adj = any(W.LADDER.index(b) - W.LADDER.index(a) == 1 for a, b in zip(tiers, tiers[1:]))
                notes.append(dict(comp=comp, display_base=disp, configs=members,
                                  note=("显示名基础相同但路径去掉 --<档位> 后不同，按冻结规则不算同一端点；"
                                        + ("若把单连字符后缀也当档位后缀，这些档位中有相邻的一对（本次不计算）"
                                           if adj else "即使合并，档位也不相邻"))))

        # ------------------------------------------------------------ eligible configs: contract fields
        elig = sorted({p["A"] for p in pairs} | {p["B"] for p in pairs})
        rows = [i for i in range(n) if col["model_config"][i] in elig]
        W.wait_ram(log, f"{comp} contract columns")
        tx = {}
        tx["problem"] = W.scan_column(pf, "problem", rows, lambda v: v)
        tx["gold"] = W.scan_column(pf, "gold_answer", rows, lambda v: v)
        tx["user_message"] = W.scan_column(pf, "user_message", rows, lambda v: v)
        um = tx["user_message"]
        am = W.scan_column(pf, "all_messages", rows, lambda v: (h16(v), W.prompt_part(v)))
        prompt = {}
        for i in rows:
            ok, why, canon = W.full_prompt_from_part(um[i], am[i][1])
            prompt[i] = (ok, why, h16(canon) if canon is not None else None)
        am = {i: am[i][0] for i in rows}
        tx["answer"] = W.scan_column(pf, "answer", rows,
                                     lambda v: (v is not None and v.strip() != "", len(v or ""), h16(v)))
        tx["parsed"] = W.scan_column(pf, "parsed_answer", rows, lambda v: (v is None, h16(v)))
        if pf.ptype["correct"] != 0:
            raise W.StopNA(f"{comp}: correct is not a BOOLEAN column (physical type {pf.ptype['correct']})")
        cnull = W.scan_column(pf, "correct", rows, lambda is_null: is_null, values=False)
        pin =W.scan_column(pf, "input_cost_per_tokens", rows, lambda v: v)
        pout = W.scan_column(pf, "output_cost_per_tokens", rows, lambda v: v)
        recs = []
        for i in rows:
            ans_ne, ans_len, ans_h = tx["answer"][i]
            pnull, ph = tx["parsed"][i]
            digest = h16(json.dumps([col[c][i] for c in meta_cols] + [h16(tx["problem"][i]), h16(tx["gold"][i]),
                                     h16(um[i]), am[i], ans_h, ph, pnull, cnull[i], pin[i], pout[i]],
                                    default=str))
            recs.append(dict(row=i, cfg=col["model_config"][i], problem_idx=col["problem_idx"][i],
                             idx=col["idx_answer"][i], problem=tx["problem"][i], gold=tx["gold"][i],
                             status=W.response_status(ans_ne, col["output_tokens"][i]), prompt_ok=prompt[i][0],
                             prompt_why=prompt[i][1], prompt=prompt[i][2], correct_is_null=cnull[i], digest=digest,
                             answer_len=ans_len, parsed_null=pnull, in_tok=col["input_tokens"][i],
                             out_tok=col["output_tokens"][i], cost=col["cost"][i], price=(pin[i], pout[i])))
        del tx
        byrow = {r["row"]: r for r in recs}

        # ------------------------------------------------------------ cells
        ep_key = lambda p: f"{p['display_base']} | {p['config_base']}"  # noqa: E731
        for p in pairs:
            valid, breaks, kept, slots = W.cell_contract(recs, problems, p["A"], p["B"])
            seen = W.SEEN.get((comp, p["A"], p["B"]))
            cell_id = f"{comp}|{p['display_base']}|{p['tier_A']}->{p['tier_B']}"
            attrs = {}
            for s, cfgv in (("A", p["A"]), ("B", p["B"])):
                rs = [r for r in recs if r["cfg"] == cfgv]
                ot = [r["out_tok"] for r in rs]
                attrs[s] = dict(config=cfgv, model_name=p[f"{s}_name"], n_rows_in_file=rows_per_cfg[cfgv],
                                output_tokens_min=min((x for x in ot if x is not None), default=None),
                                output_tokens_max=max((x for x in ot if x is not None), default=None),
                                n_output_tokens_zero_or_null=sum(1 for x in ot if not x),
                                n_nonempty_answer_zero_or_null_output_tokens=sum(
                                    1 for r in rs if r["answer_len"] and not r["out_tok"]),
                                n_saved_truncation=sum(r["status"] == W.ST_TRUNC for r in rs),
                                n_unknown_status=sum(r["status"] == W.ST_UNKNOWN for r in rs),
                                n_parsed_answer_null=sum(r["parsed_null"] for r in rs),
                                prices_per_million=sorted({str(r["price"]) for r in rs}),
                                total_cost_usd=round(sum(r["cost"] or 0 for r in rs), 6))
            n_dup = sum(1 for s in slots if str(s[4]).startswith("identical_duplicate"))
            cell = dict(cell_id=cell_id, comp=comp, endpoint=ep_key(p), display_base=p["display_base"],
                        config_base=p["config_base"], tier_A=p["tier_A"], tier_B=p["tier_B"], A=p["A"], B=p["B"],
                        A_name=p["A_name"], B_name=p["B_name"], seen=bool(seen), seen_label=seen or "",
                        n_problems_frame=len(problems), n_problems_complete=len(kept), contract_valid=valid,
                        n_problems_breaking=len(breaks), breaks=breaks,
                        na_reason="" if valid else "; ".join(sorted({x for v in breaks.values() for x in v}))[:2000],
                        kept=kept, n_identical_duplicates_collapsed=n_dup, call_condition_attributes=attrs)
            cells_out.append(cell)
            cell_rows.append([cell_id, comp, ep_key(p), p["tier_A"], p["tier_B"], p["A"], p["B"], p["A_name"],
                              p["B_name"], int(bool(seen)), seen or "", len(problems), len(kept), len(breaks),
                              "valid" if valid else "NA", cell["na_reason"]])
            for side, q, k, row, st in sorted(slots, key=lambda s: (W.pidx_key(s[1]), s[0],
                                                                  -1 if s[2] is None else s[2])):
                r = byrow.get(row, {})
                slot_rows.append([cell_id, side, q, k, "" if row is None else row, st, r.get("in_tok", ""),
                                  r.get("out_tok", ""), r.get("answer_len", ""), int(r.get("parsed_null", 0) or 0)])
            log(f"{comp} cell {p['display_base']} {p['tier_A']}->{p['tier_B']}: "
                f"{'valid' if valid else 'NA'} ({len(kept)}/{len(problems)} problems complete){' [seen]' if seen else ''}")
        for c, r in sorted(configs.items()):
            cfg_rows.append([comp, c, r.get("model_name") if isinstance(r.get("model_name"), str)
                             else json.dumps(r.get("model_name"), ensure_ascii=False), rows_per_cfg[c],
                             r.get("name_tag") or "", r.get("cfg_tag") or "", r.get("tier") or "",
                             r.get("display_base", "") or "", r.get("config_base", "") or "", r["status"],
                             r["reason"], r.get("pair_status", "")])
        for base, e in sorted(endpoints.items()):
            endpoint_rows.append([comp, base[0], base[1], ",".join(e["tiers"]), ",".join(e["configs"]), e["status"],
                                  ",".join(e["pairs"]), e["reason"]])
        comps_out[comp] = dict(status="ok", n_rows=n, n_configs=len(cfg_names), problems=problems,
                               n_problems=len(problems), n_tiered=sum(r["status"] == "tiered" for r in configs.values()),
                               n_no_tier=sum(r["status"] == "no_tier" for r in configs.values()),
                               n_config_NA=sum(r["status"] == "NA" for r in configs.values()),
                               n_endpoints_tiered=len(endpoints),
                               n_endpoints_with_pairs=sum(1 for e in endpoints.values() if e["pairs"]),
                               n_eligible_pairs=len(pairs),
                               configs={c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                                        for c, r in sorted(configs.items())},
                               endpoints={f"{b[0]} | {b[1]}": e for b, e in sorted(endpoints.items())},
                               seconds=round(time.time() - tc, 2))
        del col, recs, byrow
        log(f"{comp} done ({comps_out[comp]['seconds']} s, peak working set {C1.peak_rss_mb()[0]} MiB)")

    frame = dict(card="matharena-selection-replay", week=3, prereg=str(W.PREREG),
                 stage="frame written from metadata before any correct value is decoded (correct: null mask only)",
                 written_local=C1.now_local(), data=data, transfer=gate,
                 reader="code/pq_reader.py primitives + w3_common.scan_column (row-selective streaming; null-mask mode)",
                 ladder=W.LADDER, decided_before_correct=DECIDED, competitions=comps_out,
                 near_miss_notes=notes, cells=cells_out,
                 seen_rule={f"{k[0]}|{k[1]}->{k[2]}": v for k, v in W.SEEN.items()},
                 n_cells=len(cells_out), n_cells_valid=sum(c["contract_valid"] for c in cells_out),
                 n_cells_NA=sum(not c["contract_valid"] for c in cells_out))
    W.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    C1.write_csv(W.FRAME_DIR / "configs.csv", ["competition", "model_config", "model_name", "n_rows", "name_tag",
                                               "config_suffix_tag", "tier", "display_base", "config_base", "status",
                                               "reason", "pair_status"], cfg_rows)
    C1.write_csv(W.FRAME_DIR / "endpoints.csv", ["competition", "display_base", "config_base", "tiers", "configs",
                                                 "status", "adjacent_pairs", "reason"], endpoint_rows)
    C1.write_csv(W.FRAME_DIR / "cells.csv", ["cell_id", "competition", "endpoint", "tier_A", "tier_B", "A", "B",
                                             "A_name", "B_name", "seen", "seen_label", "n_problems_frame",
                                             "n_problems_complete", "n_problems_breaking", "contract", "na_reason"],
                 cell_rows)
    C1.write_csv(W.FRAME_DIR / "slots.csv", ["cell_id", "side", "problem_idx", "idx", "parquet_row", "status",
                                             "input_tokens", "output_tokens", "answer_chars", "parsed_answer_null"],
                 slot_rows)
    C1.write_csv(W.FRAME_DIR / "near_miss_notes.csv", ["competition", "display_base", "configs_tiers", "note"],
                 [[x["comp"], x["display_base"], json.dumps(x["configs"], ensure_ascii=False), x["note"]]
                  for x in notes])
    frame["stage1_seconds"] = round(time.time() - t0, 2)
    frame["stage1_peak_working_set_mib"] = C1.peak_rss_mb()[0]
    C1.write_json(W.FRAME_DIR / "frame.json", frame)
    for f in ("configs.csv", "endpoints.csv", "cells.csv", "slots.csv", "near_miss_notes.csv", "frame.json"):
        ro(W.FRAME_DIR / f)
    event("frame_written", n_cells=frame["n_cells"], n_valid=frame["n_cells_valid"], stage1_s=frame["stage1_seconds"],
          peak_mib=frame["stage1_peak_working_set_mib"])
    log(f"frame written read-only: {frame['n_cells']} cells, {frame['n_cells_valid']} valid, "
        f"{frame['n_cells_NA']} NA ({frame['stage1_seconds']} s, peak {frame['stage1_peak_working_set_mib']} MiB)")


if __name__ == "__main__":
    try:
        main()
    except W.StopNA as e:
        log(f"STOP: {e}")
        C1.write_json(W.W3 / "STOP_NA.json", dict(stage="frame", reason=str(e), time_local=C1.now_local()))
        sys.exit(3)

"""Week 2, stage 1: config correspondence and the 30-problem contract, written to week2/frame/ (read-only) before
any `correct` value is used.  `correct` is decoded here only to a null flag and a boolean-type flag.

Run:  python -X utf8 -I -B w2_extract.py
"""
import hashlib
import json
import os
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w2_common as W  # noqa: E402

C1, P = W.C1, W.P
LOG = W.W2 / "run_log.txt"
TIMELINE = W.W2 / "timeline.jsonl"


def log(msg):
    line = f"[{C1.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def event(name, **kw):
    with open(TIMELINE, "a", encoding="utf8") as f:
        f.write(json.dumps(dict(t_local=C1.now_local(), event=name, **kw), ensure_ascii=False) + "\n")


def write_ro_json(path, obj):
    C1.write_json(path, obj)
    os.chmod(path, stat.S_IREAD)


def stop_na(reason, extra):
    rec = dict(decision="NA", stage="extract", reason=reason, time_local=C1.now_local(), extra=extra)
    C1.write_json(W.W2 / "STOP_NA.json", rec)
    event("stop_na", reason=reason)
    log(f"NA STOP: {reason}")
    sys.exit(3)


def pidx_key(q):
    return (0, int(q)) if str(q).isdigit() else (1, str(q))


def sys_user(all_messages):
    try:
        msgs = json.loads(all_messages)
    except (TypeError, ValueError):
        return None, None, None
    if not isinstance(msgs, list):
        return None, None, None
    sysm = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    usr = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    dump = lambda xs: json.dumps(xs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # noqa: E731
    h = lambda s: hashlib.sha256(s.encode("utf8")).hexdigest()[:16]  # noqa: E731  (in-run equality only)
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    return (h(dump(sysm)) if sysm else "NONE"), (h(dump(usr)) if usr else "NONE"), roles


def main():
    W.W2.mkdir(parents=True, exist_ok=True)
    if (W.FRAME_DIR / "frame.json").exists():
        log("frame.json exists; stage 1 is not re-run")
        sys.exit(2)
    t0 = time.time()
    limits = C1.apply_limits()
    log(f"stage 1 start; pid {os.getpid()}; limits {limits}")
    event("stage1_start", pid=os.getpid(), limits=limits)

    # ------------------------------------------------------------------ transfer and resource gate
    transferred = sum(os.path.getsize(os.path.join(dp, f)) for d in (W.DATA_DIR, W.META)
                      for dp, _, fs in os.walk(d) for f in fs)
    info = C1.read_json(W.META / "hf_dataset_info.json")
    commits = C1.read_json(W.META / "hf_commits_main.json")
    data_gate = dict(repo=W.REPO, revision=W.REV, revision_is_main_head_on_fetch=info.get("sha") == W.REV,
                     revision_commit_date_utc=(commits[0] if commits and commits[0].get("id") == W.REV else {}).get("date"),
                     siblings=[s.get("rfilename") for s in info.get("siblings", [])],
                     card_download_size=info.get("cardData", {}).get("dataset_info", {}).get("download_size"),
                     card_num_examples=info.get("cardData", {}).get("dataset_info", {}).get("splits", [{}])[0].get("num_examples"),
                     license_in_card=info.get("cardData", {}).get("license"), url=W.URL,
                     parquet_bytes=os.path.getsize(W.PARQUET), parquet_read_only=not os.access(W.PARQUET, os.W_OK),
                     bytes_transferred_total=transferred, transfer_cap_bytes=W.TRANSFER_CAP, retries=0,
                     new_api_calls=0)
    log(f"data: revision {W.REV[:7]} (main head on fetch: {data_gate['revision_is_main_head_on_fetch']}), "
        f"{data_gate['parquet_bytes']} bytes, transferred {transferred} bytes (cap {W.TRANSFER_CAP})")
    if transferred > W.TRANSFER_CAP:
        stop_na("transfer cap exceeded", data_gate)

    # ------------------------------------------------------------------ metadata (all rows)
    C1.wait_ram(log, "metadata read")
    t = time.time()
    pf = P.ParquetFile(str(W.PARQUET))
    meta_cols = ["model_config", "model_name", "problem_idx", "idx_answer", "source", "input_tokens",
                 "output_tokens", "cost"]
    col = {c: pf.read_column(c) for c in meta_cols}
    n = pf.num_rows
    cfg_names = {}
    for i in range(n):
        cfg_names.setdefault(col["model_config"][i], set()).add(col["model_name"][i])
    mapping, cdetail, clear = W.match_configs(cfg_names)
    log(f"read {len(meta_cols)} metadata columns for {n} rows ({len(cfg_names)} configs); "
        f"config correspondence clear={clear}: {mapping}")
    event("config_match", clear=clear, mapping=mapping)
    if not clear:
        stop_na("config correspondence not clear", dict(detail=cdetail, data=data_gate))
    candidates = sorted({col["problem_idx"][i] for i in range(n)}, key=pidx_key)
    ab_rows = {i for i in range(n) if col["model_config"][i] in mapping.values()}

    # ------------------------------------------------------------------ A/B rows: contract fields only
    C1.wait_ram(log, "A/B row columns")
    other_cols = ["problem", "gold_answer", "user_message", "all_messages", "answer", "parsed_answer", "correct",
                  "input_cost_per_tokens", "output_cost_per_tokens"]
    oc = {c: pf.read_column(c, rows=ab_rows) for c in other_cols}
    inv = {raw: side for side, raw in mapping.items()}
    rows, extra = [], {}
    for i in sorted(ab_rows):
        rec = {c: col[c][i] for c in meta_cols}
        rec.update({c: oc[c][i] for c in other_cols})
        digest = hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
        st = W.response_status(oc["answer"][i], col["output_tokens"][i])
        pok, pwhy = W.prompt_complete(oc["user_message"][i], oc["all_messages"][i])
        sh, uh, roles = sys_user(oc["all_messages"][i])
        cv = oc["correct"][i]
        rows.append(dict(row=i, cfg=col["model_config"][i], problem_idx=col["problem_idx"][i],
                         idx=col["idx_answer"][i], problem=oc["problem"][i], gold=oc["gold_answer"][i], status=st,
                         prompt_ok=pok, prompt_why=pwhy, correct_is_null=cv is None,
                         correct_is_bool=isinstance(cv, bool), digest=digest))
        extra[i] = dict(side=inv[col["model_config"][i]], user_message=oc["user_message"][i], sys_hash=sh,
                        user_turn_hash=uh, roles=roles, answer_len=len(oc["answer"][i] or ""),
                        parsed_is_null=oc["parsed_answer"][i] is None, input_tokens=col["input_tokens"][i],
                        output_tokens=col["output_tokens"][i], cost=col["cost"][i], source=col["source"][i],
                        price_in=oc["input_cost_per_tokens"][i], price_out=oc["output_cost_per_tokens"][i])
    del oc
    t_read = round(time.time() - t, 2)
    log(f"A/B rows reduced to contract flags ({len(rows)} rows, {t_read} s, peak working set {C1.peak_rss_mb()[0]} MiB)")

    # ------------------------------------------------------------------ contract
    F, breaks, kept, slots = W.contract(rows, candidates, mapping)
    full = len(candidates) == W.TARGET_N and len(F) == W.TARGET_N
    log(f"contract: candidates {len(candidates)}, complete {len(F)}, breaks {len(breaks)}; full 30 = {full}")
    event("contract", n_candidates=len(candidates), n_F=len(F), full=full)

    # call-condition attributes (descriptive; nothing is dropped on these)
    def side_rows(s):
        return [i for i in ab_rows if extra[i]["side"] == s]

    attrs = {}
    for s in ("A", "B"):
        rs = side_rows(s)
        ots = [extra[i]["output_tokens"] for i in rs]
        its = [extra[i]["input_tokens"] for i in rs]
        attrs[s] = dict(
            raw_model_config=mapping[s], model_name=sorted(cfg_names[mapping[s]]), n_rows=len(rs),
            system_or_developer_turns=sorted({extra[i]["sys_hash"] for i in rs}),
            message_role_patterns=sorted({"/".join(extra[i]["roles"] or []) for i in rs}),
            output_tokens=dict(min=min((x for x in ots if x is not None), default=None),
                               max=max((x for x in ots if x is not None), default=None),
                               n_null=sum(x is None for x in ots), n_zero=sum(x == 0 for x in ots)),
            input_tokens=dict(min=min((x for x in its if x is not None), default=None),
                              max=max((x for x in its if x is not None), default=None),
                              n_null=sum(x is None for x in its), n_zero=sum(x == 0 for x in its)),
            nonempty_answer_with_zero_or_null_output_tokens=sum(
                1 for i in rs if extra[i]["answer_len"] and not extra[i]["output_tokens"]),
            saved_truncations=sum(1 for r in rows if extra[r["row"]]["side"] == s and r["status"] == W.ST_TRUNC),
            unknown_status=sum(1 for r in rows if extra[r["row"]]["side"] == s and r["status"] == W.ST_UNKNOWN),
            parsed_answer_null=sum(extra[i]["parsed_is_null"] for i in rs),
            prices_per_million=sorted({(extra[i]["price_in"], extra[i]["price_out"]) for i in rs}),
            total_cost_usd=round(sum(extra[i]["cost"] or 0 for i in rs), 6),
            sources=sorted({str(extra[i]["source"]) for i in rs}))
    attrs["A"]["output_tokens"]["over_official_max_tokens_32000"] = sum(
        1 for i in side_rows("A") if (extra[i]["output_tokens"] or 0) > 32000)
    same_user = {q: len({extra[kept[q][(s, k)]]["user_message"] for s in ("A", "B") for k in W.IDX}) == 1 for q in F}
    within = {s: sum(1 for q in F if len({extra[kept[q][(s, k)]]["user_message"] for k in W.IDX}) == 1)
              for s in ("A", "B")}
    attrs["prompts"] = dict(problems_with_identical_user_message_across_all_8_slots=sum(same_user.values()),
                            problems_with_identical_user_message_within_side=within, n_F=len(F))
    tok_by_idx = {s: {k: dict(n_zero_output=sum(1 for q in F if not extra[kept[q][(s, k)]]["output_tokens"]),
                              n_zero_input=sum(1 for q in F if not extra[kept[q][(s, k)]]["input_tokens"]))
                      for k in W.IDX} for s in ("A", "B")}
    attrs["token_bookkeeping_by_idx"] = tok_by_idx

    frame = dict(
        card="matharena-selection-replay", week=2, prereg=str(W.PREREG), stage="frame written before any correct value is used",
        written_local=C1.now_local(), data=data_gate,
        reader="pure-Python parquet reader code/pq_reader.py (week 1); pyarrow absent, nothing installed",
        n_rows_file=n, n_configs_file=len(cfg_names), config_match=dict(clear=clear, mapping=mapping, detail=cdetail),
        candidates=candidates, n_candidates=len(candidates), F=F, n_F=len(F), full_30_contract=full,
        breaks=breaks, kept={q: {f"{s}{k}": r for (s, k), r in v.items()} for q, v in kept.items()},
        call_condition_attributes=attrs,
        decided_before_correct=[
            "配置对应：原始 model_config 去掉固定前缀 configs/models/ 与后缀 .yaml 后等于官网标识（或以 /标识 结尾），"
            "或其行的 model_name 恰等于官网显示名；每侧恰好收集到一个原始值、两侧不同、各只带一个 model_name 才算明确，否则 NA",
            "题目按 problem_idx 识别；候选题 = 冻结副本全部行中出现的不同 problem_idx，须为 30",
            "每侧 idx 0..3 各恰一条（完全相同的重复行折叠为一条）；同键内容不同或出现 0..3 以外的 idx 即违约",
            "共同题文/gold：每题 8 条保留记录的 problem 与 gold_answer 各只有一个取值",
            "完整提示：每条 user_message 非空且 all_messages 为含 user 轮的 JSON 列表；A/B 提示是否逐字相同只作调用条件属性报告，不据此删题",
            "判分：correct 为已存布尔值（非空），gold_answer 非空",
            "响应状态：answer 非空即保留（token 记 0 也保留，usage 按原记账层级读）；answer 空且 output_tokens>0 为已保存截断，按原 correct 计；"
            "answer 空且 output_tokens 空/0 为未知响应状态，不冒充答错，构成违约",
            "任一题违约即整轮 NA：不删题、不换配置",
            "主量固定 p0 = idx0 筛选、idx1–3 复评；四槽循环（筛选槽 = idx j）只作敏感性"],
        timings=dict(read_s=t_read))
    W.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    slot_rows = []
    for side, q, k, row, st in sorted(slots, key=lambda s: (pidx_key(s[1]), s[0], -1 if s[2] is None else s[2])):
        e = extra.get(row, {})
        slot_rows.append([side, q, k, "" if row is None else row, st, e.get("input_tokens", ""),
                          e.get("output_tokens", ""), e.get("answer_len", ""), int(e.get("parsed_is_null", 0)),
                          int(same_user.get(q, False)) if row is not None else ""])
    C1.write_csv(W.FRAME_DIR / "slots.csv", ["side", "problem_idx", "idx", "parquet_row", "status", "input_tokens",
                                             "output_tokens", "answer_chars", "parsed_answer_null",
                                             "user_message_identical_across_AB"], slot_rows)
    os.chmod(W.FRAME_DIR / "slots.csv", stat.S_IREAD)
    frame["timings"]["stage1_total_s"] = round(time.time() - t0, 2)
    frame["peak_working_set_mib"] = C1.peak_rss_mb()[0]
    write_ro_json(W.FRAME_DIR / "frame.json", frame)
    event("frame_written", n_F=len(F), full=full, stage1_s=frame["timings"]["stage1_total_s"])
    log(f"frame written read-only ({frame['timings']['stage1_total_s']} s)")
    if not full:
        stop_na("30-problem contract not met", dict(n_candidates=len(candidates), n_F=len(F), breaks=breaks))


if __name__ == "__main__":
    try:
        main()
    except C1.StopGate as e:
        stop_na(f"resource: {e}", {})

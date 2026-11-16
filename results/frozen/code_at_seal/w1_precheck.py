"""Stage 1 (masked): entry / cost / execution gates, frozen manifest, seed manifest + collision check, exclusion and
role tables.  The values of `correct`, `parsed_answer` and the response text are never printed or written here:
they enter only as null-flags and inside a per-row digest used to tell identical duplicates from conflicts.

Run:  python -X utf8 -I w1_precheck.py
"""
import json
import os
import shutil
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1_common as C  # noqa: E402
import pq_reader as P  # noqa: E402

LOG = C.RESULTS / "run_log.txt"
TIMELINE = C.RESULTS / "timeline.jsonl"


def log(msg):
    line = f"[{C.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def event(name, **kw):
    with open(TIMELINE, "a", encoding="utf8") as f:
        f.write(json.dumps(dict(t_local=C.now_local(), event=name, **kw), ensure_ascii=False) + "\n")


def stop(gate, reason, extra=None):
    rec = dict(decision="STOP", gate=gate, reason=reason, time_local=C.now_local(), extra=extra or {})
    C.write_json(C.RESULTS / f"STOP_{gate}.json", rec)
    event("stop", gate=gate, reason=reason)
    log(f"STOP at {gate}: {reason}")
    sys.exit(3)


def problem_sort_key(q):
    s, i = q
    return (s, int(i) if str(i).isdigit() else 10 ** 9, str(i))


def sys_user_hashes(all_messages):
    """Hashes of the system/developer and user turns only; assistant turns are never looked at."""
    try:
        msgs = json.loads(all_messages)
    except (TypeError, ValueError):
        return dict(parse="unparseable", system_sha256=None, user_sha256=None, n_system=None, n_user=None)
    if not isinstance(msgs, list):
        return dict(parse=f"not_a_list:{type(msgs).__name__}", system_sha256=None, user_sha256=None,
                    n_system=None, n_user=None)
    sysm = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    usr = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    dump = lambda xs: json.dumps(xs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # noqa: E731
    return dict(parse="ok", n_messages=len(msgs), n_system=len(sysm), n_user=len(usr),
                system_sha256=C.sha256_text(dump(sysm)) if sysm else "NONE",
                user_sha256=C.sha256_text(dump(usr)) if usr else "NONE")


def main():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    if (C.FROZEN / "manifest_frozen.json").exists():
        log("sealed manifest already exists; stage 1 is not re-run")
        sys.exit(2)
    t_start = time.time()
    limits = C.apply_limits()
    log(f"stage 1 start; pid {os.getpid()}; limits {limits}")
    event("stage1_start", pid=os.getpid(), limits=limits)
    timings = {}

    # ------------------------------------------------------------------ entry gate (binding)
    t = time.time()
    tree = C.read_json(C.META_WEB / "hf_tree_at_rev.json")
    revinfo = C.read_json(C.META_WEB / "hf_revision_info.json")
    commits = C.read_json(C.META_WEB / "hf_commits_main.json")
    entry = next((x for x in tree if x.get("path") == C.PARQUET_PATH_IN_REPO), None)
    parquet_files = [x["path"] for x in tree if x.get("path", "").endswith(".parquet")]
    local_sha = C.sha256_file(C.PARQUET)
    local_size = os.path.getsize(C.PARQUET)
    card = (C.DATA_DIR / "README.md").read_text(encoding="utf8")
    commit = next((c for c in commits if c.get("id") == C.REV), None)
    entry_gate = dict(
        revision=C.REV, revision_api_sha=revinfo.get("sha"), revision_matches=revinfo.get("sha") == C.REV,
        parquet_files_at_revision=parquet_files, single_parquet=parquet_files == [C.PARQUET_PATH_IN_REPO],
        lfs_oid_at_revision=(entry or {}).get("lfs", {}).get("oid"), lfs_oid_registered=C.LFS_SHA256,
        local_sha256=local_sha, local_size=local_size, api_size=(entry or {}).get("size"),
        license_in_card=("license: " + C.LICENSE) in card, url=C.URL,
        snapshot_commit_date_utc=(commit or {}).get("date"), snapshot_commit_title=(commit or {}).get("title"),
        revision_is_main_head_on_fetch=bool(commits) and commits[0].get("id") == C.REV)
    entry_gate["pass"] = bool(entry_gate["revision_matches"] and entry_gate["single_parquet"]
                              and entry_gate["lfs_oid_at_revision"] == C.LFS_SHA256 and local_sha == C.LFS_SHA256
                              and local_size == entry_gate["api_size"] and entry_gate["license_in_card"])
    timings["entry_gate_s"] = round(time.time() - t, 2)
    log(f"entry gate: pass={entry_gate['pass']} (local sha256 {'==' if local_sha == C.LFS_SHA256 else '!='} "
        f"registered LFS oid; size {local_size})")
    event("entry_gate", passed=entry_gate["pass"])
    if not entry_gate["pass"]:
        C.write_json(C.RESULTS / "gates_failed_entry.json", entry_gate)
        stop("entry", "binding check failed", entry_gate)

    # ------------------------------------------------------------------ cost gate
    dl_bytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(C.DOWNLOADS) for f in fs)
    mb0 = C.wait_ram(log, "metadata read")
    cost_gate = dict(
        note="草稿未写成本门的数值阈值；按原文：保留原生生成上限、不补 retry、不重查询，故新增 API 调用为 0；"
             "资源约束（下载 <= 1 GB；启动重活前可用内存 >= 1500 MB；单进程、1 线程、BelowNormal、<= 2 GiB）。",
        new_api_calls=0, downloads_bytes=dl_bytes, downloads_limit_bytes=1024 ** 3,
        available_mb_at_start=mb0, limits=limits)
    cost_gate["pass"] = dl_bytes <= 1024 ** 3 and mb0 is not None and mb0 >= C.MIN_FREE_MB
    log(f"cost gate: pass={cost_gate['pass']} (downloads {dl_bytes / 2 ** 20:.1f} MiB, new calls 0)")
    event("cost_gate", passed=cost_gate["pass"])
    if not cost_gate["pass"]:
        stop("cost", "resource limits not met", cost_gate)

    # ------------------------------------------------------------------ read metadata (masked)
    t = time.time()
    pf = P.ParquetFile(str(C.PARQUET))
    desc = pf.describe()
    meta_cols = ["model_config", "model_name", "source", "problem_idx", "idx_answer", "problem", "gold_answer",
                 "user_message", "input_tokens", "output_tokens", "cost", "input_cost_per_tokens",
                 "output_cost_per_tokens"]
    col = {c: pf.read_column(c) for c in meta_cols}
    n = pf.num_rows
    norm = [C.normalize_config(x) if x is not None else None for x in col["model_config"]]
    ab_rows = {i for i in range(n) if norm[i] in (C.CFG_A, C.CFG_B)}
    log(f"read {len(meta_cols)} metadata columns for {n} rows; {len(ab_rows)} rows match the locked A/B configs")
    # reader sanity check on non-outcome numeric columns: cost == (in*pin + out*pout)/1e6
    chk = [i for i in range(n) if None not in (col["cost"][i], col["input_tokens"][i], col["output_tokens"][i],
                                               col["input_cost_per_tokens"][i], col["output_cost_per_tokens"][i])]
    match = sum(abs(col["cost"][i] - (col["input_tokens"][i] * col["input_cost_per_tokens"][i]
                                      + col["output_tokens"][i] * col["output_cost_per_tokens"][i]) / 1e6)
                <= 1e-6 * max(1.0, abs(col["cost"][i])) for i in chk)
    reader_check = dict(rows_checked=len(chk), cost_identity_matches=match,
                        idx_answer_range=[min(v for v in col["idx_answer"] if v is not None),
                                          max(v for v in col["idx_answer"] if v is not None)])
    log(f"reader check: cost identity holds on {match}/{len(chk)} rows")

    # masked columns: A/B rows only, reduced to status flags and a digest
    mb = C.wait_ram(log, "masked columns")
    masked = {}
    for c in ("answer", "all_messages", "parsed_answer", "correct"):
        masked[c] = pf.read_column(c, rows=ab_rows)
    rows = []
    prompt_info = {}
    for i in sorted(ab_rows):
        rec = {c: col[c][i] for c in meta_cols}
        rec.update({c: masked[c][i] for c in masked})
        digest = C.sha256_text(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str))
        ast = C.answer_status(masked["answer"][i], col["output_tokens"][i])
        su = sys_user_hashes(masked["all_messages"][i])
        prompt_info[i] = dict(user_message_sha256=C.sha256_text(col["user_message"][i] or ""),
                              problem_sha256=C.sha256_text(col["problem"][i] or ""), **su)
        rows.append(dict(row=i, cfg=norm[i], source=col["source"][i], problem_idx=col["problem_idx"][i],
                         idx=col["idx_answer"][i], problem_sha=prompt_info[i]["problem_sha256"],
                         gold=col["gold_answer"][i], answer_status=ast,
                         correct_is_null=masked["correct"][i] is None,
                         parsed_is_null=masked["parsed_answer"][i] is None, row_digest=digest,
                         answer_len=len(masked["answer"][i] or ""), output_tokens=col["output_tokens"][i],
                         input_tokens=col["input_tokens"][i]))
    del masked
    peak = C.peak_rss_mb()
    timings["read_and_mask_s"] = round(time.time() - t, 2)
    log(f"masked reduction done ({timings['read_and_mask_s']} s; peak working set {peak[0]} MiB)")

    # ------------------------------------------------------------------ execution gate
    raw_paths = {}
    for i in ab_rows:
        raw_paths.setdefault(norm[i], set()).add(col["model_config"][i])
    names = {}
    for i in ab_rows:
        names.setdefault(norm[i], set()).add(col["model_name"][i])
    candidates = sorted({(col["source"][i], col["problem_idx"][i]) for i in range(n)}, key=problem_sort_key)
    F, exclusions, kept, slot_status = C.precheck(rows, candidates)
    by_row = {r["row"]: r for r in rows}
    out_caps = {"A": 8000, "B": 64000}
    tok = {}
    for side, cfg in (("A", C.CFG_A), ("B", C.CFG_B)):
        ots = [r["output_tokens"] for r in rows if r["cfg"] == cfg and r["output_tokens"] is not None]
        tok[side] = dict(n=len(ots), min=min(ots) if ots else None, max=max(ots) if ots else None,
                         over_native_max_tokens=sum(o > out_caps[side] for o in ots),
                         native_max_tokens=out_caps[side],
                         empty_visible_answer=sum(1 for r in rows if r["cfg"] == cfg
                                                  and r["answer_status"] == C.ANSWER_EMPTY_TRUNC),
                         no_valid_response=sum(1 for r in rows if r["cfg"] == cfg
                                               and r["answer_status"] == C.ANSWER_FAIL),
                         parsed_answer_null=sum(1 for r in rows if r["cfg"] == cfg and r["parsed_is_null"]),
                         correct_null=sum(1 for r in rows if r["cfg"] == cfg and r["correct_is_null"]))
    sys_hashes = {side: sorted({prompt_info[r["row"]]["system_sha256"] or "unparsed" for r in rows if r["cfg"] == cfg})
                  for side, cfg in (("A", C.CFG_A), ("B", C.CFG_B))}
    user_tpl = {side: len({prompt_info[r["row"]]["user_message_sha256"] for r in rows if r["cfg"] == cfg})
                for side, cfg in (("A", C.CFG_A), ("B", C.CFG_B))}
    same_user_msg = sum(1 for q in F if len({prompt_info[kept[q][(s, k)]]["user_message_sha256"]
                                            for s in ("A", "B") for k in C.IDX}) == 1)
    exec_gate = dict(
        locked_A=C.CFG_A, locked_B=C.CFG_B,
        raw_path_values={k: sorted(v) for k, v in raw_paths.items()},
        model_names={k: sorted(v) for k, v in names.items()},
        normalization="only prefix configs/models/ and suffix .yaml may be removed; no fuzzy matching",
        A_present=C.CFG_A in raw_paths, B_present=C.CFG_B in raw_paths,
        n_candidates=len(candidates), target_n=C.TARGET_N, n_complete_F=len(F),
        n_excluded=len(exclusions), full_frame=len(F) == C.TARGET_N and len(candidates) == C.TARGET_N,
        rule="门：两锁定配置的路径值在该 revision 中按允许的规范化精确出现，并逐题核对 A/B 的 idx=0..3 唯一齐全；"
             "不齐全的题按草稿整题排除并报告覆盖分母；完整框 F 为空则停止。")
    exec_gate["pass"] = exec_gate["A_present"] and exec_gate["B_present"] and len(F) > 0
    log(f"execution gate: pass={exec_gate['pass']}; candidates {len(candidates)}, complete F {len(F)}, "
        f"excluded {len(exclusions)}")
    event("execution_gate", passed=exec_gate["pass"], n_F=len(F), n_candidates=len(candidates))
    if not exec_gate["pass"]:
        C.write_json(C.RESULTS / "gates_failed_execution.json", exec_gate)
        stop("execution", "locked config absent or complete frame empty", exec_gate)

    # ------------------------------------------------------------------ seeds: expand and check collisions
    t = time.time()
    man = C.seed_manifest(candidates)
    digests = [e["sha256"] for e in man]
    s63 = [e["sha63"] for e in man]
    ube = [e["u64be"] for e in man]
    ties = sum(1 for q in candidates for a in C.IDX for b in C.IDX
               if a < b and C.ns_digest(q[0], q[1], a) == C.ns_digest(q[0], q[1], b))
    C.wait_ram(log, "seed literal scan")
    wanted_dec = set(s63) | set(ube)
    scan = C.scan_tree(C.CARDS_ROOT, [C.CARD], wanted_dec, set(digests))
    ranges = C.registered_ranges(C.CARDS_ROOT)
    range_hits = [dict(range=nm, lo=lo, hi=hi, value=v) for nm, lo, hi in ranges for v in wanted_dec if lo <= v <= hi]
    collisions = []
    for v, files in scan["hits_dec"].items():
        collisions.append(dict(kind="literal_dec", value=v, files=files[:20]))
    for v, files in scan["hits_hex"].items():
        collisions.append(dict(kind="literal_hex64", value=v, files=files[:20]))
    for f in scan["ns_token_files"]:
        collisions.append(dict(kind="namespace_token_line160/", file=f))
    collisions += [dict(kind="registered_range", **h) for h in range_hits]
    seed_check = dict(
        namespace=C.NS, rule='SHA256("line160/v1|HFrevision|source|problem_idx|idx"), digest byte order, ties idx asc',
        compared_forms="full 256-bit digest vs 64-hex literals; sha63 (project convention: first 8 bytes LE, top bit "
                       "cleared) and u64 big-endian projections vs 10-20 digit literals and registered ranges; "
                       "namespace token 'line160/' anywhere",
        n_entries=len(man), distinct_digests=len(set(digests)) == len(digests),
        distinct_sha63=len(set(s63)) == len(s63), distinct_u64be=len(set(ube)) == len(ube),
        within_problem_digest_ties=ties, scan_root=str(C.CARDS_ROOT), skipped_self=[str(C.CARD)],
        reader="CreateFileW share read/write/delete; lock files and *lock* folders skipped; text files <= 20 MB",
        n_registered_ranges=len(ranges), registered_range_sources=sorted({r[0].split(':')[0] for r in ranges}),
        scan={k: v for k, v in scan.items() if k not in ("hits_dec", "hits_hex")},
        collisions=collisions)
    seed_check["collision_free"] = (seed_check["distinct_digests"] and seed_check["distinct_sha63"]
                                    and seed_check["distinct_u64be"] and not collisions)
    timings["seed_check_s"] = round(time.time() - t, 2)
    log(f"seed check: {len(man)} entries, collision_free={seed_check['collision_free']}, "
        f"{scan['n_text_files']} text files scanned ({scan['n_literal_occurrences']} literal occurrences), "
        f"{len(ranges)} registered ranges, {timings['seed_check_s']} s")
    event("seed_check", collision_free=seed_check["collision_free"])
    if not seed_check["collision_free"]:
        C.write_json(C.FROZEN / "seed_manifest.json", dict(entries=man), frozen=True)
        C.write_json(C.FROZEN / "seed_check.json", seed_check, frozen=True)
        stop("seed_collision", "namespace-hash seed collides with a registered seed (no reseeding)", seed_check)

    # ------------------------------------------------------------------ roles, manifest, seal
    order = {q: C.role_order(q) for q in candidates}
    roles_rows = []
    for q in candidates:
        o = order[q]
        inF = q in kept
        rr = [kept[q][("A", k)] if inF else "" for k in o]
        rb = [kept[q][("B", k)] if inF else "" for k in o]
        roles_rows.append([q[0], q[1], int(inF)] + o + rr + rb)
    excl_rows = [[q[0], q[1], " | ".join(rs)] for q, rs in sorted(exclusions.items(), key=lambda kv: problem_sort_key(kv[0]))]
    slot_rows = []
    for side, q, k, row, st in sorted(slot_status, key=lambda s: (problem_sort_key(s[1]), s[0], s[2] if s[2] is not None else -1)):
        pi = prompt_info.get(row, {}) if row is not None else {}
        r = by_row.get(row, {})
        slot_rows.append([side, q[0], q[1], k, "" if row is None else row, st, "UNKNOWN", "UNKNOWN",
                          pi.get("problem_sha256", ""), pi.get("user_message_sha256", ""),
                          pi.get("system_sha256", ""), r.get("output_tokens", ""), r.get("answer_status", "")])
    gh_head = C.read_json(C.META_WEB / "gh_main_head.json")
    cfg_commits = {}
    for m in ("claude-35", "claude-37"):
        cc = C.read_json(C.META_WEB / f"gh_commits_{m}.json")
        cfg_commits[m] = [dict(sha=x["sha"], date=x["commit"]["committer"]["date"],
                               title=x["commit"]["message"].splitlines()[0]) for x in cc]
    manifest = dict(
        line=160, card="matharena-selection-replay", prereg=str(C.PREREG), stage="sealed before unmasking correct",
        sealed_local=C.now_local(),
        data=dict(entry_gate=entry_gate, parquet_describe=dict(num_rows=desc["num_rows"], created_by=desc["created_by"],
                                                                 n_row_groups=len(desc["row_groups"]),
                                                                 columns=desc["columns"]),
                  reader="pure-Python parquet reader (code/pq_reader.py); pyarrow absent, no installs",
                  reader_check=reader_check),
        configs=dict(
            A=dict(locked=C.CFG_A, raw_path_values=exec_gate["raw_path_values"].get(C.CFG_A),
                   model_names=exec_gate["model_names"].get(C.CFG_A),
                   official_page="https://matharena.ai/models/anthropic_claude_35 (fetched 2026-10-23): endpoint "
                                 "claude-3-5-sonnet-20241022, Max Tokens 8000, no temperature listed",
                   github_main_config_text=(C.META_WEB / "gh_main_claude-35.yaml.txt").read_text(encoding="utf8"),
                   github_config_commits=cfg_commits["claude-35"]),
            B=dict(locked=C.CFG_B, raw_path_values=exec_gate["raw_path_values"].get(C.CFG_B),
                   model_names=exec_gate["model_names"].get(C.CFG_B),
                   official_page="https://matharena.ai/models/anthropic_claude_37 (fetched 2026-10-23): endpoint "
                                 "claude-3-7-sonnet-20250219, Max Tokens 64000, Temperature 1, thinking budget 32000",
                   github_main_config_text=(C.META_WEB / "gh_main_claude-37.yaml.txt").read_text(encoding="utf8"),
                   github_config_commits=cfg_commits["claude-37"]),
            github_head=dict(sha=gh_head.get("sha"), date=gh_head.get("commit", {}).get("committer", {}).get("date")),
            historical_config_hash="UNKNOWN", call_times="UNKNOWN", resolved_model_receipts="UNKNOWN",
            original_seed_retry="UNKNOWN", parser_commit="UNKNOWN",
            note="GitHub configs are the current main versions; the files changed after generation (commit list), "
                 "so the config at generation time is not verified."),
        prompts=dict(system_prompt_sha256_values=sys_hashes, n_distinct_user_message_hashes=user_tpl,
                     problems_in_F_with_identical_user_message_across_A_B=same_user_msg),
        tokens=tok, candidates=[list(q) for q in candidates], F=[list(q) for q in F],
        exclusions={f"{q[0]}|{q[1]}": rs for q, rs in exclusions.items()},
        time_evidence=dict(snapshot_commit_date_utc=entry_gate["snapshot_commit_date_utc"],
                           per_row_generation_time="UNKNOWN (not in the data; not invented)"),
        retry_replacement_chain="UNKNOWN for every slot",
        decided_before_unmask=[
            "候选题 = 该 revision 全部行中出现的不同 (source, problem_idx)",
            "idx=0..3 各恰一条（完全相同的重复行折叠）；同键内容不同或出现 0..3 之外的 idx 视为冲突，整题两边排除",
            "明确技术失败/无有效响应 = answer 为空或仅空白 且 output_tokens 为空或 0；answer 为空但 output_tokens>0 视为已保存截断，保留并按锁定 correct",
            "判分字段缺失 = correct 为空；parsed_answer 为空但 correct 非空视为已判分",
            "配对：每题所有保留 A/B 行的题文 SHA256 与 gold_answer 各只有一个值",
            "主量为轮换 0（p0 筛选），轮换 1..3 只作角色协议敏感性"],
        timings=timings, code_snapshot="results/frozen/code_at_seal/")
    gates = dict(entry=entry_gate["pass"], cost=cost_gate["pass"], execution=exec_gate["pass"],
                 seed_collision_free=seed_check["collision_free"], cost_gate=cost_gate, execution_gate=exec_gate)

    C.FROZEN.mkdir(parents=True, exist_ok=True)
    C.write_json(C.FROZEN / "seed_manifest.json", dict(namespace=C.NS, revision=C.REV, entries=man), frozen=True)
    C.write_json(C.FROZEN / "seed_check.json", seed_check, frozen=True)
    C.write_csv(C.FROZEN / "exclusions.csv", ["source", "problem_idx", "reasons"], excl_rows, frozen=True)
    C.write_csv(C.FROZEN / "roles.csv", ["source", "problem_idx", "in_F", "p0", "p1", "p2", "p3",
                                         "A_row_p0", "A_row_p1", "A_row_p2", "A_row_p3",
                                         "B_row_p0", "B_row_p1", "B_row_p2", "B_row_p3"], roles_rows, frozen=True)
    C.write_csv(C.FROZEN / "slots.csv", ["side", "source", "problem_idx", "idx", "parquet_row", "status",
                                         "first_request", "replacement_chain", "problem_sha256",
                                         "user_message_sha256", "system_sha256", "output_tokens", "answer_status"],
                slot_rows, frozen=True)
    C.write_json(C.FROZEN / "gates.json", gates, frozen=True)
    snap = C.FROZEN / "code_at_seal"
    snap.mkdir(parents=True, exist_ok=True)
    for fn in ("pq_reader.py", "w1_common.py", "w1_precheck.py", "w1_unseal.py", "w1_report.py", "test_w1.py"):
        src = C.HERE / fn
        if src.exists():
            dst = snap / fn
            shutil.copy2(src, dst)
            os.chmod(dst, stat.S_IREAD)
    timings["stage1_total_s"] = round(time.time() - t_start, 2)
    manifest["timings"] = timings
    manifest["peak_working_set_mib"] = C.peak_rss_mb()[0]
    C.write_json(C.FROZEN / "manifest_frozen.json", manifest, frozen=True)
    event("sealed", files=sorted(p.name for p in C.FROZEN.iterdir()))
    log(f"sealed manifest, exclusions, roles, slots, seeds and gates ({timings['stage1_total_s']} s total)")


if __name__ == "__main__":
    try:
        main()
    except C.StopGate as e:
        stop("resource", str(e))

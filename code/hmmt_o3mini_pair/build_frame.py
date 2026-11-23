"""HMMT o3-mini run, step 1: config correspondence and the 30-problem completeness check, written to results/hmmt_o3mini_pair/frame/ before
any `correct` value is used.  `correct` is decoded here only to a null flag and a boolean-type flag.

Run:  python -X utf8 -I -B build_frame.py
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmmt_pair_rules as rules  # noqa: E402

helpers, P = rules.helpers, rules.P
LOG = rules.RESULTS / "run_log.txt"


def log(msg):
    line = f"[{helpers.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def stop_na(reason, extra):
    rec = dict(decision="NA", step="extract", reason=reason, time_local=helpers.now_local(), extra=extra)
    helpers.write_json(rules.RESULTS / "STOP_NA.json", rec)
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
    rules.RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("step 1 start")

    # ------------------------------------------------------------------ transfer size and data revision
    transferred = sum(os.path.getsize(os.path.join(dp, f)) for d in (rules.DATA_DIR, rules.META)
                      for dp, _, fs in os.walk(d) for f in fs)
    info = helpers.read_json(rules.META / "hf_dataset_info.json")
    commits = helpers.read_json(rules.META / "hf_commits_main.json")
    data_gate = dict(repo=rules.REPO, revision=rules.REV, revision_is_main_head_on_fetch=info.get("sha") == rules.REV,
                     revision_commit_date_utc=(commits[0] if commits and commits[0].get("id") == rules.REV else {}).get("date"),
                     siblings=[s.get("rfilename") for s in info.get("siblings", [])],
                     card_download_size=info.get("cardData", {}).get("dataset_info", {}).get("download_size"),
                     card_num_examples=info.get("cardData", {}).get("dataset_info", {}).get("splits", [{}])[0].get("num_examples"),
                     license_in_card=info.get("cardData", {}).get("license"), url=rules.URL,
                     parquet_bytes=os.path.getsize(rules.PARQUET),
                     bytes_transferred_total=transferred, transfer_cap_bytes=rules.TRANSFER_CAP, retries=0,
                     new_api_calls=0)
    log(f"data: revision {rules.REV[:7]} (main head on fetch: {data_gate['revision_is_main_head_on_fetch']}), "
        f"{data_gate['parquet_bytes']} bytes, transferred {transferred} bytes (cap {rules.TRANSFER_CAP})")
    if transferred > rules.TRANSFER_CAP:
        stop_na("transfer cap exceeded", data_gate)

    # ------------------------------------------------------------------ metadata (all rows)
    t = time.time()
    pf = P.ParquetFile(str(rules.PARQUET))
    meta_cols = ["model_config", "model_name", "problem_idx", "idx_answer", "source", "input_tokens",
                 "output_tokens", "cost"]
    col = {c: pf.read_column(c) for c in meta_cols}
    n = pf.num_rows
    cfg_names = {}
    for i in range(n):
        cfg_names.setdefault(col["model_config"][i], set()).add(col["model_name"][i])
    mapping, cdetail, clear = rules.match_configs(cfg_names)
    log(f"read {len(meta_cols)} metadata columns for {n} rows ({len(cfg_names)} configs); "
        f"config correspondence clear={clear}: {mapping}")
    if not clear:
        stop_na("config correspondence not clear", dict(detail=cdetail, data=data_gate))
    candidates = sorted({col["problem_idx"][i] for i in range(n)}, key=pidx_key)
    ab_rows = {i for i in range(n) if col["model_config"][i] in mapping.values()}

    # ------------------------------------------------------------------ A/B rows: completeness fields only
    other_cols = ["problem", "gold_answer", "user_message", "all_messages", "answer", "parsed_answer", "correct",
                  "input_cost_per_tokens", "output_cost_per_tokens"]
    oc = {c: pf.read_column(c, rows=ab_rows) for c in other_cols}
    inv = {raw: side for side, raw in mapping.items()}
    rows, extra = [], {}
    for i in sorted(ab_rows):
        rec = {c: col[c][i] for c in meta_cols}
        rec.update({c: oc[c][i] for c in other_cols})
        digest = hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
        st = rules.response_status(oc["answer"][i], col["output_tokens"][i])
        pok, pwhy = rules.prompt_complete(oc["user_message"][i], oc["all_messages"][i])
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
    log(f"A/B rows reduced to contract flags ({len(rows)} rows, {t_read} s)")

    # ------------------------------------------------------------------ completeness check
    F, breaks, kept, slots = rules.contract(rows, candidates, mapping)
    full = len(candidates) == rules.TARGET_N and len(F) == rules.TARGET_N
    log(f"contract: candidates {len(candidates)}, complete {len(F)}, breaks {len(breaks)}; full 30 = {full}")

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
            saved_truncations=sum(1 for r in rows if extra[r["row"]]["side"] == s and r["status"] == rules.ST_TRUNC),
            unknown_status=sum(1 for r in rows if extra[r["row"]]["side"] == s and r["status"] == rules.ST_UNKNOWN),
            parsed_answer_null=sum(extra[i]["parsed_is_null"] for i in rs),
            prices_per_million=sorted({(extra[i]["price_in"], extra[i]["price_out"]) for i in rs}),
            total_cost_usd=round(sum(extra[i]["cost"] or 0 for i in rs), 6),
            sources=sorted({str(extra[i]["source"]) for i in rs}))
    attrs["A"]["output_tokens"]["over_official_max_tokens_32000"] = sum(
        1 for i in side_rows("A") if (extra[i]["output_tokens"] or 0) > 32000)
    same_user = {q: len({extra[kept[q][(s, k)]]["user_message"] for s in ("A", "B") for k in rules.IDX}) == 1 for q in F}
    within = {s: sum(1 for q in F if len({extra[kept[q][(s, k)]]["user_message"] for k in rules.IDX}) == 1)
              for s in ("A", "B")}
    attrs["prompts"] = dict(problems_with_identical_user_message_across_all_8_slots=sum(same_user.values()),
                            problems_with_identical_user_message_within_side=within, n_F=len(F))
    tok_by_idx = {s: {k: dict(n_zero_output=sum(1 for q in F if not extra[kept[q][(s, k)]]["output_tokens"]),
                              n_zero_input=sum(1 for q in F if not extra[kept[q][(s, k)]]["input_tokens"]))
                      for k in rules.IDX} for s in ("A", "B")}
    attrs["token_bookkeeping_by_idx"] = tok_by_idx

    frame = dict(
        step="frame written before any correct value is used",
        written_local=helpers.now_local(), data=data_gate,
        reader="pure-Python parquet reader code/pq_reader.py (shared with the Claude-pair run); pyarrow absent, nothing installed",
        n_rows_file=n, n_configs_file=len(cfg_names), config_match=dict(clear=clear, mapping=mapping, detail=cdetail),
        candidates=candidates, n_candidates=len(candidates), F=F, n_F=len(F), full_30_contract=full,
        breaks=breaks, kept={q: {f"{s}{k}": r for (s, k), r in v.items()} for q, v in kept.items()},
        call_condition_attributes=attrs,
        decided_before_correct=[
            "Config correspondence: a raw model_config, with only the fixed prefix configs/models/ and suffix .yaml removed, equals the official identifier (or ends with /identifier), "
            "or its rows carry model_name exactly equal to the official display name; clear only if each side collects exactly one raw value, the two differ, and each carries exactly one model_name; otherwise NA",
            "Problems are keyed by problem_idx; candidates = the distinct problem_idx in all rows of the pinned copy; must be 30",
            "Each side has each idx 0..3 exactly once (fully identical duplicate rows collapse into one); a duplicate key with differing content or an idx outside 0..3 breaks the contract",
            "Common problem text/gold: the problem and gold_answer of the 8 kept records of a problem each take one value",
            "Complete prompt: every user_message non-empty and all_messages a JSON list with a user turn; whether the A/B prompts are verbatim identical is reported as a call-condition attribute only, no problem is dropped on it",
            "Grading: correct is a stored boolean (non-null) and gold_answer is non-empty",
            "Response status: a non-empty answer is kept (also with 0 tokens; usage read at its own bookkeeping level); an empty answer with output_tokens > 0 is a saved truncation, scored by the stored correct; "
            "an empty answer with output_tokens null/0 is an unknown response status, not counted as wrong, and breaks the contract",
            "A break in any problem makes the whole round NA: no problem is deleted, no config is swapped",
            "Primary quantity: fixed p0 = idx0 screening, idx1-3 re-scoring; the four-slot cycle (screening slot = idx j) is a sensitivity analysis only"],
        timings=dict(read_s=t_read))
    rules.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    slot_rows = []
    for side, q, k, row, st in sorted(slots, key=lambda s: (pidx_key(s[1]), s[0], -1 if s[2] is None else s[2])):
        e = extra.get(row, {})
        slot_rows.append([side, q, k, "" if row is None else row, st, e.get("input_tokens", ""),
                          e.get("output_tokens", ""), e.get("answer_len", ""), int(e.get("parsed_is_null", 0)),
                          int(same_user.get(q, False)) if row is not None else ""])
    helpers.write_csv(rules.FRAME_DIR / "slots.csv", ["side", "problem_idx", "idx", "parquet_row", "status", "input_tokens",
                                             "output_tokens", "answer_chars", "parsed_answer_null",
                                             "user_message_identical_across_AB"], slot_rows)
    frame["timings"]["step1_total_s"] = round(time.time() - t0, 2)
    helpers.write_json(rules.FRAME_DIR / "frame.json", frame)
    log(f"frame written ({frame['timings']['step1_total_s']} s)")
    if not full:
        stop_na("30-problem contract not met", dict(n_candidates=len(candidates), n_F=len(F), breaks=breaks))


if __name__ == "__main__":
    main()

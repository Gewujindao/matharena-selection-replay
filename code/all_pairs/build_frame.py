"""All-pairs run, step 1: from metadata only, enumerate every config of the four competitions, assign tiers / base endpoints
by the tier rule, list the eligible adjacent pairs and every ineligible config with its reason, run the per-cell
completeness check and write it all to results/all_pairs/frame/ before any `correct` value is decoded.  For `correct` only the
parquet definition levels (null mask) are read; its value bytes are never decoded here.

Run:  python -X utf8 -I -B build_frame.py
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all_pairs_rules as rules  # noqa: E402

helpers, P = rules.helpers, rules.P
LOG = rules.RESULTS / "run_log.txt"


def log(msg):
    line = f"[{helpers.now_local()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def h16(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()[:16] if s is not None else "NULL"   # in-run equality only


DECIDED = [
    "Tier ladder minimal < low < medium < high < xhigh; a trailing display-name parenthesis counts as a tier tag only if its content is exactly a ladder word (case-insensitive); other trailing parentheses (Think, 05-06, preview, ...) do not count and stay in the base name",
    "The path suffix is only the part after the last \"--\" in the last path component that is exactly a ladder word; a single hyphen (-low, -high) is not a --<tier> suffix; only the fixed prefix configs/models/ and suffix .yaml are removed from the path",
    "The tier is taken from the display-name tag, else from the path suffix; both present and different -> config NA; a config value carrying more than one model_name -> config NA",
    "Base endpoint = (display name without the trailing tier parenthesis and the whitespace before it, path without the --<tier> suffix); same endpoint only if both are equal; more than one config on one tier of an endpoint -> that endpoint is NA in that competition",
    "Eligible pair: same endpoint, two tiers adjacent on the ladder, A = the lower tier; non-adjacent tiers are not joined",
    "Problem frame = the distinct problem_idx in all rows of the competition's pinned copy (the source column is empty everywhere and is not part of the key)",
    "Per cell and problem: A and B each have idx 0..3 exactly once (rows identical in every column except correct collapse; whether correct is also identical is checked in step 2); differing duplicates or an idx outside 0..3 break the contract; any missing record breaks the contract",
    "Problem text/gold: the problem and gold_answer of the 8 kept records of a problem each take one value; gold non-empty",
    "Full prompt = (user_message, canonical JSON with sorted keys of every non-assistant entry of all_messages in original order); user_message must be non-empty and all_messages a JSON list with a user turn; the eight slots must be byte-identical",
    "Stored correct present: only the parquet definition levels (null mask) are read, values are not decoded; physical type must be BOOLEAN",
    "Response status (as in the HMMT o3-mini run): a non-empty answer is kept (also with 0 tokens; usage read at its own bookkeeping level); an empty answer with output_tokens > 0 is a saved truncation, scored by the stored correct; an empty answer with output_tokens null/0 is an unknown response status -> contract break",
    "A break in any problem -> that cell is NA; no intersection of the two sides, no problem deleted, no config swapped; NA cells stay in the coverage denominator",
    "Primary slot fixed: idx0 screening, idx1-3 re-scoring; the four-slot cycle is a sensitivity analysis only",
    "Seen cell: HMMT o3-mini run, HMMT openai/o3-mini--medium -> openai/o3-mini; the Claude-pair run (AIME claude-35 -> claude-37) is not a tier pair and cannot appear among the eligible cells",
    "Pre-specified descriptive stratum: the cached accuracy of A (lower tier) over all four slots of the full problem frame lies in the closed interval [30%, 70%]; descriptive only, not used for selection",
]


def main():
    rules.RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("step 1 start")

    # ------------------------------------------------------------------ data check (revisions, transfer)
    data = {}
    transferred = 0
    for comp in rules.COMP_ORDER:
        spec = rules.COMPS[comp]
        pq = spec["parquet"]
        info = {}
        if spec["new_download"]:
            name = spec["repo"].split("/")[1]
            info = helpers.read_json(rules.META / f"hf_{name}_revision_info.json")
            tree = helpers.read_json(rules.META / f"hf_{name}_tree_data.json")
            size_api = tree[0]["size"] if tree else None
            transferred += os.path.getsize(pq)
        else:
            info = helpers.read_json(rules.DL / "meta_web_hmmt_o3mini_pair" / "hf_dataset_info.json")
            size_api = None
        data[comp] = dict(repo=spec["repo"], revision=spec["rev"], revision_found=info.get("sha") == spec["rev"],
                          revision_last_modified=info.get("lastModified"),
                          license=(info.get("cardData") or {}).get("license"), parquet=str(pq),
                          parquet_bytes=os.path.getsize(pq), size_listed_at_revision=size_api,
                          source="new download (all-pairs run)" if spec["new_download"] else "reused HMMT o3-mini run copy")
    meta_bytes = sum(os.path.getsize(rules.META / f) for f in os.listdir(rules.META) if f.endswith(".json"))
    transferred += meta_bytes
    gate = dict(bytes_transferred=transferred, metadata_json_bytes=meta_bytes, transfer_cap=rules.TRANSFER_CAP,
                retries=0, new_api_calls_to_models=0)
    log(f"data gate: {json.dumps({c: (d['revision'][:7], d['revision_found'], d['parquet_bytes']) for c, d in data.items()})}; transferred {transferred}")
    if transferred > rules.TRANSFER_CAP:
        raise rules.StopNA("transfer cap exceeded")

    comps_out, cells_out, cfg_rows, cell_rows, slot_rows, endpoint_rows, notes = {}, [], [], [], [], [], []
    for comp in rules.COMP_ORDER:
        spec = rules.COMPS[comp]
        if not data[comp]["revision_found"]:
            comps_out[comp] = dict(status="NA", reason="pinned revision not found")
            continue
        tc = time.time()
        pf = P.ParquetFile(str(spec["parquet"]))
        meta_cols = ["model_config", "model_name", "problem_idx", "idx_answer", "source", "input_tokens",
                     "output_tokens", "cost"]
        col = {c: pf.read_column(c) for c in meta_cols}
        n = pf.num_rows
        cfg_names = {}
        for i in range(n):
            cfg_names.setdefault(col["model_config"][i], set()).add(col["model_name"][i])
        configs, endpoints, pairs = rules.enumerate_pairs(cfg_names)
        problems = sorted({col["problem_idx"][i] for i in range(n)}, key=rules.pidx_key)
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
                tiers = sorted({t for _, t in members}, key=rules.LADDER.index)
                adj = any(rules.LADDER.index(b) - rules.LADDER.index(a) == 1 for a, b in zip(tiers, tiers[1:]))
                notes.append(dict(comp=comp, display_base=disp, configs=members,
                                  note=("same display base but different paths after removing --<tier>, so not the same endpoint under the tier rule; "
                                        + ("if single-hyphen suffixes were also read as tier suffixes, these tiers would include an adjacent pair (not computed)"
                                           if adj else "even if merged, the tiers are not adjacent"))))

        # ------------------------------------------------------------ eligible configs: completeness fields
        elig = sorted({p["A"] for p in pairs} | {p["B"] for p in pairs})
        rows = [i for i in range(n) if col["model_config"][i] in elig]
        tx = {}
        tx["problem"] = rules.scan_column(pf, "problem", rows, lambda v: v)
        tx["gold"] = rules.scan_column(pf, "gold_answer", rows, lambda v: v)
        tx["user_message"] = rules.scan_column(pf, "user_message", rows, lambda v: v)
        um = tx["user_message"]
        am = rules.scan_column(pf, "all_messages", rows, lambda v: (h16(v), rules.prompt_part(v)))
        prompt = {}
        for i in rows:
            ok, why, canon = rules.full_prompt_from_part(um[i], am[i][1])
            prompt[i] = (ok, why, h16(canon) if canon is not None else None)
        am = {i: am[i][0] for i in rows}
        tx["answer"] = rules.scan_column(pf, "answer", rows,
                                     lambda v: (v is not None and v.strip() != "", len(v or ""), h16(v)))
        tx["parsed"] = rules.scan_column(pf, "parsed_answer", rows, lambda v: (v is None, h16(v)))
        if pf.ptype["correct"] != 0:
            raise rules.StopNA(f"{comp}: correct is not a BOOLEAN column (physical type {pf.ptype['correct']})")
        cnull = rules.scan_column(pf, "correct", rows, lambda is_null: is_null, values=False)
        pin =rules.scan_column(pf, "input_cost_per_tokens", rows, lambda v: v)
        pout = rules.scan_column(pf, "output_cost_per_tokens", rows, lambda v: v)
        recs = []
        for i in rows:
            ans_ne, ans_len, ans_h = tx["answer"][i]
            pnull, ph = tx["parsed"][i]
            digest = h16(json.dumps([col[c][i] for c in meta_cols] + [h16(tx["problem"][i]), h16(tx["gold"][i]),
                                     h16(um[i]), am[i], ans_h, ph, pnull, cnull[i], pin[i], pout[i]],
                                    default=str))
            recs.append(dict(row=i, cfg=col["model_config"][i], problem_idx=col["problem_idx"][i],
                             idx=col["idx_answer"][i], problem=tx["problem"][i], gold=tx["gold"][i],
                             status=rules.response_status(ans_ne, col["output_tokens"][i]), prompt_ok=prompt[i][0],
                             prompt_why=prompt[i][1], prompt=prompt[i][2], correct_is_null=cnull[i], digest=digest,
                             answer_len=ans_len, parsed_null=pnull, in_tok=col["input_tokens"][i],
                             out_tok=col["output_tokens"][i], cost=col["cost"][i], price=(pin[i], pout[i])))
        del tx
        byrow = {r["row"]: r for r in recs}

        # ------------------------------------------------------------ cells
        ep_key = lambda p: f"{p['display_base']} | {p['config_base']}"  # noqa: E731
        for p in pairs:
            valid, breaks, kept, slots = rules.cell_contract(recs, problems, p["A"], p["B"])
            seen = rules.SEEN.get((comp, p["A"], p["B"]))
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
                                n_saved_truncation=sum(r["status"] == rules.ST_TRUNC for r in rs),
                                n_unknown_status=sum(r["status"] == rules.ST_UNKNOWN for r in rs),
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
            for side, q, k, row, st in sorted(slots, key=lambda s: (rules.pidx_key(s[1]), s[0],
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
        log(f"{comp} done ({comps_out[comp]['seconds']} s)")

    frame = dict(step="frame written from metadata before any correct value is decoded (correct: null mask only)",
                 written_local=helpers.now_local(), data=data, transfer=gate,
                 reader="code/pq_reader.py primitives + all_pairs_rules.scan_column (row-selective streaming; null-mask mode)",
                 ladder=rules.LADDER, decided_before_correct=DECIDED, competitions=comps_out,
                 near_miss_notes=notes, cells=cells_out,
                 seen_rule={f"{k[0]}|{k[1]}->{k[2]}": v for k, v in rules.SEEN.items()},
                 n_cells=len(cells_out), n_cells_valid=sum(c["contract_valid"] for c in cells_out),
                 n_cells_NA=sum(not c["contract_valid"] for c in cells_out))
    rules.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    helpers.write_csv(rules.FRAME_DIR / "configs.csv", ["competition", "model_config", "model_name", "n_rows", "name_tag",
                                               "config_suffix_tag", "tier", "display_base", "config_base", "status",
                                               "reason", "pair_status"], cfg_rows)
    helpers.write_csv(rules.FRAME_DIR / "endpoints.csv", ["competition", "display_base", "config_base", "tiers", "configs",
                                                 "status", "adjacent_pairs", "reason"], endpoint_rows)
    helpers.write_csv(rules.FRAME_DIR / "pairs.csv", ["cell_id", "competition", "endpoint", "tier_A", "tier_B", "A", "B",
                                             "A_name", "B_name", "seen", "seen_label", "n_problems_frame",
                                             "n_problems_complete", "n_problems_breaking", "contract", "na_reason"],
                 cell_rows)
    helpers.write_csv(rules.FRAME_DIR / "slots.csv", ["cell_id", "side", "problem_idx", "idx", "parquet_row", "status",
                                             "input_tokens", "output_tokens", "answer_chars", "parsed_answer_null"],
                 slot_rows)
    helpers.write_csv(rules.FRAME_DIR / "near_miss_notes.csv", ["competition", "display_base", "configs_tiers", "note"],
                 [[x["comp"], x["display_base"], json.dumps(x["configs"], ensure_ascii=False), x["note"]]
                  for x in notes])
    frame["step1_seconds"] = round(time.time() - t0, 2)
    helpers.write_json(rules.FRAME_DIR / "frame.json", frame)
    log(f"frame written: {frame['n_cells']} cells, {frame['n_cells_valid']} valid, "
        f"{frame['n_cells_NA']} NA ({frame['step1_seconds']} s)")


if __name__ == "__main__":
    try:
        main()
    except rules.StopNA as e:
        log(f"STOP: {e}")
        helpers.write_json(rules.RESULTS / "STOP_NA.json", dict(step="frame", reason=str(e), time_local=helpers.now_local()))
        sys.exit(3)

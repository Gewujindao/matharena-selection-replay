# MathArena Wrong-Answer Screening Replay

**中文简介**：本仓库用 MathArena 公开的模型作答记录做回放，量化"按某个配置答错来筛题"带来的选择效应。把两个配置记为 A、B：在 A 首次作答答错的题上，B 的提升（G）中有多少只是 A 自己重答也能拿回的（R），剩下的配置差（D = G − R）与全部题目上的配置差（D_F）相比如何。研究分三个阶段，每个阶段都先冻结预注册，再取数计算；第三阶段覆盖四场 2025 最终答案比赛中同一基础模型相邻推理档位的全部配置对。数据不随仓库分发。

## At a glance

- Replays MathArena's cached responses to measure how screening on a lower reasoning tier's wrong answers inflates the next tier's apparent gain, across all adjacent o3-mini / o4-mini tier pairs in four 2025 competitions.
- Macro-averaged over the 9 valid cells, the apparent gain on screened problems is G = +50.70 pp, of which R = +25.14 pp is recovered by the lower tier's own re-attempts, while the gap on all problems is D_F = +15.83 pp.
- Each stage was preregistered and frozen before scoring, and an independent recomputation with a separate Parquet decoder and implementation agrees on every reported item.

![Apparent gain G, lower-tier recovery R and all-problem gap D_F for the 9 valid adjacent-tier cells and their macro average](figures/summary.png)

---

## 1. Research question

Benchmarks are often built by keeping the problems that some reference model gets wrong. When a newer configuration B is then compared with the reference configuration A on those problems, part of B's apparent gain is regression to the mean: A itself would have recovered some of them on a fresh attempt. This repository measures that effect on MathArena's cached responses, where every model configuration answered every problem four times (`idx` 0–3).

1. On problems that A answered wrong in its screening attempt, how much of B's gain is recovered by A's own other attempts?
2. How does the remaining configuration gap on the screened problems compare with the gap on all problems?
3. Across every adjacent reasoning-tier pair of the same base model in four 2025 final-answer competitions, how large are these quantities and how often do they change sign?

## 2. Definitions (from the frozen preregistrations)

For each problem, one attempt is the screening slot and the other three are re-scoring slots. Week 1 assigns the screening slot by a hash-based rotation (`p0`); weeks 2 and 3 fix it at `idx0`.

- `A₀`: correctness of A's screening attempt (0/1).
- `A₁`, `B₁`: mean correctness of A and of B over the three re-scoring attempts. B's screening-slot record is not used, so both sides are scored on the same number of attempts.
- `S = {problems with A₀ = 0}`, `K = |S|`. G, R, D are undefined when K = 0.
- **G** = mean over S of `(B₁ − A₀)`: the apparent gain of B on the screened problems (called `G_naive` in week 1).
- **R** = mean over S of `(A₁ − A₀)`: recovery of A's own cached records after screening.
- **D** = mean over S of `(B₁ − A₁)`: the configuration gap on the screened set. `G − R = D` holds exactly.
- **D_F** = mean over the complete frame F (all problems with full records) of `(B₁ − A₁)`: the configuration gap on all problems.
- **C** = `D − D_F`: the composition difference created by screening. Week 3 also reports `G − D_F` (`= R + C`) and the sign-reversal flag `1[D · D_F < 0]`.

All quantities are computed as exact fractions on a fixed, finite set of records and reported in percentage points (pp). Results are finite-ledger descriptions of these records; no confidence intervals are attached, because the original repeat mechanism (seeds, retries, request history) is not documented. A four-slot rotation of the screening slot is reported as a sensitivity analysis only.

Week-3 aggregation (frozen weights): within each competition, average the valid adjacent pairs of a base endpoint with equal weight, then average endpoints with equal weight; the macro average gives equal weight to each competition with at least one valid cell.

## 3. Main results

### 3.1 Week 3: all adjacent reasoning-tier pairs, four competitions (`week3/report.txt`)

- 210 configurations across AIME, HMMT Feb, BRUMO and CMIMC 2025. The frozen tier rule yields 12 adjacent-tier cells, all from the `o3-mini` and `o4-mini` base endpoints; 9 meet the data contract and 3 are NA because the eight-slot prompts are not byte-identical (AIME `o3-mini` medium→high; both CMIMC cells). The weighted summary therefore covers AIME, HMMT and BRUMO.
- Across the 9 valid cells (screening slot `idx0`), R ranges from +0.00 to +50.00 pp and C from −9.44 to +22.22 pp (7 positive, 1 zero, 1 negative). D and D_F have strictly opposite signs in 1 of 9 cells (BRUMO `o4-mini` medium→high, K = 4).
- Frozen-weight macro averages over the three competitions:

| Set | G | R | D | D_F | C = D − D_F | G − D_F |
|---|---:|---:|---:|---:|---:|---:|
| All valid cells | +50.70 | +25.14 | +25.57 | +15.83 | +9.73 | +34.87 |
| Excluding the cell already used in week 2 | +51.33 | +26.23 | +25.10 | +16.11 | +8.99 | +35.22 |

On the problems a lower tier gets wrong, about half of the higher tier's apparent gain (R = +25.14 of G = +50.70 pp) is recovered by the lower tier's own other attempts. The remaining gap D is still larger than the all-problem gap D_F by C = +9.73 pp.

- An independent recomputation (separate Parquet decoder and separate implementation of the tier, contract and aggregation rules, `code/week3/w3_independent.py`) agrees on every item. A second independent recomputation is in `code/week3_review/` with output `week3/review_recompute.json`.

### 3.2 Week 2: HMMT Feb 2025, A = o3-mini (medium), B = o3-mini (high) (`week2/report.txt`)

K = 11 of 30 problems; G = +27.27 pp, R = +0.00 pp (0 of 33 re-scoring records correct), D = +27.27 pp; D_F = +16.67 pp, D − D_F = +10.61 pp.

### 3.3 Week 1: AIME 2025, A = Claude-3.5-Sonnet, B = Claude-3.7-Sonnet (Think) (`results/report.txt`)

K = 28 of 30 problems; G = 10/21 (+47.6 pp), R = 1/84 (+1.2 pp), D = 13/28 (+46.4 pp); D_F = 41/90 (+45.6 pp), D − D_F = 11/1260 (+0.9 pp). A answers 4 of 120 records correctly, so screening on A's errors barely changes the problem set; this is a floor case.

## 4. Data sources (not distributed)

Cached model responses published by MathArena on Hugging Face, license **cc-by-nc-sa-4.0**. The data are not redistributed in this repository, and per-item tables derived from them (slot-level correctness and slot listings) are omitted; the scripts regenerate those tables from the data. Download the data at the pinned revisions:

| Competition | Dataset | Revision |
|---|---|---|
| AIME 2025 | `MathArena/aime_2025_outputs` | `16e93d3bb9b0a25ea9422893464d46b3c5ff0ae9` |
| HMMT Feb 2025 | `MathArena/hmmt_feb_2025_outputs` | `e3fb6c74f24eabf83413eb9e62b9816c1f3f7d2e` |
| BRUMO 2025 | `MathArena/brumo_2025_outputs` | `3b234efea64d03f80ec57a85e29879fc41b04864` |
| CMIMC 2025 | `MathArena/cmimc_2025_outputs` | `27826187da75fb124dca803cc0274d3c5694df5c` |

Each dataset is a single file `data/train-00000-of-00001.parquet`, e.g.
`https://huggingface.co/datasets/MathArena/aime_2025_outputs/resolve/16e93d3bb9b0a25ea9422893464d46b3c5ff0ae9/data/train-00000-of-00001.parquet`.

The scripts expect the files at:

```
downloads/hf_aime_2025_outputs_16e93d3/train-00000-of-00001.parquet       (week 1)
downloads/hf_hmmt_feb_2025_outputs_e3fb6c7/train-00000-of-00001.parquet   (weeks 2, 3)
downloads/w3_hf_aime_2025_outputs_16e93d3/train-00000-of-00001.parquet    (week 3)
downloads/w3_hf_brumo_2025_outputs_3b234ef/train-00000-of-00001.parquet   (week 3)
downloads/w3_hf_cmimc_2025_outputs_2782618/train-00000-of-00001.parquet   (week 3)
```

The stage-1 scripts also read metadata snapshots saved at run time (Hugging Face revision info and file trees, MathArena config history on GitHub) from `downloads/meta_web/`, `downloads/w2_meta_web/` and `downloads/w3_meta_web/`; these snapshots are not included.

## 5. Repository layout

```
预注册_第一周_冻结_2026-10-23.md   week-1 frozen preregistration (AIME, Claude-3.5 vs 3.7 Think)
预注册_第二周_冻结_2026-10-23.md   week-2 frozen preregistration (HMMT, o3-mini medium vs high)
预注册_第三周_冻结_2026-10-23.md   week-3 frozen preregistration; Part 1 holds the operational definitions
code/
  pq_reader.py                      pure-Python Parquet reader
  w1_common.py, w1_precheck.py      week 1: gates, frozen manifest, role tables (correct column masked)
  w1_unseal.py, w1_report.py        week 1: unseal correct, compute, write results/report.txt
  test_w1.py                        week-1 unit tests
  diag/                             week-1 added diagnostics and report regeneration
  week2/                            w2_extract -> w2_compute -> w2_independent -> w2_report, tests
  week3/                            w3_frame -> w3_compute -> w3_independent -> w3_report, probes, prompt diagnostic, tests
  week3_review/                     second independent week-3 recomputation (own Parquet decoder)
results/                            week 1 outputs
  report.txt, summary.json, run_log.txt, timeline.jsonl
  frozen/                           manifest, slots, roles, exclusions, gates, seed manifest, code at sealing time
  tables/                           rotations, per-problem table, per-slot correctness
  diag/                             official-page accuracy check
week2/                              week 2 outputs: report.txt, summary.json, independent_recompute.json,
                                    frame/ (written before reading correct), tables/
week3/                              week 3 outputs: report.txt, summary.json, independent_recompute.json,
                                    review_recompute.json, frame/ (configs, endpoints, cells, slots,
                                    near-miss notes; written before reading correct), tables/
```

Reports and preregistrations are in Chinese. Per-slot tables contain record keys, token counts, text hashes and 0/1 correctness only; no problem statements or response texts are included.

## 6. Reproduction

Requirements: Python 3.10+, Windows (process limits use Win32 job objects through `ctypes`), standard library plus the `zstandard` package (imported by some of the Parquet decoders). No model API calls are made.

1. Download the Parquet files listed in section 4 into `downloads/` at the pinned revisions.
2. Week 1, from `code/`:
   ```
   python -X utf8 -I test_w1.py
   python -X utf8 -I w1_precheck.py      # stage 1: seal frame and roles (requires downloads/meta_web/)
   python -X utf8 -I w1_unseal.py        # stage 2: read correct, compute, write report
   ```
3. Week 2, from `code/week2/`:
   ```
   python -X utf8 -I -B w2_extract.py
   python -X utf8 -I -B w2_compute.py
   python -X utf8 -I -B w2_independent.py
   python -X utf8 -I -B w2_report.py
   ```
4. Week 3, from `code/week3/`:
   ```
   python -X utf8 -I -B w3_frame.py      # metadata only: configs, tiers, pairs, contracts -> week3/frame/
   python -X utf8 -I -B w3_compute.py    # decode correct, per-cell quantities and aggregates
   python -X utf8 -I -B w3_independent.py
   python -X utf8 -I -B w3_report.py
   ```
   Second recomputation: `python code/week3_review/rv_recompute.py`, then `python code/week3_review/rv_compare.py`.

The stage-1 outputs (`results/frozen/`, `week2/frame/`, `week3/frame/`) were stored read-only; clear the read-only attribute before a rerun overwrites them.

## License

Code and text in this repository: MIT, see `LICENSE`. The MathArena data are licensed separately (cc-by-nc-sa-4.0) and are not covered by this license.

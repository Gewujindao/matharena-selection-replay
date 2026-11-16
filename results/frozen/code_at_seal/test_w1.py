"""Unit tests (unittest; pytest is not installed).  Run:  python -X utf8 -I test_w1.py"""
import os
import struct
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1_common as C  # noqa: E402
import pq_reader as P  # noqa: E402


# ----------------------------------------------------------------------------- tiny thrift / parquet writer (tests)

def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def zz(n):
    return varint((n << 1) ^ (n >> 63))


TYPE = {"i32": 5, "i64": 6, "bin": 8, "list": 9, "struct": 12}


def enc_value(v):
    kind = v[0]
    if kind in ("i32", "i64"):
        return zz(v[1])
    if kind == "bin":
        return varint(len(v[1])) + v[1]
    if kind == "list":
        et, items = v[1], v[2]
        h = bytes([(len(items) << 4) | TYPE[et]]) if len(items) < 15 else bytes([0xF0 | TYPE[et]]) + varint(len(items))
        return h + b"".join(enc_value((et, x)) if et != "struct" else enc_struct(x) for x in items)
    if kind == "struct":
        return enc_struct(v[1])
    raise ValueError(kind)


def enc_struct(fields):
    out = bytearray()
    last = 0
    for fid in sorted(fields):
        v = fields[fid]
        t = TYPE[v[0]]
        d = fid - last
        if 0 < d <= 15:
            out.append((d << 4) | t)
        else:
            out.append(t)
            out += zz(fid)
        out += enc_value(v)
        last = fid
    out.append(0)
    return bytes(out)


def snappy_literal_only(raw):
    out = bytearray(varint(len(raw)))
    for i in range(0, len(raw), 60):
        ch = raw[i:i + 60]
        out.append((len(ch) - 1) << 2)
        out += ch
    return bytes(out)


def rle_run(value, n, width=1):
    return varint(n << 1) + int(value).to_bytes((width + 7) // 8, "little")


def def_levels(defs):
    out = bytearray()
    i = 0
    while i < len(defs):
        j = i
        while j < len(defs) and defs[j] == defs[i]:
            j += 1
        out += rle_run(defs[i], j - i)
        i = j
    return bytes(out)


def plain_bytes(vals):
    return b"".join(struct.pack("<i", len(v)) + v for v in vals)


def write_parquet(path, pages_by_col, codec=0):
    """pages_by_col: {name: (ptype, [list of page value lists with None for nulls])}; one row group."""
    body = bytearray(b"PAR1")
    chunks = []
    nrows = None
    for name, (ptype, pages) in pages_by_col.items():
        start = len(body)
        total = 0
        for vals in pages:
            defs = [0 if v is None else 1 for v in vals]
            nn = [v for v in vals if v is not None]
            if ptype == 6:
                vb = plain_bytes(nn)
            elif ptype == 2:
                vb = struct.pack(f"<{len(nn)}q", *nn)
            elif ptype == 0:
                bits = sum(1 << i for i, v in enumerate(nn) if v)
                vb = bits.to_bytes((len(nn) + 7) // 8, "little")
            dl = def_levels(defs)
            raw = struct.pack("<i", len(dl)) + dl + vb
            comp = snappy_literal_only(raw) if codec == 1 else raw
            ph = enc_struct({1: ("i32", 0), 2: ("i32", len(raw)), 3: ("i32", len(comp)),
                             5: ("struct", {1: ("i32", len(vals)), 2: ("i32", 0), 3: ("i32", 3), 4: ("i32", 3)})})
            body += ph + comp
            total += len(vals)
        if nrows is None:
            nrows = total
        md = {1: ("i32", ptype), 2: ("list", "i32", [0, 3]), 3: ("list", "bin", [name.encode()]),
              4: ("i32", codec), 5: ("i64", total), 6: ("i64", len(body) - start), 7: ("i64", len(body) - start),
              9: ("i64", start)}
        chunks.append({2: ("i64", start), 3: ("struct", md)})
    schema = [{4: ("bin", b"schema"), 5: ("i32", len(pages_by_col))}]
    schema += [{1: ("i32", pt), 3: ("i32", 1), 4: ("bin", n.encode())} for n, (pt, _) in pages_by_col.items()]
    fmd = enc_struct({1: ("i32", 1), 2: ("list", "struct", schema), 3: ("i64", nrows),
                      4: ("list", "struct", [{1: ("list", "struct", chunks), 2: ("i64", 0), 3: ("i64", nrows)}])})
    body += fmd + struct.pack("<i", len(fmd)) + b"PAR1"
    with open(path, "wb") as f:
        f.write(body)


# ----------------------------------------------------------------------------- reader tests


class TestReader(unittest.TestCase):
    def test_thrift_struct(self):
        buf = bytes([0x15, 0x0A, 0x18, 0x02]) + b"ab" + bytes([0x29, 0x25, 0x02, 0x04, 0x05, 0x28, 0x05, 0x00])
        self.assertEqual(P.ThriftReader(buf).struct(), {1: 5, 2: b"ab", 4: [1, 2], 20: -3})

    def test_thrift_roundtrip_writer(self):
        s = {1: ("i32", -7), 3: ("i64", 1 << 40), 30: ("bin", b"xyz"), 31: ("list", "i32", list(range(20)))}
        self.assertEqual(P.ThriftReader(enc_struct(s)).struct(), {1: -7, 3: 1 << 40, 30: b"xyz", 31: list(range(20))})

    def test_snappy_copy1_overlap(self):
        stream = bytes([13, 0x08]) + b"abc" + bytes([0x15, 0x03, 0x00]) + b"X"
        self.assertEqual(P.snappy_decompress(stream), b"abcabcabcabcX")

    def test_snappy_copy2(self):
        stream = bytes([11, 0x14]) + b"hello " + bytes([0x12, 0x06, 0x00])
        self.assertEqual(P.snappy_decompress(stream), b"hello hello")

    def test_snappy_long_literal(self):
        raw = bytes(range(256)) * 3
        stream = varint(len(raw)) + bytes([61 << 2]) + (len(raw) - 1).to_bytes(2, "little") + raw
        self.assertEqual(P.snappy_decompress(stream), raw)
        self.assertEqual(P.snappy_decompress(snappy_literal_only(raw)), raw)

    def test_snappy_length_mismatch(self):
        with self.assertRaises(ValueError):
            P.snappy_decompress(bytes([5, 0x08]) + b"abc")

    def test_rle_hybrid(self):
        buf = rle_run(1, 5) + rle_run(0, 3)
        self.assertEqual(P.rle_hybrid(buf, 0, len(buf), 1, 8), [1, 1, 1, 1, 1, 0, 0, 0])
        vals = list(range(8))
        packed = sum(v << (3 * i) for i, v in enumerate(vals)).to_bytes(3, "little")
        buf = bytes([0x03]) + packed
        self.assertEqual(P.rle_hybrid(buf, 0, len(buf), 3, 8), vals)
        self.assertEqual(P.rle_hybrid(buf, 0, len(buf), 3, 5), vals[:5])

    def test_plain(self):
        self.assertEqual(P.plain(plain_bytes([b"hi", b""]), 6, 2), [b"hi", b""])
        self.assertEqual(P.plain(struct.pack("<2q", -1, 5), 2, 2), [-1, 5])
        self.assertEqual(P.plain(struct.pack("<d", 0.25), 5, 1), [0.25])
        self.assertEqual(P.plain(bytes([0b00000101, 0b1]), 0, 9), [True, False, True] + [False] * 5 + [True])

    def test_read_column_roundtrip(self):
        cols = {"s": (6, [["a", None, "ccc"], ["dé", "e"]]), "n": (2, [[1, 2, None], [4, 5]]),
                "b": (0, [[True, False, True], [None, True]])}
        enc = {k: (pt, [[(v.encode() if isinstance(v, str) else v) for v in pg] for pg in pages])
               for k, (pt, pages) in cols.items()}
        for codec in (0, 1):
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "t.parquet")
                write_parquet(p, enc, codec=codec)
                pf = P.ParquetFile(p)
                self.assertEqual(pf.num_rows, 5)
                self.assertEqual(pf.read_column("s"), ["a", None, "ccc", "dé", "e"])
                self.assertEqual(pf.read_column("n"), [1, 2, None, 4, 5])
                self.assertEqual(pf.read_column("b"), [True, False, True, None, True])
                part = pf.read_column("s", rows={3})
                self.assertIs(part[0], P.SKIPPED)
                self.assertEqual(part[3:], ["dé", "e"])


# ----------------------------------------------------------------------------- precheck / roles / metrics


def mk(side_cfg, q, k, row, **kw):
    r = dict(row=row, cfg=side_cfg, source=q[0], problem_idx=q[1], idx=k, problem_sha="h" + q[1], gold="7",
             answer_status=C.ANSWER_OK, correct_is_null=False, row_digest=f"d{row}")
    r.update(kw)
    return r


def full(q, start=0):
    rows = []
    for j, cfg in enumerate((C.CFG_A, C.CFG_B)):
        for k in C.IDX:
            rows.append(mk(cfg, q, k, start + 10 * j + k))
    return rows


class TestPrecheck(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(C.normalize_config("configs/models/anthropic/claude-35.yaml"), "anthropic/claude-35")
        self.assertEqual(C.normalize_config("anthropic/claude-35"), "anthropic/claude-35")
        self.assertNotEqual(C.normalize_config("models/anthropic/claude-35.yml"), "anthropic/claude-35")
        self.assertNotEqual(C.normalize_config("anthropic/Claude-35"), "anthropic/claude-35")

    def test_answer_status(self):
        self.assertEqual(C.answer_status("x", 0), C.ANSWER_OK)
        self.assertEqual(C.answer_status("  ", 5), C.ANSWER_EMPTY_TRUNC)
        self.assertEqual(C.answer_status(None, None), C.ANSWER_FAIL)
        self.assertEqual(C.answer_status("", 0), C.ANSWER_FAIL)

    def test_complete_problem_passes(self):
        q = ("aime", "1")
        F, ex, kept, _ = C.precheck(full(q), [q])
        self.assertEqual(F, [q])
        self.assertEqual(ex, {})
        self.assertEqual(kept[q][("A", 2)], 2)
        self.assertEqual(kept[q][("B", 3)], 13)

    def test_missing_b_excludes_both_sides(self):
        q = ("aime", "1")
        rows = [r for r in full(q) if not (r["cfg"] == C.CFG_B and r["idx"] == 1)]
        F, ex, _, st = C.precheck(rows, [q])
        self.assertEqual(F, [])
        self.assertIn("B: missing idx 1", ex[q])
        self.assertIn(("B", q, 1, None, "missing"), st)

    def test_identical_duplicate_collapsed(self):
        q = ("aime", "1")
        rows = full(q)
        dup = dict(rows[0])
        dup["row"] = 99
        F, ex, kept, st = C.precheck(rows + [dup], [q])
        self.assertEqual(F, [q])
        self.assertEqual(kept[q][("A", 0)], 0)
        self.assertTrue(any(s[3] == 99 and s[4].startswith("identical_duplicate") for s in st))

    def test_conflicting_duplicate_excluded(self):
        q = ("aime", "1")
        rows = full(q)
        dup = dict(rows[0], row=99, row_digest="other")
        F, ex, _, _ = C.precheck(rows + [dup], [q])
        self.assertEqual(F, [])
        self.assertTrue(any("conflicting duplicate" in r for r in ex[q]))

    def test_gold_or_text_mismatch_excluded(self):
        q = ("aime", "1")
        rows = full(q)
        rows[5]["gold"] = "8"
        self.assertEqual(C.precheck(rows, [q])[0], [])
        rows = full(q)
        rows[6]["problem_sha"] = "zz"
        self.assertEqual(C.precheck(rows, [q])[0], [])

    def test_failure_and_grade_missing_excluded_truncation_kept(self):
        q = ("aime", "1")
        rows = full(q)
        rows[1]["answer_status"] = C.ANSWER_FAIL
        self.assertEqual(C.precheck(rows, [q])[0], [])
        rows = full(q)
        rows[2]["correct_is_null"] = True
        self.assertEqual(C.precheck(rows, [q])[0], [])
        rows = full(q)
        rows[3]["answer_status"] = C.ANSWER_EMPTY_TRUNC
        self.assertEqual(C.precheck(rows, [q])[0], [q])

    def test_unexpected_idx_and_other_config_ignored(self):
        q = ("aime", "1")
        rows = full(q) + [mk(C.CFG_A, q, 4, 50)]
        self.assertEqual(C.precheck(rows, [q])[0], [])
        rows = full(q) + [mk("anthropic/claude-opus-4", q, 0, 60, gold="9")]
        self.assertEqual(C.precheck(rows, [q])[0], [q])

    def test_candidate_absent_everywhere(self):
        q1, q2 = ("aime", "1"), ("aime", "2")
        F, ex, _, _ = C.precheck(full(q1), [q1, q2])
        self.assertEqual(F, [q1])
        self.assertEqual(len(ex[q2]), 8)


class TestRoles(unittest.TestCase):
    def test_string_format(self):
        self.assertEqual(C.ns_string("aime_2025", "3", 2), f"line160/v1|{C.REV}|aime_2025|3|2")

    def test_order_by_digest_bytes_ties_by_idx(self):
        d = {0: b"\x05", 1: b"\x01", 2: b"\x05", 3: b"\x00"}
        self.assertEqual(C.role_order(("s", "1"), digest_fn=d.get), [3, 1, 0, 2])

    def test_order_real_hash_is_permutation(self):
        o = C.role_order(("aime_2025", "1"))
        self.assertEqual(sorted(o), [0, 1, 2, 3])
        dg = [C.ns_digest("aime_2025", "1", k) for k in o]
        self.assertEqual(dg, sorted(dg))

    def test_seed_projections(self):
        d = bytes(range(32))
        self.assertEqual(C.u64be(d), int.from_bytes(bytes(range(8)), "big"))
        self.assertEqual(C.sha63(d), int.from_bytes(bytes(range(8)), "little") & ((1 << 63) - 1))
        m = C.seed_manifest([("aime_2025", "1")])
        self.assertEqual(len(m), 4)
        self.assertEqual(sorted(e["position"] for e in m), [0, 1, 2, 3])

    def test_literal_regex(self):
        b = b"x1234567890y 123456789 123456789012345678901 9999999999,"
        self.assertEqual([m.group(1) for m in C.LIT.finditer(b)], [b"1234567890", b"9999999999"])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.q1, self.q2, self.q3 = ("s", "1"), ("s", "2"), ("s", "3")
        self.order = {q: [0, 1, 2, 3] for q in (self.q1, self.q2, self.q3)}
        c = {}
        # q1: A0=0, A rest (1,0,0) -> A1=1/3 ; B rest (1,1,0) -> B1=2/3
        for k, v in enumerate([0, 1, 0, 0]):
            c[("A", self.q1, k)] = v
        for k, v in enumerate([1, 1, 1, 0]):
            c[("B", self.q1, k)] = v
        # q2: A0=1 (not screened); A1=1, B1=1
        for k in range(4):
            c[("A", self.q2, k)] = 1
            c[("B", self.q2, k)] = 1
        # q3: A0=0; A1=0; B1=1
        for k in range(4):
            c[("A", self.q3, k)] = 0
            c[("B", self.q3, k)] = 1
        self.c = c

    def test_primary(self):
        out, per = C.replay_metrics([self.q1, self.q2, self.q3], self.order, self.c, j=0)
        self.assertEqual(out["K"], 2)
        self.assertEqual(out["A1_S"], Fraction(1, 6))
        self.assertEqual(out["B1_S"], Fraction(5, 6))
        self.assertEqual(out["G_naive"], Fraction(5, 6))
        self.assertEqual(out["R"], Fraction(1, 6))
        self.assertEqual(out["D"], Fraction(2, 3))
        self.assertEqual(out["G_naive"] - out["R"], out["D"])
        # D_F over all three: (1/3 + 0 + 1)/3 = 4/9
        self.assertEqual(out["D_F"], Fraction(4, 9))
        self.assertEqual(out["D_minus_D_F"], Fraction(2, 9))

    def test_rotation_uses_other_slot(self):
        out, per = C.replay_metrics([self.q1], self.order, self.c, j=1)
        self.assertEqual(out["K"], 0)                       # A[1]=1 -> q1 not screened
        self.assertIsNone(out["D"])
        self.assertEqual(per[0]["rescore_idx"], [0, 2, 3])

    def test_k_zero_undefined(self):
        out, _ = C.replay_metrics([self.q2], self.order, self.c, j=0)
        self.assertEqual(out["K"], 0)
        self.assertIsNone(out["G_naive"])
        self.assertIsNone(out["D_minus_D_F"])
        self.assertEqual(out["D_F"], Fraction(0))


class TestPipelineSmoke(unittest.TestCase):
    """build_summary + write_report on synthetic records (full frame, K=0, and a reduced frame)."""

    def fake(self, nF, excluded=None):
        import w1_unseal as U
        F = [("aime_2025", str(i)) for i in range(1, nF + 1)]
        order = {q: C.role_order(q) for q in F}
        man = dict(data=dict(entry_gate=dict(snapshot_commit_date_utc="2026-09-11T10:55:10.000Z", local_size=142685920),
                             reader_check=dict(cost_identity_matches=7915, rows_checked=7915)),
                   configs=dict(A=dict(raw_path_values=[C.CFG_A], model_names=["a"]),
                                B=dict(raw_path_values=[C.CFG_B], model_names=["b"])),
                   tokens=dict(A=dict(over_native_max_tokens=0, empty_visible_answer=0),
                               B=dict(over_native_max_tokens=0, empty_visible_answer=1)),
                   decided_before_unmask=["x"], timings=dict(stage1_total_s=1.0, seed_check_s=0.5),
                   peak_working_set_mib=10.0, candidates=[list(q) for q in F] + ([["aime_2025", "99"]] if excluded else []),
                   exclusions=excluded or {})
        gates = dict(entry=True, cost=True, execution=True, seed_collision_free=True,
                     cost_gate=dict(downloads_bytes=143000000))
        seeds = dict(n_entries=4 * nF, scan=dict(n_text_files=10), n_registered_ranges=3)
        return U, F, order, man, gates, seeds

    def run_case(self, nF, corr_fn, excluded=None):
        import w1_report as RP
        U, F, order, man, gates, seeds = self.fake(nF, excluded)
        corr = {(s, q, k): corr_fn(s, i, k) for i, q in enumerate(F) for s in ("A", "B") for k in C.IDX}
        summary, rots, _ = U.build_summary(man, gates, F, order, corr, {}, dict(stage2_read_s=0.1))
        summary["timings"]["stage2_total_s"] = 0.2
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "r.txt")
            text = RP.write_report(summary, man, seeds, gates, out_path=out)
        for o in rots:
            if o["D"] is not None:
                self.assertEqual(o["G_naive"] - o["R"], o["D"])
        return summary, text

    def test_full_frame(self):
        summary, text = self.run_case(30, lambda s, i, k: int((s == "B") or ((i + k) % 3 == 0)))
        self.assertGreater(summary["primary"]["K"], 0)
        self.assertIn("缓存记录回升", text)
        self.assertIn("错筛组成差", text)
        self.assertIn("未参与筛选的旧配置角色", text)
        self.assertIn("不强加 95% CI", text)

    def test_k_zero(self):
        summary, text = self.run_case(30, lambda s, i, k: 1)
        self.assertEqual(summary["primary"]["K"], 0)
        self.assertIn("未定义", text)

    def test_reduced_frame(self):
        summary, text = self.run_case(29, lambda s, i, k: int(k % 2 == 0), excluded={"aime_2025|99": ["B: missing idx 1"]})
        self.assertIn("完整缓存子框", text)
        self.assertIn("排除 aime_2025|99", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)

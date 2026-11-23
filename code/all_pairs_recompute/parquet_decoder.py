# Minimal Parquet reader for the second recomputation of the all-pairs run (independent of every other module in code/).
# Supports: Thrift compact footer + page headers, UNCOMPRESSED / SNAPPY / ZSTD / GZIP,
# data page v1 and v2, PLAIN and (RLE_|PLAIN_)DICTIONARY values, RLE/bit-packed hybrid levels.
import struct
import zlib

# ---------------- Thrift compact protocol ----------------

class TReader:
    __slots__ = ("b", "p")

    def __init__(self, b, p=0):
        self.b = b
        self.p = p

    def varint(self):
        r = 0
        s = 0
        b = self.b
        while True:
            x = b[self.p]
            self.p += 1
            r |= (x & 0x7F) << s
            if x < 0x80:
                return r
            s += 7

    def zz(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def value(self, t):
        if t == 1:
            return True
        if t == 2:
            return False
        if t == 3:
            v = self.b[self.p]
            self.p += 1
            return v - 256 if v > 127 else v
        if t in (4, 5, 6):
            return self.zz()
        if t == 7:
            v = struct.unpack_from("<d", self.b, self.p)[0]
            self.p += 8
            return v
        if t == 8:
            n = self.varint()
            v = bytes(self.b[self.p:self.p + n])
            self.p += n
            return v
        if t in (9, 10):
            h = self.b[self.p]
            self.p += 1
            n = h >> 4
            et = h & 0x0F
            if n == 15:
                n = self.varint()
            out = []
            for _ in range(n):
                if et in (1, 2):
                    x = self.b[self.p]
                    self.p += 1
                    out.append(x == 1)
                else:
                    out.append(self.value(et))
            return out
        if t == 11:
            n = self.varint()
            if n == 0:
                return {}
            h = self.b[self.p]
            self.p += 1
            kt, vt = h >> 4, h & 0x0F
            return {self.value(kt): self.value(vt) for _ in range(n)}
        if t == 12:
            return self.struct()
        raise ValueError("thrift type %d" % t)

    def struct(self):
        out = {}
        last = 0
        while True:
            h = self.b[self.p]
            self.p += 1
            if h == 0:
                return out
            t = h & 0x0F
            d = h >> 4
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.value(t)


# ---------------- Snappy (raw block) ----------------

def snappy_decompress(src):
    src = memoryview(src)
    p = 0
    n = 0
    s = 0
    while True:
        x = src[p]
        p += 1
        n |= (x & 0x7F) << s
        if x < 0x80:
            break
        s += 7
    out = bytearray()
    L = len(src)
    while p < L:
        tag = src[p]
        p += 1
        kind = tag & 3
        if kind == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(src[p:p + nb], "little")
                p += nb
            ln += 1
            out += src[p:p + ln]
            p += ln
            continue
        if kind == 1:
            ln = 4 + ((tag >> 2) & 7)
            off = ((tag >> 5) << 8) | src[p]
            p += 1
        elif kind == 2:
            ln = 1 + (tag >> 2)
            off = src[p] | (src[p + 1] << 8)
            p += 2
        else:
            ln = 1 + (tag >> 2)
            off = int.from_bytes(src[p:p + 4], "little")
            p += 4
        if off == 0 or off > len(out):
            raise ValueError("bad snappy offset")
        start = len(out) - off
        if off >= ln:
            out += out[start:start + ln]
        else:
            pat = bytes(out[start:])
            out += (pat * (ln // off + 1))[:ln]
    if len(out) != n:
        raise ValueError("snappy length mismatch %d vs %d" % (len(out), n))
    return bytes(out)


_zstd = None


def decompress(codec, data, usize):
    if codec == 0:
        return bytes(data)
    if codec == 1:
        return snappy_decompress(data)
    if codec == 2:
        return zlib.decompress(bytes(data), 47)
    if codec == 6:
        global _zstd
        if _zstd is None:
            import zstandard
            _zstd = zstandard.ZstdDecompressor()
        return _zstd.decompress(bytes(data), max_output_size=usize)
    raise ValueError("codec %d unsupported" % codec)


# ---------------- RLE / bit-packed hybrid ----------------

def rle_hybrid(buf, p, end, bw, count):
    out = []
    if bw == 0:
        return [0] * count
    bytew = (bw + 7) // 8
    mask = (1 << bw) - 1
    while len(out) < count and p < end:
        h = 0
        s = 0
        while True:
            x = buf[p]
            p += 1
            h |= (x & 0x7F) << s
            if x < 0x80:
                break
            s += 7
        if h & 1:
            groups = h >> 1
            nbytes = groups * bw
            v = int.from_bytes(buf[p:p + nbytes], "little")
            p += nbytes
            for _ in range(groups * 8):
                out.append(v & mask)
                v >>= bw
        else:
            rep = h >> 1
            v = int.from_bytes(buf[p:p + bytew], "little")
            p += bytew
            out.extend([v] * rep)
    return out[:count]


def bitwidth(maxv):
    return maxv.bit_length()


# ---------------- PLAIN values ----------------

def plain_values(ptype, buf, p, n, tlen=None):
    if ptype == 0:  # BOOLEAN
        out = []
        for i in range(n):
            out.append(bool((buf[p + (i >> 3)] >> (i & 7)) & 1))
        return out
    if ptype == 1:
        return list(struct.unpack_from("<%di" % n, buf, p))
    if ptype == 2:
        return list(struct.unpack_from("<%dq" % n, buf, p))
    if ptype == 4:
        return list(struct.unpack_from("<%df" % n, buf, p))
    if ptype == 5:
        return list(struct.unpack_from("<%dd" % n, buf, p))
    if ptype == 6:
        out = []
        for _ in range(n):
            ln = struct.unpack_from("<I", buf, p)[0]
            p += 4
            out.append(bytes(buf[p:p + ln]))
            p += ln
        return out
    if ptype == 7:
        out = []
        for _ in range(n):
            out.append(bytes(buf[p:p + tlen]))
            p += tlen
        return out
    raise ValueError("ptype %d" % ptype)


# ---------------- File / schema ----------------

class PFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(size - 8)
            tail = f.read(8)
            if tail[4:] != b"PAR1":
                raise ValueError("not parquet")
            flen = struct.unpack("<I", tail[:4])[0]
            f.seek(size - 8 - flen)
            self.meta = TReader(f.read(flen)).struct()
        self.num_rows = self.meta[3]
        self._schema()

    def _schema(self):
        els = self.meta[2]
        self.leaves = []  # dicts: path, ptype, max_def, max_rep, conv, logical
        idx = [1]

        def walk(prefix, d, r):
            e = els[idx[0]]
            idx[0] += 1
            name = e[4].decode()
            rep = e.get(3, 0)  # 0 REQUIRED 1 OPTIONAL 2 REPEATED
            nd = d + (1 if rep in (1, 2) else 0)
            nr = r + (1 if rep == 2 else 0)
            path = prefix + (name,)
            nc = e.get(5, 0)
            if nc:
                for _ in range(nc):
                    walk(path, nd, nr)
            else:
                self.leaves.append({"path": path, "ptype": e.get(1), "tlen": e.get(2),
                                    "max_def": nd, "max_rep": nr, "conv": e.get(6),
                                    "logical": e.get(10)})

        root = els[0]
        for _ in range(root.get(5, 0)):
            walk((), 0, 0)

    def leaf(self, dotted):
        for i, lf in enumerate(self.leaves):
            if ".".join(lf["path"]) == dotted:
                return i, lf
        raise KeyError(dotted)

    def chunks(self, leaf_index):
        for rg in self.meta[4]:
            cc = rg[1][leaf_index]
            md = cc[3]
            yield rg[3], md  # rows in group, ColumnMetaData

    def iter_pages(self, leaf_index):
        """Yield per page: (def_levels or None, rep_levels or None, values list (non-null only))."""
        lf = self.leaves[leaf_index]
        ptype, tlen = lf["ptype"], lf["tlen"]
        md_ = lf["max_def"]
        mr_ = lf["max_rep"]
        with open(self.path, "rb") as f:
            for nrows, md in self.chunks(leaf_index):
                codec = md[4]
                start = md.get(11) if md.get(11) is not None else md[9]
                if md.get(11) is not None and md[9] < start:
                    start = md[9]
                total = md[7]
                f.seek(start)
                raw = f.read(total)
                p = 0
                dictionary = None
                seen_vals = 0
                nvals = md[5]
                while p < len(raw) and seen_vals < nvals:
                    tr = TReader(raw, p)
                    ph = tr.struct()
                    p = tr.p
                    ptype_page = ph[1]
                    usize = ph[2]
                    csize = ph[3]
                    body = raw[p:p + csize]
                    p += csize
                    if ptype_page == 2:  # dictionary
                        dh = ph[7]
                        data = decompress(codec, body, usize)
                        dictionary = plain_values(ptype, data, 0, dh[1], tlen)
                        continue
                    if ptype_page == 0:  # data page v1
                        dh = ph[5]
                        n = dh[1]
                        enc = dh[2]
                        data = decompress(codec, body, usize)
                        q = 0
                        rl = dl = None
                        if mr_ > 0:
                            ln = struct.unpack_from("<I", data, q)[0]
                            q += 4
                            rl = rle_hybrid(data, q, q + ln, bitwidth(mr_), n)
                            q += ln
                        if md_ > 0:
                            ln = struct.unpack_from("<I", data, q)[0]
                            q += 4
                            dl = rle_hybrid(data, q, q + ln, bitwidth(md_), n)
                            q += ln
                    elif ptype_page == 3:  # data page v2
                        dh = ph[8]
                        n = dh[1]
                        enc = dh[4]
                        dlen = dh[5]
                        rlen = dh[6]
                        is_comp = dh.get(7, True)
                        rl = dl = None
                        if mr_ > 0:
                            rl = rle_hybrid(body, 0, rlen, bitwidth(mr_), n)
                        if md_ > 0:
                            dl = rle_hybrid(body, rlen, rlen + dlen, bitwidth(md_), n)
                        rest = body[rlen + dlen:]
                        if is_comp:
                            data = decompress(codec, rest, usize - rlen - dlen)
                        else:
                            data = bytes(rest)
                        q = 0
                    else:
                        continue
                    nn = n if dl is None else sum(1 for x in dl if x == md_)
                    if enc == 0:
                        vals = plain_values(ptype, data, q, nn, tlen)
                    elif enc in (2, 8):
                        bw = data[q]
                        idxs = rle_hybrid(data, q + 1, len(data), bw, nn)
                        vals = [dictionary[i] for i in idxs]
                    elif enc == 3 and ptype == 0:
                        vals = [bool(x) for x in rle_hybrid(data, q + 4, len(data), 1, nn)]
                    else:
                        raise ValueError("encoding %d unsupported (%s)" % (enc, lf["path"]))
                    seen_vals += n
                    yield dl, rl, vals

    def read_flat(self, dotted):
        """Flat (non-repeated) column -> list of python values with None for nulls, length num_rows."""
        i, lf = self.leaf(dotted)
        if lf["max_rep"] != 0:
            raise ValueError("not flat")
        out = []
        md_ = lf["max_def"]
        for dl, rl, vals in self.iter_pages(i):
            if dl is None:
                out.extend(vals)
            else:
                it = iter(vals)
                for d in dl:
                    out.append(next(it) if d == md_ else None)
        if len(out) != self.num_rows:
            raise ValueError("row count mismatch for %s: %d vs %d" % (dotted, len(out), self.num_rows))
        return out

    def null_mask_flat(self, dotted):
        """Definition-level null mask only (no value decode is needed, but pages are still parsed)."""
        i, lf = self.leaf(dotted)
        md_ = lf["max_def"]
        out = []
        for dl, rl, vals in self.iter_pages(i):
            if dl is None:
                out.extend([False] * len(vals))
            else:
                out.extend([d != md_ for d in dl])
        return out


def _page_rows(ph):
    if ph[1] == 0:
        return ph[5][1]
    if ph[1] == 3:
        return ph[8][3]
    return 0


def read_rows(pf, dotted, wanted, fn):
    """Flat column; return {row: fn(value)} for rows in `wanted` (set of ints), decoding only pages
    that contain at least one wanted row. Nulls give fn(None)."""
    i, lf = pf.leaf(dotted)
    if lf["max_rep"] != 0:
        raise ValueError("not flat")
    ptype, tlen, md_ = lf["ptype"], lf["tlen"], lf["max_def"]
    out = {}
    row0 = 0
    with open(pf.path, "rb") as f:
        for nrows, md in pf.chunks(i):
            codec = md[4]
            start = md[9]
            if md.get(11) is not None and md[11] < start:
                start = md[11]
            f.seek(start)
            raw = f.read(md[7])
            p = 0
            dict_body = None
            dictionary = None
            r = row0
            while p < len(raw) and r < row0 + nrows:
                tr = TReader(raw, p)
                ph = tr.struct()
                p = tr.p
                csize = ph[3]
                body = raw[p:p + csize]
                p += csize
                if ph[1] == 2:
                    dict_body = (body, ph[2], ph[7][1])
                    dictionary = None
                    continue
                n = _page_rows(ph)
                if n == 0:
                    continue
                hit = [x for x in range(r, r + n) if x in wanted]
                if not hit:
                    r += n
                    continue
                if ph[1] == 0:
                    dh = ph[5]
                    enc = dh[2]
                    data = decompress(codec, body, ph[2])
                    q = 0
                    dl = None
                    if md_ > 0:
                        ln = struct.unpack_from("<I", data, q)[0]
                        q += 4
                        dl = rle_hybrid(data, q, q + ln, bitwidth(md_), n)
                        q += ln
                else:
                    dh = ph[8]
                    enc = dh[4]
                    dlen, rlen = dh[5], dh[6]
                    dl = rle_hybrid(body, rlen, rlen + dlen, bitwidth(md_), n) if md_ > 0 else None
                    rest = body[rlen + dlen:]
                    data = decompress(codec, rest, ph[2] - rlen - dlen) if dh.get(7, True) else bytes(rest)
                    q = 0
                nn = n if dl is None else sum(1 for x in dl if x == md_)
                if enc == 0:
                    vals = plain_values(ptype, data, q, nn, tlen)
                elif enc in (2, 8):
                    if dictionary is None:
                        b, us, nv = dict_body
                        dictionary = plain_values(ptype, decompress(codec, b, us), 0, nv, tlen)
                    bw = data[q]
                    vals = [dictionary[k] for k in rle_hybrid(data, q + 1, len(data), bw, nn)]
                elif enc == 3 and ptype == 0:
                    vals = [bool(x) for x in rle_hybrid(data, q + 4, len(data), 1, nn)]
                else:
                    raise ValueError("encoding %d" % enc)
                it = iter(vals)
                for k in range(n):
                    v = next(it) if (dl is None or dl[k] == md_) else None
                    if (r + k) in wanted:
                        out[r + k] = fn(v)
                r += n
            if r != row0 + nrows:
                raise ValueError("row accounting mismatch in %s" % dotted)
            row0 += nrows
    return out

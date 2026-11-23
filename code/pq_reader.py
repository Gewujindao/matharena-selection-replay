"""Minimal pure-Python Parquet reader for flat files (no pyarrow in the pinned environment, which is kept unchanged).

Scope: flat schemas (no nesting), physical types BOOLEAN / INT32 / INT64 / DOUBLE / BYTE_ARRAY, data pages v1 and v2,
PLAIN, PLAIN_DICTIONARY / RLE_DICTIONARY and RLE (booleans) encodings, codecs UNCOMPRESSED / SNAPPY / GZIP.
Anything else raises NotImplementedError instead of guessing.  The file is only parsed, never executed.
"""
import struct
import zlib

# ----------------------------------------------------------------------------- thrift compact protocol

CT_STOP, CT_TRUE, CT_FALSE, CT_BYTE, CT_I16, CT_I32, CT_I64, CT_DOUBLE, CT_BINARY, CT_LIST, CT_SET, CT_MAP, \
    CT_STRUCT = range(13)


class ThriftReader:
    """Decodes thrift compact structs into {field_id: value} dicts (structs nested the same way)."""

    def __init__(self, buf, pos=0):
        self.buf = buf
        self.pos = pos

    def byte(self):
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def varint(self):
        shift = 0
        out = 0
        while True:
            b = self.byte()
            out |= (b & 0x7F) << shift
            if not b & 0x80:
                return out
            shift += 7

    def zigzag(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self):
        n = self.varint()
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return out

    def value(self, t):
        if t == CT_TRUE:
            return True
        if t == CT_FALSE:
            return False
        if t == CT_BYTE:
            b = self.byte()
            return b - 256 if b > 127 else b
        if t in (CT_I16, CT_I32, CT_I64):
            return self.zigzag()
        if t == CT_DOUBLE:
            v = struct.unpack_from("<d", self.buf, self.pos)[0]
            self.pos += 8
            return v
        if t == CT_BINARY:
            return self.binary()
        if t in (CT_LIST, CT_SET):
            h = self.byte()
            size, et = h >> 4, h & 0x0F
            if size == 15:
                size = self.varint()
            if et in (CT_TRUE, CT_FALSE):           # bools inside containers are one byte each
                return [self.byte() == CT_TRUE for _ in range(size)]
            return [self.value(et) for _ in range(size)]
        if t == CT_MAP:
            size = self.varint()
            if size == 0:
                return {}
            kv = self.byte()
            kt, vt = kv >> 4, kv & 0x0F
            return {self.value(kt): self.value(vt) for _ in range(size)}
        if t == CT_STRUCT:
            return self.struct()
        raise ValueError(f"unknown thrift compact type {t}")

    def struct(self):
        out = {}
        last = 0
        while True:
            h = self.byte()
            t = h & 0x0F
            if t == CT_STOP:
                return out
            delta = h >> 4
            fid = last + delta if delta else self.zigzag()
            out[fid] = self.value(t)
            last = fid


# ----------------------------------------------------------------------------- codecs


def snappy_decompress(src):
    """Raw (unframed) snappy block, as used by Parquet."""
    src = memoryview(src)
    pos = 0
    n = 0
    shift = 0
    while True:
        b = src[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            break
        shift += 7
    out = bytearray()
    end = len(src)
    while pos < end:
        tag = src[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:                                   # literal
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(src[pos:pos + nb], "little")
                pos += nb
            ln += 1
            out += src[pos:pos + ln]
            pos += ln
            continue
        if kind == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | src[pos]
            pos += 1
        elif kind == 2:
            ln = (tag >> 2) + 1
            off = src[pos] | (src[pos + 1] << 8)
            pos += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(src[pos:pos + 4], "little")
            pos += 4
        if off == 0 or off > len(out):
            raise ValueError("corrupt snappy stream (bad copy offset)")
        start = len(out) - off
        if off >= ln:
            out += out[start:start + ln]
        else:                                           # overlapping copy: repeat the period
            chunk = out[start:]
            reps, rem = divmod(ln, off)
            out += chunk * reps + chunk[:rem]
    if len(out) != n:
        raise ValueError(f"snappy length mismatch: header {n}, got {len(out)}")
    return bytes(out)


def decompress(codec, data, uncompressed_size):
    if codec == 0:
        out = bytes(data)
    elif codec == 1:
        out = snappy_decompress(data)
    elif codec == 2:
        out = zlib.decompress(bytes(data), 47)          # gzip or zlib header
    else:
        raise NotImplementedError(f"parquet codec {codec} not supported by this reader")
    if uncompressed_size is not None and len(out) != uncompressed_size:
        raise ValueError(f"decompressed size {len(out)} != header {uncompressed_size}")
    return out


# ----------------------------------------------------------------------------- encodings


def bit_width(max_value):
    return max_value.bit_length()


def rle_hybrid(buf, pos, end, width, count):
    """Decode `count` values of the RLE / bit-packed hybrid in buf[pos:end]."""
    out = []
    if width == 0:
        return [0] * count
    nbytes = (width + 7) // 8
    mask = (1 << width) - 1
    while len(out) < count:
        if pos >= end:
            raise ValueError("RLE hybrid stream ended early")
        h = 0
        shift = 0
        while True:
            b = buf[pos]
            pos += 1
            h |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        if h & 1:                                       # bit-packed groups of 8
            groups = h >> 1
            nb = groups * width
            chunk = int.from_bytes(buf[pos:pos + nb], "little")
            pos += nb
            for i in range(groups * 8):
                out.append((chunk >> (i * width)) & mask)
        else:
            run = h >> 1
            v = int.from_bytes(buf[pos:pos + nbytes], "little")
            pos += nbytes
            out.extend([v] * run)
    return out[:count]


def plain(buf, ptype, count):
    if ptype == 6:                                      # BYTE_ARRAY
        out = []
        pos = 0
        for _ in range(count):
            ln = int.from_bytes(buf[pos:pos + 4], "little")
            pos += 4
            out.append(bytes(buf[pos:pos + ln]))
            pos += ln
        return out
    if ptype == 2:
        return list(struct.unpack_from(f"<{count}q", buf, 0))
    if ptype == 1:
        return list(struct.unpack_from(f"<{count}i", buf, 0))
    if ptype == 5:
        return list(struct.unpack_from(f"<{count}d", buf, 0))
    if ptype == 0:
        bits = int.from_bytes(buf[:(count + 7) // 8], "little")
        return [bool((bits >> i) & 1) for i in range(count)]
    raise NotImplementedError(f"physical type {ptype} not supported")


# ----------------------------------------------------------------------------- file


class ParquetFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(size - 8)
            tail = f.read(8)
            if tail[4:] != b"PAR1":
                raise ValueError("not a parquet file (bad tail magic)")
            flen = int.from_bytes(tail[:4], "little")
            f.seek(size - 8 - flen)
            footer = f.read(flen)
            f.seek(0)
            if f.read(4) != b"PAR1":
                raise ValueError("not a parquet file (bad head magic)")
        self.size = size
        self.meta = ThriftReader(footer).struct()
        schema = self.meta[2]
        root, leaves = schema[0], schema[1:]
        if any(5 in el and el[5] for el in leaves):
            raise NotImplementedError("nested schema not supported")
        self.columns = [el[4].decode("utf8") for el in leaves]
        self.ptype = {el[4].decode("utf8"): el.get(1) for el in leaves}
        self.rep = {el[4].decode("utf8"): el.get(3, 0) for el in leaves}
        if any(r == 2 for r in self.rep.values()):
            raise NotImplementedError("repeated columns not supported")
        self.num_rows = self.meta[3]
        self.row_groups = self.meta[4]
        self.created_by = self.meta.get(6, b"").decode("utf8", "replace")
        self.kv = {k[1].decode(): (k.get(2) or b"").decode("utf8", "replace") for k in self.meta.get(5, [])}

    def describe(self):
        rg = []
        for g in self.row_groups:
            cols = {}
            for cc in g[1]:
                md = cc[3]
                name = b".".join(md[3]).decode()
                cols[name] = dict(codec=md[4], encodings=md[2], num_values=md[5], uncompressed=md[6],
                                  compressed=md[7])
            rg.append(dict(num_rows=g[3], columns=cols))
        return dict(num_rows=self.num_rows, created_by=self.created_by, columns=self.columns,
                    ptype=self.ptype, repetition=self.rep, row_groups=rg)

    def read_column(self, name, rows=None, decode_utf8=True):
        """All values of column `name` (None for nulls), or only the rows in the set `rows` (global row numbers;
        others come back as the sentinel SKIPPED and their pages are not decompressed)."""
        ptype = self.ptype[name]
        max_def = 1 if self.rep[name] == 1 else 0
        out = []
        base = 0
        with open(self.path, "rb") as f:
            for g in self.row_groups:
                cc = next(c for c in g[1] if b".".join(c[3][3]).decode() == name)
                md = cc[3]
                codec = md[4]
                start = md.get(11) if md.get(11) is not None else md[9]
                if md.get(11) is not None and md[9] < start:
                    start = md[9]
                f.seek(start)
                chunk = f.read(md[7])
                pos = 0
                dictionary = None
                got = 0
                while got < md[5]:
                    tr = ThriftReader(chunk, pos)
                    ph = tr.struct()
                    pos = tr.pos
                    ptyp, usize, csize = ph[1], ph[2], ph[3]
                    body = chunk[pos:pos + csize]
                    pos += csize
                    if ptyp == 2:                                   # dictionary page
                        dh = ph[7]
                        raw = decompress(codec, body, usize)
                        dictionary = plain(raw, ptype, dh[1])
                        continue
                    if ptyp == 0:
                        dh = ph[5]
                        nval = dh[1]
                        lo, hi = base + got, base + got + nval
                        if rows is not None and not any(lo <= r < hi for r in rows):
                            out.extend([SKIPPED] * nval)
                            got += nval
                            continue
                        raw = decompress(codec, body, usize)
                        p = 0
                        if max_def:
                            ln = int.from_bytes(raw[p:p + 4], "little")
                            p += 4
                            defs = rle_hybrid(raw, p, p + ln, 1, nval)
                            p += ln
                        else:
                            defs = [1] * nval
                        vals_buf = raw[p:]
                        enc = dh[2]
                    elif ptyp == 3:
                        dh = ph[8]
                        nval, dlen, rlen = dh[1], dh[5], dh[6]
                        lo, hi = base + got, base + got + nval
                        if rows is not None and not any(lo <= r < hi for r in rows):
                            out.extend([SKIPPED] * nval)
                            got += nval
                            continue
                        if rlen:
                            raise NotImplementedError("repetition levels present")
                        defs = rle_hybrid(body, 0, dlen, 1, nval) if max_def else [1] * nval
                        rest = body[rlen + dlen:]
                        comp = dh.get(7, True)
                        vals_buf = decompress(codec, rest, usize - rlen - dlen) if comp else bytes(rest)
                        enc = dh[4]
                    else:
                        raise NotImplementedError(f"page type {ptyp}")
                    nnon = sum(defs)
                    if enc == 0:
                        vals = plain(vals_buf, ptype, nnon)
                    elif enc in (2, 8):
                        if dictionary is None:
                            raise ValueError("dictionary-encoded page without dictionary")
                        w = vals_buf[0]
                        idx = rle_hybrid(vals_buf, 1, len(vals_buf), w, nnon)
                        vals = [dictionary[i] for i in idx]
                    elif enc == 3 and ptype == 0:               # RLE booleans: 4-byte length prefix
                        ln = int.from_bytes(vals_buf[:4], "little")
                        vals = [bool(v) for v in rle_hybrid(vals_buf, 4, 4 + ln, 1, nnon)]
                    else:
                        raise NotImplementedError(f"encoding {enc} for type {ptype}")
                    it = iter(vals)
                    out.extend(next(it) if d else None for d in defs)
                    got += nval
                base += g[3]
        if ptype == 6 and decode_utf8:
            out = [v.decode("utf8") if isinstance(v, bytes) else v for v in out]
        if len(out) != self.num_rows:
            raise ValueError(f"column {name}: {len(out)} values for {self.num_rows} rows")
        return out


class _Skipped:
    def __repr__(self):
        return "SKIPPED"


SKIPPED = _Skipped()

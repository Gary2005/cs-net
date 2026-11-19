"""
Lossless columnar binary codec for extracted demo data.

Why not plain JSON
------------------
A demo's per-tick block is ~10^6 scalars, most of them smooth float32 signals
that arrived from the game as float32 in the first place.  Written as JSON text
each one costs 5-20 bytes of ASCII; written as a typed column it costs 4 bytes,
and 4 bytes of *byte-planed* float32 is exactly what zstd is good at.

The format is self-describing (no schema on the reading side), so `loads(dumps(x))`
is a pure round trip.

Guarantees
----------
`loads(dumps(x)) == x` is exact, not approximate:

* float columns are stored as float32 **only** when every value survives the
  f64 -> f32 -> f64 trip *bit for bit* (NaN payloads included, compared on the
  raw bit pattern).  Otherwise the column silently widens to float64.
* integer columns are stored as delta + zigzag varint, which is exact for
  arbitrary-precision Python ints.
* nothing is quantised, rounded, clamped or dropped.  There is no lossy option
  in this module at all.

Layout
------
    container = b"CSZ5" | u8 version | u32 raw_len | zstd_frame
    raw_len lets `load` tell a truncated/corrupt file from a valid one.

    generic value tags:
        00 None          01 False        02 True        03 int
        04 float64       05 str          06 bytes       07 list (generic)
        08 dict
    typed array tags (these are where the compression comes from):
        0A float32[]     0B float64[]    09 int[] (delta+varint)
        0C bool[]        10 empty list   0F bytes[]
        0E str[] (with a dictionary when it pays off)
        0D table: list[dict] with an identical key set, stored column-major
"""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised by tests only when zstd is present
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

MAGIC = b"CSZ5"
VERSION = 2  # 2: 2nd-order delta fixed (v1 had a wrong initial condition)

# ── tags ─────────────────────────────────────────────────────────────────────

T_NONE = 0x00
T_FALSE = 0x01
T_TRUE = 0x02
T_INT = 0x03
T_FLOAT = 0x04
T_STR = 0x05
T_BYTES = 0x06
T_LIST = 0x07
T_DICT = 0x08
T_IARR = 0x09
T_F32 = 0x0A
T_F64 = 0x0B
T_BITS = 0x0C
T_TABLE = 0x0D
T_STRDICT = 0x0E
T_BLOBS = 0x0F
T_EMPTY = 0x10
T_F32D = 0x11  # float32, 1st-order uint32 delta + byte-plane
T_F64D = 0x12  # float64, 1st-order uint64 delta + byte-plane
T_F32D2 = 0x13 # float32, 2nd-order delta (better for positions)
T_F64D2 = 0x14 # float64, 2nd-order delta

DEFAULT_LEVEL = 19


# ── primitives ───────────────────────────────────────────────────────────────


def _uvarint(n: int) -> bytes:
    if n < 0:
        raise ValueError("uvarint needs a non-negative int")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _zigzag(n: int) -> int:
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _unzigzag(n: int) -> int:
    return (n >> 1) if not (n & 1) else -((n + 1) >> 1)


class _Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.buf):
            raise ValueError("truncated payload")
        out = self.buf[self.pos:end]
        self.pos = end
        return out

    def u8(self) -> int:
        if self.pos >= len(self.buf):
            raise ValueError("truncated payload")
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def uvarint(self) -> int:
        shift = 0
        out = 0
        while True:
            if self.pos >= len(self.buf):
                raise ValueError("truncated varint")
            b = self.buf[self.pos]
            self.pos += 1
            out |= (b & 0x7F) << shift
            if not (b & 0x80):
                return out
            shift += 7

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]


def _w(out: bytearray, payload: bytes) -> None:
    out += _uvarint(len(payload))
    out += payload


# ── float column helpers ─────────────────────────────────────────────────────


def _f32_exact_bits(vals: list[float]) -> bool:
    """True when every value survives f64 -> f32 -> f64 bit-for-bit."""
    a = np.asarray(vals, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        b = a.astype(np.float32).astype(np.float64)
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def _plane_f32(arr: np.ndarray) -> bytes:
    """(N,) float32 -> byte-planed bytes: all byte0, then all byte1, ..."""
    return np.ascontiguousarray(arr).view(np.uint8).reshape(-1, 4).T.tobytes()


def _plane_f64(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).view(np.uint8).reshape(-1, 8).T.tobytes()


def _plane_delta(arr: np.ndarray, idt, order: int = 1) -> bytes:
    """Byte-planed modular difference of the raw bit patterns.

    Differencing the *bit pattern* (as an unsigned int) rather than the value is
    exactly invertible, and for smooth signals it collapses the high bytes into
    long identical runs.  `order=1` is the usual first difference; `order=2`
    (second difference) is measurably better for position tracks, which are
    near-constant-acceleration over a tick, and measurably worse for angles and
    velocities.  The caller picks per column by trial compression.
    """
    u = np.ascontiguousarray(arr).view(idt)
    n = len(u)
    d = np.empty_like(u)
    if order == 1:
        d[0] = u[0]
        np.subtract(u[1:], u[:-1], out=d[1:])
    else:
        # Decoding is cumsum applied twice, which returns u only if the first
        # cumulative sum is the *first difference* of u.  That pins the second
        # element to d[1] = u[1] - 2*u[0]; with d[1] = u[1] the round trip
        # comes back as 2k*u[0] + u[k] (correct only when u[0] == 0).
        d[0] = u[0]
        if n > 1:
            d[1:2] = u[1:2]
            d[1:2] -= u[0:1]
            d[1:2] -= u[0:1]
        if n > 2:
            # (u2-u1) - (u1-u0), all in modular unsigned arithmetic
            np.subtract(u[2:], u[1:-1], out=d[2:])
            d[2:] -= np.subtract(u[1:-1], u[:-2])
    width = idt().itemsize
    return np.ascontiguousarray(d).view(np.uint8).reshape(-1, width).T.tobytes()


def _unplane_delta(buf: bytes, n: int, width: int, idt, order: int = 1) -> np.ndarray:
    d = _unplane(buf, n, width, idt)
    u = np.cumsum(d, dtype=idt)
    if order == 2:
        u = np.cumsum(u, dtype=idt)
    return u.view(np.float32 if width == 4 else np.float64)


def _unplane(buf: bytes, n: int, width: int, dtype) -> np.ndarray:
    if len(buf) != n * width:
        raise ValueError(f"plane size mismatch: {len(buf)} != {n * width}")
    a = np.frombuffer(buf, dtype=np.uint8).reshape(width, n).T
    return np.ascontiguousarray(a).view(dtype).reshape(n)


# ── encoder ──────────────────────────────────────────────────────────────────


def _enc(out: bytearray, v: Any) -> None:
    if v is None:
        out.append(T_NONE)
        return
    tv = type(v)
    if tv is bool:
        out.append(T_TRUE if v else T_FALSE)
        return
    if tv is int:
        out.append(T_INT)
        out += _uvarint(_zigzag(v))
        return
    if tv is float:
        out.append(T_FLOAT)
        out += struct.pack("<d", v)
        return
    if tv is str:
        out.append(T_STR)
        _w(out, v.encode("utf-8"))
        return
    if tv is bytes or tv is bytearray:
        out.append(T_BYTES)
        _w(out, bytes(v))
        return
    if tv is dict:
        out.append(T_DICT)
        out += _uvarint(len(v))
        for k, val in v.items():
            if type(k) is not str:
                raise TypeError(f"dict keys must be str, got {type(k)}")
            _w(out, k.encode("utf-8"))
            _enc(out, val)
        return
    if tv is list:
        _enc_list(out, v)
        return
    if tv is tuple:
        _enc_list(out, list(v))
        return
    if isinstance(v, (np.integer,)):
        _enc(out, int(v))
        return
    if isinstance(v, (np.floating,)):
        _enc(out, float(v))
        return
    if isinstance(v, np.bool_):
        _enc(out, bool(v))
        return
    raise TypeError(f"cannot encode {type(v)}")


def _enc_list(out: bytearray, v: list) -> None:
    n = len(v)
    if n == 0:
        out.append(T_EMPTY)
        return

    types = set(map(type, v))

    # list of dicts with an identical key set -> store column-major
    if types == {dict} and n > 1:
        keys = list(v[0].keys())
        if all(len(d) == len(keys) and all(k in d for k in keys) for d in v):
            out.append(T_TABLE)
            out += _uvarint(len(keys))
            for k in keys:
                _w(out, k.encode("utf-8"))
            for k in keys:
                _enc_list(out, [d[k] for d in v])
            return

    if types == {bool}:
        out.append(T_BITS)
        out += _uvarint(n)
        out += np.packbits(np.asarray(v, dtype=np.bool_)).tobytes()
        return

    if types == {int}:
        out.append(T_IARR)
        out += _uvarint(n)
        prev = 0
        for x in v:
            out += _uvarint(_zigzag(x - prev))
            prev = x
        return

    if types == {float}:
        f32 = _f32_exact_bits(v)
        a = np.asarray(v, dtype=np.float32 if f32 else np.float64)
        tag, payload = _best_float_plane(a, f32)
        out.append(tag)
        out += _uvarint(n)
        out += payload
        return

    if types == {bytes}:
        out.append(T_BLOBS)
        out += _uvarint(n)
        for b in v:
            out += _uvarint(len(b))
        for b in v:
            out += b
        return

    if types == {str}:
        uniq = sorted(set(v))
        if len(uniq) < n:
            idx = {s: i for i, s in enumerate(uniq)}
            out.append(T_STRDICT)
            out += _uvarint(n)
            out += _uvarint(len(uniq))
            for s in uniq:
                _w(out, s.encode("utf-8"))
            for s in v:
                out += _uvarint(idx[s])
            return

    out.append(T_LIST)
    out += _uvarint(n)
    for x in v:
        _enc(out, x)


# Trial compression picks the cheapest exactly-invertible form per column.
# Level 1 on a ~20 KB column is well under a millisecond, and the choice is made
# on real bytes rather than a heuristic (zero-byte counts turn out to predict it
# badly: yaw has *fewer* zero bytes with the 2nd difference yet is 6% larger).
_TRIAL = zstd.ZstdCompressor(level=1) if zstd is not None else None


def _best_float_plane(a: np.ndarray, f32: bool) -> tuple[int, bytes]:
    plain_tag = T_F32 if f32 else T_F64
    plain = _plane_f32(a) if f32 else _plane_f64(a)
    if len(a) < 64 or _TRIAL is None:
        return plain_tag, plain
    idt = np.uint32 if f32 else np.uint64
    cands = [
        (plain_tag, plain),
        (T_F32D if f32 else T_F64D, _plane_delta(a, idt, 1)),
        (T_F32D2 if f32 else T_F64D2, _plane_delta(a, idt, 2)),
    ]
    return min(cands, key=lambda c: len(_TRIAL.compress(c[1])))


# ── decoder ──────────────────────────────────────────────────────────────────


def _dec(r: _Reader) -> Any:
    tag = r.u8()
    if tag == T_NONE:
        return None
    if tag == T_FALSE:
        return False
    if tag == T_TRUE:
        return True
    if tag == T_INT:
        return _unzigzag(r.uvarint())
    if tag == T_FLOAT:
        return r.f64()
    if tag == T_STR:
        return r.take(_uvarint_from(r)).decode("utf-8")
    if tag == T_BYTES:
        return r.take(_uvarint_from(r))
    if tag == T_DICT:
        n = r.uvarint()
        return dict((r.take(_uvarint_from(r)).decode("utf-8"), _dec(r)) for _ in range(n))
    if tag == T_LIST:
        return [_dec(r) for _ in range(r.uvarint())]
    if tag == T_EMPTY:
        return []
    if tag == T_IARR:
        n = r.uvarint()
        out = []
        acc = 0
        for _ in range(n):
            acc += _unzigzag(r.uvarint())
            out.append(acc)
        return out
    if tag == T_BITS:
        n = r.uvarint()
        raw = r.take((n + 7) // 8)
        return [bool(b) for b in np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n]]
    if tag in (T_F32, T_F64, T_F32D, T_F64D, T_F32D2, T_F64D2):
        n = r.uvarint()
        width = 4 if tag in (T_F32, T_F32D, T_F32D2) else 8
        buf = r.take(n * width)
        if tag in (T_F32D, T_F64D):
            arr = _unplane_delta(buf, n, width, np.uint32 if width == 4 else np.uint64, 1)
        elif tag in (T_F32D2, T_F64D2):
            arr = _unplane_delta(buf, n, width, np.uint32 if width == 4 else np.uint64, 2)
        else:
            arr = _unplane(buf, n, width, np.float32 if width == 4 else np.float64)
        return [float(x) for x in arr]
    if tag == T_BLOBS:
        n = r.uvarint()
        lens = [r.uvarint() for _ in range(n)]
        return [r.take(L) for L in lens]
    if tag == T_STRDICT:
        n = r.uvarint()
        k = r.uvarint()
        table = [r.take(_uvarint_from(r)).decode("utf-8") for _ in range(k)]
        return [table[r.uvarint()] for _ in range(n)]
    if tag == T_TABLE:
        nk = r.uvarint()
        keys = [r.take(_uvarint_from(r)).decode("utf-8") for _ in range(nk)]
        cols = [_dec(r) for _ in range(nk)]
        n = len(cols[0]) if cols else 0
        return [dict(zip(keys, (cols[j][i] for j in range(nk)))) for i in range(n)]
    raise ValueError(f"unknown tag 0x{tag:02x} at {r.pos - 1}")


def _uvarint_from(r: _Reader) -> int:
    return r.uvarint()


# ── public API ───────────────────────────────────────────────────────────────


def dumps(data: Any, level: int = DEFAULT_LEVEL) -> bytes:
    """Serialise to a compressed, self-describing container."""
    raw = bytearray()
    _enc(raw, data)
    body = bytes(raw)
    header = MAGIC + bytes([VERSION]) + struct.pack("<I", len(body))
    if zstd is None:
        return header + body
    cctx = zstd.ZstdCompressor(level=level, write_checksum=True, write_content_size=True)
    return header + cctx.compress(body)


def loads(blob: bytes, verify: bool = True) -> Any:
    """Inverse of :func:`dumps`."""
    if not blob.startswith(MAGIC):
        raise ValueError("not a CSZ5 container")
    ver = blob[4]
    if ver != VERSION:
        raise ValueError(f"unsupported container version {ver}")
    (raw_len,) = struct.unpack("<I", blob[5:9])
    body = blob[9:]
    if zstd is not None and raw_len:
        try:
            body = zstd.ZstdDecompressor().decompress(body, max_output_size=raw_len or None)
        except zstd.ZstdError:
            body = body  # stored uncompressed
    if verify and len(body) != raw_len:
        raise ValueError(f"payload length {len(body)} != declared {raw_len}")
    r = _Reader(body)
    val = _dec(r)
    if verify and r.pos != len(body):
        raise ValueError(f"{len(body) - r.pos} trailing bytes")
    return val


def save(data: Any, path: str | Path, level: int = DEFAULT_LEVEL) -> Path:
    """Write `data` to `path` (a single `.csz` file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = dumps(data, level=level)
    p.write_bytes(blob)
    return p


def load(path: str | Path, verify: bool = True) -> Any:
    """Read back a file written by :func:`save`."""
    return loads(Path(path).read_bytes(), verify=verify)


def raw_blob(data: Any) -> bytes:
    """Uncompressed body — for inspecting what actually costs bytes."""
    raw = bytearray()
    _enc(raw, data)
    return bytes(raw)


# ─ JSON interchange (human-readable, not the compact path) ───────────────────


class _JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            return {"__b64__": base64.b64encode(o).decode("ascii")}
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _from_jsonable(o):
    if isinstance(o, dict):
        if set(o.keys()) == {"__b64__"}:
            return base64.b64decode(o["__b64__"])
        return {k: _from_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_from_jsonable(v) for v in o]
    return o


def save_json(data: Any, path: str | Path, indent: int | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".gz":
        import gzip

        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(data, f, cls=_JsonEncoder, ensure_ascii=False, indent=indent)
    else:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=_JsonEncoder, ensure_ascii=False, indent=indent)
    return p


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if p.suffix == ".gz":
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as f:
            return _from_jsonable(json.load(f))
    with open(p, "r", encoding="utf-8") as f:
        return _from_jsonable(json.load(f))


# ── equality that survives NaN ───────────────────────────────────────────────


def equal(a: Any, b: Any, path: str = "$") -> tuple[bool, str]:
    """Structural equality where NaN == NaN (by bit pattern for floats).

    Returns ``(ok, where)``; `where` points at the first difference.
    """
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not (
            isinstance(a, bool) or isinstance(b, bool)
        ):
            pass
        else:
            return False, f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return False, f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            ok, w = equal(a[k], b[k], f"{path}.{k}")
            if not ok:
                return ok, w
        return True, path
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, w = equal(x, y, f"{path}[{i}]")
            if not ok:
                return ok, w
        return True, path
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True, path
        if a != b:
            return False, f"{path}: {a!r} vs {b!r}"
        return True, path
    if a != b:
        return False, f"{path}: {a!r} vs {b!r}"
    return True, path
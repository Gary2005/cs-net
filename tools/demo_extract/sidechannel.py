"""
Runs the Go side-channel and parses its binary output.

demoparser2 cannot reach the inferno fire cells (`m_firePositions.*` and
friends) or the smoke voxel journal (`m_VoxelFrameData`) on any of its three
access paths, so those come from demoinfocs-golang through a small helper
binary.  See `sidechannel_go/main.go` for the format writer.

Blob layout (little-endian):
    "CS2SC2" | u8 version | u8 f32_exact | f64 tickrate | u32 voxel_capacity
    u32 n_ticks | i32[n_ticks]
    u32 n_infernos
      u32 entity_id | i32 first_tick | i32 last_tick | i32 inferno_type
      u32 n_changes
        i32 tick | u16 count | count x (f32 px,py,pz, nx,ny,nz, qx,qy,qz, u8 burning)
    u32 n_smokes
      u32 entity_id | i32 first_tick | i32 last_tick
      u32 n_changes
        i32 tick | u16 size | u16 update | u8 mode | u32 len | u8[len]
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

MAGIC = b"CS2SC2"
VERSION = 2
MODE_APPEND, MODE_FULL = "append", "full"
_MODE_BY_ID = {0: MODE_APPEND, 1: MODE_FULL}
_ENTITY_PANICS_RE = re.compile(r"entity_panics=(\d+)")

HERE = Path(__file__).resolve().parent
GO_DIR = HERE / "sidechannel_go"
#: `go build -o` 出来的名字。Windows 上必须带 `.exe`，否则 `is_file()` 找不到，
#: 更糟的是找到了也执行不了（PE 加载器不认无扩展名的文件）。
BINARY_NAME = "cs2sidechannel.exe" if os.name == "nt" else "cs2sidechannel"

# 9 float32 + 1 byte per fire cell
_CELL = struct.Struct("<9fB")
_CELL_SIZE = _CELL.size  # 37


class SideChannelError(RuntimeError):
    pass


def binary_path(build_dir: str | Path | None = None) -> Path:
    """Where the helper binary lives (or should be built to)."""
    if build_dir is not None:
        return Path(build_dir) / BINARY_NAME
    env = os.environ.get("CS2_SIDECHANNEL_BIN")
    if env:
        return Path(env)
    return GO_DIR / BINARY_NAME


def build(build_dir: str | Path | None = None, verbose: bool = False) -> Path:
    """Compile the Go helper. Requires a Go toolchain and network on first run."""
    out = binary_path(build_dir)
    go = shutil.which("go")
    if go is None:
        raise SideChannelError(
            "the Go toolchain is required to build the side-channel helper "
            "(install Go, or set CS2_SIDECHANNEL_BIN to a prebuilt binary)"
        )
    env = dict(os.environ)
    root = HERE.parent.parent  # repository root
    env.setdefault("GOPATH", str(root / ".build" / "gopath"))
    env.setdefault("GOCACHE", str(root / ".build" / "gocache"))
    env.setdefault("GOFLAGS", "-mod=mod")
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [go, "build", "-o", str(out), "."],
        cwd=str(GO_DIR),
        env=env,
        capture_output=not verbose,
        text=True,
    )
    if proc.returncode != 0:
        raise SideChannelError(
            f"go build failed:\n{proc.stderr if proc.stderr else proc.stdout}"
        )
    return out


def ensure_binary(build_dir: str | Path | None = None, auto_build: bool = True) -> Path:
    p = binary_path(build_dir)
    if p.is_file():
        return p
    if not auto_build:
        raise SideChannelError(f"side-channel binary not found at {p}")
    return build(build_dir)


def run(
    demo_path: str | Path,
    ticks: list[int],
    out_path: str | Path,
    build_dir: str | Path | None = None,
    auto_build: bool = True,
    verbose: bool = False,
) -> Path:
    """Run the helper over `demo_path`, sampling exactly `ticks`."""
    exe = ensure_binary(build_dir, auto_build=auto_build)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tick_file = out_path.with_suffix(out_path.suffix + ".ticks")
    tick_file.write_text("\n".join(str(int(t)) for t in ticks))
    try:
        proc = subprocess.run(
            [str(exe), "-demo", str(demo_path), "-ticks", str(tick_file), "-out", str(out_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SideChannelError(
                f"side-channel failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
            )
        # demoinfocs 会为每个损坏的 PacketEntities 包 panic；辅助程序打开了
        # IgnorePacketEntitiesPanic，这些包被丢弃、解析继续，末尾汇总成 entity_panics=N。
        # 丢包意味着那些位置的火/烟状态可能有缺口，所以即使成功也要说出来。
        m = _ENTITY_PANICS_RE.search(proc.stderr)
        if m and int(m.group(1)):
            print(
                f"[sidechannel] 警告：丢弃了 {m.group(1)} 个损坏的 PacketEntities 包，"
                f"这些 tick 附近的火/烟状态可能有缺口",
                file=sys.stderr,
            )
        if verbose and proc.stderr.strip():
            print(f"[sidechannel] {proc.stderr.strip()}", file=sys.stderr)
    finally:
        tick_file.unlink(missing_ok=True)
    return out_path


class _R:
    __slots__ = ("b", "o")

    def __init__(self, b: bytes) -> None:
        self.b = b
        self.o = 0

    def u8(self) -> int:
        v = self.b[self.o]
        self.o += 1
        return v

    def u16(self) -> int:
        (v,) = struct.unpack_from("<H", self.b, self.o)
        self.o += 2
        return v

    def u32(self) -> int:
        (v,) = struct.unpack_from("<I", self.b, self.o)
        self.o += 4
        return v

    def i32(self) -> int:
        (v,) = struct.unpack_from("<i", self.b, self.o)
        self.o += 4
        return v

    def f64(self) -> float:
        (v,) = struct.unpack_from("<d", self.b, self.o)
        self.o += 8
        return v

    def take(self, n: int) -> bytes:
        v = self.b[self.o:self.o + n]
        if len(v) != n:
            raise SideChannelError("truncated side-channel blob")
        self.o += n
        return v


def parse(path: str | Path) -> dict:
    """Parse the helper's blob into plain Python structures."""
    blob = Path(path).read_bytes()
    if not blob.startswith(MAGIC):
        raise SideChannelError("bad side-channel magic")
    r = _R(blob)
    r.take(len(MAGIC))
    version = r.u8()
    if version != VERSION:
        raise SideChannelError(f"unsupported side-channel version {version}")
    f32_exact = bool(r.u8())
    tickrate = r.f64()
    voxel_capacity = r.u32()

    n_ticks = r.u32()
    ticks = [r.i32() for _ in range(n_ticks)]

    infernos = []
    for _ in range(r.u32()):
        eid = r.u32()
        first = r.i32()
        last = r.i32()
        typ = r.i32()
        changes = []
        for _ in range(r.u32()):
            tick = r.i32()
            count = r.u16()
            pos: list[float] = []
            nrm: list[float] = []
            par: list[float] = []
            burning: list[bool] = []
            for _i in range(count):
                vals = _CELL.unpack_from(blob, r.o)
                r.o += _CELL_SIZE
                pos.extend(vals[0:3])
                nrm.extend(vals[3:6])
                par.extend(vals[6:9])
                burning.append(bool(vals[9]))
            changes.append(
                {"tick": tick, "count": count, "pos": pos, "normal": nrm,
                 "parent": par, "burning": burning}
            )
        infernos.append(
            {"entity_id": eid, "inferno_type": typ, "first_tick": first,
             "last_tick": last, "changes": changes}
        )

    smokes = []
    for _ in range(r.u32()):
        eid = r.u32()
        first = r.i32()
        last = r.i32()
        changes = []
        for _ in range(r.u32()):
            tick = r.i32()
            size = r.u16()
            update = r.u16()
            mode = _MODE_BY_ID[r.u8()]
            data = r.take(r.u32())
            changes.append(
                {"tick": tick, "size": size, "update": update, "mode": mode, "data": data}
            )
        smokes.append(
            {"entity_id": eid, "first_tick": first, "last_tick": last, "changes": changes}
        )

    if r.o != len(blob):
        raise SideChannelError(f"{len(blob) - r.o} trailing bytes in side-channel blob")

    return {
        "format": "cs2.sidechannel.v2",
        "tickrate": tickrate,
        "f32_exact": f32_exact,
        "voxel_capacity": voxel_capacity,
        "sampled_ticks": ticks,
        "infernos": infernos,
        "smokes": smokes,
    }


def summary(sc: dict) -> str:
    return (
        f"infernos={len(sc['infernos'])} "
        f"({sum(len(i['changes']) for i in sc['infernos'])} fire snapshots), "
        f"smokes={len(sc['smokes'])} "
        f"({sum(len(s['changes']) for s in sc['smokes'])} voxel records)"
    )
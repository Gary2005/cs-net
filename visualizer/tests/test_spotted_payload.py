#!/usr/bin/env python3
"""
`spotted`（谁看得见谁）从 .csz 到浏览器状态对象的端到端一致性。

这个功能只有一件事会错，而且错了**不会报错、只会画反**：`spotted[E]` 记的是
「**能看到 E 的敌人**」，所以" S 看得见 E "⟺ `S ∈ spotted[E]`。搞反了照样有
图标，只是全都标在错的人头上。所以这个测试分两层：

  1. **语义**：拿**几何**独立验证方向 —— 把 `spotted[i]` 里单向出现
     （`i ∉ spotted[j]`）的 (i, j) 对挑出来，量两个人的朝向和连线的夹角。
     如果 `spotted` 真是"谁看得见 i"，那么 **j** 应该在看着 i。
  2. **链路**：用真正的 `server._build_round_payload` 造载荷、用真正的
     `replay-core.js`（node 跑）解码，再和 `.csz` 里的原始列表逐位比较。
     顺便验证 `visibleEnemiesOf()` 只返回敌方、而且按 alive 过滤掉了
     `approximate_spotted_by` 死亡后残留的脏值。

    python visualizer/tests/test_spotted_payload.py /path/to/match.csz
    CS2_CSZ=/path/to/match.csz python visualizer/tests/test_spotted_payload.py

需要 node。没有 .csz / node / spotted 列时打印 SKIP 并以 0 退出。
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_JS_DIR = _HERE.parent / "static" / "js"
_TOOLS = _HERE.parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402

from demo_extract import codec  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# 用真正的 replay-core.js 解码（它 import 了 three，但解码路径不碰 DOM / WebGL）。
# argv[2] = 回合载荷，argv[3] = 摘要（`loadMatchData` 的输入，`hasSpottedData()`
# 只信摘要 —— 载荷里"没有这一列"和"这一列全是空"长得一模一样）。
_JS = r"""
import { decodeRoundPayload, setRound, visibleEnemiesOf, spottedBy,
         hasSpottedData, loadMatchData } from '%(core)s';
import { readFileSync } from 'node:fs';
const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const summary = JSON.parse(readFileSync(process.argv[3], 'utf8'));
loadMatchData(summary);
const rnd = decodeRoundPayload(payload);
setRound(rnd);
const T = rnd.ticks.length, P = rnd.players.length;
const visible = [], by = [];
for (let s = 0; s < T; s++) {
  const rowV = [], rowB = [];
  for (let p = 0; p < P; p++) {
    rowV.push(visibleEnemiesOf(p, s));
    rowB.push(spottedBy(p, s));
  }
  visible.push(rowV); by.push(rowB);
}
console.log(JSON.stringify({ visible, by, has_spotted: hasSpottedData(), teams: rnd.teams }));
"""


def _angle_to(P, a: int, b: int, i: int):
    """a 的朝向 与 a→b 连线的夹角（度）。b 用躯干中心、a 用眼高。"""
    dx = P[b]["x"][i] - P[a]["x"][i]
    dy = P[b]["y"][i] - P[a]["y"][i]
    dz = (P[b]["z"][i] + 40.0) - (P[a]["z"][i] + 64.0938)
    dist = math.hypot(math.hypot(dx, dy), dz)
    if dist < 1.0:
        return None
    yaw = math.radians(P[a]["yaw"][i])
    pitch = math.radians(P[a]["pitch"][i])
    fx = math.cos(pitch) * math.cos(yaw)
    fy = math.cos(pitch) * math.sin(yaw)
    fz = -math.sin(pitch)
    dot = (fx * dx + fy * dy + fz * dz) / dist
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def check_direction(data: dict) -> None:
    """几何独立验证 `spotted[i]` 的方向（这是整个功能唯一会"静默画反"的地方）。"""
    print("\ndirection (independent geometric check)")
    a_owner: list[float] = []   # spotted[i] 的持有者 i 在看着 j？
    a_other: list[float] = []   # 还是 j 在看着 i？
    for rnd in data.get("rounds", []):
        P = rnd["players"]
        n = len(rnd["ticks"])
        for i in range(0, n, 3):
            for pi in range(len(P)):
                if not P[pi]["alive"][i]:
                    continue
                seen = P[pi]["spotted"][i]
                for j in seen:
                    if not P[j]["alive"][i]:
                        continue
                    if pi in P[j]["spotted"][i]:
                        continue          # 互相看得见 → 两个角度都小，没有区分度
                    x = _angle_to(P, pi, j, i)
                    y = _angle_to(P, j, pi, i)
                    if x is not None:
                        a_owner.append(x)
                    if y is not None:
                        a_other.append(y)
    if len(a_owner) < 50:
        print(f"       (only {len(a_owner)} one-way pairs; skipped)")
        return

    def med(v):
        v = sorted(v)
        return v[len(v) // 2]

    frac_other = sum(1 for v in a_other if v < 20.0) / len(a_other)
    frac_owner = sum(1 for v in a_owner if v < 20.0) / len(a_owner)
    print(f"       {len(a_owner)} one-way pairs · median j→i {med(a_other):.1f}°, "
          f"i→j {med(a_owner):.1f}° · within 20°: j {frac_other:.1%} vs i {frac_owner:.1%}")
    check("`spotted[i]` contains j means **j is looking at i**",
          frac_other > 0.80 and frac_other > frac_owner + 0.15,
          f"j {frac_other:.3f} vs i {frac_owner:.3f}")
    check("the two directions are clearly distinguishable (not symmetric)",
          med(a_other) < med(a_owner) - 5.0,
          f"{med(a_other):.1f} vs {med(a_owner):.1f}")


def main() -> int:
    csz = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CS2_CSZ")
    if not csz:
        for cand in (_HERE.parent / "outputs" / "parivision-vs-tyloo-m1-inferno.csz",
                     _HERE.parent / "outputs" / "test.csz"):
            if cand.exists():
                csz = str(cand)
                break
    if not csz or not Path(csz).exists():
        print("SKIP: no .csz (pass one, or set CS2_CSZ)")
        return 0
    node = shutil.which("node")
    if not node:
        print("SKIP: no node")
        return 0

    data = codec.load(csz)
    print(f"csz   : {csz}")
    print(f"format: {data.get('format')}")

    rounds = data.get("rounds", [])
    if not rounds or not rounds[0]["players"] or "spotted" not in rounds[0]["players"][0]:
        print("SKIP: this .csz has no `spotted` column (extracted with --no-spotted)")
        return 0

    import server  # noqa: E402  (visualizer/server.py)

    summary = server._summary(data, "t.csz")
    check("the summary advertises has_spotted", summary["has_spotted"])

    # 挑一个 spotted 最活跃的回合：没活动的回合验不出方向，也验不出图标
    def activity(r) -> int:
        return sum(len(v) for p in r["players"] for v in p["spotted"])

    ridx = max(range(len(rounds)), key=lambda i: activity(rounds[i]))
    rnd = rounds[ridx]
    print(f"round : idx={ridx} id={rnd['id']} samples={len(rnd['ticks'])} "
          f"spotted pairs={activity(rnd)}")

    payload = server._build_round_payload(data, ridx)
    raw = payload["players"][0]["spotted"]
    print(f"transport: {sum(len(p['spotted']) for p in payload['players'])} sparse changes "
          f"for {len(rnd['ticks'])} x {len(rnd['players'])} samples "
          f"(player 0: {len(raw)})")

    core = (_JS_DIR / "replay-core.js").as_uri()
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(_JS % {"core": core})
        script = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        payload_file = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(summary, fh)
        summary_file = fh.name
    try:
        proc = subprocess.run([node, script, payload_file, summary_file],
                              capture_output=True, text=True, timeout=300)
    finally:
        os.unlink(script)
        os.unlink(payload_file)
        os.unlink(summary_file)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        check("node decoded the payload", False, f"exit {proc.returncode}")
        return 1
    js = json.loads(proc.stdout)

    T, P = len(rnd["ticks"]), len(rnd["players"])
    check("JS sees the spotted column through the summary", js["has_spotted"] is True)

    print("\ntransport round trip")
    mismatch = 0
    for p in range(P):
        ref = [sorted(int(x) for x in v) for v in rnd["players"][p]["spotted"]]
        got = [sorted(int(x) for x in v) for v in (js["by"][s][p] for s in range(T))]
        for s in range(T):
            if ref[s] != got[s]:
                mismatch += 1
    check("`spotted` survives the sparse transport exactly", mismatch == 0,
          f"{mismatch} sample mismatches out of {T * P}")

    print("\nvisibleEnemiesOf")
    teams = rnd["teams"]
    bad_dir = bad_team = bad_dead = 0
    expected_pairs = 0
    for v in range(P):
        for s in range(T):
            want = set()
            # 独立重算一遍：S 看得见 E ⟺ S ∈ spotted[E]，且双方都活着、互为敌方
            if rnd["players"][v]["alive"][s]:
                for e in range(P):
                    if e == v or teams[e] == teams[v]:
                        continue
                    if not rnd["players"][e]["alive"][s]:
                        continue
                    if v in rnd["players"][e]["spotted"][s]:
                        want.add(e)
            got = set(int(x) for x in js["visible"][s][v])
            if got != want:
                bad_dir += 1
            for e in got:
                if teams[e] == teams[v]:
                    bad_team += 1
                if not rnd["players"][e]["alive"][s] or not rnd["players"][v]["alive"][s]:
                    bad_dead += 1
            expected_pairs += len(want)
    check("visibleEnemiesOf matches the reverse-lookup reference", bad_dir == 0,
          f"{bad_dir} of {T * P} (player, sample) cells differ")
    check("only enemies are ever returned", bad_team == 0, f"{bad_team} same-team hits")
    check("no dead viewer / dead enemy ever produces an icon", bad_dead == 0,
          f"{bad_dead} dead hits")
    check("this round actually has something to show", expected_pairs > 0,
          f"{expected_pairs} (viewer, enemy, sample) triples")

    # 反面：数据里确实有"死人还留着 spotted"的脏值，过滤必须真的把它挡掉
    stale = 0
    for p in range(P):
        for s in range(T):
            if not rnd["players"][p]["alive"][s] and rnd["players"][p]["spotted"][s]:
                stale += 1
    print(f"       (data carries {stale} dead-player samples with a stale spotted list)")

    check_direction(data)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("spotted payload: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

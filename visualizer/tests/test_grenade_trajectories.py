#!/usr/bin/env python3
"""
投掷物轨迹分组的回归测试。

`replay-core.js` 的 `buildGrenadeTrajectories()` 必须做到：**一个投掷物 = 一条轨迹**。

曾经的 bug：它按 `thrower|type` 分组，再用"和上一点距离 < 300 HU 就接上"的启发式
切分。同一个玩家连丢两颗同类投掷物时，第二颗的起点常常落在第一颗终点 300 HU 以内，
于是两颗被并成一条轨迹 —— 表现是"第一颗落地后不消失，继续飘到第二颗的落点"。
实测 mercury 在 de_inferno 第 7 回合的两颗闪光只差 74 HU，被并成 1 条 208 点的轨迹。

现在改成直接按 `grenade_entity_id` 分组。这个测试守住这条不变量：

    对每个回合，JS 产出的轨迹集合 == 「采样数 >= 2 的 eid」集合
    （逐条比对点数，合并了的话点数就会变大）

需要一个带投掷物采样的 .csz。用法::

    python visualizer/tests/test_grenade_trajectories.py /path/to/match.csz
    CS2_CSZ=/path/to/match.csz python visualizer/tests/test_grenade_trajectories.py

需要 node（只跑一个几行的脚本，不需要 npm install）。
没有 .csz 或没有 node 时打印 SKIP 并以 0 退出。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_JS_DIR = _HERE.parent / "static" / "js"
_TOOLS = _HERE.parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

from demo_extract import codec  # noqa: E402

MAX_ROUNDS = 12


def _has_duplicate_grenades(round_: dict) -> bool:
    """这个回合里有没有"同一玩家丢 >=2 个同种投掷物"（也就是旧实现的触发条件）。"""
    per: dict[tuple, set] = defaultdict(set)
    for g in round_.get("events", {}).get("grenades", []) or []:
        per[(g["th"], g["ty"])].add(g["eid"])
    return any(len(v) >= 2 for v in per.values())


def build_fixture(data: dict) -> list[dict]:
    """挑出会触发旧 bug 的回合 + 每个投掷物的期望点数。"""
    cases = []
    for idx, rnd in enumerate(data["rounds"]):
        if not _has_duplicate_grenades(rnd):
            continue
        grenades = rnd["events"]["grenades"]
        per_eid: dict[int, list] = defaultdict(list)
        for g in grenades:
            per_eid[g["eid"]].append(g)
        cases.append({
            "idx": idx,
            "round": {
                "id": rnd["id"],
                "idx": idx,
                "ticks": rnd["ticks"],
                "round_seconds": rnd["round_seconds"],
                "events": {"grenades": grenades},
            },
            # 期望：每个采样数 >= 2 的 eid 恰好一条轨迹，点数是它的采样数
            "expected": sorted(
                len(v) for v in per_eid.values() if len(v) >= 2
            ),
            "expected_eids": sorted(e for e, v in per_eid.items() if len(v) >= 2),
        })
        if len(cases) >= MAX_ROUNDS:
            break
    return cases


NODE_RUNNER = r"""
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [fixturePath, modulePath] = process.argv.slice(2);
const { setRound, buildGrenadeTrajectories } = await import(pathToFileURL(modulePath).href);
const cases = JSON.parse(readFileSync(fixturePath, 'utf8'));

const out = [];
for (const c of cases) {
  setRound(c.round);
  const trajs = buildGrenadeTrajectories();
  out.push({
    idx: c.idx,
    trajs: trajs.map((t) => ({
      eid: t.eid, type: t.type, thrower: t.thrower, n: t.points.length,
      t0: t.points[0].t, t1: t.points[t.points.length - 1].t, settle: t.settleTime,
    })),
  });
}
console.log(JSON.stringify(out));
"""


def main() -> int:
    csz = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CS2_CSZ")
    if not csz:
        print("SKIP: 没给 .csz（用法: test_grenade_trajectories.py match.csz 或设 CS2_CSZ）")
        return 0
    if not Path(csz).is_file():
        print(f"SKIP: 文件不存在 {csz}")
        return 0
    node = shutil.which("node")
    if not node:
        print("SKIP: 没找到 node")
        return 0

    data = codec.load(csz)
    cases = build_fixture(data)
    if not cases:
        print("SKIP: 这个 .csz 里没有「同一玩家丢 >=2 个同种投掷物」的回合")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="grenade_traj_"))
    fixture = tmp / "fixture.json"
    runner = tmp / "runner.mjs"
    fixture.write_text(json.dumps(cases))
    runner.write_text(NODE_RUNNER)

    proc = subprocess.run(
        [node, str(runner), str(fixture), str(_JS_DIR / "replay-core.js")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout.strip())
        print(proc.stderr.strip()[-3000:], file=sys.stderr)
        print("FAIL: node 跑挂了")
        return 1

    results = {r["idx"]: r["trajs"] for r in json.loads(proc.stdout)}
    names = {i: p["name"] for i, p in enumerate(data["players"])}

    failures: list[str] = []
    checked = 0
    merged_examples: list[str] = []

    for case in cases:
        idx = case["idx"]
        trajs = results.get(idx, [])
        got = sorted(t["n"] for t in trajs)
        want = case["expected"]
        checked += len(want)

        if got != want:
            failures.append(
                f"回合 idx={idx}: 轨迹点数 {got} != 期望 {want}（合并/拆分了投掷物）"
            )
            continue
        got_eids = sorted(t["eid"] for t in trajs)
        if got_eids != case["expected_eids"]:
            failures.append(
                f"回合 idx={idx}: 轨迹 eid {got_eids} != 期望 {case['expected_eids']}"
            )
            continue
        # 每条轨迹的 eid 必须非空（分组键就是它）
        for t in trajs:
            if t["eid"] is None:
                failures.append(f"回合 idx={idx}: 有条轨迹没有 eid")
        # 记录旧实现在这些回合会合并多少（只用于报告）
        per: dict[tuple, list] = defaultdict(list)
        for g in case["round"]["events"]["grenades"]:
            per[(g["th"], g["ty"])].append(g)
        for (th, ty), ev in per.items():
            eids = {g["eid"] for g in ev}
            if len(eids) >= 2:
                merged_examples.append(
                    f"    idx={idx} {names.get(th, '?')} {ty.replace('Projectile', '')} "
                    f"{len(eids)} 个投掷物 -> {len(eids)} 条轨迹 ✅"
                )

    print(f"检查了 {len(cases)} 个回合 / {checked} 个投掷物")
    if merged_examples:
        print("旧实现会合并的案例（现在都分开了）:")
        for line in merged_examples[:15]:
            print(line)
        if len(merged_examples) > 15:
            print(f"    ... 还有 {len(merged_examples) - 15} 条")

    # 每个回合里 (thrower, type) 的轨迹条数必须等于投掷物个数
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: 每条轨迹恰好对应一个 grenade_entity_id，没有投掷物被合并")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

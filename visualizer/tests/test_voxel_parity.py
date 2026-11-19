#!/usr/bin/env python3
"""
浏览器端烟雾体素解码器 vs Python 参考实现的逐位一致性测试。

`visualizer/static/js/smoke-voxels.js` 是 `tools/demo_extract/voxel.py` 的 JS 移植。
渲染出来的烟雾只有在两者**算出同一套体素**时才是"真实数据"，所以这里直接对比：

    对若干真实烟雾实例、若干 tick：
      Python: voxel.solid_voxels(journal_at(inst, tick))        -> 实心云栅格
      JS    : SmokeVoxelCache(inst).at(tick).grid               -> 同上
    要求 32×32×32 每一位都相同。同时对照**边界壳**模式（`boundaryGrid`）。

需要一个 .csz 文件（里面必须有烟）。用法::

    python visualizer/tests/test_voxel_parity.py /path/to/match.csz
    CS2_CSZ=/path/to/match.csz python visualizer/tests/test_voxel_parity.py

需要 node（只用它跑一个几行的脚本，不需要 npm install）。
没有 .csz 或没有 node 时打印 SKIP 并以 0 退出，不会让 CI 变红。
"""

from __future__ import annotations

import base64
import json
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

from demo_extract import codec, voxel  # noqa: E402

MAX_INSTANCES = 6


def build_fixture(data: dict) -> list[dict]:
    """挑几个有 journal 的烟雾实例，每个抽查 5 个 tick，顺带算出 Python 的答案。"""
    cases = []
    for rnd in data["rounds"]:
        for inst in rnd.get("smoke", []):
            if not inst.get("changes"):
                continue
            ticks = [c["tick"] for c in inst["changes"]]
            picks = sorted({ticks[0], ticks[len(ticks) // 4], ticks[len(ticks) // 2],
                            ticks[(3 * len(ticks)) // 4], ticks[-1]})
            checks = []
            for t in picks:
                journal = voxel.journal_at(inst, t)
                if journal is None:
                    continue
                checks.append({
                    "tick": t,
                    "solid": sorted(voxel.solid_voxels(journal)),
                    "boundary": sorted(voxel.boundary_voxels(journal)),
                })
            if not checks:
                continue
            cases.append({
                "entity_id": inst["entity_id"],
                "detonation": inst["detonation"],
                "first_tick": inst["first_tick"],
                "last_tick": inst["last_tick"],
                "changes": [
                    {"tick": c["tick"], "size": c["size"], "update": c["update"],
                     "mode": c["mode"],
                     "data": base64.b64encode(c["data"]).decode("ascii")}
                    for c in inst["changes"]
                ],
                "checks": checks,
            })
            if len(cases) >= MAX_INSTANCES:
                return cases
    return cases


NODE_RUNNER = r"""
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [fixturePath, modulePath] = process.argv.slice(2);
const { SmokeVoxelCache } = await import(pathToFileURL(modulePath).href);
const cases = JSON.parse(readFileSync(fixturePath, 'utf8'));

let pass = 0, fail = 0;
const gridList = (grid) => {
  const out = [];
  for (let x = 0; x < 32; x++) for (let y = 0; y < 32; y++) for (let z = 0; z < 32; z++)
    if (grid[(x * 32 + y) * 32 + z]) out.push([x, y, z]);
  return out;
};

for (const c of cases) {
  const solidCache = new SmokeVoxelCache(c, { solid: true });
  const boundCache = new SmokeVoxelCache(c, { solid: false });
  for (const chk of c.checks) {
    for (const [cache, key, label] of [[solidCache, 'solid', 'solid'],
                                       [boundCache, 'boundary', 'boundary']]) {
      const r = cache.at(chk.tick);
      const got = r ? gridList(r.grid) : [];
      const exp = chk[key];
      if (JSON.stringify(got) === JSON.stringify(exp)) { pass++; continue; }
      fail++;
      console.log(`MISMATCH ${label} eid=${c.entity_id} tick=${chk.tick} got=${got.length} exp=${exp.length}`);
      for (let i = 0; i < Math.max(got.length, exp.length); i++) {
        if (JSON.stringify(got[i]) !== JSON.stringify(exp[i])) {
          console.log(`  first diff @${i}: got=${JSON.stringify(got[i])} exp=${JSON.stringify(exp[i])}`);
          break;
        }
      }
    }
  }
}
console.log(JSON.stringify({ pass, fail }));
process.exit(fail ? 1 : 0);
"""


def main() -> int:
    csz = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CS2_CSZ")
    if not csz:
        print("SKIP: 没给 .csz（用法: test_voxel_parity.py match.csz 或设 CS2_CSZ）")
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
        print("SKIP: 这个 .csz 里没有带 journal 的烟雾（提取时 --no-smoke？）")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="voxel_parity_"))
    fixture = tmp / "fixture.json"
    runner = tmp / "runner.mjs"
    fixture.write_text(json.dumps(cases))
    runner.write_text(NODE_RUNNER)

    proc = subprocess.run(
        [node, str(runner), str(fixture), str(_JS_DIR / "smoke-voxels.js")],
        capture_output=True, text=True,
    )
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip()[-2000:], file=sys.stderr)

    n_checks = sum(len(c["checks"]) for c in cases) * 2  # solid + boundary
    if proc.returncode != 0:
        print(f"FAIL: {len(cases)} 个烟雾实例 / {n_checks} 个检查点里有不一致")
        return 1
    print(f"PASS: {len(cases)} 个烟雾实例 × {n_checks // len(cases)} 个 (tick, 模式) 检查点，"
          f"JS 与 voxel.py 逐位相同")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
下蹲 / 眼高：从 .csz 一路到浏览器状态对象的端到端一致性。

链路上有三处会各自出错，而且都不会报错、只会画歪：

  1. `extract.py`   —— `m_flDuckAmount` 有没有真的落进 .csz（NaN 有没有变 0）
  2. `server.py`    —— 量化成 Uint8、以及 `crouching` 塞进 flags 的 bit 4
  3. `replay-core.js` —— 解回来之后 crouch / crouching / eye_height 对不对

所以这个测试**不信任中间任何一步**：它把 server 产出的真实 base64 载荷喂给
真正的 `replay-core.js`（用 node 跑），再拿 JS 算出的状态跟 Python 的
`recipes` 参考实现对逐位比较。

    python visualizer/tests/test_crouch_payload.py /path/to/match.csz
    CS2_CSZ=/path/to/match.csz python visualizer/tests/test_crouch_payload.py

需要 node。没有 .csz / node / v7 数据时打印 SKIP 并以 0 退出。
"""
from __future__ import annotations

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
sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402

from demo_extract import codec, recipes  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# 用真正的 replay-core.js 解码。它 import 了 three，所以这里把解码需要的那部分
# 单独取出来跑：模块本身是纯函数，不碰 DOM。
_JS = r"""
import { decodeRoundPayload, getPlayerState, setRound, eyeHeightFor, hullHeightFor,
         STANDING_EYE_HEIGHT, CROUCH_EYE_DROP } from '%(core)s';
import { readFileSync } from 'node:fs';
// the payload goes through a file, not argv: an all-tick round is far past
// the OS argument-list limit
const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const rnd = decodeRoundPayload(payload);
setRound(rnd);                       // getPlayerState reads module state
const out = { eye: [], hull: [], crouch: [], standing: STANDING_EYE_HEIGHT,
              drop: CROUCH_EYE_DROP, has_crouch: rnd.players.map(p => p.has_crouch) };
for (let s = 0; s < rnd.ticks.length; s++) {
  const row = { eye: [], hull: [], crouch: [] };
  for (let p = 0; p < rnd.players.length; p++) {
    const st = getPlayerState(p, s);
    row.eye.push(st.eye_height);
    row.hull.push(hullHeightFor(st.crouching));
    row.crouch.push(st.crouch);
  }
  out.eye.push(row.eye); out.hull.push(row.hull); out.crouch.push(row.crouch);
}
console.log(JSON.stringify(out));
"""


def main() -> int:
    csz = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CS2_CSZ")
    if not csz:
        for cand in (Path("/tmp/crouch_test.csz"),
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
    if data.get("format") != "cs2.demo.v7":
        print(f"SKIP: needs cs2.demo.v7, got {data.get('format')}")
        return 0

    import server  # noqa: E402  (visualizer/server.py)

    # build the payload through the real builder, not a hand-rolled copy of it:
    # a hand-rolled one would silently disagree with the server the moment the
    # two drift apart, which is exactly the failure this test exists to catch.
    ridx = next((i for i, r in enumerate(data["rounds"])
                 if any(c > 0.5 for c in r["players"][0]["crouch"])), 0)
    payload = server._build_round_payload(data, ridx)
    print(f"round : idx={ridx} id={payload['id']} samples={payload['sample_count']}")
    core = (_JS_DIR / "replay-core.js").as_uri()
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(_JS % {"core": core})
        script = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        payload_file = fh.name
    try:
        proc = subprocess.run([node, script, payload_file],
                              capture_output=True, text=True, timeout=300)
    finally:
        os.unlink(script)
        os.unlink(payload_file)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        check("node decoded the payload", False, f"exit {proc.returncode}")
        return 1
    js = json.loads(proc.stdout)
    rnd = data["rounds"][ridx]
    js_eye = np.asarray(js["eye"], dtype=np.float64)      # (T, P)
    js_hull = np.asarray(js["hull"], dtype=np.float64)
    js_crouch = np.asarray(js["crouch"], dtype=np.float64)

    print("\ntransport / quantisation")
    check("every player reports has_crouch", all(js["has_crouch"]),
          str(js["has_crouch"]))
    ref_crouch = recipes.crouch_amount(rnd)               # (T, P)
    worst_q = float(np.abs(js_crouch - ref_crouch).max())
    check("crouch survives the Uint8 quantisation (<= half a step)",
          worst_q <= 0.5 / 255.0 + 1e-6, f"max error {worst_q:.8f}")
    check("quantisation error is sub-millimetre in eye height",
          worst_q * recipes.CROUCH_EYE_DROP_HU <= 0.5 / 255.0 * 18.0 + 1e-3,
          f"{worst_q * 18.0:.4f} HU")

    print("\nflags byte / bit 4")
    ref_hull = np.where(recipes.crouching_mask(rnd), 54.0, 72.0)
    check("crouching round-trips through flags bit 4",
          np.array_equal(js_hull, ref_hull),
          f"{int((js_hull != ref_hull).sum())} mismatches")
    check("the hull only ever takes 72 or 54",
          set(np.unique(js_hull).tolist()) <= {72.0, 54.0}, str(np.unique(js_hull)))

    print("\neye height")
    ref_eye = recipes.eye_height(rnd)
    worst_e = float(np.abs(js_eye - ref_eye).max())
    check("JS eye height matches the Python reference",
          worst_e <= 18.0 / 510.0 + 1e-4, f"max error {worst_e:.6f} HU")
    check("standing eye height is the documented 64.0938",
          abs(js["standing"] - 64.0938) < 1e-9, str(js["standing"]))
    check("crouch drop is the measured 18 HU", abs(js["drop"] - 18.0) < 1e-9, str(js["drop"]))
    check("eye height stays inside [46.09, 64.09]",
          js_eye.min() >= 46.0 and js_eye.max() <= 64.1, f"{js_eye.min()}..{js_eye.max()}")

    print("\nthe two quantities are genuinely different")
    # The whole reason `crouch` and `crouching` are stored separately: mid-ramp
    # the hull and the eye disagree.  Two independent disagreements exist and
    # both are visible in real data:
    #   * hull still 72 while the eye has already dropped  -> the hull is a step
    #   * hull already 54 while the eye is still up        -> the flag is not a
    #     function of the amount; it tracks the duck *key*, so it can snap early
    mid = (js_crouch > 0.1) & (js_crouch < 0.9)
    n_mid = int(mid.sum())
    if n_mid:
        stand_eye = js["standing"]
        crouch_eye = stand_eye - js["drop"]
        lag = int(((js_hull == 72.0) & (js_eye < stand_eye - 0.5))[mid].sum())
        lead = int(((js_hull == 54.0) & (js_eye > crouch_eye + 0.5))[mid].sum())
        check("mid-ramp samples exist to compare", n_mid > 0, str(n_mid))
        check("the hull can still be 72 while the eye is already down", lag > 0,
              f"{lag}/{n_mid}")
        check("the hull can already be 54 while the eye is still up", lead > 0,
              f"{lead}/{n_mid}")
        check("crouch and crouching are not the same predicate",
              not np.array_equal(js_hull == 54.0, js_crouch > 0.5))
        print(f"       {n_mid} mid-ramp samples: {lag} hull-lagging, {lead} hull-leading, "
              f"eye {js_eye[mid].min():.1f}..{js_eye[mid].max():.1f} HU")
    else:
        print("       (no mid-ramp sample in this round; skipped)")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("crouch payload: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

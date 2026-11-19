#!/usr/bin/env python3
"""
启动横幅的方框对齐（`visualizer/banner.py`）。

这个 bug 用户一眼就能看见，而所有测试都看不见：**每一行右边框差几个格子**。
根因是 `f"{x:<49}"` 数的是字符、终端里中文和全角括号占两列，所以
`python -m app.server --port 5000` 打出来就是个歪框；URL 那几行更直接，用的是
`' ' * max(0, 22 - len(host) - len(port))` —— host / port 一变长就自己算错。

所以这里不测"某个字符串长什么样"（那样改个标签就红了），而是测**性质**：

  1. `display_width` 把 CJK / 全角算 2 列、ASCII 算 1 列、组合符 0 列
  2. 框里**每一行的显示宽度都相同**（= 上下边框的长度），不管内容多长、
     中英文怎么混、有没有超长 Windows 路径
  3. 值的**起始列一致**（标签列对齐），包括最长的那个标签
  4. 两个真实的横幅（`app.server.banner` / `visualizer.server.banner`）都满足上面两条
     —— 这才是回归测试：以后再往横幅里塞一行，忘了对齐也照样红
  5. 长短 host / port、找不到地图、超长数据目录都不许把框撑歪

    python visualizer/tests/test_banner.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualizer.banner import (  # noqa: E402
    _BG_DEFAULT, box, display_width, logo_lines, logo_width, rows_of, startup,
    supports_color,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

FAILED: list[str] = []
PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILED.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))


def widths(banner: str) -> list[int]:
    return [display_width(ln) for ln in banner.splitlines()]


def value_col(line: str) -> int | None:
    """这一行的**值**从第几列开始（标签 + 中间那串空格之后）。

    从渲染出来的字符串反推，而不是复用 `box()` 内部的算术 —— 这样"值的列对齐"才是
    独立量出来的。
    """
    m = re.match(r"^(\s*)(\S+:)(\s+)(\S)", line)
    return display_width(m.group(1) + m.group(2) + m.group(3)) if m else None


def rows_kv(banner: str) -> dict[str, str]:
    """把渲染出来的框解析回 ``{标签: 值}`` —— 不依赖中间空格的个数。

    标签里可以有空格（``Demo 库:``），值里也可以有冒号（``http://``、``D:\\``），
    所以标签那一组要**贪心**匹配：只有"冒号后面跟着空格"的那个冒号才可能是标签的
    结尾（URL 的 ``http:`` 后面是 ``//``）。
    """
    out = {}
    for line in rows_of(banner):
        m = re.match(r"^\s*(.+?:)\s+(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
#: 带捕获组的那一份，给解码器用（`ANSI_RE` 故意不带组，`strip_ansi` 用它做替换）
ANSI_SEQ = re.compile(r"\x1b\[([0-9;]*)m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def box_lines_of(banner: str) -> list[str]:
    """输出里属于**框**的那些行（带 ANSI 也没关系：剥掉再找 ║）。"""
    return [strip_ansi(ln) for ln in banner.splitlines() if "║" in strip_ansi(ln)]


class FakeTTY:
    """给 supports_color 用的假流（真终端在测试里不存在）。"""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def decode_ansi(lines: list[str], palette: dict) -> list[str]:
    """把带 ANSI 的行解回半高像素网格（和 ``logo_art.ROWS`` 同构）。

    这是这一层最该有的测试：编码器要错，几乎不会错在"颜色值选错了"，而是错在
    **状态残留** —— 背景色是个状态，设过之后后面那些空格会带着它一起画，"透明"的地方
    就被上一段的颜色糊满，整幅图变成一块实心色板。所有只检查"有没有 ANSI""有没有 ▀"
    的断言在这种情况下全部照常通过（真出过一次，在浅色终端里一眼就能看见）。
    这里反过来把终端会画出来的东西解出来，和源数据逐位比。

    语义按终端的实际行为来：先铺整格背景（``bg``，默认就是终端自己的底色），再画
    ``▀``/``▄`` 覆盖一半（用 ``fg``）；空格只铺背景。
    """
    rev = {tuple(int(c) for c in v): k for k, v in palette.items()}
    out: list[str] = []
    fg = bg = None
    for line in lines:
        top: list[str] = []
        bottom: list[str] = []
        i = 0
        while i < len(line):
            m = ANSI_SEQ.match(line, i)
            if m:
                p = m.group(1).split(";")
                if p == ["0"]:
                    fg = bg = None
                elif p == ["49"]:
                    bg = None
                elif p[0] == "48" and p[1:2] == ["2"]:
                    bg = tuple(int(v) for v in p[2:5])
                elif p[0] == "38" and p[1:2] == ["2"]:
                    fg = tuple(int(v) for v in p[2:5])
                i = m.end()
                continue
            ch = line[i]
            i += 1
            base = "." if bg is None else rev.get(bg, "?")
            if ch == " ":
                top.append(base)
                bottom.append(base)
            elif ch == "▀":
                top.append(rev.get(fg, "?"))
                bottom.append(base)
            elif ch == "▄":
                top.append(base)
                bottom.append(rev.get(fg, "?"))
            else:
                top.append("@")
                bottom.append("@")
        out.append("".join(t + b for t, b in zip(top, bottom)))
    return out


# ── display_width ───────────────────────────────────────────────────────────


def test_width() -> None:
    print("\ndisplay_width：终端里占几列")
    check("ASCII", display_width("abc") == 3)
    check("中文 = 两列", display_width("中文") == 4)
    check("全角括号 = 两列", display_width("（）") == 4)
    check("中英混排", display_width("数据目录: D:\\a") == 8 + 2 + 3 + 1)
    check("空串 = 0", display_width("") == 0)
    check("组合符不算列（e + ́）", display_width("e\u0301") == 1)
    check("ambiguous 的 · 按一列（Windows 等宽字体就是这样）",
          display_width("·") == 1)
    check("边框字符自己也算一列", display_width("║═╔") == 3)


# ── box ─────────────────────────────────────────────────────────────────────


def test_box() -> None:
    print("\nbox：右边框对齐")
    plain = box([("a:", "1"), ("bb:", "2")], title="t")
    check("纯 ASCII：每行等宽", len(set(widths(plain))) == 1, f"{widths(plain)}")

    long_path = "D:\\project\\cs-net-v5\\maps\\optimized_obj_files"
    mixed = box([("数据目录:", "D:\\project\\cs-net-v5\\app\\data"),
                 ("地图:", long_path),
                 ("提取器:", "demo_extract（每个 tick）")],
                title="cs-net · Demo 库 + 3D 回放")
    check("中英混排 + 全角括号：每行等宽", len(set(widths(mixed))) == 1,
          f"{widths(mixed)}")

    # 框宽 = 最长那一行 + 左右内边距 + 两个边框；内容不能被截断
    label_w = display_width("数据目录:")            # 最长的那个标签
    longest = label_w + 2 + display_width(long_path)
    check("框宽 = 最长那一行 + 内边距 + 边框", max(widths(mixed)) == longest + 4 + 2,
          f"{max(widths(mixed))} vs {longest + 6}")
    check("长路径整段都在框里（没有被截断）", long_path in mixed)

    rows = rows_of(mixed)
    check("rows_of 拿到的是内容行，不含边框",
          len(rows) == len(mixed.splitlines()) - 2)
    check("标题下面留了一个空行", rows[1].strip() == "")

    cols = [value_col(ln) for ln in rows]
    cols = [c for c in cols if c is not None]
    check("值的起始列一致（标签列对齐）", len(cols) == 3 and len(set(cols)) == 1, f"{cols}")

    blank = box([("a:", "1"), None, ("b:", "2")]).splitlines()[2]
    check("None = 空行",
          blank.startswith("║") and blank.endswith("║") and blank.strip("║").strip() == "",
          repr(blank))
    check("空行也参与框宽", len(set(widths(box([("a:", "1"), None, ("b:", "2")])))) == 1)
    check("没有内容行时不炸", len(set(widths(box([], title="只有标题")))) == 1)
    check("超长单行不会把别的行挤歪",
          len(set(widths(box([("x:", "y" * 200)])))) == 1)

    # 全角标签比 ASCII 标签"看起来"更长，但列数要对齐
    cjk_label = box([("提取器:", "v"), ("Open:", "v")])
    rows2 = rows_of(cjk_label)
    check("全角标签和 ASCII 标签混排时值也对齐",
          value_col(rows2[0]) == value_col(rows2[1]) is not None, f"{rows2}")


# ── logo ────────────────────────────────────────────────────────────────────

ROWS = [("Demo 库:", "http://127.0.0.1:5000/library"),
        ("玩家:", "http://127.0.0.1:5000/players"),
        ("数据目录:", "D:\\project\\cs-net-v5\\app\\data")]


def test_logo_art() -> None:
    print("\nlogo 字符画（generated data）")
    from visualizer import logo_art as art

    check("有一份字符画", len(art.ROWS) > 8)
    check("每行长度 = 2 × WIDTH（一格两像素）",
          {len(r) for r in art.ROWS} == {art.WIDTH * 2},
          f"{sorted({len(r) for r in art.ROWS})} vs {art.WIDTH * 2}")
    check("像素行数是偶数（半格方块填满）", art.PIXEL_ROWS == len(art.ROWS) * 2)
    alphabet = {ch for r in art.ROWS for ch in r}
    check("字符只有调色板 + 透明标记", alphabet <= set(art.PALETTE) | {"."},
          f"{sorted(alphabet)}")
    check("三种品牌色都在用", set(art.PALETTE) <= alphabet, f"{sorted(alphabet)}")
    check("调色板颜色是 0..255 的 RGB 三元组",
          all(isinstance(v, tuple) and len(v) == 3 and all(0 <= c <= 255 for c in v)
              for v in art.PALETTE.values()))
    # 有内容但不满屏 —— 全空说明渲染挂了，全满说明阈值/裁剪错了
    ink = sum(1 for r in art.ROWS for ch in r if ch != ".")
    check("既有内容又不是一整块", 0.05 < ink / (art.WIDTH * art.PIXEL_ROWS) < 0.6,
          f"{ink}/{art.WIDTH * art.PIXEL_ROWS}")
    check("源文件指向 SVG", art.SOURCE.endswith("logo.svg"))


def test_logo_render() -> None:
    print("\nlogo 渲染：▀ 半格方块 + 24 位 ANSI")
    from visualizer import logo_art as art

    lines = logo_lines(color=True)
    check("行数和网格一致", len(lines) == len(art.ROWS))
    check("带颜色", all(ANSI_RE.search(ln) for ln in lines))
    check("每行都复位（不然颜色会漏到后面的框上）",
          all(ln.endswith("\x1b[0m") for ln in lines))

    plain = [strip_ansi(ln) for ln in lines]
    check("剥掉 ANSI 后每行正好 WIDTH 格",
          {display_width(ln) for ln in plain} == {art.WIDTH},
          f"{sorted({display_width(ln) for ln in plain})}")
    check("只用 ▀ / ▄ / 空格，没有别的字符",
          all(set(ln) <= {"▀", "▄", " "} for ln in plain),
          f"{sorted({c for ln in plain for c in ln})}")
    check("没有把透明画成黑底（浅色终端会糊出黑边）",
          "48;2;0;0;0" not in "".join(lines))
    check("用默认背景画半格", "\x1b[49m" in "".join(lines))
    check("logo_width() 和网格一致（不能拿带 ANSI 的串去量宽）",
          logo_width() == art.WIDTH)
    check("没有颜色时干脆不打", logo_lines(color=False) == [])
    check("缩进只是行首空格（不破坏图案）",
          all(strip_ansi(ln).startswith("  ") for ln in logo_lines(color=True, indent=2)))

    # ── 逐位对拍：把终端会画出来的东西解回来，必须跟源数据一模一样 ──────────
    decoded = decode_ansi(lines, art.PALETTE)
    check("编码 → 解码 == 源网格（每个半像素都落在该落的颜色上）",
          decoded == art.ROWS,
          "第一处不同：" + next(
              (f"第 {i} 行\n  源  {art.ROWS[i]}\n  解回 {decoded[i]}"
               for i in range(min(len(decoded), len(art.ROWS))) if decoded[i] != art.ROWS[i]),
              f"行数 {len(decoded)} vs {len(art.ROWS)}"))
    check("没有把透明区糊成上一段的颜色（背景色不会残留）",
          "?" not in "".join(decoded))
    # 缩进版也要逐位一致（缩进 = 行首多出 indent 个格子，图案本身不能变）
    indented = decode_ansi(logo_lines(True, indent=2), art.PALETTE)
    check("带缩进时解码结果一样（缩进不改变图案）",
          [row[2 * 2:] for row in indented] == art.ROWS)

    # 灵敏度自检：把"透明段前面的背景复位"删掉 —— 就是真出过的那个 bug —— 解码必须
    # 和源数据不一致。不然上面那条只是在自我复读，真坏了也照样绿。
    flooded = [re.sub(re.escape(_BG_DEFAULT) + r"( +)", r"\1", ln) for ln in lines]
    check("去掉背景复位后必须能测出来（灵敏度自检）",
          decode_ansi(flooded, art.PALETTE) != art.ROWS,
          "解码器认不出「整片被上一段颜色糊满」，那就守不住这个 bug")


def test_supports_color() -> None:
    print("\nsupports_color：什么时候该打颜色")
    import os as _os

    saved = {k: _os.environ.get(k) for k in ("NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE", "TERM")}
    try:
        for k in saved:
            _os.environ.pop(k, None)
        check("不是终端 → 不打", not supports_color(FakeTTY(False)))
        # Windows 上"是终端"还要看 conhost 能不能开 VT（在这个进程里可能开不了），
        # 所以这一条断言的是"决策只剩平台那一步"，不是写死 True。
        from visualizer.banner import _enable_vt

        check("是终端 → 交给平台判断（非 Windows 上就是 True）",
              supports_color(FakeTTY(True)) == _enable_vt())
        _os.environ["NO_COLOR"] = "1"
        check("NO_COLOR 覆盖终端判断", not supports_color(FakeTTY(True)))
        _os.environ.pop("NO_COLOR")
        _os.environ["TERM"] = "dumb"
        check("TERM=dumb → 不打", not supports_color(FakeTTY(True)))
        _os.environ.pop("TERM")
        _os.environ["FORCE_COLOR"] = "1"
        check("FORCE_COLOR 连管道都打（CI 里要图的时候）", supports_color(FakeTTY(False)))
        _os.environ["FORCE_COLOR"] = "0"
        check("FORCE_COLOR=0 不算强制", not supports_color(FakeTTY(False)))
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def test_startup_layout() -> None:
    print("\nstartup：窄终端上下排、宽终端并排")
    from visualizer import logo_art as art

    narrow = startup(ROWS, title="t", columns=70, color=True)
    wide = startup(ROWS, title="t", columns=200, color=True)
    off = startup(ROWS, title="t", columns=200, color=False)

    check("关掉颜色就只剩框", "\x1b[" not in off and "▀" not in off)
    check("关掉颜色时框仍然对齐",
          len(set(widths(off))) == 1, f"{widths(off)}")

    # 窄：logo 在上、框在下，logo 按框宽居中
    n_lines = narrow.splitlines()
    box_w = max(widths(box(ROWS, title="t")))
    logo_plain = [strip_ansi(ln) for ln in n_lines if "║" not in strip_ansi(ln)]
    art_rows = [ln for ln in logo_plain if "▀" in ln or "▄" in ln]
    check("窄终端：logo 有内容", len(art_rows) == len(art.ROWS), f"{len(art_rows)}")
    # 每行 = 居中缩进 + 整幅宽（logo 左边透明的那几格是行内空格，不能当缩进量）
    indent = max(0, (box_w - art.WIDTH) // 2)
    check("窄终端：logo 按框宽居中（缩进 + 幅宽恒定）",
          {display_width(ln) for ln in art_rows} == {indent + art.WIDTH},
          f"{sorted({display_width(ln) for ln in art_rows})} vs {indent + art.WIDTH} (box_w={box_w})")
    check("窄终端：框没被 logo 挤歪",
          len(set(widths("\n".join(box_lines_of(narrow))))) == 1)
    check("窄终端：logo 和框之间有空行",
          any(ln.strip() == "" for ln in n_lines[: len(art.ROWS) + 2]))

    # 宽：并排，框的左边框在 logo 右边
    w_lines = [strip_ansi(ln) for ln in wide.splitlines()]
    check("宽终端：行数 = max(logo, 框)", len(w_lines) == len(art.ROWS))
    check("宽终端：logo 和框在同一行上",
          all("▀" in ln or "▄" in ln or ln.strip() == "" for ln in w_lines)
          and any("║" in ln for ln in w_lines))
    check("宽终端：框正好接在 logo 右边（logo 宽 + 2 空格）",
          {ln.index("║") for ln in w_lines if "║" in ln} == {art.WIDTH + 2},
          f"{sorted({ln.index('║') for ln in w_lines if '║' in ln})}")
    box_part = [" " * 0 + ln[ln.index("║"):] for ln in w_lines if "║" in ln]
    check("宽终端：框自己仍然每行等宽",
          len({display_width(ln) for ln in box_part}) == 1,
          f"{sorted({display_width(ln) for ln in box_part})}")
    check("宽终端：没有超出给定的列数",
          all(display_width(ln) <= 200 for ln in w_lines))


# ── 两个真实横幅 ────────────────────────────────────────────────────────────


def test_real_banners() -> None:
    print("\n真实横幅：app.server / visualizer.server")
    from app.server import banner as app_banner
    from visualizer.server import banner as vis_banner

    cases = [
        (app_banner("127.0.0.1", 5000, maps=Path("D:/project/cs-net-v5/maps/optimized_obj_files"),
                    data_dir=Path("D:/project/cs-net-v5/app/data"), sampling="每个 tick"),
         "app：默认 host / 长路径"),
        (app_banner("0.0.0.0", 50000, maps=None,
                    data_dir=Path("C:/a/very/long/data/directory/that/goes/on/and/on"),
                    sampling="每 8 tick"),
         "app：端口变长 / 没有地图 / 超长数据目录"),
        (app_banner("localhost", 80, maps=Path("D:/m"), data_dir=Path("D:/d"),
                    sampling="每 0.25 秒"),
         "app：短 host / 短路径"),
        (vis_banner("127.0.0.1", 5000, maps=Path("D:/project/cs-net-v5/maps/optimized_obj_files")),
         "visualizer：默认"),
        (vis_banner("0.0.0.0", 5099, maps=None), "visualizer：没有地图"),
    ]

    for banner, label in cases:
        check(f"{label}：每行等宽", len(set(widths(banner))) == 1, f"{widths(banner)}")
        check(f"{label}：上下边框一样长",
              widths(banner)[0] == widths(banner)[-1] == max(widths(banner)))

    app_default = cases[0][0]
    kv = rows_kv(app_default)
    for label, value in (("Demo 库:", "http://127.0.0.1:5000/library"),
                         ("玩家:", "http://127.0.0.1:5000/players"),
                         ("回放页:", "http://127.0.0.1:5000/replay"),
                         ("数据目录:", "D:\\project\\cs-net-v5\\app\\data"),
                         ("地图:", "D:\\project\\cs-net-v5\\maps\\optimized_obj_files"),
                         ("提取器:", "demo_extract（每个 tick）")):
        check(f"app 横幅里「{label}」指向 {value}", kv.get(label) == value,
              f"实际是 {kv.get(label)!r}（标签和值必须还在同一行、且没被拼错）")

    vis_default = cases[3][0]
    kvv = rows_kv(vis_default)
    for label, value in (("Open:", "http://127.0.0.1:5000/"),
                         ("提取:", "demo_extract（采样在提取时定死，这里只读 .csz）")):
        check(f"visualizer 横幅里「{label}」指向 {value}", kvv.get(label) == value,
              f"实际是 {kvv.get(label)!r}")


def main() -> int:
    test_width()
    test_box()
    test_logo_art()
    test_logo_render()
    test_supports_color()
    test_startup_layout()
    test_real_banners()
    print(f"\n{PASSED} passed, {len(FAILED)} failed" + (f": {FAILED}" if FAILED else ""))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

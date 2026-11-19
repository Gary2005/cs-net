"""启动时打的那一块：logo 字符画 + 信息框。

两个入口都会打它（``app.server`` 和 ``visualizer/server.py``），所以放在这里；
``app`` 本来就依赖 ``visualizer``，反过来不成立，所以归属 ``visualizer``。

logo 是**生成的**，不是手画的字符画
-----------------------------------
``visualizer/static/logo.svg`` 经 ``scripts/make_logo_art.py`` 渲染成
``visualizer/logo_art.py``（半格像素网格）；设计改色 / 改形状之后重新生成一次就同步。
这里只负责把网格画成 ``▀`` + 24 位 ANSI。没有颜色就不打 logo —— 那三个品牌色是
它的全部信息（见 `logo_lines`）。

框为什么不能手写空格
--------------------
中文、全角括号（`（）：`）在终端里占 **2 列**，而 ``f"{x:<49}"`` 数的是**字符**。
于是每行的右边框都会差几个格子 —— ``python -m app.server --port 5000`` 打出来的框
就是歪的。更糟的是 URL 那几行原本写的是
``' ' * max(0, 22 - len(host) - len(port))``：host / port 一变长就自己算错，和"框到底
有多宽"毫无关系。这里统一成"先把内容准备好，再按最长的一行补齐"。

宽度按什么算
------------
东亚宽度 ``W``（宽）/ ``F``（全角）算 2 列，其余算 1 列 —— **包括 ``A``（ambiguous）**：
``·``、``—``、以及这个框自己的 ``║ ═ ╔`` 都属于 ambiguous。Windows 的等宽字体
（Consolas / Cascadia Mono / 默认的点阵字体）把它们画成 1 列；这个框本身用的就是这批
字符，如果按 2 列算，边框自己就先错开了，所以只有按 1 列算才自洽。
（代价写在明处：真正的 CJK 字体配上"ambiguous 当宽"的终端时，``·`` 会占 2 列，
那一行会差一格。要绝对安全就把 ambiguous 字符从横幅里去掉。）
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from typing import Iterable, Optional, Sequence, Union

#: 一行内容：``(标签, 值)`` 或 ``None``（空行）。
Row = Union[tuple, None]


def display_width(text: str) -> int:
    """这段文字在终端里占几列（CJK / 全角 = 2，其余 = 1，组合符 = 0）。"""
    width = 0
    for ch in str(text):
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _spaces(n: int) -> str:
    return " " * max(0, n)


def box(rows: Iterable[Row], *, title: Optional[str] = None, pad: int = 2,
        gap: int = 2) -> str:
    """把若干行放进一个 ``╔═╗`` 框里，右边框一定对齐。

    ``rows`` 里每一项是 ``(标签, 值)``：标签会按最长的那个对齐，值从同一条竖线开始
    （不用自己在字符串里数空格）。``None`` 表示一个空行。

    ``title`` 居中放在最上面，下面自动跟一个空行（两个横幅原来都是这个形状）。
    框的宽度由**内容**决定：路径长就把框撑宽，不会截断，也不会歪。
    """
    items = list(rows)
    label_w = max((display_width(item[0]) for item in items if item and item[0]),
                  default=0)

    # 内容行先排好（标签列对齐），再统一按最长的一行补右边
    lines: list[Optional[str]] = []
    for item in items:
        if item is None:
            lines.append(None)
            continue
        label, value = item
        lines.append(label + _spaces(label_w - display_width(label) + gap) + str(value))

    inner = max([display_width(t) for t in lines if t is not None]
                + ([display_width(title)] if title else [0]))
    inner += pad * 2

    out = ["╔" + "═" * inner + "╗"]
    if title is not None:
        left = pad + max(0, (inner - pad * 2 - display_width(title)) // 2)
        out.append("║" + _spaces(left) + title
                   + _spaces(inner - left - display_width(title)) + "║")
        out.append("║" + _spaces(inner) + "║")
    for text in lines:
        if text is None:
            out.append("║" + _spaces(inner) + "║")
            continue
        out.append("║" + _spaces(pad) + text
                   + _spaces(inner - pad - display_width(text)) + "║")
    out.append("╚" + "═" * inner + "╝")
    return "\n".join(out)


def rows_of(banner: str) -> Sequence[str]:
    """框里的内容行（去掉边框）。测试和调试用。"""
    return [ln[1:-1] for ln in banner.splitlines()
            if len(ln) >= 2 and ln[0] == "║" and ln[-1] == "║"]


# ── logo（字符画在 visualizer/logo_art.py，由 scripts/make_logo_art.py 生成）─────

#: 24 位 ANSI
_RESET = "\x1b[0m"
_BG_DEFAULT = "\x1b[49m"


def ansi_logo(rows, palette, indent: int = 0) -> list[str]:
    """字符画行 → 带 ANSI 的文本行：``▀`` 前景=上半格、背景=下半格。

    一格两色是半格方块的全部意义 —— 字符格宽:高 ≈ 1:2，用两个像素填一格，竖直分辨率
    翻倍，logo 才不会被压成一半高。

    透明的那半边用 **默认背景**（``▄``/``▀`` 只设前景），不是画成黑色：浅色终端里
    画黑会糊出一圈黑边。
    """
    rgb = {k: tuple(int(c) for c in v) for k, v in palette.items()}
    out: list[str] = []
    for row in rows:
        cells = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
        line = " " * indent
        i = 0
        while i < len(cells):
            j = i
            while j < len(cells) and cells[j] == cells[i]:
                j += 1                       # 相邻同色的格子合成一段
            top, bottom = cells[i]
            fg, bg = rgb.get(top), rgb.get(bottom)
            if fg is None and bg is None:
                # 先复位背景再打空格！背景色是**状态**，会一直留到被改掉为止 ——
                # 上一段刚设过 \x1b[48;2;…m，这里如果直接打空格，"透明"的那片就会
                # 被上一段的颜色糊满（真机上整幅图变成一块实心色板，笔画全看不见）。
                line += _BG_DEFAULT + " " * (j - i)
            elif bg is None:
                line += f"{_BG_DEFAULT}\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m" + "▀" * (j - i)
            elif fg is None:
                line += f"{_BG_DEFAULT}\x1b[38;2;{bg[0]};{bg[1]};{bg[2]}m" + "▄" * (j - i)
            else:
                line += (f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m"
                         f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m") + "▀" * (j - i)
            i = j
        out.append(line + _RESET)
    return out


def logo_width() -> int:
    """logo 的可视宽度（列）。**不能**拿带 ANSI 的字符串去量 —— 转义序列也算字符。"""
    from visualizer import logo_art

    return logo_art.WIDTH


def logo_lines(color: bool = True, indent: int = 0) -> list[str]:
    """logo 字符画，每行带 ANSI。

    ``color=False`` 返回空表：这个 logo 的全部信息就是那三个品牌色，没有颜色只剩几个
    方块，打在日志里只是噪音。

    生成物缺失（有人删了 ``logo_art.py`` / 忘了提交）时也返回空表 —— 一个 logo 不值得
    让服务起不来；这种情况由 ``test_banner.py`` 在 CI 里喊。
    """
    if not color:
        return []
    try:
        from visualizer import logo_art
    except ImportError:                              # pragma: no cover - 生成物缺失
        return []

    return ansi_logo(logo_art.ROWS, logo_art.PALETTE, indent=indent)


def _enable_vt() -> bool:
    """Windows 的 conhost 默认不解释 ANSI 转义，顺手打开 VT 处理。

    失败（重定向到了文件、老系统、没有控制台）就当不支持颜色 —— 调用的地方会退回
    纯文本框，而不是打出一堆 ``←[38;2;...``。
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:                                # pragma: no cover - 平台相关
        return False


def supports_color(stream=None) -> bool:
    """这个输出流能不能吃 24 位 ANSI。

    * 不是终端（管道 / 重定向进文件）→ 不打：日志里不需要图形。
    * ``NO_COLOR`` 非空 → 不打（大家都认的那条约定）。
    * ``FORCE_COLOR`` / ``CLICOLOR_FORCE`` 非空且不是 ``0`` → 打（CI 里想要图的时候）。
    * ``TERM=dumb`` → 不打。
    """
    env = os.environ
    forced = env.get("FORCE_COLOR") or env.get("CLICOLOR_FORCE")
    if forced and forced != "0":
        return True
    if env.get("NO_COLOR"):
        return False
    out = sys.stdout if stream is None else stream
    if not getattr(out, "isatty", lambda: False)():
        return False
    if env.get("TERM") == "dumb":
        return False
    return _enable_vt()


def startup(rows: Iterable[Row], *, title: Optional[str] = None,
            columns: Optional[int] = None, color: Optional[bool] = None) -> str:
    """启动时打的那一整块：logo + 信息框。

    版式看终端宽度：够宽就 **logo 在左、框在右**（框竖直居中），否则 logo 居中压在框
    上面。宽度不够还硬并排的话，框的右边框会被终端折行折断 —— 那是比没有 logo 更难看
    的结果，所以这里量一次宽度。
    """
    body = box(rows, title=title)
    if color is None:
        color = supports_color()
    lines = logo_lines(color)
    if not lines:
        return body

    lw = logo_width()
    box_lines = body.splitlines()
    box_w = max(display_width(ln) for ln in box_lines)
    if columns is None:
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns

    if columns >= box_w + lw + 2:
        # 并排：框在右边竖直居中
        top = max(0, (len(lines) - len(box_lines)) // 2)
        height = max(len(lines), len(box_lines) + top)
        out = []
        for i in range(height):
            left = lines[i] if i < len(lines) else " " * lw
            j = i - top
            out.append(left + "  " + (box_lines[j] if 0 <= j < len(box_lines) else ""))
        return "\n".join(out)

    indent = max(0, (box_w - lw) // 2)
    art = logo_lines(color, indent=indent)
    # 前面那个空行由调用方 print() 负责 —— 这里再来一个就连着两行了
    return "\n".join(art + ["", body])

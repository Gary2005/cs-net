"""关注名单：跟的是 **Steam 账号**，不是一个游戏内名字。

为什么名单里还能放名字
----------------------
最初这个名单只存游戏内名字，用**子串**匹配（"Zyw" 也能命中 "ZywOo"）。它对
"我关注这个人"够用，但对"我关注这个账号"不够：同一个人在 5E 上叫一个名字、
在官匹上叫另一个，按名字关注只跟得住其中一半，而且换一次名字就丢了。

SteamID64 是账号本身的标识，跨平台、跨改名都不变。所以名单现在两种条目都收，
一个条目一种解释：

    76561198851261234    17 位纯数字 → 按 SteamID **精确**匹配
    Zyw                  其余        → 按游戏内名字**子串**匹配（老用法）

一个列表而不是两个配置项，是因为用户的心智模型就是"一行一个人"：他既可以从
玩家页点「加入关注」（写 SteamID），也可以自己在设置里手打半截名字。老配置
不用迁移，仍然有效。

所有"这个人算不算被关注"的判断都必须走这里 —— 库卡片的高亮、玩家列表的
"只看关注"、库的 `watched=1` 筛选、`watched_json` 的重算，四处口径要一致。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

#: SteamID64 的形状：17 位纯数字。个人账号一律以 7656 开头，但这里不校验前缀
#: —— 换个平台导出个别的 ID 也未必是个人账号，长度够就是 ID，不是名字。
_STEAMID_RE = re.compile(r"^\d{17}$")

#: 名字条目最短的有效长度。一个字符的条目会把所有名字都点亮（"a" 命中一大片），
#: 和"关注某个人"没有关系，所以直接忽略。
_MIN_NAME = 1


def normalize(entry: Any) -> str:
    """条目的比较键：SteamID 原样，名字小写去空白。"""
    text = str(entry or "").strip()
    return text if is_steamid(text) else text.lower()


def is_steamid(entry: Any) -> bool:
    """这个条目是不是一个 SteamID64。"""
    return bool(_STEAMID_RE.match(str(entry or "").strip()))


def split(entries: Optional[Iterable[Any]]) -> tuple[list[str], list[str]]:
    """把一份名单拆成 ``(名字列表, SteamID 列表)``，各自去重并保持顺序。"""
    names: list[str] = []
    ids: list[str] = []
    for raw in entries or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if is_steamid(text):
            if text not in ids:
                ids.append(text)
        elif len(text) >= _MIN_NAME:
            low = text.lower()
            if low not in names:
                names.append(low)
    return names, ids


def matches(entry: Any, *, name: Any = "", steamid: Any = "") -> bool:
    """这一条目是否命中给定的人（名字 + SteamID）。"""
    text = str(entry or "").strip()
    if not text:
        return False
    if is_steamid(text):
        return text == str(steamid or "").strip()
    return text.lower() in str(name or "").lower()


def matched_entry(
    entries: Optional[Iterable[Any]], *, name: Any = "", steamid: Any = ""
) -> Optional[str]:
    """名单里第一个命中的条目（原始写法），没有则 ``None``。

    SteamID 优先：名字是会被改的显示信息，账号才是身份。
    """
    candidates = [str(e or "").strip() for e in (entries or []) if str(e or "").strip()]
    for entry in candidates:
        if is_steamid(entry) and entry == str(steamid or "").strip():
            return entry
    for entry in candidates:
        if not is_steamid(entry) and entry.lower() in str(name or "").lower():
            return entry
    return None


def is_watched(
    entries: Optional[Iterable[Any]], *, name: Any = "", steamid: Any = ""
) -> bool:
    return matched_entry(entries, name=name, steamid=steamid) is not None

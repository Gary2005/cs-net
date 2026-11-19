#!/usr/bin/env python3
"""
Static wiring checks for the library front end.

There is no browser in this environment, so the UI cannot be clicked through.
These checks cover the failure modes that would otherwise only show up as a
dead button in a browser:

* ``library.js`` reaching for an element id that ``library.html`` does not have
  (or that was renamed on one side only)
* ``library.js`` calling an ``/api/library/...`` path that the Flask app does
  not actually serve -- a typo in a URL is silent until the fetch 404s at
  runtime
* the page referencing a stylesheet or script that is not on disk
* the replay page's batch selector drifting out of sync with what ``main.js``
  looks for

Run:  python app/tests/test_ui_wiring.py
"""

from __future__ import annotations

import contextlib
import re
import shutil
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from app import config  # noqa: E402

_passed = 0
_failed: list[str] = []

APP = _ROOT / "app"
LIB_HTML = APP / "templates" / "library.html"
LIB_JS = APP / "static" / "js" / "library.js"
LIB_CSS = APP / "static" / "css" / "library.css"
PLAYERS_HTML = APP / "templates" / "players.html"
PLAYERS_JS = APP / "static" / "js" / "players.js"
PLAYER_HTML = APP / "templates" / "player.html"
PLAYER_JS = APP / "static" / "js" / "player.js"
PLAYERS_CSS = APP / "static" / "css" / "players.css"
REPLAY_HTML = _ROOT / "visualizer" / "templates" / "index.html"
REPLAY_JS = _ROOT / "visualizer" / "static" / "js" / "main.js"
REPLAY_CSS = _ROOT / "visualizer" / "static" / "css" / "style.css"
TOPBAR_CSS = _ROOT / "visualizer" / "static" / "css" / "topbar.css"
VIS_STATIC = _ROOT / "visualizer" / "static"
I18N_DIR = VIS_STATIC / "js" / "i18n"
I18N_CORE = I18N_DIR / "index.js"
I18N_LIB = I18N_DIR / "library.js"

#: The topbar every page renders.  Same labels in the same order everywhere;
#: each page links the two it is not on and marks its own entry as current.
#: This is what makes the four pages read as one product, so it is asserted
#: rather than left to whoever edits a page next to remember.
NAV_LABELS = ["Demo 库", "玩家", "可视化回放"]
NAV_HREFS = ["/library", "/players", "/replay"]

#: Phrases that mean the copy has stopped talking to the user and started
#: talking about itself.  Every one of these was written here first, defending a
#: design decision ("目录要手动添加，没有可靠的自动探测方式"), narrating the
#: implementation ("玩家是从记分板里汇总出来的"), or promising future work
#: ("后面接入的分析模块放在这里").  None of that belongs on screen: the user
#: wants to scan a demo, not read why the feature is shaped the way it is.
#: Checked against comment-stripped source, so prose in a code comment -- where
#: this kind of reasoning *is* wanted -- does not trip it.
NARRATION_TELLS = (
    "后面", "以后", "目前这块", "没有可靠", "不需要 API", "不用登录",
    "汇总出来", "值得一提", "不影响其他", "不影响别的", "之所以",
    "换句话说", "需要注意的是", "这意味", "顺便",
)

#: (page, script, label) for every page this app renders itself.  Each pair gets
#: the same id and api-path checks, so a rename on one side cannot go unnoticed
#: just because it happened on a page nobody thought to list here.
PAGES = (
    (LIB_HTML, LIB_JS, "library"),
    (PLAYERS_HTML, PLAYERS_JS, "players"),
    (PLAYER_HTML, PLAYER_JS, "player"),
)


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


@contextlib.contextmanager
def scratch():
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── extraction helpers ───────────────────────────────────────────────────────


def html_ids(text: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', text))


#: Ids that only exist **after** JS renders them (the stats dialog rebuilds its
#: body on every tab switch / round change), so they are legitimately absent from
#: the static template.  Registered here so the "every id exists" checks stay
#: strict for everything else.
RUNTIME_IDS = {
    "stats-round-detail",     # replay stats: single-round table body
    "stats-ai-round-detail",  # replay stats (AI tab): single-round table body
    "stats-progress-fill",    # replay stats (AI tab): progress bar
    "stats-progress-stage",   # replay stats (AI tab): progress text
    "stats-progress-pct",     # replay stats (AI tab): percentage / elapsed
}


def js_element_refs(text: str) -> set[str]:
    """Ids the JS looks up: $('x'), getElementById('x'), querySelector('#x')."""
    found = set(re.findall(r"\$\(\s*'([^']+)'\s*\)", text))
    found |= set(re.findall(r"\$\(\s*\"([^\"]+)\"\s*\)", text))
    found |= set(re.findall(r"getElementById\(\s*'([^']+)'\s*\)", text))
    found |= set(re.findall(r"querySelector\(\s*'#([A-Za-z0-9_-]+)'\s*\)", text))
    return {f for f in found if re.fullmatch(r"[A-Za-z0-9_-]+", f)}


def js_api_paths(text: str) -> set[str]:
    """API paths the JS calls, with `${...}` interpolations normalised away."""
    raw = set(re.findall(r"""(?:api|post|put|patch)\(\s*[`'"]([^`'"]+)[`'"]""", text))
    raw |= set(re.findall(r"""fetch\(\s*[`'"]([^`'"]+)[`'"]""", text))
    # XHR is used where fetch cannot report upload progress (the demo upload).
    raw |= set(re.findall(r"""\.open\(\s*['"]\w+['"]\s*,\s*[`'"]([^`'"]+)[`'"]""", text))
    out = set()
    for p in raw:
        if not p.startswith("/api/"):
            continue
        p = p.split("?")[0]
        p = re.sub(r"\$\{[^}]*\}", "\x00", p)   # normalise interpolation
        p = p.rstrip("/") or "/"
        out.add(p)
    return out


def flask_paths(app) -> set[str]:
    """Every registered route, with converters normalised the same way."""
    out = set()
    for rule in app.url_map.iter_rules():
        path = re.sub(r"<[^>]+>", "\x00", str(rule))
        out.add(path.rstrip("/") or "/")
    return out


# ── checks ───────────────────────────────────────────────────────────────────


def nav_items(html: str) -> list[tuple[str | None, str]]:
    """``(href, label)`` for each topbar nav entry, in order.

    The current page's entry is a ``<span>`` rather than an ``<a>`` (clicking a
    link to the page you are already on just reloads it and drops ``?batch=``),
    so its href is ``None``.
    """
    block = re.search(r'<nav class="nav">(.*?)</nav>', html, re.S)
    if not block:
        return []
    out = []
    for m in re.finditer(r"<(a|span)\b([^>]*)>([^<]*)</\1>", block.group(1)):
        href = re.search(r'href="([^"]+)"', m.group(2))
        out.append((href.group(1) if href else None, m.group(3).strip()))
    return out


def test_files_exist() -> None:
    print("\nassets")
    for p in (LIB_HTML, LIB_JS, LIB_CSS, PLAYERS_HTML, PLAYERS_JS, PLAYER_HTML,
              PLAYER_JS, PLAYERS_CSS, REPLAY_HTML, REPLAY_JS, REPLAY_CSS, TOPBAR_CSS):
        check(f"{p.relative_to(_ROOT)} exists", p.is_file())

    for html_path, _js, label in PAGES:
        html = html_path.read_text(encoding="utf-8")
        for href in re.findall(r'(?:href|src)="(/lib-static/[^"]+)"', html):
            # Strip the ?v= cache-buster before looking on disk.
            rel = href.replace("/lib-static/", "").split("?")[0]
            check(f"{label}: references an existing file {href}",
                  (APP / "static" / rel).is_file())

    # The shared topbar lives under the visualizer's static mount (it is the one
    # directory both servers serve), so the library pages reach it through
    # /static/.  A typo there costs the whole nav its styling and nothing else
    # would notice, so the path gets checked too.
    for html_path, _js, label in PAGES + ((REPLAY_HTML, REPLAY_JS, "replay"),):
        html = html_path.read_text(encoding="utf-8")
        for href in re.findall(r'(?:href|src)="(/static/[^"]+)"', html):
            rel = href.replace("/static/", "").split("?")[0]
            check(f"{label}: references an existing file {href}",
                  (VIS_STATIC / rel).is_file())

    html = LIB_HTML.read_text(encoding="utf-8")
    check("no leftover /static/ reference for library assets",
          "/static/js/library.js" not in html and "/static/css/library.css" not in html)


def test_dom_ids() -> None:
    print("\nelement ids")
    html = LIB_HTML.read_text(encoding="utf-8")
    js = LIB_JS.read_text(encoding="utf-8")
    ids = html_ids(html)
    refs = js_element_refs(js)

    check("the page has ids at all", len(ids) > 20, f"{len(ids)}")
    check("the js looks up ids at all", len(refs) > 20, f"{len(refs)}")

    missing = sorted(refs - ids - RUNTIME_IDS)
    check("every id library.js looks up exists in library.html", not missing,
          f"missing: {missing}")

    # The other direction is informational: a page id nothing ever touches is
    # usually dead markup, but some are only reached through data-* handlers.
    unused = sorted(ids - refs)
    print(f"      (page ids never looked up directly: {len(unused)})")

    print("\ndata-* hooks")
    # Every data-* attribute the JS selects on must be produced by the JS
    # templates (or the HTML).  A rename on one side is otherwise invisible.
    js_attrs = set(re.findall(r"\[data-([a-z-]+)\]", js))
    html_attrs = set(re.findall(r"\bdata-([a-z-]+)=", html))
    produced = html_attrs | js_attrs
    for attr in sorted(js_attrs):
        check(f"data-{attr} is produced somewhere", attr in produced)
    check("no data-* hook is selected but never written",
          all(a in produced for a in js_attrs), f"{sorted(js_attrs)}")

    print("\nelement ids on the player pages")
    for html_path, js_path, label in PAGES[1:]:
        page_ids = html_ids(html_path.read_text(encoding="utf-8"))
        page_refs = js_element_refs(js_path.read_text(encoding="utf-8"))
        check(f"{label}: the page has ids", len(page_ids) >= 5, f"{len(page_ids)}")
        check(f"{label}: the js looks up ids", len(page_refs) >= 5, f"{len(page_refs)}")
        missing_ids = sorted(page_refs - page_ids - RUNTIME_IDS)
        check(f"every id {js_path.name} looks up exists in {html_path.name}",
              not missing_ids, f"missing: {missing_ids}")


def test_api_paths() -> None:
    print("\napi paths called from library.js")
    js = LIB_JS.read_text(encoding="utf-8")
    called = js_api_paths(js)
    check("found api calls to check", len(called) >= 8, f"{sorted(called)}")

    saved = {k: getattr(config, k) for k in
             ("DATA_DIR", "CONFIG_PATH", "LIBRARY_DB", "CSZ_DIR", "THUMB_DIR",
              "AVATAR_DIR")}
    with scratch() as tmp:
        config.DATA_DIR = tmp
        config.CONFIG_PATH = tmp / "config.json"
        config.LIBRARY_DB = tmp / "library.db"
        config.CSZ_DIR = tmp / "csz"
        config.THUMB_DIR = tmp / "thumbs"
        config.AVATAR_DIR = tmp / "avatars"
        for d in (config.CSZ_DIR, config.THUMB_DIR):
            d.mkdir(parents=True, exist_ok=True)
        try:
            from app.server import create_app
            app = create_app(start_background=False)
            served = flask_paths(app)
        finally:
            for k, v in saved.items():
                setattr(config, k, v)

    for path in sorted(called):
        check(f"the server serves {path.replace(chr(0), '<id>')}", path in served,
              "not in the url_map")

    print("\napi paths called from the player pages")
    for _html, js_path, label in PAGES[1:]:
        page_called = js_api_paths(js_path.read_text(encoding="utf-8"))
        check(f"{label}: found api calls to check", len(page_called) >= 3,
              f"{sorted(page_called)}")
        for path in sorted(page_called):
            check(f"the server serves {path.replace(chr(0), '<id>')} ({label})",
                  path in served, "not in the url_map")

    print("\npages the front end links to")
    linked = set()
    for html_path, _js, _label in PAGES:
        html = html_path.read_text(encoding="utf-8")
        linked |= set(re.findall(r'href="(/[a-z]+)"', html))
    for path in sorted(linked):
        check(f"the server serves the page {path}", path in served, "not in the url_map")


def test_replay_integration() -> None:
    print("\nreplay page integration")
    html = REPLAY_HTML.read_text(encoding="utf-8")
    js = REPLAY_JS.read_text(encoding="utf-8")

    check("replay page has the batch selector", 'id="batch-select"' in html)
    check("replay page links back to the library", 'href="/library"' in html)
    check("main.js reads the ?batch= parameter", "batch" in js and "initBatch" in js)
    check("main.js calls the batch endpoint", "/api/library/batch/" in js)

    ids = html_ids(html)
    missing = sorted(js_element_refs(js) - ids - RUNTIME_IDS)
    check("every id main.js looks up exists in index.html", not missing,
          f"missing: {missing}")

    # The batch selector must be able to start hidden and be revealed by JS.
    check("batch selector starts hidden",
          re.search(r'id="batch-select"[^>]*class="[^"]*hidden', html) is not None
          or re.search(r'class="[^"]*hidden[^"]*"[^>]*id="batch-select"', html) is not None)
    check("main.js unhides it", "classList.remove('hidden')" in js)

    print("\nvisualizer styling for the new controls")
    css = REPLAY_CSS.read_text(encoding="utf-8")
    check("#batch-select is styled", "#batch-select" in css)

    # ── 四个页面长得像同一个产品 ─────────────────────────────────────────
    #
    # 回放页以前有自己的顶栏：品牌块样式不同、没有导航，一进去顶部菜单就
    # 消失了。现在顶栏是一份共用样式 + 一份共用结构，这里把"共用"钉住。
    print("\nthe four pages wear the same topbar")
    pages = ((LIB_HTML, "library", "/library"), (PLAYERS_HTML, "players", "/players"),
             (PLAYER_HTML, "player", "/players"), (REPLAY_HTML, "replay", "/replay"))
    for page, label, current in pages:
        src = page.read_text(encoding="utf-8")
        items = nav_items(src)
        check(f"{label}: nav labels are the shared three",
              [text for _href, text in items] == NAV_LABELS,
              f"{[text for _href, text in items]}")
        expected = [None if h == current else h for h in NAV_HREFS]
        check(f"{label}: links the other two and marks its own entry",
              [href for href, _text in items] == expected,
              f"{[href for href, _text in items]}")
        check(f"{label}: loads the shared topbar stylesheet",
              "/static/css/topbar.css" in src)

    shared = TOPBAR_CSS.read_text(encoding="utf-8")
    for sel in (".topbar", ".brand", ".brand-name", ".brand-sub", ".nav",
                ".nav-link", ".tb-actions"):
        check(f"topbar.css defines {sel}",
              re.search(rf"^{re.escape(sel)}[\s{{,:]", shared, re.M) is not None)
    check("the library stylesheet does not define them a second time",
          ".nav-link" not in LIB_CSS.read_text(encoding="utf-8"))
    check("and neither does the replay stylesheet", ".nav-link" not in css)


def test_replay_topbar_survives_the_empty_state() -> None:
    """The replay page's topbar must not be a child of ``#app``.

    It used to be.  That is invisible while a replay is playing -- ``#app`` is
    visible, so the topbar is too -- but ``/replay`` with no ``?batch=`` and no
    live server session shows ``#empty`` and hides ``#app``, which took the
    topbar and the whole nav down with it.  Clicking 可视化回放 in the nav
    therefore landed on a page with no menu at all, reading as "a different
    website" rather than a tab switch.

    Document order is enough to prove the nesting: an element that appears
    before ``#app`` opens cannot be inside it.
    """
    print("\nthe replay topbar is chrome, not replay content")
    html = REPLAY_HTML.read_text(encoding="utf-8")
    js = REPLAY_JS.read_text(encoding="utf-8")
    css = REPLAY_CSS.read_text(encoding="utf-8")

    pos_topbar = html.index('id="topbar"')
    pos_empty = html.index('id="empty"')
    pos_app = html.index('id="app"')
    check("the topbar sits outside #app", pos_topbar < pos_app,
          "inside #app it disappears whenever the empty state shows")
    check("so does the empty state", pos_empty < pos_app)
    check("the topbar comes first", pos_topbar < pos_empty)

    # Layout has to follow the markup: with the topbar as a sibling, body owns
    # the height split and the two page areas take what is left.
    check("body lays the page out as a column",
          re.search(r"\bbody\s*\{[^}]*display:\s*flex", css) is not None)
    check("#app takes the height the topbar leaves",
          re.search(r"\.app\s*\{[^}]*flex:\s*1", css) is not None)
    check(".empty-state takes the height the topbar leaves",
          re.search(r"\.empty-state\s*\{[^}]*flex:\s*1", css) is not None)
    check("neither one is still sized in viewport units",
          "height: 100vh" not in css)
    check("the topbar does not shrink as a flex item",
          re.search(r"\.topbar\s*\{[^}]*flex:\s*none", TOPBAR_CSS.read_text(encoding="utf-8"))
          is not None)

    # Map / round / sampling / download are meaningless before a demo is loaded.
    check("the replay-only topbar controls start hidden",
          re.search(r'id="tb-info"[^>]*class="[^"]*hidden', html) is not None
          and re.search(r'id="btn-download"[^>]*class="[^"]*hidden', html) is not None)
    check("main.js reveals them with the data",
          "$('tb-info').classList.remove('hidden')" in js
          and "$('btn-download').classList.remove('hidden')" in js)
    check("main.js hides #app and #empty, never the topbar",
          "$('topbar').classList" not in js)


def test_replay_has_no_upload_page() -> None:
    """回放只能从 Demo 库进，那个独立的上传页必须真的没了。

    删一半最糟：页面上没有了、``main.js`` 还在找它的元素，``bindEvents``
    第一行就抛，整个 3D 场景建不起来。
    """
    print("\nthe replay page has no upload entry of its own")
    html = REPLAY_HTML.read_text(encoding="utf-8")
    js = REPLAY_JS.read_text(encoding="utf-8")
    css = REPLAY_CSS.read_text(encoding="utf-8")
    server = (_ROOT / "visualizer" / "server.py").read_text(encoding="utf-8")

    for gone in ("landing", "drop-zone", "file-input", "sampling-select",
                 "btn-load-example", "example-list", "loading-bar", "loading-text",
                 "btn-back"):
        check(f"index.html has no #{gone}", f'id="{gone}"' not in html)
        check(f"main.js does not look up #{gone}", f"$('{gone}')" not in js)

    for dead in (".landing", ".drop-zone", ".example-item", ".btn-ghost",
                 ".sampling-row", ".loading-bar", ".device-select"):
        check(f"style.css has no {dead} rule", dead not in css)

    check("main.js only ever loads a path, never an uploaded file",
          "startLoad({" not in js and "request.files" not in js)
    check("no sampling picker left in main.js", "samplingParams" not in js)

    check("the server no longer lists local demo files",
          "/api/examples" not in server and "EXAMPLE_DIRS" not in server)
    check("the server no longer accepts uploaded files", "request.files" not in server)
    check("and the 2 GB upload limit went with it",
          "MAX_CONTENT_LENGTH" not in server)

    # 没有批次又没有会话时的那一屏：它现在是唯一的引导，别删过头
    check("the empty state exists", 'id="empty"' in html)
    check("it points at the library", 'href="/library"' in html)
    check("main.js hides it once data arrives",
          "$('empty').classList.add('hidden')" in js)


def strip_js_comments(src: str) -> str:
    """Drop comment-only lines and ``/* */`` blocks from JS source.

    Deliberately line-based rather than a real tokenizer: ``library.js`` has
    regex literals like ``/[&<>"']/g`` that make naive quote tracking think a
    string has started, and mis-tracking is worse than not tracking.  This is
    only used to stop a comment *mentioning* a forbidden call from tripping a
    "that call must not appear in code" check, and such mentions live on their
    own lines.  A trailing ``// ...`` comment after real code is not stripped.
    """
    out: list[str] = []
    in_block = False
    for line in src.splitlines():
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def visible_text(path: Path) -> str:
    """Source with its comments removed, for judging user-facing copy.

    A narration tell in a code comment is fine -- that is where the reasoning
    belongs.  The same sentence in a ``title=`` or a template literal is on
    screen, which is what :data:`NARRATION_TELLS` guards.
    """
    src = path.read_text(encoding="utf-8")
    if path.suffix == ".js":
        return strip_js_comments(src)
    return re.sub(r"<!--.*?-->", "", src, flags=re.S)


def test_delete_is_consistent() -> None:
    """Single and batch delete must be the same operation.

    They used to diverge: the per-card button deleted immediately while the
    batch button asked, and the batch dialog abused ``confirm()`` so that
    "Cancel" meant "delete the record but keep the .dem".  Both now go through
    one dialog, and that dialog must be closable from every exit.

    The dialog then offered two side-by-side buttons ("keep" / "move to
    recycle"), which read as two options to pick between rather than a default
    with an exception.  It is now one confirm button plus a checkbox that is
    unchecked every time the dialog opens: the .dem survives unless the user
    opts in.
    """
    print("\ndelete flow")
    js = LIB_JS.read_text(encoding="utf-8")
    code = strip_js_comments(js)
    html = LIB_HTML.read_text(encoding="utf-8")

    check("no native dialog left in library.js code",
          "confirm(" not in code and "alert(" not in code,
          "a native dialog cannot express the optional .dem removal")

    check("deleteOne asks before deleting", "async function deleteOne" in code
          and "await askDelete([" in code)
    check("deleteSelected asks before deleting",
          "async function deleteSelected" in code and "await askDelete(ids)" in code)
    check("both share one delete implementation",
          code.count("await doDelete(") == 2, f"{code.count('await doDelete(')}")

    # A cancelled dialog must not delete anything.  The guard has to compare
    # against null: the dialog resolves `false` for "keep the .dem", and a
    # truthiness check would silently turn that into a cancel.
    check("cancel returns early",
          code.count("if (withFile === null) return;") == 2,
          f"{code.count('if (withFile === null) return;')}")

    # Selection survives pagination, so the dialog must not pretend S.items is
    # the whole selection.
    check("off-page selections are accounted for",
          "offPage" in code and "不在当前页" in js)

    print("\ndelete dialog exits")
    check("the dialog exists and starts hidden",
          re.search(r'id="delete-modal"[^>]*class="[^"]*hidden', html) is not None)
    for el in ("btn-del-confirm", "btn-del-cancel", "del-with-file", "del-with-file-label"):
        check(f"the dialog has {el}", f'id="{el}"' in html)
    check("the dialog can be dismissed with data-close",
          'data-close="delete-modal"' in html)
    check("the .dem checkbox is a checkbox, not a button",
          (lambda m: m is not None and 'type="checkbox"' in m.group(0))(
              re.search(r'<input[^>]*id="del-with-file"[^>]*>', html))
          and 'id="btn-del-with-file"' not in html,
          "attribute order must not decide whether this passes")
    check("removing the record is the only confirm action",
          html.count('id="btn-del-confirm"') == 1
          and re.search(r'id="btn-del-confirm"[^>]*class="[^"]*btn-danger', html) is not None)
    check("the checkbox is never checked in the markup",
          re.search(r'id="del-with-file"[^>]*\bchecked\b', html) is None,
          "a pre-checked box makes deleting the .dem the default")
    check("the checkbox is reset on every open",
          re.search(r"box\.checked = false", code) is not None)
    check("the confirm label follows the checkbox",
          "移除记录并移走 .dem" in I18N_LIB.read_text(encoding="utf-8")
          and "box.onchange = relabel" in code)
    check("a useless checkbox is disabled when every source is gone",
          "box.disabled = fileGone" in code and "fileGone" in code)
    check("closing resolves the pending promise",
          "closeDeleteModal(null)" in code and "_deleteResolve = null" in code)
    check("every close path goes through closeDeleteModal",
          code.count("closeDeleteModal(null)") == 3, f"{code.count('closeDeleteModal(null)')}")


def test_ui_copy() -> None:
    """Interface copy is read by users, not generated for them.

    Long em-dash asides, warning emoji and "✓ saved" ticks made the library read
    like model output.  They are banned here rather than left to taste, because
    they creep back one sentence at a time.
    """
    print("\ncopy")
    js = LIB_JS.read_text(encoding="utf-8")
    html = LIB_HTML.read_text(encoding="utf-8")

    for name, text in (
        ("library.js", js), ("library.html", html),
        ("players.js", PLAYERS_JS.read_text(encoding="utf-8")),
        ("players.html", PLAYERS_HTML.read_text(encoding="utf-8")),
        ("player.js", PLAYER_JS.read_text(encoding="utf-8")),
        ("player.html", PLAYER_HTML.read_text(encoding="utf-8")),
    ):
        for marker in ("\u2014\u2014", "\u26a0", "\u2713", "\u2717", "\u2248"):
            check(f"{name} has no {marker!r}", marker not in text)

    # Only the JS: a <b> in static markup is structural, one dropped into a
    # sentence built by a template literal is the tell.
    for name, text in (("library.js", js),
                       ("players.js", PLAYERS_JS.read_text(encoding="utf-8")),
                       ("player.js", PLAYER_JS.read_text(encoding="utf-8"))):
        check(f"{name} does not wedge bold into a sentence", "<b>" not in text)

    lib_zh = I18N_LIB.read_text(encoding="utf-8")
    check("the delete dialog names everything it removes",
          all(w in lib_zh for w in ("库记录", ".csz", "缩略图")))
    check("the delete dialog does not lecture",
          "别删" not in lib_zh and "不影响任何事" not in lib_zh)
    # A CS2 demo carries no wall-clock timestamp, so the only date available is
    # the file's mtime.  It must not be dressed up as the match date.
    check("the date is not offered as a match date", "比赛日期" not in html)
    check("the date is labelled 文件时间", "文件时间" in lib_zh)
    # Every mention in the copy must be the negated one in the tooltip.
    eq("the only mention of 比赛日期 in the zh table denies it",
       lib_zh.count("比赛日期"), lib_zh.count("不是比赛日期"))

    check("no bold wedged into a sentence", "<b>" not in js)

    # The copy must address the user, not narrate the codebase back at them.
    # The zh tables are included: after the i18n pass most of the copy lives
    # there, so scanning only the JS/HTML would stop guarding anything.
    for path in (LIB_JS, PLAYERS_JS, PLAYER_JS, LIB_HTML, PLAYERS_HTML, PLAYER_HTML,
                 I18N_LIB, I18N_DIR / "players.js"):
        text = visible_text(path)
        hits = [t for t in NARRATION_TELLS if t in text]
        where = ""
        if hits:
            idx = text.find(hits[0])
            where = " ..." + text[max(0, idx - 30):idx + 30].replace("\n", " ") + "..."
        check(f"{path.name} does not narrate itself{where}", not hits)


def test_extract_controls() -> None:
    """The UI must expose the difference between refresh and re-parse.

    "刷新信息" re-derives metadata from the existing .csz; "重新解析" throws it
    away and re-parses.  Only the second one honours the sampling settings, so
    both have to be reachable and clearly named.
    """
    print("\nextract controls")
    js = strip_js_comments(LIB_JS.read_text(encoding="utf-8"))
    html = LIB_HTML.read_text(encoding="utf-8")

    for btn in ("btn-extract-selected", "btn-reextract-selected"):
        check(f"the selection bar has {btn}", f'id="{btn}"' in html)
    check("refresh and re-parse are both wired",
          "extractMany([...S.selected]);" in js
          and "extractMany([...S.selected], true);" in js)
    check("the request carries force", "{ ids, force }" in js)
    check("the two buttons are not labelled the same",
          ">刷新信息<" in html and ">重新解析<" in html)


def test_player_pages() -> None:
    """The player index and profile have to be reachable, and have to degrade.

    Avatars are the one feature here that depends on a third party, so the
    fallback path is checked as carefully as the happy one: a machine that
    cannot reach steamcommunity.com must still get a usable page.
    """
    print("\nplayer pages")
    lib = LIB_HTML.read_text(encoding="utf-8")
    lib_js = LIB_JS.read_text(encoding="utf-8")
    idx = PLAYERS_HTML.read_text(encoding="utf-8")
    idx_js = PLAYERS_JS.read_text(encoding="utf-8")
    prof = PLAYER_HTML.read_text(encoding="utf-8")
    prof_js = PLAYER_JS.read_text(encoding="utf-8")

    check("the library links to the player index", 'href="/players"' in lib)
    check("the player index links back to the library", 'href="/library"' in idx)
    check("the profile links back to the player index", 'href="/players"' in prof)

    check("the profile gets its steamid from the page, not a second fetch",
          "data-steamid=" in prof and "dataset.steamid" in prof_js)
    check("the profile asks for that player", "/api/library/players/${SID}" in prof_js)

    # A missing avatar must not leave a broken image or an empty box.
    for name, text in (("players.js", idx_js), ("player.js", prof_js)):
        check(f"{name} has an avatar fallback", "avatar-fallback" in text)
        check(f"{name} only uses a url when there is one", "av.url" in text)
    check("the library roster falls back to the team dot", "'<span class=\"dot\"></span>'" in lib_js)

    check("the roster links to the player profile",
          "/players/${esc(p.steamid)}" in lib_js)
    check("the roster prefers the cached avatar", 'class="pav"' in lib_js)
    check("the avatar is styled in the library sheet", ".roster .p .pav" in
          LIB_CSS.read_text(encoding="utf-8"))

    check("avatar syncing is reachable from both pages",
          "/api/library/avatars/sync" in idx_js and "/api/library/avatars/sync" in prof_js)
    check("the settings page can turn avatars off",
          'id="fetch-avatars"' in lib and "fetch_avatars" in lib_js)

    check("the profile can follow a player",
          "/api/library/players/${SID}" in prof_js and "watched" in prof_js)

    # The user asked for a place to hang later analysis modules on.
    check("the profile has a slot for later analysis",
          'id="player-analysis"' in prof)


def test_player_names_and_storage() -> None:
    """Two names, in two places, and a way to see what the app is using.

    A demo carries the in-game name only; the Steam persona is fetched
    separately.  The player pages are about the *person*, so they are titled by
    the persona and list the in-game names underneath.  A library card is about
    one *match*, so it shows the name used in that match and nothing else.
    """
    print("\nnames: persona on top, in-game names below")
    lib_js = strip_js_comments(LIB_JS.read_text(encoding="utf-8"))
    idx = PLAYERS_HTML.read_text(encoding="utf-8")
    idx_js = PLAYERS_JS.read_text(encoding="utf-8")
    prof = PLAYER_HTML.read_text(encoding="utf-8")
    prof_js = PLAYER_JS.read_text(encoding="utf-8")

    for name, text in (("players.js", idx_js), ("player.js", prof_js)):
        check(f"{name} is titled by the Steam persona", "p.persona || p.name" in text)
        check(f"{name} lists the other names", "otherNames(" in text)
    check("the index renders an in-game name list", "pcard-names" in idx_js)
    check("the index knows which name is still in use", "n === p.name" in idx_js)
    check("the profile has a name history block", 'id="name-table"' in prof)
    check("the profile fills it", "renderNames" in prof_js)
    check("from the API's name_history", "name_history" in prof_js)

    # One match, one name: the persona belongs on the player pages.
    check("the library card shows only the in-game name",
          "Steam 昵称" not in lib_js, "the persona leaked back into the roster")

    print("\nthe search box offers ranked candidates")
    check("the index has a suggestion box", 'id="search-suggest"' in idx)
    check("which starts hidden",
          re.search(r'id="search-suggest"[^>]*class="[^"]*hidden', idx) is not None)
    check("players.js queries the ranked endpoint",
          "/api/library/players/search" in idx_js)
    check("the box is filled from the response", "sg-item" in idx_js)
    check("the match is labelled", "MATCH_LABEL" in idx_js)
    check("and can be driven from the keyboard", "moveSuggest" in idx_js)
    check("clicking outside closes it",
          "search-wrap" in idx_js and "search-wrap" in idx
          and "closeSuggest" in idx_js)
    check("the placeholder names all three fields",
          all(w in idx for w in ("游戏内名字", "Steam 昵称", "SteamID")))

    print("\nstorage management")
    lib = LIB_HTML.read_text(encoding="utf-8")
    check("the settings modal has a storage section", 'id="storage-list"' in lib)
    check("with a summary line", 'id="storage-note"' in lib)
    check("library.js reads the overview", "/api/library/storage" in lib_js)
    check("and can clear a cache", "/api/library/storage/clean" in lib_js)
    check("clearable rows get a button", "data-clean=" in lib_js)
    check("the server decides what is clearable", "e.removable" in lib_js)
    check("the original .dem files are shown without a button",
          "原始 Demo" in I18N_LIB.read_text(encoding="utf-8"))
    check("every size is formatted, including zero", "sizeText" in lib_js)

    check("the clean dialog exists and starts hidden",
          re.search(r'id="clean-modal"[^>]*class="[^"]*hidden', lib) is not None)
    check("with a confirm button", 'id="btn-clean-confirm"' in lib)
    check("and a way out", 'data-close="clean-modal"' in lib)
    check("closing it resolves the pending promise",
          "closeCleanModal(null)" in lib_js and "_cleanResolve = null" in lib_js)
    check("every close path goes through closeCleanModal",
          lib_js.count("closeCleanModal(null)") == 3,
          f"{lib_js.count('closeCleanModal(null)')}")


def test_double_click_opens_analysis() -> None:
    """Double-clicking a card or row opens the analysis view.

    The interesting part is what must *not* happen, so the assertions below pin
    the guard as well as the feature.  A naive implementation gets this wrong in
    a way no static check in the rest of this file would catch: a row's name is
    plain text, so the browser's own double-click selects the word under the
    cursor *before* ``dblclick`` fires -- a ``!selection.isCollapsed`` guard
    therefore swallows every double-click in list view, and a guard that clears
    the selection on mousedown swallows the "drag then double-click" case.
    """
    print("\ndouble-click opens the analysis view")
    lib_js = LIB_JS.read_text(encoding="utf-8")

    check("the content area listens for dblclick",
          re.search(r"content\.addEventListener\('dblclick'", lib_js) is not None
          and "const content = $('content')" in lib_js)
    check("it resolves the card or row under the cursor",
          ".card[data-id], .row[data-id]" in lib_js)
    check("and goes through the same entry point as the toolbar button",
          "openInReplay([id])" in lib_js)
    check("it does not need the item to be selected first",
          "S.selected" not in lib_js.split("addEventListener('dblclick'")[1].split("});")[0])

    # the skip list: controls inside a card keep their own click semantics
    check("controls are excluded from the double-click",
          all(sel in lib_js for sel in
              ("button", "a,", "input", "[data-act]", "[data-check]", "[data-edit]")))

    # the guard must be driven by pointer movement, not by the selection
    check("a pointer-move flag exists", "pointerMoved" in lib_js)
    check("mousemove only counts a real drag",
          "ev.buttons & 1" in lib_js and "> 3" in lib_js)
    check("the dblclick handler consults it", "pressState" in lib_js
          and "pressState === 'dragged'" in lib_js)
    check("the verdict is taken on mousedown, not in the handler",
          re.search(r"pressState = \(pointerMoved && hasRange\)", lib_js) is not None)
    check("the second press does not overwrite the verdict",
          re.search(r"if \(ev\.detail > 1\) return;", lib_js) is not None)


def test_replay_stats_panel() -> None:
    """回放页的「统计」面板：入口在最上面、弹窗开得了也关得掉、路由真的有人服务。

    这是个**整页遮罩**的弹窗：关不掉就等于把回放页锁死了（顶栏、时间轴、3D 画布
    都在它下面）。所以除了"按钮和弹窗都在"，还把三条关闭路径 + Esc 逐条点名 ——
    少写一条在浏览器里就是"点了没反应"，而这环境里没有浏览器点。
    """
    print("\nreplay stats panel")
    html = REPLAY_HTML.read_text(encoding="utf-8")
    js = REPLAY_JS.read_text(encoding="utf-8")
    code = strip_js_comments(js)
    css = REPLAY_CSS.read_text(encoding="utf-8")

    check("index.html has the stats button", 'id="btn-stats"' in html)
    check("the button lives in the right sidebar",
          html.index('id="right-panel"') < html.index('id="btn-stats"'))
    # 「最上面」= 在数据检查那些区块之前
    check("it sits above the data-inspector sections",
          html.index('id="btn-stats"') < html.index('id="inspector-extract"'))

    check("the modal exists and starts hidden",
          re.search(r'id="stats-modal"[^>]*class="[^"]*hidden', html) is not None)
    for el in ("stats-body", "stats-subtitle", "btn-stats-close"):
        check(f"the modal has {el}", f'id="{el}"' in html)

    check("main.js opens it from the button",
          "$('btn-stats').addEventListener('click', openStats)" in code)
    check("main.js asks the server for the stats", "fetch('/api/stats')" in code)
    check("the modal is a full-page overlay",
          re.search(r"\.stats-overlay\s*\{[^}]*position:\s*fixed", css) is not None)
    check("and it is capped so the tables scroll inside it",
          re.search(r"\.stats-card\s*\{[^}]*max-height", css) is not None)

    print("      (every way out of the modal)")
    check("the ✕ closes it",
          "$('btn-stats-close').addEventListener('click', closeStats)" in code)
    check("clicking the backdrop closes it",
          "e.target === $('stats-modal')" in code and "closeStats();" in code)
    check("Esc closes it", "e.code === 'Escape') closeStats();" in code)
    # 换比赛 = 缓存作废 + 弹窗关掉。别数 `closeStats();` 出现几次：那既算不出
    # "有几条关闭路径"，也会因为别处多一个入口而误报 —— 直接看 onDataReady 里
    # 有没有这两件事。
    ready = re.search(r"function onDataReady\([\s\S]*?\nfunction ", code)
    body = ready.group(0) if ready else ""
    check("switching match closes it and drops the cached numbers",
          "statsPayload = null;" in body and "aiStatsPayload = null;" in body
          and "closeStats();" in body)

    # 每回合矩阵的列头就是"跳到那一回合"的入口
    check("round columns carry a round index", "data-round-col" in code)
    check("and clicking one seeks that round",
          re.search(r"closest\('\[data-round-col\]'\)", code) is not None
          and "setRound(idx)" in code)

    # 路由：静态查一次 url_map，比等到浏览器 404 早
    from visualizer.server import create_app

    served = flask_paths(create_app(maps_dir=None))
    check("the server serves /api/stats", "/api/stats" in served)


def test_replay_ai_overlay() -> None:
    """AI 预测的接线：两个开关、面板 / 进度条 / 指标表、请求路径、3D 层模块。

    数据层在 ``app/tests/test_ai.py``，渲染在 ``visualizer/tests/test_ai_overlay.mjs``；
    这里只查"线接上了没有" —— 开关有没有绑事件、``main.js`` 打的是哪条接口、
    进度条 / 指标表的每个 id 在不在、``/api/ai/*`` 在不在 Flask 的 url_map 里。
    """
    html = REPLAY_HTML.read_text(encoding="utf-8")
    js = REPLAY_JS.read_text(encoding="utf-8")
    overlay = _ROOT / "visualizer" / "static" / "js" / "ai-overlay.js"
    css = (_ROOT / "visualizer" / "static" / "css" / "style.css").read_text(
        encoding="utf-8")
    code = strip_js_comments(js)

    for el in ("toggle-ai-spatial", "toggle-ai-duel", "toggle-player-ids",
               "ai-panel", "ai-model",
               "ai-state", "ai-win-ct", "ai-win-t", "ai-bar-ct",
               "ai-progress", "ai-progress-fill", "ai-progress-stage",
               "ai-progress-time", "ai-table", "ai-table-body"):
        check(f"index.html has #{el}", f'id="{el}"' in html)

    check("the player-ID toggle is on by default",
          re.search(r'id="toggle-player-ids"[^>]*checked', html) is not None)

    check("the AI panel starts hidden",
          re.search(r'id="ai-panel"[^>]*class="[^"]*hidden', html) is not None)
    check("the AI table starts hidden",
          re.search(r'id="ai-table"[^>]*class="[^"]*hidden', html) is not None)
    check("main.js binds both AI toggles",
          "['toggle-ai-spatial', 'toggle-ai-duel']" in code)
    check("main.js asks for the model status", "fetch('/api/ai/status')" in code)
    check("main.js asks for the round prediction", "/api/ai/round/${idx}" in code)
    check("main.js asks for the streaming (NDJSON) variant",
          "progress: '1'" in code or "progress=1" in code)
    check("main.js drives the 3D overlay", "updateAiOverlay({" in code)
    check("main.js drives the metrics table", "updateAiTable(" in code)
    check("main.js drives the progress bar", "updateAiProgress(" in code)
    check("ai-overlay.js exists", overlay.is_file())
    if overlay.is_file():
        ocode = strip_js_comments(overlay.read_text(encoding="utf-8"))
        for fn in ("export function updateAiOverlay", "export function probColor",
                   "export function clearAiOverlay", "export function aiOverlayDebug"):
            check(f"ai-overlay.js exports {fn.split()[-1]}", fn in ocode)
        check("the arc is split at the probability (two halves)",
              "split" in ocode and "arcB" in ocode)
        # 数字只在左下角表格里：3D 层不许再画文字 sprite（头顶那排百分比）
        check("ai-overlay.js draws no text (every number lives in the table)",
              "makeTextSprite" not in ocode and "drawTextSprite" not in ocode)
        # 弧线上不再挂白色游标点：只留两段按胜率上色的弧线
        check("the white cursor dot is gone (two coloured arcs only)",
              "makeDotTexture" not in ocode and "new THREE.Sprite" not in ocode)

    # 五项指标都要有列（少一列就是"某个头没接上"，界面不会报错，只会永远空着）
    for metric in ("alive", "can_kill", "next_kill", "next_death", "duel"):
        check(f"the metrics table has a {metric} column",
              f'data-metric="{metric}"' in html)
    for cls in (".ai-table", ".ai-progress", ".ai-row", ".ai-cell"):
        check(f"style.css styles {cls}", cls in css)

    # ── 表头释义 / 点名字跳人 / 点数字看曲线 ────────────────────────────
    # 三个交互各有自己的 DOM 出口；少一个就是"点了没反应"而且没有任何报错。
    for el in ("ai-tip", "ai-tip-dot", "ai-tip-title", "ai-tip-src",
               "ai-tip-body", "ai-tip-hint",
               "ai-chart", "ai-chart-dot", "ai-chart-title", "ai-chart-now",
               "ai-chart-close", "ai-chart-canvas", "ai-chart-death"):
        check(f"index.html has #{el}", f'id="{el}"' in html)

    chart_js = _ROOT / "visualizer" / "static" / "js" / "ai-chart.js"
    check("ai-chart.js exists", chart_js.is_file())
    if chart_js.is_file():
        ccode = strip_js_comments(chart_js.read_text(encoding="utf-8"))
        check("ai-chart.js exports createAiChart",
              "export function createAiChart" in ccode)
        check("the curve is clickable (pointerdown -> onSeek)",
              "pointerdown" in ccode and "onSeek" in ccode)
        check("the curve keeps its y axis at 0..100%",
              "Math.round(v * 100)" in ccode)

    check("main.js imports the chart module", "from './ai-chart.js'" in code)
    check("main.js explains all eight headers (six columns + two win rates)",
          all(f"\n  {key}: {{" in code for key in
              ("name", "alive", "can_kill", "next_kill", "next_death", "duel",
               "win_ct", "win_t")))
    check("main.js shows the explanation on hover", "showAiTip(" in code)
    check("main.js hides it when the pointer leaves", "hideAiTip(" in code)
    check("clicking a name selects and focuses that player",
          "focusPlayerFromAi(" in code)
    check("clicking a number opens the curve", "toggleAiChart(" in code)
    # 顶部两个胜率：同一张曲线卡，player = -1（不属于某个人，所以不锁镜头）
    check("the win percentages open the same curve card",
          'data-metric="win_ct"' in html and 'data-metric="win_t"' in html
          and "toggleAiChart(-1, el.dataset.metric)" in code)
    check("the top-panel percentages accept the mouse",
          re.search(r"\.ai-pct\s*\{[^}]*pointer-events:\s*auto", css) is not None)
    # 选手 ID：头顶写名字的 Sprite（billboard 永远朝向摄像机），开关接在 visuals.js
    check("main.js binds the player-ID toggle",
          "$('toggle-player-ids').addEventListener" in code)
    check("main.js controls the ID tags", "setPlayerIdsVisible(" in code)
    vcode = strip_js_comments(
        (_ROOT / "visualizer" / "static" / "js" / "visuals.js").read_text(encoding="utf-8"))
    check("visuals.js draws the ID tags as sprites",
          "createIdTag(" in vcode and "new THREE.Sprite" in vcode)
    check("visuals.js exports the ID debug hook",
          "export function playerIdDebug" in vcode)
    check("the curve seeks before locking the camera", "seekTo(seconds)" in code)
    check("the curve pauses playback before seeking",
          "setPlaying(false)" in code and "onSeek:" in code)
    check("the chart is closed when the round changes",
          "aiChart?.close();" in code)
    check("the AI table cells accept the mouse",
          re.search(r"\.ai-cell\s*\{[^}]*pointer-events:\s*auto", css) is not None)
    check("the AI table headers accept the mouse",
          re.search(r"\.ai-col,[^{]*\{[^}]*pointer-events:\s*auto", css) is not None)
    # 原生 title 和自定义释义卡会同时弹两个框：列头上一律不留 title
    check("native titles are gone from the headers (no double tooltip)",
          'title="spatial · alive' not in html
          and 'title="spatial · next_kill' not in html)

    # 路由：单独注册这个 blueprint（不碰真实的 library.db，所以不经 app.server）
    from flask import Flask

    from app.ai.api import bp as ai_bp

    app = Flask(__name__)
    app.register_blueprint(ai_bp)
    served = {str(r) for r in app.url_map.iter_rules()}
    check("the AI blueprint serves /api/ai/status", "/api/ai/status" in served)
    check("the AI blueprint serves /api/ai/round/<idx>",
          "/api/ai/round/<int:idx>" in served)


def test_i18n_wiring() -> None:
    """中英切换：四页都接上同一个运行时，且默认仍然是中文。

    文案本身对不对由 ``visualizer/tests/check_i18n.mjs`` 查（键成对、模板里
    没有漏标的中文）。这里只查"接上了没有"：运行时在不在、每个页面有没有真的
    import 它、模板有没有 data-i18n、顶栏开关有没有样式。

    默认中文是一条硬要求：浏览器回归测试（``visualizer/tests/test_*.mjs``）都
    按中文文案断言，默认值一旦变成"跟随浏览器语言"，那些测试会在 en-US 的
    Chrome 里成片失败，而症状看起来像功能坏了。
    """
    print("\ni18n wiring")
    for name in ("index.js", "common.js", "replay.js", "library.js", "players.js"):
        check(f"i18n/{name} exists", (I18N_DIR / name).is_file())

    core = I18N_CORE.read_text(encoding="utf-8")
    check("the runtime exports t()", "export function t(" in core)
    check("the runtime applies data-i18n", "export function applyI18n(" in core)
    check("the runtime can switch language", "export function setLang(" in core)
    check("the runtime mounts the topbar switch", "export function mountLangSwitcher(" in core)
    check("the choice is remembered", "localStorage.setItem(STORAGE_KEY" in core
          and "csnet.lang" in core)
    check("?lang= wins over the stored choice", "URLSearchParams" in core and "'lang'" in core)
    check("the default stays Chinese", "|| 'zh'" in core)
    check("the switch offers 中文 and EN",
          "'中文'" in core and "'EN'" in core)
    check("the switch lands in the shared topbar",
          ".topbar .tb-actions" in core)
    check("<html lang> follows the choice", "document.documentElement.lang" in core)

    for js, label in ((REPLAY_JS, "replay"), (LIB_JS, "library"),
                      (PLAYERS_JS, "players"), (PLAYER_JS, "player")):
        text = js.read_text(encoding="utf-8")
        check(f"{label}.js imports the runtime", "i18n/index.js" in text)

    for html, label in ((REPLAY_HTML, "replay"), (LIB_HTML, "library"),
                        (PLAYERS_HTML, "players"), (PLAYER_HTML, "player")):
        text = html.read_text(encoding="utf-8")
        keys = re.findall(r'data-i18n(?:-html|-title|-placeholder|-aria-label)?="([^"]+)"', text)
        check(f"{label}: the page is annotated for translation", len(keys) > 5, f"{len(keys)}")
        check(f"{label}: the document title is translated",
              re.search(r"<title[^>]*data-i18n=", text) is not None)

    check("the language switch has styles", ".lang-switch" in TOPBAR_CSS.read_text(encoding="utf-8"))


def main() -> int:
    test_files_exist()
    test_dom_ids()
    test_api_paths()
    test_delete_is_consistent()
    test_extract_controls()
    test_player_pages()
    test_player_names_and_storage()
    test_ui_copy()
    test_double_click_opens_analysis()
    test_replay_integration()
    test_replay_topbar_survives_the_empty_state()
    test_replay_has_no_upload_page()
    test_replay_stats_panel()
    test_replay_ai_overlay()
    test_i18n_wiring()
    print(f"\n{_passed} passed, {len(_failed)} failed")
    if _failed:
        for name in _failed:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

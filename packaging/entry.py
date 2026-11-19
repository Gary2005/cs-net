"""PyInstaller 入口：真正的代码在 ``app.desktop.main``。

不能直接把 ``app/desktop.py`` 当入口脚本 —— 它里面全是 ``from . import config``
这种相对导入，而 PyInstaller 是把入口脚本当 ``__main__`` 执行的，没有父包，
相对导入会直接炸。这里做一层绝对导入的壳，PyInstaller 顺着它就能把 ``app``
当成普通包收进去。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `app` 可导入：源码树里跑（pyinstaller 分析阶段）靠这一行，
# 冻结后 `app` 已经是包内模块，这一行是空操作。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.desktop import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

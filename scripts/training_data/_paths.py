"""把仓库的 `tools/` 加进 sys.path，让 `demo_extract` 可导入。

`demo_extract` 不是安装好的包，就躺在 `tools/demo_extract/`。
任何需要它的模块，在 import 之前先 `from . import _paths`。

打包成 exe 之后（PyInstaller）`scripts/` 在 `_internal/` 里，仓库根是 exe 所在
目录；而 `demo_extract` 是被当作普通模块收进去的，`tools/` 那条 sys.path 通常
已经用不上了（插一个不存在的路径是安全的）。这里两个都试。
"""

from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    REPO_ROOT = Path(sys.executable).resolve().parent
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

"""把仓库的 `tools/` 加进 sys.path，让 `demo_extract` 可导入。

`demo_extract` 不是安装好的包，就躺在 `tools/demo_extract/`。
任何需要它的模块，在 import 之前先 `from . import _paths`。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

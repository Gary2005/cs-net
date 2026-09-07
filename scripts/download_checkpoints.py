#!/usr/bin/env python3
"""
从 Hugging Face 下载 cs-net-v4 模型 checkpoint。

用法:
    python scripts/download_checkpoints.py                          # 全部 4 个 ckpt → checkpoints/
    python scripts/download_checkpoints.py --out-dir models         # 自定义保存目录
    python scripts/download_checkpoints.py --pretrain-only          # 只下载路径预测模型
    python scripts/download_checkpoints.py --repo-id your/cs-net-v4 # 镜像仓库

依赖: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FILES = [
    "cs-net-v4-pro.pt",                 # 路径预测（预训练）模型
    "pretrain-v4-pro-win_rate.pt",      # spatial-only: 胜率
    "pretrain-v4-pro-alive_end.pt",     # spatial-only: 回合末存活
    "pretrain-v4-pro-future_kill.pt",   # spatial-only: 未来击杀
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default="gary2oos/cs-net-v4",
                    help="Hugging Face 仓库（默认 gary2oos/cs-net-v4）")
    ap.add_argument("--out-dir", default="checkpoints",
                    help="保存目录（默认 checkpoints/，已在 .gitignore 中）")
    ap.add_argument("--pretrain-only", action="store_true",
                    help="只下载路径预测模型 cs-net-v4-pro.pt")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("缺少 huggingface_hub，请先安装:  pip install huggingface_hub",
              file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = FILES[:1] if args.pretrain_only else FILES

    print(f"下载 {args.repo_id} → {out}/")
    for name in files:
        local = hf_hub_download(repo_id=args.repo_id, filename=name,
                                local_dir=str(out))
        size_mb = Path(local).stat().st_size / 1024 / 1024
        print(f"  ✓ {name}  ({size_mb:.0f} MB)")

    print("\n完成。接下来验证模型能正确加载：")
    print(f"  python scripts/test_checkpoints.py --models-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

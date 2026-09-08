#!/usr/bin/env python3
"""
从 Hugging Face 下载 cs-net-v4 模型 checkpoint。

用法:
    python scripts/download_checkpoints.py                          # 全部 4 个 ckpt → checkpoints/
    python scripts/download_checkpoints.py --out-dir models         # 自定义保存目录
    python scripts/download_checkpoints.py --pretrain-only          # 只下载路径预测模型
    python scripts/download_checkpoints.py --repo-id your/cs-net-v4 # 镜像仓库

依赖: pip install huggingface_hub

说明:
    - 大文件（555MB）开始时进度条可能停留 0% 约 1 分钟（连接/预检阶段），
      之后会自动开始下载；如长时间无速度，可 Ctrl+C 重跑（自动断点续传）。
    - 若下载偏慢，建议升级传输器:
        pip install -U huggingface_hub        # 新版支持并行分块、进度更好
        pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
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

_MIN_RECOMMENDED_VERSION = (1, 20)  # 低于此版本建议升级（下载器/进度显示改进）


def _print_version_tip() -> None:
    try:
        from huggingface_hub import __version__
        ver = tuple(int(x) for x in __version__.split(".")[:2])
        if ver < _MIN_RECOMMENDED_VERSION:
            print(f"提示: 当前 huggingface_hub={__version__} 版本较旧，下载大文件可能"
                  f"偏慢/进度条长时间停在 0%（连接阶段属正常）。建议升级：")
            print("      pip install -U huggingface_hub")
            print()
    except Exception:
        pass


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

    _print_version_tip()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = FILES[:1] if args.pretrain_only else FILES

    print(f"下载 {args.repo_id} → {out}/")
    for name in files:
        print(f"  连接中: {name} ...")
        print("    （大文件首次连接可能需要约 1 分钟，进度条才会开始移动；")
        print("      中途网络超时会自动断点续传，耐心等待即可）")
        local = hf_hub_download(repo_id=args.repo_id, filename=name,
                                local_dir=str(out))
        size_mb = Path(local).stat().st_size / 1024 / 1024
        print(f"  ✓ {name}  ({size_mb:.0f} MB)")

    print("\n完成。接下来验证模型能正确加载：")
    print(f"  python scripts/test_checkpoints.py --models-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

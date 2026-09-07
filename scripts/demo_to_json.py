#!/usr/bin/env python3
"""Thin wrapper — delegates to ``python -m demo_parser``."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `demo_parser` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_parser.cli import main

if __name__ == "__main__":
    main()

"""
CS2 Demo Parser — Optimized V2 JSON Format.

Usage:
    from demo_parser import parse_demo, save_demo_json

    data = parse_demo("match.dem", interval=0.5)
    save_demo_json(data, "output.json")
"""

from .extract import parse_demo, save_demo_json

__all__ = ["parse_demo", "save_demo_json"]

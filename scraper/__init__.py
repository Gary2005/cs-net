"""
HLTV match scraper → demo download → .csz pipeline (v5).

Usage:
    python -m scraper --setup         # one-time: login to HLTV
    python -m scraper --count 10      # scrape + download + parse 10 demos
    python -m scraper --count 10 --interval 0.25   # 4 Hz instead of every tick
"""

from .pipeline import DemoPipeline

__all__ = ["DemoPipeline"]

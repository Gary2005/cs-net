"""What does demoparser2's own field vocabulary actually contain?

`list_updated_fields()` returns every netprop the demo transmits. Search it for
the concepts demoinfocs exposes as first-class accessors, so "demoparser2 cannot
do X" is checked against demoparser2's real vocabulary rather than my guesses.
"""
import re
import sys

from demoparser2 import DemoParser

DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
fields = DemoParser(DEMO).list_updated_fields()
print(f"total fields: {len(fields)}\n")

PATTERNS = {
    "button / input": r"button|usercmd|input",
    "spotted / visibility": r"spotted|visible|seen",
    "duck / crouch": r"duck|crouch",
    "reload": r"reload",
    "plant / defuse": r"plant|defus",
    "eye angles / view": r"eyeangle|eye_ang|viewoffset|view_offset|viewangle",
    "velocity modifier": r"velocitymodifier|velocity_modifier",
    "money / economy": r"money|account|equipmentvalue|equipment_value|balance",
    "stats (kda/damage/mvp)": r"^kills$|^deaths$|^assists$|mvp|totaldamage|total_damage|utilitydamage",
    "clan / viewmodel / color": r"clantag|clan_tag|viewmodel|^color$",
    "rank": r"rank|competitive",
    "shots fired": r"shotsfired|shots_fired",
    "item identity / skin": r"itemdefinitionindex|item_def|paintkit|paint_kit|fallback",
}

for label, pat in PATTERNS.items():
    rx = re.compile(pat, re.I)
    hits = sorted(f for f in fields if rx.search(f))
    print(f"--- {label}: {len(hits)}")
    for h in hits[:40]:
        print(f"      {h}")
    if len(hits) > 40:
        print(f"      ... (+{len(hits)-40} more)")
    print()
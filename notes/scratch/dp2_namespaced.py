"""Test demoparser2 with FULLY-QUALIFIED sendtable names.

demoparser2's `list_updated_fields()` returns 966 namespaced netprops
(`CCSPlayerPawn.*`, `Weapon.*`, `Grenade.*`, `CCSGameRulesProxy.*`).
Short/friendly names are silently dropped, which can look like "the data is
missing" when it is really just a name-lookup failure.
"""
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = sys.argv[1] if len(sys.argv) > 1 else "/Users/wanjungu/Downloads/test.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 220)

TICKS = list(range(4800, 4820))

CASES = [
    ("buttons (input mask)", "CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev"),
    ("toggle-button mask", "CCSPlayerPawn.CCSPlayer_MovementServices.m_nToggleButtonDownMask"),
    ("spotted-by mask", "CCSPlayerPawn.m_bSpottedByMask"),
    ("duck amount", "CCSPlayerPawn.CCSPlayer_MovementServices.m_flDuckAmount"),
    ("in crouch", "CCSPlayerPawn.CCSPlayer_MovementServices.m_bInCrouch"),
    ("is defusing", "CCSPlayerPawn.m_bIsDefusing"),
    ("is planting (controller)", "CCSPlayerController.m_bPawnHasDefuser"),
    ("eye angles", "CCSPlayerPawn.m_angEyeAngles"),
    ("equipment value", "CCSPlayerPawn.m_unCurrentEquipmentValue"),
    ("money (controller)", "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iAccount"),
    ("cash spent this round", "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iCashSpentThisRound"),
    ("shots fired", "CCSPlayerPawn.m_iShotsFired"),
    ("velocity modifier", "CCSPlayerPawn.m_flVelocityModifier"),
    ("mvps", "CCSPlayerController.m_iMVPs"),
    ("viewmodel fov", "CCSPlayerPawn.m_flViewmodelFOV"),
    ("rank", "CCSPlayerController.m_iCompetitiveRanking"),
    ("WEAPON item def index", "Weapon.m_iItemDefinitionIndex"),
    ("WEAPON paint kit", "Weapon.m_nFallbackPaintKit"),
    ("WEAPON fallback wear", "Weapon.m_flFallbackWear"),
    ("WEAPON in reload", "Weapon.m_bInReload"),
    ("GRENADE item def index", "Grenade.m_iItemDefinitionIndex"),
    ("GRENADE paint kit", "Grenade.m_nFallbackPaintKit"),
    ("GRENADE fallback wear", "Grenade.m_flFallbackWear"),
    ("GRENADE voxel frame data", "Grenade.m_VoxelFrameData"),
    ("GRENADE fire positions", "Grenade.m_firePositions"),
    ("GRENADE fire count", "Grenade.m_fireCount"),
    ("GRENADE fire is burning", "Grenade.m_bFireIsBurning"),
    ("rules bomb planted", "CCSGameRulesProxy.CCSGameRules.m_bBombPlanted"),
    ("rules round time", "CCSGameRulesProxy.CCSGameRules.m_iRoundTime"),
    ("rules total rounds", "CCSGameRulesProxy.CCSGameRules.m_totalRoundsPlayed"),
]

print("=" * 90)
print("demoparser2 with FULLY-QUALIFIED names   demo =", DEMO)
print("=" * 90)
for label, name in CASES:
    try:
        df = p.parse_ticks(wanted_props=[name], ticks=TICKS)
        present = name in df.columns
        if not present:
            print(f"  DROPPED  {label:30s} {name}")
            continue
        s = df[name]
        nn = int(s.notna().sum())
        try:
            uniq = s.dropna().unique()[:3]
            sample = str(list(uniq))[:70]
        except Exception:
            sample = "(unprintable)"
        print(f"  OK       {label:30s} nonnull={nn:<5} {sample}")
    except BaseException as e:
        print(f"  RAISED   {label:30s} {str(e).splitlines()[0][:70]}")
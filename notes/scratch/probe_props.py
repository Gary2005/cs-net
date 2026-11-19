"""Probe which entity props are retrievable via parse_ticks.

demoparser2 raises if a prop name is unknown, so we test each candidate
individually and record (ok, dtype, sample values, nonnull count).
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)

# tick window well inside a live round (~derived later); we just need data
TICKS = list(range(30000, 30100))

CANDIDATES = [
    # --- basic transform / team ---
    "X", "Y", "Z", "x", "y", "z", "origin", "vecX", "vecY", "vecZ",
    "m_vecX", "m_vecY", "m_vecZ",
    "yaw", "pitch", "roll", "m_angEyeAngles", "eye_angles", "ang_rotation",
    # --- health / armor / status ---
    "health", "m_iHealth", "armor", "m_ArmorValue", "helmet", "has_helmet",
    "m_bHasHelmet", "has_defuser", "m_bHasDefuser", "is_alive",
    "m_bPawnIsAlive", "life_state", "m_lifeState",
    "is_defusing", "m_bIsDefusing", "is_planting", "is_scoped", "m_bIsScoped",
    "is_walking", "m_bIsWalking", "is_airborne", "m_bIsAirborne",
    "ducking", "m_bDucking", "m_flDuckAmount",
    # --- economy ---
    "balance", "m_iAccount", "money", "equipment_value", "m_unCurrentEquipmentValue",
    "m_unRoundStartEquipmentValue", "m_unFreezetimeEndEquipmentValue",
    "cash_spent", "m_iCashSpentThisRound",
    # --- weapons ---
    "active_weapon", "m_hActiveWeapon", "active_weapon_name", "weapon_name",
    "inventory", "m_hMyWeapons", "inventory_as_ids", "weapons",
    "m_iItemDefinitionIndex", "current_weapon",
    "ammo", "m_iAmmo", "clip1", "m_iClip1", "m_iClip2",
    # --- flash / misc pawn state ---
    "flash_duration", "m_flFlashDuration", "m_flFlashMaxAlpha", "m_bFlashing",
    "velocity", "m_vecVelocity", "m_vecBaseVelocity", "speed",
    "m_fFlags", "flags", "m_bInBuyZone", "m_bInBombZone",
    "m_szLastPlaceName", "last_place_name",
    "m_iShotsFired", "shots_fired", "m_flVelocityModifier",
    "m_flFallVelocity", "m_bSpotted",
    # --- controller ---
    "team_num", "m_iTeamNum", "team_number", "player_name", "m_iszPlayerName",
    "m_iPawnHealth", "m_iPawnArmor", "m_bPawnHasDefuser", "m_bPawnHasHelmet",
    "ping", "m_iPing", "score", "m_iScore", "kills", "deaths", "assists",
    "m_nTickBase", "tick_base", "m_iCompetitiveRanking", "m_szClan",
    "is_bomb_planted", "m_bBombPlanted", "is_freeze_period", "m_bFreezePeriod",
    "game_phase", "m_gamePhase", "m_iRoundTime", "round_time",
    "m_eRoundWinReason", "m_iRoundWinStatus", "m_totalRoundsPlayed",
    "m_iRoundStartRoundNumber", "m_nRoundStartCount", "m_fRoundStartTime",
    "is_warmup", "m_bWarmupPeriod",
    # --- grenade / inferno props (the big question) ---
    "m_bFireIsBurning", "m_fireCount", "m_nFireCount", "m_firePositions",
    "m_fireParentPositions", "m_nInfernoType", "m_nFireLifetime",
    "m_nFireEffectTickBegin", "m_BurnNormal",
    "m_fireXDelta", "m_fireYDelta", "m_fireZDelta", "m_bFireIsBurning_",
    "m_bDidSmokeEffect", "m_nSmokeEffectTickBegin", "m_vSmokeDetonationPos",
    "m_vSmokeColor", "m_flDetonateTime", "m_nVoxelFrameDataSize",
    "m_VoxelFrameData", "m_nVoxelUpdate", "m_bIsIncGrenade",
    "m_flThrowStrength", "m_vInitialPosition", "m_vInitialVelocity",
    "m_hThrower", "m_nExplodeEffectTickBegin", "m_vecExplodeEffectOrigin",
]

out = {}
for prop in CANDIDATES:
    try:
        df = p.parse_ticks([prop], ticks=TICKS)
        cols = [c for c in df.columns if c != "tick"]
        info = {"status": "ok", "returned_cols": cols}
        for c in cols:
            s = df[c]
            nn = int(s.notna().sum())
            info.setdefault("details", {})[c] = {
                "dtype": str(s.dtype),
                "nonnull": nn,
                "sample": [repr(v)[:120] for v in s.dropna().head(3).tolist()],
            }
        out[prop] = info
    except BaseException as e:  # noqa
        out[prop] = {"status": "ERROR", "err": f"{type(e).__name__}: {str(e)[:200]}"}

json.dump(out, open("props_result.json", "w"), indent=1, ensure_ascii=False)

ok = [k for k, v in out.items() if v["status"] == "ok"]
bad = [k for k, v in out.items() if v["status"] != "ok"]
print(f"OK ({len(ok)}):")
for k in ok:
    d = out[k]
    det = d.get("details", {})
    summ = "; ".join(f"{c}:{v['dtype']}(nn={v['nonnull']},eg={v['sample'][:1]})" for c, v in det.items())
    print(f"  {k:34s} -> {summ}")
print(f"\nFAILED ({len(bad)}):")
for k in bad:
    print(f"  {k:34s} !! {out[k]['err']}")

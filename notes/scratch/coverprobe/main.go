// Coverage probe: does demoinfocs-golang v4 deliver everything the cs-net
// pipeline currently consumes from demoparser2, and what does it expose extra?
// Research probe. Not project code.
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/sendtables"
)

var counts = map[string]int{}

func bump(k string) { counts[k]++ }

// safe runs f and converts a panic into an inline marker, so one bad accessor
// doesn't abort the whole probe.
func safe(f func() string) (out string) {
	defer func() {
		if r := recover(); r != nil {
			out = "PANIC(" + fmt.Sprint(r) + ")"
			fmt.Fprintf(os.Stderr, "RECOVERED: %v\n", r)
		}
	}()
	return f()
}

// pv reads a raw sendtable property. PropertyValue.String() can itself panic on
// some Source 2 property types, so it is guarded as well.
func pv(e sendtables.Entity, name string) (out string, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			out, ok = "PANIC", false
		}
	}()
	if e == nil {
		return "", false
	}
	v, exists := e.PropertyValue(name)
	if !exists {
		return "", false
	}
	return v.String(), true
}

func main() {
	path := os.Args[1]
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	// ── The 13 events the cs-net pipeline asks demoparser2 for ──────────────
	p.RegisterEventHandler(func(events.RoundFreezetimeEnd) { bump("round_freeze_end") })
	p.RegisterEventHandler(func(events.RoundEnd) { bump("round_end") })
	p.RegisterEventHandler(func(events.Kill) { bump("player_death") })
	p.RegisterEventHandler(func(events.PlayerHurt) { bump("player_hurt") })
	p.RegisterEventHandler(func(events.WeaponFire) { bump("weapon_fire") })
	p.RegisterEventHandler(func(events.Footstep) { bump("player_footstep") })
	p.RegisterEventHandler(func(events.BombPlanted) { bump("bomb_planted") })
	p.RegisterEventHandler(func(events.BombExplode) { bump("bomb_exploded") })
	p.RegisterEventHandler(func(events.BombDefused) { bump("bomb_defused") })
	p.RegisterEventHandler(func(events.SmokeStart) { bump("smokegrenade_detonate") })
	p.RegisterEventHandler(func(events.SmokeExpired) { bump("smokegrenade_expired") })
	p.RegisterEventHandler(func(events.InfernoStart) { bump("inferno_startburn") })
	p.RegisterEventHandler(func(events.InfernoExpired) { bump("inferno_expire") })

	// ── Extra events demoparser2 does not surface ──────────────────────────
	p.RegisterEventHandler(func(events.PlayerFlashed) { bump("X player_flashed") })
	p.RegisterEventHandler(func(events.PlayerJump) { bump("X player_jump") })
	p.RegisterEventHandler(func(events.PlayerSound) { bump("X player_sound") })
	p.RegisterEventHandler(func(events.BulletDamage) { bump("X bullet_damage") })
	p.RegisterEventHandler(func(events.PlayerButtonsStateUpdate) { bump("X buttons_state_update") })
	p.RegisterEventHandler(func(events.PlayerSpottersChanged) { bump("X player_spotters_changed") })
	p.RegisterEventHandler(func(events.BombPlantBegin) { bump("X bomb_plant_begin") })
	p.RegisterEventHandler(func(events.BombPlantAborted) { bump("X bomb_plant_aborted") })
	p.RegisterEventHandler(func(events.BombDefuseStart) { bump("X bomb_defuse_start") })
	p.RegisterEventHandler(func(events.BombDefuseAborted) { bump("X bomb_defuse_aborted") })
	p.RegisterEventHandler(func(events.BombDropped) { bump("X bomb_dropped") })
	p.RegisterEventHandler(func(events.BombPickup) { bump("X bomb_pickup") })
	p.RegisterEventHandler(func(events.WeaponReload) { bump("X weapon_reload") })
	p.RegisterEventHandler(func(events.GrenadeProjectileBounce) { bump("X grenade_bounce") })
	p.RegisterEventHandler(func(events.ItemPickup) { bump("X item_pickup") })
	p.RegisterEventHandler(func(events.ItemDrop) { bump("X item_drop") })
	p.RegisterEventHandler(func(events.RankUpdate) { bump("X rank_update") })
	p.RegisterEventHandler(func(events.TeamSideSwitch) { bump("X team_side_switch") })
	p.RegisterEventHandler(func(events.GameHalfEnded) { bump("X game_half_ended") })
	p.RegisterEventHandler(func(events.RoundMVPAnnouncement) { bump("X round_mvp") })
	p.RegisterEventHandler(func(events.ChatMessage) { bump("X chat_message") })
	p.RegisterEventHandler(func(events.ConVarsUpdated) { bump("X convars_updated") })
	p.RegisterEventHandler(func(events.PlayerTeamChange) { bump("X player_team_change") })

	// ── Kill richness (compare with demoparser2 player_death columns) ──────
	var killExtra = map[string]int{}
	p.RegisterEventHandler(func(e events.Kill) {
		if e.IsWallBang() {
			killExtra["wallbang"]++
		}
		if e.NoScope {
			killExtra["noscope"]++
		}
		if e.Distance > 0 {
			killExtra["distance>0"]++
		}
		if e.PenetratedObjects > 0 {
			killExtra["penetrated>0"]++
		}
		if e.Weapon != nil {
			killExtra["weapon!=nil"]++
		}
	})

	// ─ Grenade trajectory length (demoparser2 parse_grenades equivalent) ──
	trajMax, trajN, trajEntries := 0, 0, 0
	var bestTraj []common.TrajectoryEntry
	var bestKind string
	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) {
		trajN++
		n := len(e.Projectile.Trajectory2)
		if n > trajMax {
			trajMax = n
			bestTraj = e.Projectile.Trajectory2
			if e.Projectile.WeaponInstance != nil {
				bestKind = e.Projectile.WeaponInstance.String()
			}
		}
		trajEntries += n
	})

	// Raw game-event layer: covers any field a typed event struct omits.
	var geNames = map[string]int{}
	var deathRaw, plantRaw string
	p.RegisterEventHandler(func(e events.GenericGameEvent) {
		geNames[e.Name]++
		snap := func() string {
			keys := make([]string, 0, len(e.Data))
			for k := range e.Data {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			var parts []string
			for _, k := range keys {
				v := e.Data[k]
				parts = append(parts, fmt.Sprintf("%s=%v", k, v.GetValString()))
			}
			return strings.Join(parts, " ")
		}
		if e.Name == "player_death" && deathRaw == "" {
			deathRaw = snap()
		}
		if e.Name == "bomb_planted" && plantRaw == "" {
			plantRaw = snap()
		}
	})

	target := -1
	dumped := false

	p.RegisterEventHandler(func(events.RoundFreezetimeEnd) {
		if target < 0 {
			target = p.GameState().IngameTick() + 640
		}
	})

	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		if dumped || target < 0 || gs.IngameTick() < target {
			return
		}
		dumped = true
		tick := gs.IngameTick()

		var b strings.Builder
		W := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }

		// Everything below is wrapped: a single unsupported accessor on Source 2
		// must not kill the run, and partial output must survive.
		safe(func() string {
		W("\n################ PLAYER-TICK DUMP @ tick %d ################\n", tick)

		players := gs.Participants().Playing()
		W("playing players: %d\n", len(players))

		for i, pl := range players {
			if pl == nil {
				continue
			}
			safe(func() string {
			posS := safe(func() string { v := pl.Position(); return fmt.Sprintf("(%.2f, %.2f, %.2f)", v.X, v.Y, v.Z) })
			velS := safe(func() string { v := pl.Velocity(); return fmt.Sprintf("(%.2f, %.2f, %.2f)", v.X, v.Y, v.Z) })
			eyesS := safe(func() string { v := pl.PositionEyes(); return fmt.Sprintf("(%.2f, %.2f, %.2f)", v.X, v.Y, v.Z) })
			vmS := safe(func() string { v := pl.ViewmodelOffset(); return fmt.Sprintf("(%.2f, %.2f, %.2f)", v.X, v.Y, v.Z) })

			var inv []string
			for _, eq := range pl.Weapons() {
				inv = append(inv, safe(func() string {
					return fmt.Sprintf("%s#%d(mag=%d type=%d res=%d zoom=%d recoil=%.2f sil=%v cls=%v)",
						eq.String(), int(eq.Type), eq.AmmoInMagazine(), eq.AmmoType(),
						eq.AmmoReserve(), int(eq.ZoomLevel()), eq.RecoilIndex(),
						eq.Silenced(), eq.Class())
				}))
			}
			sort.Strings(inv)

			pawn := pl.PlayerPawnEntity()
			W("\n--- player[%d] %s ---\n", i, pl.Name)
			W("  steamid64=%d team=%v bot=%v connected=%v\n", pl.SteamID64, pl.Team, pl.IsBot, pl.IsConnected)
			W("  pos=%s eyes=%s\n", posS, eyesS)
			W("  vel=%s  yaw=%.2f pitch=%.2f\n",
				velS, pl.ViewDirectionY(), pl.ViewDirectionX())
			W("  hp=%d armor=%d helmet=%v defuser=%v alive=%v\n",
				pl.Health(), pl.Armor(), pl.HasHelmet(), pl.HasDefuseKit(), pl.IsAlive())
			W("  activeWeapon=%v  nWeapons=%d\n", pl.ActiveWeapon(), len(pl.Weapons()))
			W("  flash=%v flashRemain=%v flashTick=%d\n",
				pl.GetFlashDuration(), pl.FlashDurationTimeRemaining(), pl.FlashTick)
			W("  lastPlaceName=%q\n", pl.LastPlaceName())
			W("  flags: walking=%v scoped=%v ducking=%v stand=%v airborne=%v reloading=%v\n",
				pl.IsWalking(), pl.IsScoped(), pl.IsDucking(), pl.IsStanding(),
				pl.IsAirborne(), pl.IsReloading)
			W("  flags: planting=%v defusing=%v hostage=%v bombZone=%v buyZone=%v\n",
				pl.IsPlanting, pl.IsDefusing, pl.IsGrabbingHostage(), pl.IsInBombZone(), pl.IsInBuyZone())
			W("  money=%d eqCur=%d eqRoundStart=%d eqFreezeEnd=%d spentTotal=%d spentRound=%d\n",
				pl.Money(), pl.EquipmentValueCurrent(), pl.EquipmentValueRoundStart(),
				pl.EquipmentValueFreezeTimeEnd(), pl.MoneySpentTotal(), pl.MoneySpentThisRound())
			W("  ping=%d score=%d kda=%d/%d/%d mvp=%d dmg=%d utilityDmg=%d\n",
				pl.Ping(), pl.Score(), pl.Kills(), pl.Deaths(), pl.Assists(),
				pl.MVPs(), pl.TotalDamage(), pl.UtilityDamage())
			W("  rankType=%d rank=%d wins=%d clantag=%q crosshair=%q\n",
				pl.RankType(), pl.Rank(), pl.CompetitiveWins(), pl.ClanTag(), pl.CrosshairCode())
			W("  viewmodelFOV=%.1f viewmodelOff=%s color=%v\n",
				pl.ViewmodelFOV(), vmS, pl.Color())
			W("  buttonsPressedState=0x%x attack=%v jump=%v duck=%v fwd=%v back=%v use=%v reload=%v zoom=%v\n",
				pl.ButtonsPressedState, pl.IsPressingButton(common.ButtonAttack),
				pl.IsPressingButton(common.ButtonJump), pl.IsPressingButton(common.ButtonDuck),
				pl.IsPressingButton(common.ButtonForward), pl.IsPressingButton(common.ButtonBack),
				pl.IsPressingButton(common.ButtonUse), pl.IsPressingButton(common.ButtonReload),
				pl.IsPressingButton(common.ButtonZoom))
			W("  ammoLeft[0..7]=%v\n", pl.AmmoLeft[0:8])
			W("  inventory: %s\n", strings.Join(inv, " | "))

			// spotted: pairwise
			var spotted []string
			for j, other := range players {
				if other == nil || other == pl {
					continue
				}
				if pl.IsSpottedBy(other) {
					spotted = append(spotted, fmt.Sprintf("%d", j))
				}
			}
			W("  spottedBy(idx)=%v  hasSpottedBridge=%v\n",
				spotted, pl.HasSpotted(players[(i+1)%len(players)]))

			// raw property fallbacks (things demoinfocs has no accessor for)
			W("  RAW %s\n", rawProbe(pawn))
			return ""
			})
		}

		// ── match-level extras ────────────────────────────────────────────
		W("\n################ MATCH / RULES ################\n")
		W("tickrate=%.4f ingameTick=%d totalRoundsPlayed=%d matchStarted=%v\n",
			p.TickRate(), tick, gs.TotalRoundsPlayed(), gs.IsMatchStarted())
		if rt, err := gs.Rules().RoundTime(); err == nil {
			W("rules.roundTime=%v\n", rt)
		}
		if ft, err := gs.Rules().FreezeTime(); err == nil {
			W("rules.freezeTime=%v\n", ft)
		}
		if bt, err := gs.Rules().BombTime(); err == nil {
			W("rules.bombTime=%v\n", bt)
		}
		ge := gs.Rules().Entity()
		for _, n := range []string{"m_bBombPlanted", "m_bBombDropped", "m_iRoundTime",
			"m_totalRoundsPlayed", "m_bFreezePeriod", "m_iNumCT", "m_iNumTerrorist"} {
			v, ok := pv(ge, n)
			W("RAW rules.%s -> %q ok=%v\n", n, v, ok)
		}
		W("bomb: carrier=%v position=%v\n", safe(func() string {
			if b := gs.Bomb(); b != nil {
				if b.Carrier != nil {
					return b.Carrier.Name
				}
				return "(no carrier)"
			}
			return "(nil bomb)"
		}), safe(func() string {
			if b := gs.Bomb(); b != nil {
				pp := b.Position()
				return fmt.Sprintf("(%.1f, %.1f, %.1f)", pp.X, pp.Y, pp.Z)
			}
			return "(nil bomb)"
		}))

		// ─ what properties the raw pawn entity actually exposes ──────────
		W("\n################ PAWN RAW PROPERTY INVENTORY (sample) ################\n")
		if pawn := players[0].PlayerPawnEntity(); pawn != nil {
			var names []string
			for _, pr := range pawn.Properties() {
				names = append(names, pr.Name())
			}
			sort.Strings(names)
			W("pawn property count=%d\n", len(names))
			var eyeish []string
			for _, n := range names {
				ln := strings.ToLower(n)
				if strings.Contains(ln, "eye") || strings.Contains(ln, "viewoffset") ||
					strings.Contains(ln, "viewofs") || strings.Contains(ln, "lookat") ||
					strings.Contains(ln, "spotted") || strings.Contains(ln, "button") {
					if v, ok := pv(pawn, n); ok {
						eyeish = append(eyeish, fmt.Sprintf("%s=%s", n, v))
					} else {
						eyeish = append(eyeish, n+"=<unreadable>")
					}
				}
			}
			W("pawn view/spotted/button props:\n")
			for _, e := range eyeish {
				W("    %s\n", e)
			}
		}
		return ""
		})
		fmt.Print(b.String())
	})

	// ── Focused second sample: mid-round, when spotting / shooting is live ──
	dumped2 := false
	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		if target < 0 || !dumped || dumped2 || gs.IngameTick() < target+3000 {
			return
		}
		dumped2 = true
		var b strings.Builder
		W := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }

		safe(func() string {
			W("\n################ FOCUSED DUMP v2 @ tick %d ################\n", gs.IngameTick())
			playing := gs.Participants().Playing()

			for i, pl := range playing {
				pawn := pl.PlayerPawnEntity()
				mask, mok := pv(pawn, "m_bSpottedByMask")
				off, ook := pv(pawn, "m_vecViewOffset")
				spotted := "n/a"
				if len(playing) > 1 {
					spotted = fmt.Sprint(pl.IsSpottedBy(playing[(i+3)%len(playing)]))
				}
				W("p%d %-30s mask=%-46q ok=%-5v viewOffset=%-28q ok=%-5v spottedBy(+3)=%s\n",
					i, pl.Name, mask, mok, off, ook, spotted)

				for _, eq := range pl.Weapons() {
					if eq.Entity == nil {
						continue
					}
					for _, n := range []string{
						"m_iItemDefinitionIndex",
						"m_AttributeManager.m_Item.m_iItemDefinitionIndex",
						"m_AttributeManager.m_Item.m_iEntityQuality",
						"m_AttributeManager.m_Item.m_iEntityLevel",
						"m_AttributeManager.m_Item.m_iAccountID",
						"m_nFallbackPaintKit", "m_nFallbackSeed", "m_flFallbackWear",
						"m_iItemIDHigh", "m_iItemIDLow", "m_szCustomName",
					} {
						if v, ok := pv(eq.Entity, n); ok {
							W("      %-16s %-52s = %s\n", eq.String(), n, v)
						}
					}
				}
			}

			ge := gs.Rules().Entity()
			if ge != nil {
				var names []string
				for _, pr := range ge.Properties() {
					names = append(names, pr.Name())
				}
				sort.Strings(names)
				W("\nrules entity class=%s property count=%d\n", ge.ServerClass().Name(), len(names))
				for _, n := range names {
					ln := strings.ToLower(n)
					if strings.Contains(ln, "bomb") || strings.Contains(ln, "round") ||
						strings.Contains(ln, "freeze") || strings.Contains(ln, "time") ||
						strings.Contains(ln, "numct") || strings.Contains(ln, "terrorist") {
						if v, ok := pv(ge, n); ok {
							W("  rules.%s = %s\n", n, v)
						}
					}
				}
			}
			return ""
		})
		fmt.Print(b.String())
	})

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	fmt.Printf("\n################ EVENT COVERAGE ################\n")
	var used, extra []string
	for k := range counts {
		if strings.HasPrefix(k, "X ") {
			extra = append(extra, k)
		} else {
			used = append(used, k)
		}
	}
	sort.Strings(used)
	sort.Strings(extra)
	fmt.Printf("-- pipeline events (demoparser2 equivalents) --\n")
	for _, k := range used {
		fmt.Printf("  %-26s %d\n", k, counts[k])
	}
	fmt.Printf("-- extra events not in the pipeline's demoparser2 usage --\n")
	for _, k := range extra {
		fmt.Printf("  %-26s %d\n", strings.TrimPrefix(k, "X "), counts[k])
	}

	fmt.Printf("\n################ KILL EVENT RICHNESS ################\n")
	var kk []string
	for k := range killExtra {
		kk = append(kk, k)
	}
	sort.Strings(kk)
	for _, k := range kk {
		fmt.Printf("  %-16s %d\n", k, killExtra[k])
	}

	fmt.Printf("\n################ GRENADE TRAJECTORIES ################\n")
	fmt.Printf("grenade destroy events=%d  total trajectory entries=%d  max entries/grenade=%d\n",
		trajN, trajEntries, trajMax)
	fmt.Printf("densest grenade kind=%s  entries=%d\n", bestKind, len(bestTraj))
	for i, t := range bestTraj {
		if i < 30 {
			fmt.Printf("   traj[%2d] frame=%-7d t=%-12v pos=(%.1f, %.1f, %.1f)\n",
				i, t.FrameID, t.Time, t.Position.X, t.Position.Y, t.Position.Z)
		}
	}

	fmt.Printf("\n################ GENERIC GAME-EVENT LAYER ################\n")
	var gn []string
	for k := range geNames {
		gn = append(gn, k)
	}
	sort.Strings(gn)
	fmt.Printf("distinct raw game events seen: %d\n", len(gn))
	for _, k := range gn {
		fmt.Printf("  %-34s %d\n", k, geNames[k])
	}
	fmt.Printf("\nraw player_death payload:\n  %s\n", deathRaw)
	fmt.Printf("\nraw bomb_planted payload:\n  %s\n", plantRaw)
	for i, t := range bestTraj {
		if i < 30 {
			fmt.Printf("   traj[%2d] frame=%-7d t=%-12v pos=(%.1f, %.1f, %.1f)\n",
				i, t.FrameID, t.Time, t.Position.X, t.Position.Y, t.Position.Z)
		}
	}
}

func rawProbe(e sendtables.Entity) string {
	if e == nil {
		return "(pawn entity nil)"
	}
	names := []string{
		"m_flFlashMaxAlpha", "m_flFlashDuration", "m_flFlashBangTime",
		"m_bSpottedByMask", "m_bSpotted",
		"m_iItemDefinitionIndex", "m_iEntityQuality", "m_iEntityLevel",
		"m_ArmorValue", "m_bHasHelmet", "m_bHasDefuser",
		"m_iHealth", "m_lifeState", "m_iTeamNum",
		"m_nButtons", "m_nButtonDownMaskPrev",
		"m_angEyeAngles", "m_angEyeAngles[0]", "m_angEyeAngles[1]",
		"m_vecVelocity", "m_vecAbsVelocity",
		"m_flVelocityModifier", "m_flStamina", "m_flGroundAccelLinearFrac",
		"m_bIsWalking", "m_bIsDefusing", "m_bIsGrabbingHostage",
		"m_bInBombZone", "m_bInBuyZone",
		"m_iShotsFired", "m_iAmmo", "m_bBombPlanted",
		"m_szLastPlaceName", "m_iLastPlaceName",
		"m_iPing", "m_iScore", "m_iMVPs", "m_iTotalDamage",
		"m_pInGameMoneyServices", "m_pActionTrackingServices",
	}
	var out []string
	hit, miss := 0, 0
	for _, n := range names {
		v, ok := pv(e, n)
		if ok {
			hit++
			out = append(out, fmt.Sprintf("%s=%s", n, v))
		} else {
			miss++
		}
	}
	return fmt.Sprintf("raw pawn props: %d present, %d absent -> %s", hit, miss, strings.Join(out, " ; "))
}
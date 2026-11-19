// Counterpart probe: does demoinfocs-golang actually reach the grenade-projectile
// properties that demoparser2 silently drops (fire cells, smoke voxels, throw
// physics)? Also dumps per-tick button masks for a demoparser2 cross-check.
// Research probe. Not project code.
package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/sendtables"
)

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

// properties demoparser2 silently drops (verified separately)
var dropped = []string{
	// inferno / fire
	"m_firePositions", "m_fireCount", "m_bFireIsBurning", "m_fireParentPositions",
	"m_nInfernoType", "m_nFireLifetime", "m_nFireEffectTickBegin",
	// smoke
	"m_VoxelFrameData", "m_nVoxelFrameDataSize", "m_nVoxelUpdate",
	"m_bDidSmokeEffect", "m_nSmokeEffectTickBegin", "m_vSmokeColor", "m_vSmokeDetonationPos",
	// projectile physics
	"m_vInitialVelocity", "m_nBounces", "m_flDetonateTime", "m_bIsIncGrenade",
	// control: retrievable via demoparser2 too
	"m_iItemDefinitionIndex", "m_nFallbackPaintKit",
}

func main() {
	path := os.Args[1]
	ticks := map[int]bool{}
	if len(os.Args) > 2 {
		for _, s := range strings.Split(os.Args[2], ",") {
			if n, err := strconv.Atoi(s); err == nil {
				ticks[n] = true
			}
		}
	}

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	p := demoinfocs.NewParser(f)
	defer p.Close()

	seen := map[int]bool{}
	var classCount = map[string]int{}
	var claim = map[string]int{}
	var miss = map[string]int{}

	// group by the server class of the entity the props belong to
	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		t := gs.IngameTick()

		if ticks[t] {
			for _, pl := range gs.Participants().Playing() {
				if pl == nil {
					continue
				}
				pawn := pl.PlayerPawnEntity()
				v, ok := pv(pawn, "m_pMovementServices.m_nButtonDownMaskPrev")
				if !ok {
					v, ok = pv(pawn, "m_nButtonDownMaskPrev")
				}
				if ok {
					fmt.Printf("BTN %d %d %s\n", t, pl.SteamID64, v)
				}
			}
		}

		// one sample per grenade projectile entity
		for id, proj := range gs.GrenadeProjectiles() {
			if seen[id] || proj == nil || proj.Entity == nil {
				continue
			}
			seen[id] = true
			cls := proj.Entity.ServerClass().Name()
			classCount[cls]++
			kind := "?"
			if proj.WeaponInstance != nil {
				kind = proj.WeaponInstance.String()
			}
			fmt.Printf("\n--- nade entity %d  class=%s  kind=%s ---\n", id, cls, kind)
			for _, n := range dropped {
				v, ok := pv(proj.Entity, n)
				if ok {
					claim[n]++
					if len(v) > 150 {
						v = v[:150] + fmt.Sprintf("...(len %d)", len(v))
					}
					fmt.Printf("    OK   %-28s = %s\n", n, v)
				} else {
					miss[n]++
					fmt.Printf("    MISS %-28s\n", n)
				}
			}
		}
	})

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	fmt.Printf("\n########## SUMMARY ##########\n")
	fmt.Printf("distinct grenade projectile entities: %d\n", len(seen))
	var cl []string
	for c := range classCount {
		cl = append(cl, c)
	}
	sort.Strings(cl)
	for _, c := range cl {
		fmt.Printf("   class %-34s n=%d\n", c, classCount[c])
	}
	fmt.Printf("\nper-property reachability across those entities:\n")
	for _, n := range dropped {
		fmt.Printf("   %-30s present=%d absent=%d\n", n, claim[n], miss[n])
	}
}
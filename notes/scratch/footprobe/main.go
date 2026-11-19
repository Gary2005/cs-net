// footprobe — count player_footstep game events with demoinfocs-golang.
//
// Purpose: decide whether the very low player_footstep count seen through
// demoparser2 (~291 in a whole match, against ~14k expected from distance run)
// is a *parser* gap or the game genuinely emitting that few. demoinfocs is an
// independent implementation, so if it agrees, the number is real.
//
// It also counts every other game event by name, so we can see whether
// footsteps have a sibling event that demoparser2 misses.
package main

import (
	"flag"
	"math"
	"fmt"
	"os"
	"sort"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
)

func main() {
	demo := flag.String("demo", "", "path to .dem")
	flag.Parse()
	if *demo == "" {
		fmt.Fprintln(os.Stderr, "usage: footprobe -demo <file.dem>")
		os.Exit(2)
	}

	f, err := os.Open(*demo)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	counts := map[string]int{}
	// Ticks at which a footstep happened, plus the player's speed at that
	// moment, so we can tell "running" from "standing".
	type foot struct {
		tick    int
		steamID uint64
		speed   float64
	}
	var feet []foot

	p.RegisterEventHandler(func(e events.Footstep) {
		counts["player_footstep"]++
		speed := 0.0
		if e.Player != nil {
			v := e.Player.Velocity()
			speed = math.Hypot(float64(v.X), float64(v.Y))
		}
		feet = append(feet, foot{p.GameState().IngameTick(), e.Player.SteamID64, speed})
	})

	// Count every other event type we know how to name, for comparison.
	p.RegisterEventHandler(func(e events.WeaponFire) { counts["weapon_fire"]++ })
	p.RegisterEventHandler(func(e events.PlayerJump) { counts["player_jump"]++ })
	p.RegisterEventHandler(func(e events.PlayerHurt) { counts["player_hurt"]++ })
	p.RegisterEventHandler(func(e events.Kill) { counts["player_death"]++ })
	p.RegisterEventHandler(func(e events.SmokeStart) { counts["smokegrenade_detonate"]++ })
	p.RegisterEventHandler(func(e events.InfernoStart) { counts["inferno_startburn"]++ })
	p.RegisterEventHandler(func(e events.RoundStart) { counts["round_start"]++ })
	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) { counts["round_freeze_end"]++ })

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	fmt.Printf("demo      : %s\n", *demo)
	fmt.Printf("tickrate  : %.4f\n", p.TickRate())
	fmt.Printf("footsteps : %d\n", len(feet))
	fmt.Println("\nevent counts (demoinfocs):")
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("  %-24s %d\n", k, counts[k])
	}

	// speed distribution at footsteps
	if len(feet) > 0 {
		sum, mx, mn := 0.0, 0.0, 1e18
		for _, f := range feet {
			sum += f.speed
			if f.speed > mx {
				mx = f.speed
			}
			if f.speed < mn {
				mn = f.speed
			}
		}
		fmt.Printf("\nspeed at footstep (HU/s): min=%.0f mean=%.0f max=%.0f\n",
			mn, sum/float64(len(feet)), mx)
		// how many footsteps happen while standing still?
		slow := 0
		for _, f := range feet {
			if f.speed < 100 {
				slow++
			}
		}
		fmt.Printf("footsteps with speed < 100 HU/s: %d / %d\n", slow, len(feet))
	}
}

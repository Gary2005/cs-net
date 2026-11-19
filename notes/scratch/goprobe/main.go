// Probe v3: REAL CS2 inferno fire-cell footprints + smoke voxel occupancy field.
// Research probe for the cs-net-v5 plan. Not project code.
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
)

type snap struct {
	tick  int
	cells [][3]float64
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

	best := map[int64]*snap{}
	order := []int64{}
	smokeDone := false
	classes := map[string]int{}

	p.RegisterEventHandler(func(e events.FrameDone) {
		gs := p.GameState()

		// ---------- inferno fire cells: keep the fullest spread per inferno ----------
		for _, inf := range gs.Infernos() {
			uid := inf.UniqueID()
			active := inf.Fires().Active().List()
			if len(active) == 0 {
				continue
			}
			cur := best[uid]
			if cur == nil {
				order = append(order, uid)
			}
			if cur == nil || len(active) > len(cur.cells) {
				s := &snap{tick: gs.IngameTick()}
				for _, fr := range active {
					s.cells = append(s.cells, [3]float64{fr.Vector.X, fr.Vector.Y, fr.Vector.Z})
				}
				best[uid] = s
			}
		}

		// ---------- smoke voxel occupancy ----------
		if !smokeDone {
			for _, ent := range gs.Entities() {
				if ent == nil {
					continue
				}
				cls := ent.ServerClass().Name()
				if strings.Contains(strings.ToLower(cls), "smoke") {
					classes[cls]++
				}
				if !strings.Contains(strings.ToLower(cls), "smokegrenadeprojectile") {
					continue
				}

				var voxelN, voxelNonZero int
				var head []string
				var scalarHits []string
				sizeVal := ""
				for _, pr := range ent.Properties() {
					n := pr.Name()
					if n == "m_nVoxelFrameDataSize" {
						sizeVal = pr.Value().String()
					}
					if strings.HasPrefix(n, "m_VoxelFrameData.") {
						voxelN++
						s := pr.Value().String()
						if s != "0" {
							voxelNonZero++
						}
						if len(head) < 10 {
							head = append(head, fmt.Sprintf("%s=%s", strings.TrimPrefix(n, "m_VoxelFrameData."), s))
						}
						continue
					}
					ln := strings.ToLower(n)
					if strings.Contains(ln, "voxel") || strings.Contains(ln, "smoke") {
						scalarHits = append(scalarHits, fmt.Sprintf("%s=%s", n, pr.Value().String()))
					}
				}
				// only report once the smoke has actually bloomed
				if voxelN == 0 || (voxelNonZero == 0 && sizeVal == "0") {
					continue
				}
				// dump full voxel array in index order for offline grid analysis
				if os.Getenv("DUMP_VOXELS") != "" {
					var idx []string
					for _, pr := range ent.Properties() {
						if strings.HasPrefix(pr.Name(), "m_VoxelFrameData.") {
							idx = append(idx, pr.Value().String())
						}
					}
					det := ent.Property("m_vSmokeDetonationPos").Value().String()
					fmt.Printf("VOXELJSON {\"n\":%d,\"det\":%q,\"vals\":[%s]}\n",
						len(idx), det, strings.Join(idx, ","))
					smokeDone = true
					break
				}
				sort.Strings(scalarHits)
				fmt.Printf("\n=== SMOKE PROJECTILE class=%s (BLOOMED) ===\n", cls)
				fmt.Printf("  m_VoxelFrameData elements=%d  nonZero=%d\n", voxelN, voxelNonZero)
				fmt.Printf("  voxel head: %v\n", head)
				fmt.Printf("  other smoke props: %v\n", scalarHits)
				smokeDone = true
				break
			}
		}
	})

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	fmt.Printf("\n########## INFERNO FIRE-CELL FOOTPRINTS (max spread) ##########\n")
	fmt.Printf("distinct infernos observed: %d\n", len(order))
	maxCells, shown := 0, 0
	for _, uid := range order {
		s := best[uid]
		if len(s.cells) > maxCells {
			maxCells = len(s.cells)
		}
		if len(s.cells) < 4 || shown >= 3 {
			continue
		}
		shown++
		fmt.Printf("\n--- inferno uid=%d  fullest tick=%d  burning cells=%d ---\n", uid, s.tick, len(s.cells))
		for i, c := range s.cells {
			fmt.Printf("   cell[%2d] (%.1f, %.1f, %.1f)\n", i, c[0], c[1], c[2])
		}
	}
	fmt.Printf("\n>>> MAX real fire cells in one inferno: %d   (Valve cap = 16)\n", maxCells)
	fmt.Printf(">>> smoke classes seen: %v\n", classes)
}

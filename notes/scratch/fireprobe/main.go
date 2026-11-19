// Fire probe: dump the full per-cell 3D geometry of every inferno.
// Source 2 flattens fixed-size arrays into INDEXED property names, so the cells
// live at m_firePositions.0000 .. m_firePositions.%04d-1 -- there is no property
// literally called "m_firePositions". Research probe. Not project code.
package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/golang/geo/r3"
	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/sendtables"
)

func ivec(e sendtables.Entity, name string) (v r3.Vector, ok bool) {
	defer func() { recover() }()
	if e == nil {
		return v, false
	}
	p := e.Property(name)
	if p == nil {
		return v, false
	}
	return p.Value().R3Vec(), true
}

func ival(e sendtables.Entity, name string) (n int, ok bool) {
	defer func() { recover() }()
	if e == nil {
		return 0, false
	}
	val, exists := e.PropertyValue(name)
	if !exists {
		return 0, false
	}
	return val.Int(), true
}

func ibool(e sendtables.Entity, name string) (b bool, ok bool) {
	defer func() { recover() }()
	if e == nil {
		return false, false
	}
	val, exists := e.PropertyValue(name)
	if !exists {
		return false, false
	}
	return val.BoolVal(), true
}

type cell struct {
	p, parent, normal r3.Vector
	burning           bool
	hasP, hasPar      bool
	hasN              bool
}

func cells(e sendtables.Entity, count int) []cell {
	out := make([]cell, 0, count)
	for i := 0; i < count; i++ {
		s := fmt.Sprintf("%04d", i)
		var c cell
		if v, ok := ivec(e, "m_firePositions."+s); ok {
			c.p, c.hasP = v, true
		}
		if v, ok := ivec(e, "m_fireParentPositions."+s); ok {
			c.parent, c.hasPar = v, true
		}
		if v, ok := ivec(e, "m_BurnNormal."+s); ok {
			c.normal, c.hasN = v, true
		}
		if b, ok := ibool(e, "m_bFireIsBurning."+s); ok {
			c.burning = b
		}
		out = append(out, c)
	}
	return out
}

type snap struct {
	tick, count int
	cs          []cell
}

func main() {
	path := os.Args[1]
	mode := "summary"
	if len(os.Args) > 2 {
		mode = os.Args[2]
	}

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	p := demoinfocs.NewParser(f)
	defer p.Close()

	best := map[int]snap{}
	countRange := map[int]map[int]bool{}
	printedNames := map[int]bool{}

	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		t := gs.IngameTick()
		for id, inf := range gs.Infernos() {
			if inf == nil || inf.Entity == nil {
				continue
			}
			e := inf.Entity

			if mode == "names" {
				if !printedNames[id] {
					printedNames[id] = true
					all := e.Properties()
					fmt.Printf("\n### inferno %d class=%s -- %d properties total\n",
						id, e.ServerClass().Name(), len(all))
					for _, pr := range all {
						l := toLower(pr.Name())
						if contains(l, "fire") || contains(l, "burn") || contains(l, "flame") {
							fmt.Printf("    %-34s type=%-10v elem=%-10v\n",
								pr.Name(), pr.Type(), pr.ArrayElementType())
						}
					}
				}
				continue
			}

			cnt, _ := ival(e, "m_fireCount")
			if mode == "count" {
				if fs := os.Getenv("FIRE_ENTITY"); fs == "" || fs == fmt.Sprint(id) {
					fmt.Printf("COUNT %d %d %d\n", id, t, cnt)
				}
			}
			if _, ok := countRange[id]; !ok {
				countRange[id] = map[int]bool{}
			}
			countRange[id][cnt] = true
			cs := cells(e, cnt)
			if cur, ok := best[id]; !ok || len(cs) > len(cur.cs) {
				best[id] = snap{tick: t, count: cnt, cs: cs}
			}
		}
	})

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	ids := make([]int, 0, len(best))
	for id := range best {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	if mode == "names" {
		return
	}

	if mode == "raw" {
		for _, id := range ids {
			s := best[id]
			fmt.Printf("\n=== inferno %d  peak tick %d  fireCount=%d ===\n", id, s.tick, s.count)
			fmt.Printf("   %-4s %-24s %-24s %-22s %s\n", "i", "firePos", "parentPos", "burnNormal", "burning")
			for i, c := range s.cs {
				nz := "n/a"
				if c.hasN {
					nz = fmt.Sprintf("(%.3f %.3f %.3f)", c.normal.X, c.normal.Y, c.normal.Z)
				}
				par := "n/a"
				if c.hasPar {
					par = fmt.Sprintf("(%.1f %.1f %.1f)", c.parent.X, c.parent.Y, c.parent.Z)
				}
				pp := "ABSENT"
				if c.hasP {
					pp = fmt.Sprintf("(%.1f %.1f %.1f)", c.p.X, c.p.Y, c.p.Z)
				}
				fmt.Printf("   [%2d] %-24s %-24s %-22s %v\n", i, pp, par, nz, c.burning)
			}
		}
		return
	}

	fmt.Printf("infernos: %d\n", len(ids))
	cntHist := map[int]int{}
	for _, id := range ids {
		cntHist[best[id].count]++
	}
	ks := []int{}
	for k := range cntHist {
		ks = append(ks, k)
	}
	sort.Ints(ks)
	fmt.Print("peak fireCount histogram: ")
	for _, k := range ks {
		fmt.Printf("%d x%d   ", k, cntHist[k])
	}
	fmt.Println()

	spread := 0
	for _, id := range ids {
		if len(countRange[id]) > 1 {
			spread++
		}
	}
	fmt.Printf("infernos whose fireCount changed during life: %d/%d\n\n", spread, len(ids))

	fmt.Println("3D geometry of the cells at peak (HU):")
	fmt.Printf("  %-6s %4s %6s %6s %6s %8s %9s %8s %8s %6s\n",
		"ent", "n", "dX", "dY", "dZ", "maxRad", "planeRes", "nzMin", "nzMax", "tilt")
	worstZ, worstZID, worstRes, worstResID := 0.0, 0, 0.0, 0
	tiltCnt, parDiffCnt, noPos, nPar := 0, 0, 0, 0
	maxRadAll := 0.0
	for _, id := range ids {
		s := best[id]
		pts := []r3.Vector{}
		for _, c := range s.cs {
			if c.hasP {
				pts = append(pts, c.p)
			}
		}
		if len(pts) == 0 {
			noPos++
			continue
		}
		if len(pts) < 3 {
			continue
		}
		mnx, mny, mnz := pts[0].X, pts[0].Y, pts[0].Z
		mxx, mxy, mxz := mnx, mny, mnz
		for _, v := range pts {
			mnx, mxx = fmin(mnx, v.X), fmax(mxx, v.X)
			mny, mxy = fmin(mny, v.Y), fmax(mxy, v.Y)
			mnz, mxz = fmin(mnz, v.Z), fmax(mxz, v.Z)
		}
		n := float64(len(pts))
		var sx, sy, sz, sxx, sxy, syy, sxz, syz float64
		for _, v := range pts {
			sx += v.X
			sy += v.Y
			sz += v.Z
			sxx += v.X * v.X
			sxy += v.X * v.Y
			syy += v.Y * v.Y
			sxz += v.X * v.Z
			syz += v.Y * v.Z
		}
		det := (sxx - sx*sx/n) * (syy - sy*sy/n)
		det -= (sxy - sx*sy/n) * (sxy - sx*sy/n)
		res := 0.0
		if det > 1e-9 {
			a := ((sxz-sx*sz/n)*(syy-sy*sy/n) - (syz-sy*sz/n)*(sxy-sx*sy/n)) / det
			b := ((syz-sy*sz/n)*(sxx-sx*sx/n) - (sxz-sx*sz/n)*(sxy-sx*sy/n)) / det
			c := sz/n - a*sx/n - b*sy/n
			for _, v := range pts {
				d := v.Z - (a*v.X + b*v.Y + c)
				if d < 0 {
					d = -d
				}
				res = fmax(res, d)
			}
		}
		cx, cy, cz := sx/n, sy/n, sz/n
		maxRad := 0.0
		for _, v := range pts {
			maxRad = fmax(maxRad, dist3(v, r3.Vector{X: cx, Y: cy, Z: cz}))
		}
		maxRadAll = fmax(maxRadAll, maxRad)

		nzMin, nzMax := 1.0, -1.0
		seen, parDiff, anyPar := false, false, false
		for _, c := range s.cs {
			if c.hasN {
				if !seen {
					nzMin, nzMax, seen = c.normal.Z, c.normal.Z, true
				}
				nzMin, nzMax = fmin(nzMin, c.normal.Z), fmax(nzMax, c.normal.Z)
			}
			if c.hasPar {
				anyPar = true
				if c.hasP && dist3(c.p, c.parent) > 1.0 {
					parDiff = true
				}
			}
		}
		if anyPar {
			nPar++
		}
		if parDiff {
			parDiffCnt++
		}
		tilt := "flat"
		if nzMin < 0.9 {
			tilt = "TILTED"
			tiltCnt++
		}
		fmt.Printf("  %-6d %4d %6.1f %6.1f %6.1f %8.1f %9.2f %8.3f %8.3f %6s\n",
			id, len(pts), mxx-mnx, mxy-mny, mxz-mnz, maxRad, res, nzMin, nzMax, tilt)
		if mxz-mnz > worstZ {
			worstZ, worstZID = mxz-mnz, id
		}
		if res > worstRes {
			worstRes, worstResID = res, id
		}
	}
	fmt.Printf("\n  infernos with no absolute positions at all: %d\n", noPos)
	fmt.Printf("  max vertical spread dZ : %.1f HU (entity %d)\n", worstZ, worstZID)
	fmt.Printf("  max plane-fit residual : %.2f HU (entity %d)\n", worstRes, worstResID)
	fmt.Printf("  max 3D radius          : %.1f HU\n", maxRadAll)
	fmt.Printf("  infernos with a tilted normal (nz<0.9)  : %d/%d\n", tiltCnt, len(ids))
	fmt.Printf("  infernos where parentPos exists         : %d/%d\n", nPar, len(ids))
	fmt.Printf("  infernos where firePos != parentPos     : %d/%d\n", parDiffCnt, len(ids))
}

func toLower(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 32
		}
	}
	return string(b)
}

func contains(h, n string) bool {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return true
		}
	}
	return false
}

func dist3(a, b r3.Vector) float64 {
	dx, dy, dz := a.X-b.X, a.Y-b.Y, a.Z-b.Z
	return fsqrt(dx*dx + dy*dy + dz*dz)
}

func fsqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 60; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func fmin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func fmax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// Voxel probe: dump the raw m_VoxelFrameData array for smoke projectiles at given
// ticks, so the single scalar demoparser2 returns can be identified by comparison.
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

func bytesOf(e sendtables.Entity) (out []int, kind string) {
	defer func() {
		if r := recover(); r != nil {
			out, kind = nil, fmt.Sprint("PANIC:", r)
		}
	}()
	if e == nil {
		return nil, "nil entity"
	}
	v, ok := e.PropertyValue("m_VoxelFrameData")
	if !ok {
		return nil, "absent"
	}
	arr := v.S2Array()
	kind = fmt.Sprintf("S2Array[%d]", len(arr))
	if len(arr) > 0 {
		kind += fmt.Sprintf(" elemType=%T", arr[0])
	}
	out = make([]int, 0, len(arr))
	for _, x := range arr {
		switch n := x.(type) {
		case uint8:
			out = append(out, int(n))
		case int8:
			out = append(out, int(n))
		case uint16:
			out = append(out, int(n))
		case int32:
			out = append(out, int(n))
		case uint32:
			out = append(out, int(n))
		case int:
			out = append(out, n)
		case float32:
			out = append(out, int(n))
		case float64:
			out = append(out, int(n))
		default:
			txt := fmt.Sprint(x)
			iv, err := strconv.Atoi(txt)
			if err != nil {
				out = append(out, -1)
			} else {
				out = append(out, iv)
			}
		}
	}
	return out, kind
}

func main() {
	path := os.Args[1]
	want := map[int]bool{}
	if len(os.Args) > 2 {
		for _, s := range strings.Split(os.Args[2], ",") {
			if n, err := strconv.Atoi(s); err == nil {
				want[n] = true
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

	type rec struct {
		tick, size, upd int
		bs              []int
	}
	best := map[int]rec{}
	summary := os.Getenv("SUMMARY") != ""

	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		t := gs.IngameTick()

		if summary {
			for id, proj := range gs.GrenadeProjectiles() {
				if proj == nil || proj.Entity == nil ||
					!strings.Contains(proj.Entity.ServerClass().Name(), "Smoke") {
					continue
				}
				sz, _ := pv(proj.Entity, "m_nVoxelFrameDataSize")
				up, _ := pv(proj.Entity, "m_nVoxelUpdate")
				szN, _ := strconv.Atoi(sz)
				upN, _ := strconv.Atoi(up)
				if cur, ok := best[id]; ok && cur.size >= szN {
					continue
				}
				bs, _ := bytesOf(proj.Entity)
				best[id] = rec{tick: t, size: szN, upd: upN, bs: bs}
			}
		}

		if !want[t] {
			return
		}
		for id, proj := range gs.GrenadeProjectiles() {
			if proj == nil || proj.Entity == nil {
				continue
			}
			if !strings.Contains(proj.Entity.ServerClass().Name(), "Smoke") {
				continue
			}
			size, _ := pv(proj.Entity, "m_nVoxelFrameDataSize")
			upd, _ := pv(proj.Entity, "m_nVoxelUpdate")
			bs, kind := bytesOf(proj.Entity)
			if len(bs) == 0 {
				fmt.Printf("VOX %d %d size=%s upd=%s <no array: %s>\n", t, id, size, upd, kind)
				continue
			}
			n := len(bs)
			head := bs
			if len(head) > 12 {
				head = head[:12]
			}
			tailStart := n - 4
			if tailStart < 0 {
				tailStart = 0
			}
			nonzero := 0
			for _, b := range bs {
				if b != 0 {
					nonzero++
				}
			}
			fmt.Printf("VOX %d %d size=%s upd=%s kind=%s n=%d bytes0to11=%v bytes_last4=%v nonzero=%d maxbyte=%d\n",
				t, id, size, upd, kind, n, head, bs[tailStart:], nonzero, maxInt(bs))
			if os.Getenv("FULLDUMP") != "" {
				parts := make([]string, len(bs))
				for i, b := range bs {
					parts[i] = strconv.Itoa(b)
				}
				fmt.Printf("FULL %d %d size=%s upd=%s %s\n", t, id, size, upd, strings.Join(parts, " "))
			}
		}
	})

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	if summary {
		fmt.Printf("\n########## SMOKE JOURNAL SUMMARY (largest journal per entity) ##########\n")
		ids := make([]int, 0, len(best))
		for id := range best {
			ids = append(ids, id)
		}
		sort.Ints(ids)
		totK, totD, totH, totOne := 0, 0, 0, 0
		for _, id := range ids {
			r := best[id]
			size := r.size
			if size > len(r.bs) {
				size = len(r.bs)
			}
			off, nK, nD, nH, nRec := 0, 0, 0, 0, 0
			for off+4 <= size {
				_, ln := le16(r.bs, off)
				if off+4+ln > size {
					break
				}
				ft := -1
				if ln > 1 {
					ft = r.bs[off+5]
				}
				switch ft {
				case 3:
					nK++
				case 2:
					nD++
				case 0:
					nH++
				}
				nRec++
				off += 4 + ln
			}
			totK += nK
			totD += nD
			totH += nH
			if nRec <= 1 {
				totOne++
			}
			fmt.Printf("  entity %-6d lastTick=%-7d size=%-6d upd=%-5d records=%-4d KEYFRAME=%-3d DELTA=%-4d HEARTBEAT=%-4d consumed=%d/%d\n",
				id, r.tick, r.size, r.upd, nRec, nK, nD, nH, off, size)
		}
		fmt.Printf("\n  smoke entities: %d   with only 1 record: %d\n", len(ids), totOne)
		fmt.Printf("  totals: KEYFRAME=%d DELTA=%d HEARTBEAT=%d\n", totK, totD, totH)
	}
}

func le16(b []int, off int) (int, int) {
	if off+2 > len(b) {
		return 0, 0
	}
	return b[off] | b[off+1]<<8, b[off+2] | b[off+3]<<8
}

func maxInt(a []int) int {
	m := 0
	for _, x := range a {
		if x > m {
			m = x
		}
	}
	return m
}
// Format probe: decide the on-disk encoding for fire + smoke by measuring
//   * is m_nVoxelFrameDataSize monotonic per smoke entity?
//   * are the first `size` bytes of m_VoxelFrameData ever rewritten in place?
//   * how many *distinct* fire states does an inferno pass through?
// Research probe. Not project code.
package main

import (
	"bytes"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/sendtables"
)

func prop(e sendtables.Entity, name string) (v sendtables.PropertyValue, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	if e == nil {
		return v, false
	}
	v, ok = e.PropertyValue(name)
	return
}

func ivecStr(e sendtables.Entity, name string) string {
	defer func() { recover() }()
	if e == nil {
		return "?"
	}
	p := e.Property(name)
	if p == nil {
		return "?"
	}
	v := p.Value().R3Vec()
	return strconv.FormatFloat(v.X, 'b', -1, 64) + "," +
		strconv.FormatFloat(v.Y, 'b', -1, 64) + "," +
		strconv.FormatFloat(v.Z, 'b', -1, 64)
}

func iboolStr(e sendtables.Entity, name string) string {
	v, ok := prop(e, name)
	if !ok {
		return "?"
	}
	defer func() { recover() }()
	if v.BoolVal() {
		return "1"
	}
	return "0"
}

func iint(e sendtables.Entity, name string) (int, bool) {
	v, ok := prop(e, name)
	if !ok {
		return 0, false
	}
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	return v.Int(), true
}

// rawBytes returns the m_VoxelFrameData S2Array as bytes, or nil.
func rawBytes(e sendtables.Entity) (out []byte, arrLen int, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	v, exists := prop(e, "m_VoxelFrameData")
	if !exists {
		return nil, 0, false
	}
	arr := v.S2Array()
	arrLen = len(arr)
	out = make([]byte, 0, len(arr))
	for _, x := range arr {
		switch n := x.(type) {
		case uint8:
			out = append(out, n)
		case int8:
			out = append(out, byte(n))
		case uint16:
			out = append(out, byte(n))
		case uint32:
			out = append(out, byte(n))
		case uint64:
			out = append(out, byte(n))
		case int64:
			out = append(out, byte(n))
		case int:
			out = append(out, byte(n))
		default:
			iv, err := strconv.Atoi(fmt.Sprint(x))
			if err != nil {
				return nil, arrLen, false
			}
			out = append(out, byte(iv))
		}
	}
	return out, arrLen, true
}

func fireState(e sendtables.Entity, cnt int) string {
	var b strings.Builder
	b.WriteString("c")
	b.WriteString(strconv.Itoa(cnt))
	for i := 0; i < cnt; i++ {
		s := fmt.Sprintf("%04d", i)
		b.WriteString("|")
		b.WriteString(ivecStr(e, "m_firePositions."+s))
		b.WriteString(";")
		b.WriteString(ivecStr(e, "m_BurnNormal."+s))
		b.WriteString(";")
		b.WriteString(ivecStr(e, "m_fireParentPositions."+s))
		b.WriteString(";")
		b.WriteString(iboolStr(e, "m_bFireIsBurning."+s))
	}
	return b.String()
}

type smState struct {
	first, last   int
	frames        int
	nonMono       int
	rewrite       int
	distinct      int
	prevSize      int
	prevBytes     []byte
	maxSize       int
	maxArrLen     int
	maxJournal    int
	lastKey       string
	inited        bool
	firstNonEmpty int
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

	fmt.Printf("TICKRATE %f\n", p.TickRate())

	infFirst := map[int]int{}
	infLast := map[int]int{}
	infFrames := map[int]int{}
	infStates := map[int]int{}
	infPeak := map[int]int{}
	infLastKey := map[int]string{}
	sm := map[int]*smState{}
	smLastKey := map[int]string{}
	smLastUpd := map[int]int{}

	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		t := gs.IngameTick()

		for id, inf := range gs.Infernos() {
			if inf == nil || inf.Entity == nil {
				continue
			}
			cnt, _ := iint(inf.Entity, "m_fireCount")
			key := fireState(inf.Entity, cnt)
			if _, ok := infFirst[id]; !ok {
				infFirst[id] = t
			}
			infLast[id] = t
			infFrames[id]++
			if infLastKey[id] != key {
				infStates[id]++
				infLastKey[id] = key
			}
			if cnt > infPeak[id] {
				infPeak[id] = cnt
			}
		}

		for id, proj := range gs.GrenadeProjectiles() {
			if proj == nil || proj.Entity == nil {
				continue
			}
			if !strings.Contains(proj.Entity.ServerClass().Name(), "Smoke") {
				continue
			}
			e := proj.Entity
			st := sm[id]
			if st == nil {
				st = &smState{first: t, prevSize: -1}
				sm[id] = st
			}
			st.last = t
			st.frames++
			size, _ := iint(e, "m_nVoxelFrameDataSize")
			upd, _ := iint(e, "m_nVoxelUpdate")
			bs, arrLen, ok := rawBytes(e)
			if arrLen > st.maxArrLen {
				st.maxArrLen = arrLen
			}
			if !ok {
				continue
			}
			if size > st.maxSize {
				st.maxSize = size
			}
			if size <= len(bs) && size > st.maxJournal {
				st.maxJournal = size
			}
			// monotonicity
			if st.prevSize >= 0 && size < st.prevSize {
				st.nonMono++
			}
			// in-place rewrite of the already-published prefix
			if st.prevSize > 0 && st.prevBytes != nil {
				n := st.prevSize
				if n > len(bs) {
					n = len(bs)
				}
				if n > len(st.prevBytes) {
					n = len(st.prevBytes)
				}
				if !bytes.Equal(bs[:n], st.prevBytes[:n]) {
					st.rewrite++
				}
			}
			key := strconv.Itoa(size) + "/" + strconv.Itoa(upd)
			if !st.inited || key != smLastKey[id] {
				st.distinct++
				smLastKey[id] = key
				st.inited = true
			}
			if size > 0 && st.firstNonEmpty == 0 {
				st.firstNonEmpty = t
			}
			st.prevSize = size
			st.prevBytes = bs
			smLastUpd[id] = upd
		}
	})

	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}

	infIDs := keys(infFirst)
	fmt.Printf("INFERNOS %d\n", len(infIDs))
	totStates, totFrames := 0, 0
	for _, id := range infIDs {
		totStates += infStates[id]
		totFrames += infFrames[id]
		fmt.Printf("  INF %-6d first=%-7d last=%-7d frames=%-5d states=%-4d peak=%d\n",
			id, infFirst[id], infLast[id], infFrames[id], infStates[id], infPeak[id])
	}
	fmt.Printf("  inferno frames=%d distinct-states=%d ratio=%.3f\n",
		totFrames, totStates, float64(totStates)/float64(totFrames))

	smIDs := make([]int, 0, len(sm))
	for id := range sm {
		smIDs = append(smIDs, id)
	}
	sort.Ints(smIDs)
	fmt.Printf("SMOKES %d\n", len(smIDs))
	nonMonoTot, rewriteTot, distinctTot, maxJournal, maxArr := 0, 0, 0, 0, 0
	for _, id := range smIDs {
		st := sm[id]
		nonMonoTot += st.nonMono
		rewriteTot += st.rewrite
		distinctTot += st.distinct
		if st.maxJournal > maxJournal {
			maxJournal = st.maxJournal
		}
		if st.maxArrLen > maxArr {
			maxArr = st.maxArrLen
		}
		fmt.Printf("  SMK %-6d first=%-7d last=%-7d frames=%-5d nonMono=%-3d rewrite=%-3d distinct=%-4d maxSize=%-5d arrLen=%-5d journal=%d firstNonEmpty=%d\n",
			id, st.first, st.last, st.frames, st.nonMono, st.rewrite, st.distinct,
			st.maxSize, st.maxArrLen, st.maxJournal, st.firstNonEmpty)
	}
	fmt.Printf("  TOTALS nonMono=%d rewrite=%d distinct=%d maxJournal=%d maxArrLen=%d\n",
		nonMonoTot, rewriteTot, distinctTot, maxJournal, maxArr)
}

func keys(m map[int]int) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

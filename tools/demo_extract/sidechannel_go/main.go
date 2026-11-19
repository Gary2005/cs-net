// cs2sidechannel — extracts the two things demoparser2 cannot reach:
//
//   * inferno (fire) per-cell 3D geometry: m_firePositions / m_BurnNormal /
//     m_fireParentPositions / m_bFireIsBurning, which Source 2 flattens into
//     indexed property names (m_firePositions.0000 .. .0015). There is no
//     property literally called "m_firePositions".
//   * smoke voxel content: the raw m_VoxelFrameData byte journal.
//
// Both are sampled on a caller-supplied tick list and emitted as a compact
// little-endian binary blob. The output is *exact* at those ticks: a record is
// written for every sampled tick at which the state differs from the previous
// written record, so "state at tick t" == "payload of the last record with
// tick <= t" reproduces the sampled ground truth bit for bit.
package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/sendtables"
)

const (
	magic          = "CS2SC2" // side-channel format, version 2
	fireFields     = 16       // m_firePositions has 16 fixed slots
	modeAppend     = 0
	modeFull       = 1
	maxVoxelRecord = 1 << 16 // sanity bound on a payload
)

// ── property helpers ─────────────────────────────────────────────────────────

func prop(e sendtables.Entity, name string) (v sendtables.PropertyValue, ok bool) {
	if e == nil {
		return v, false
	}
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	v, ok = e.PropertyValue(name)
	return
}

func ivec(e sendtables.Entity, name string) (x, y, z float32, ok bool) {
	if e == nil {
		return 0, 0, 0, false
	}
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	p := e.Property(name)
	if p == nil {
		return 0, 0, 0, false
	}
	v := p.Value().R3Vec()
	return float32(v.X), float32(v.Y), float32(v.Z), true
}

func ibool(e sendtables.Entity, name string) bool {
	v, ok := prop(e, name)
	if !ok {
		return false
	}
	defer func() {
		if r := recover(); r != nil {
		}
	}()
	return v.BoolVal()
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

// voxelBytes reads m_VoxelFrameData (S2Array, uint64 elements holding byte
// values) and returns it as a byte slice. The returned slice is a fresh
// allocation owned by the caller.
func voxelBytes(e sendtables.Entity) (out []byte, arrLen int, ok bool) {
	v, exists := prop(e, "m_VoxelFrameData")
	if !exists {
		return nil, 0, false
	}
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	// demoinfocs v5 把 PropertyValue.S2Array() 改名成了 Array()。
	arr := v.Array()
	arrLen = len(arr)
	out = make([]byte, len(arr))
	for i, x := range arr {
		switch n := x.(type) {
		case uint8:
			out[i] = n
		case int8:
			out[i] = byte(n)
		case uint16:
			out[i] = byte(n)
		case int16:
			out[i] = byte(n)
		case uint32:
			out[i] = byte(n)
		case int32:
			out[i] = byte(n)
		case uint64:
			out[i] = byte(n)
		case int64:
			out[i] = byte(n)
		case int:
			out[i] = byte(n)
		default:
			iv, err := strconv.Atoi(fmt.Sprint(x))
			if err != nil || iv < 0 || iv > 255 {
				return nil, arrLen, false
			}
			out[i] = byte(iv)
		}
	}
	return out, arrLen, true
}

// ── recorded state ───────────────────────────────────────────────────────────

type fireCell struct {
	p, n, q [3]float32
	burning bool
}

type fireRec struct {
	tick  int
	cells []fireCell
}

type fireInst struct {
	id       int
	typ      int
	first    int
	last     int
	lastSeen int
	recs     []fireRec
	cells    []fireCell
	prevKey  []byte
	keyBuf   []byte
}

type smokeRec struct {
	tick   int
	size   int
	update int
	mode   uint8
	data   []byte
}

type smokeInst struct {
	id       int
	first    int
	last     int
	lastSeen int
	recs     []smokeRec
	prev     []byte // published journal as of the last written record
	prevSize int
	prevUpd  int
}

var (
	f32Exact      = true
	maxVoxelCap   int
)

func noteF32(orig float64, narrowed float32) {
	if float64(narrowed) != orig {
		f32Exact = false
	}
}

func (fi *fireInst) read(e sendtables.Entity, count int) {
	fi.keyBuf = fi.keyBuf[:0]
	fi.keyBuf = binary.LittleEndian.AppendUint16(fi.keyBuf, uint16(count))
	fi.cells = fi.cells[:0]
	for i := 0; i < count; i++ {
		s := fmt.Sprintf("%04d", i)
		var c fireCell
		var okp, okn, okq bool
		var ox, oy, oz float64
		c.p[0], c.p[1], c.p[2], okp = ivec(e, "m_firePositions."+s)
		c.n[0], c.n[1], c.n[2], okn = ivec(e, "m_BurnNormal."+s)
		c.q[0], c.q[1], c.q[2], okq = ivec(e, "m_fireParentPositions."+s)
		c.burning = ibool(e, "m_bFireIsBurning."+s)
		fi.keyBuf = append(fi.keyBuf, boolByte(okp), boolByte(okn), boolByte(okq), boolByte(c.burning))
		for _, f := range [3][3]float32{c.p, c.n, c.q} {
			for k := 0; k < 3; k++ {
				fi.keyBuf = binary.LittleEndian.AppendUint32(fi.keyBuf, math.Float32bits(f[k]))
			}
		}
		fi.cells = append(fi.cells, c)
		_ = ox
		_ = oy
		_ = oz
	}
}

func boolByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

// ── main ─────────────────────────────────────────────────────────────────────

func main() {
	demoPath := flag.String("demo", "", "path to .dem")
	tickFile := flag.String("ticks", "", "file with one sampled tick per line")
	outPath := flag.String("out", "", "output binary path")
	ignoreEntityPanic := flag.Bool("ignore-entity-panic", true,
		"skip PacketEntities parsing panics instead of aborting the whole demo "+
			"(demoinfocs workaround: some broadcast/POV demos delta-reference entity "+
			"indices that were never created; the affected packet is dropped)")
	flag.Parse()

	if *demoPath == "" || *tickFile == "" || *outPath == "" {
		fmt.Fprintln(os.Stderr, "usage: cs2sidechannel -demo X.dem -ticks ticks.txt -out out.bin")
		os.Exit(2)
	}

	ticks, err := readTicks(*tickFile)
	if err != nil {
		fatal(err)
	}
	tickSet := make(map[int]struct{}, len(ticks))
	for _, t := range ticks {
		tickSet[t] = struct{}{}
	}

	f, err := os.Open(*demoPath)
	if err != nil {
		fatal(err)
	}
	defer f.Close()
	// IgnorePacketEntitiesPanic: 有些直播/POV demo 的 PacketEntities 会引用一个从未被
	// 创建过的实体下标（"unable to find existing entity N"），demoinfocs 默认会 panic
	// 并让整场解析失败。打开后该包被丢弃、解析继续，只留一条 ParserWarn。
	// 必须从 DefaultParserConfig 出发，否则 MsgQueueBufferSize 会变成 0（强制串行解析）。
	cfg := demoinfocs.DefaultParserConfig
	cfg.IgnorePacketEntitiesPanic = *ignoreEntityPanic
	p := demoinfocs.NewParserWithConfig(f, cfg)
	defer p.Close()

	var entityPanics int
	p.RegisterEventHandler(func(w events.ParserWarn) {
		if w.Type != events.WarnTypePacketEntitiesPanic {
			return
		}
		entityPanics++
		if entityPanics <= 5 {
			fmt.Fprintf(os.Stderr, "sidechannel: warning: %s\n", w.Message)
		}
	})

	var (
		fireList  []*fireInst
		smokeList []*smokeInst
		curFire   = map[int]*fireInst{}
		curSmoke  = map[int]*smokeInst{}
	)

	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		t := gs.IngameTick()
		_, sampled := tickSet[t]

		// ── infernos ─────────────────────────────────────────────────────
		for id, inf := range gs.Infernos() {
			if inf == nil || inf.Entity == nil {
				continue
			}
			e := inf.Entity
			cur := curFire[id]
			if cur == nil || t-cur.lastSeen > 1 {
				cur = &fireInst{id: id, first: t, lastSeen: t}
				if v, ok := iint(e, "m_nInfernoType"); ok {
					cur.typ = v
				}
				curFire[id] = cur
				fireList = append(fireList, cur)
			}
			cur.lastSeen = t
			cur.last = t
			if !sampled {
				continue
			}
			count, ok := iint(e, "m_fireCount")
			if !ok {
				count = 0
			}
			if count > fireFields {
				count = fireFields
			}
			if count < 0 {
				count = 0
			}
			cur.read(e, count)
			if !bytes.Equal(cur.keyBuf, cur.prevKey) {
				cur.prevKey = append(cur.prevKey[:0], cur.keyBuf...)
				cells := make([]fireCell, len(cur.cells))
				copy(cells, cur.cells)
				cur.recs = append(cur.recs, fireRec{tick: t, cells: cells})
			}
		}

		// ── smoke projectiles ────────────────────────────────────────────
		for id, proj := range gs.GrenadeProjectiles() {
			if proj == nil || proj.Entity == nil {
				continue
			}
			e := proj.Entity
			if !strings.Contains(e.ServerClass().Name(), "Smoke") {
				continue
			}
			cur := curSmoke[id]
			if cur == nil || t-cur.lastSeen > 1 {
				cur = &smokeInst{id: id, first: t, lastSeen: t}
				curSmoke[id] = cur
				smokeList = append(smokeList, cur)
			}
			cur.lastSeen = t
			cur.last = t
			if !sampled {
				continue
			}
			size, _ := iint(e, "m_nVoxelFrameDataSize")
			upd, _ := iint(e, "m_nVoxelUpdate")

			raw, arrLen, ok := voxelBytes(e)
			if !ok {
			    continue
			}
			if arrLen > maxVoxelCap {
			    maxVoxelCap = arrLen
			}
			if size < 0 || size > len(raw) {
			    size = len(raw)
			}
			raw = raw[:size]

			// The contract is "exact state at every sampled tick", so the whole journal
			// is compared on every sampled tick. No heuristic, no (size,update) shortcut.
			fresh := len(cur.recs) == 0
			changed := fresh || !bytes.Equal(raw, cur.prev) ||
			    size != cur.prevSize || upd != cur.prevUpd
			if !changed {
			    continue
			}

			mode := uint8(modeFull)
			if len(raw) >= len(cur.prev) && bytes.HasPrefix(raw, cur.prev) {
			    // append-only journal: store just the newly published tail
			    mode = modeAppend
			}
			var payload []byte
			if mode == modeAppend {
			    payload = raw[len(cur.prev):]
			} else {
			    payload = raw
			}
			if len(payload) > maxVoxelRecord {
			    fmt.Fprintf(os.Stderr, "payload too large (%d) at tick %d\n", len(payload), t)
			    os.Exit(1)
			}
			cp := make([]byte, len(payload))
			copy(cp, payload)
			cur.recs = append(cur.recs, smokeRec{tick: t, size: size, update: upd, mode: mode, data: cp})
			cur.prev = append(cur.prev[:0], raw...)
			cur.prevSize = size
			cur.prevUpd = upd
		}
	})

	if err := p.ParseToEnd(); err != nil {
		fatal(err)
	}

	tickrate := p.TickRate()

	var buf bytes.Buffer
	buf.WriteString(magic)
	buf.WriteByte(2) // version
	putU8(&buf, boolByte(f32Exact))
	putF64(&buf, tickrate)
	putU32(&buf, uint32(maxVoxelCap))

	putU32(&buf, uint32(len(ticks)))
	for _, t := range ticks {
		putI32(&buf, int32(t))
	}

	putU32(&buf, uint32(len(fireList)))
	for _, fi := range fireList {
		putU32(&buf, uint32(fi.id))
		putI32(&buf, int32(fi.first))
		putI32(&buf, int32(fi.last))
		putI32(&buf, int32(fi.typ))
		putU32(&buf, uint32(len(fi.recs)))
		for _, r := range fi.recs {
			putI32(&buf, int32(r.tick))
			putU16(&buf, uint16(len(r.cells)))
			for _, c := range r.cells {
				for k := 0; k < 3; k++ {
					putF32(&buf, c.p[k])
				}
				for k := 0; k < 3; k++ {
					putF32(&buf, c.n[k])
				}
				for k := 0; k < 3; k++ {
					putF32(&buf, c.q[k])
				}
				putU8(&buf, boolByte(c.burning))
			}
		}
	}

	putU32(&buf, uint32(len(smokeList)))
	for _, si := range smokeList {
		putU32(&buf, uint32(si.id))
		putI32(&buf, int32(si.first))
		putI32(&buf, int32(si.last))
		putU32(&buf, uint32(len(si.recs)))
		for _, r := range si.recs {
			putI32(&buf, int32(r.tick))
			putU16(&buf, uint16(r.size))
			putU16(&buf, uint16(r.update))
			putU8(&buf, r.mode)
			putU32(&buf, uint32(len(r.data)))
			buf.Write(r.data)
		}
	}

	if err := os.WriteFile(*outPath, buf.Bytes(), 0o644); err != nil {
		fatal(err)
	}

	nf, ns, nfr, nsr := len(fireList), len(smokeList), 0, 0
	for _, x := range fireList {
		nfr += len(x.recs)
	}
	for _, x := range smokeList {
		nsr += len(x.recs)
	}
	fmt.Fprintf(os.Stderr,
		"sidechannel: tickrate=%.4f infernos=%d fire_records=%d smokes=%d smoke_records=%d bytes=%d f32_exact=%v entity_panics=%d\n",
		tickrate, nf, nfr, ns, nsr, buf.Len(), f32Exact, entityPanics)
	_ = sort.Ints
}

func readTicks(path string) ([]int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []int
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil {
			return nil, fmt.Errorf("bad tick %q: %w", line, err)
		}
		out = append(out, n)
	}
	return out, nil
}

// ── little-endian writers ────────────────────────────────────────────────────

func putU8(b *bytes.Buffer, v byte)  { b.WriteByte(v) }
func putU16(b *bytes.Buffer, v uint16) { var t [2]byte; binary.LittleEndian.PutUint16(t[:], v); b.Write(t[:]) }
func putU32(b *bytes.Buffer, v uint32) { var t [4]byte; binary.LittleEndian.PutUint32(t[:], v); b.Write(t[:]) }
func putI32(b *bytes.Buffer, v int32)  { var t [4]byte; binary.LittleEndian.PutUint32(t[:], uint32(v)); b.Write(t[:]) }
func putF32(b *bytes.Buffer, v float32) { var t [4]byte; binary.LittleEndian.PutUint32(t[:], math.Float32bits(v)); b.Write(t[:]) }
func putF64(b *bytes.Buffer, v float64) { var t [8]byte; binary.LittleEndian.PutUint64(t[:], math.Float64bits(v)); b.Write(t[:]) }

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

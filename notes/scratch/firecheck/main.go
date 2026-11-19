// Independent check: print (tick, fireCount, burning-bits, first cell xyz) for
// one inferno entity, at a caller-given stride, straight from demoinfocs.
package main

import (
	"fmt"
	"os"
	"strconv"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/sendtables"
)

func ival(e sendtables.Entity, n string) int {
	defer func() { recover() }()
	v, ok := e.PropertyValue(n)
	if !ok {
		return -1
	}
	return v.Int()
}

func ib(e sendtables.Entity, n string) int {
	defer func() { recover() }()
	v, ok := e.PropertyValue(n)
	if !ok {
		return -1
	}
	if v.BoolVal() {
		return 1
	}
	return 0
}

func v3(e sendtables.Entity, n string) (float64, float64, float64) {
	defer func() { recover() }()
	p := e.Property(n)
	if p == nil {
		return 0, 0, 0
	}
	v := p.Value().R3Vec()
	return v.X, v.Y, v.Z
}

func main() {
	path, want := os.Args[1], 0
	if len(os.Args) > 2 {
		want, _ = strconv.Atoi(os.Args[2])
	}
	stride := 1
	if len(os.Args) > 3 {
		stride, _ = strconv.Atoi(os.Args[3])
	}
	f, _ := os.Open(path)
	defer f.Close()
	p := demoinfocs.NewParser(f)
	defer p.Close()
	p.RegisterEventHandler(func(events.FrameDone) {
		gs := p.GameState()
		t := gs.IngameTick()
		if t%stride != 0 {
			return
		}
		inf, ok := gs.Infernos()[want]
		if !ok || inf == nil || inf.Entity == nil {
			return
		}
		e := inf.Entity
		c := ival(e, "m_fireCount")
		bits := ""
		for i := 0; i < c; i++ {
			bits += strconv.Itoa(ib(e, fmt.Sprintf("m_bFireIsBurning.%04d", i)))
		}
		x, y, z := v3(e, "m_firePositions.0000")
		fmt.Printf("t=%d count=%d burning=%s p0=(%.2f %.2f %.2f)\n", t, c, bits, x, y, z)
	})
	if err := p.ParseToEnd(); err != nil {
		panic(err)
	}
}

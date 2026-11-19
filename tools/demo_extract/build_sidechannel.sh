#!/usr/bin/env bash
# Build the demoinfocs-golang side-channel helper (fire cells + smoke voxels).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export GOPATH="${GOPATH:-$ROOT/.build/gopath}"
export GOCACHE="${GOCACHE:-$ROOT/.build/gocache}"
export GOFLAGS="${GOFLAGS:--mod=mod}"
cd "$HERE/sidechannel_go"
go build -o cs2sidechannel .
echo "built $HERE/sidechannel_go/cs2sidechannel"

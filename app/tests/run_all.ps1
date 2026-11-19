# Run every demo-library check in one go.
#
# Usage:  pwsh -File app/tests/run_all.ps1
# Exit code is non-zero if any check fails.
#
# Note on stderr: demoparser2 is Rust, and `test_library.py` deliberately feeds
# it malformed files to prove the panic handling holds.  Those panics print a
# backtrace to stderr, which PowerShell turns into a NativeCommandError and an
# exit code of 1 even though the script succeeded -- so stderr is redirected to
# $null here and the real exit code is what decides.  Running any single test by
# hand will show the same misleading exit code for the same reason.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $root

$python = if ($env:CSNET_PYTHON) { $env:CSNET_PYTHON } else { 'python' }
$failed = @()

function Step($name, $script) {
    Write-Host ""
    Write-Host "=== $name " -NoNewline
    Write-Host ("=" * [Math]::Max(0, 58 - $name.Length))
    & $python $script 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> $name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        $script:failed += $name
    }
}

Step 'paths (dev / frozen / %LOCALAPPDATA%)'          'app/tests/test_paths.py'
Step 'library units (config / metadata / db / probe)' 'app/tests/test_library.py'
Step 'radar transform + sampling'                     'app/tests/test_radar.py'
Step 'player aggregation + steam avatars'             'app/tests/test_players.py'
Step 'HTTP API'                                       'app/tests/test_api.py'
Step 'front-end wiring (ids / api paths)'             'app/tests/test_ui_wiring.py'
# Model inputs / duel pairing / cache keys / the two /api/ai routes.  Does NOT
# load the real checkpoints (that is test_ai_overlay.mjs, needs a server).
Step 'AI prediction (pairing / windows / cache / api)' 'app/tests/test_ai.py'
# The five AI-stats numbers (kill / round impact, position risk, duel score), the
# per-match aggregate + its disk cache, and the two player-profile endpoints.
# Uses hand-computed fixtures, so no GPU and no real checkpoints.
Step 'AI stats (impact / position / duel score)'       'app/tests/test_ai_stats.py'
# scripts/download_models.py (mirror fallback, resume, size checks) and the two
# /api/ai/models* routes.  A local HTTP server impersonates the HF repo, so this
# neither downloads 665 MB nor needs the real checkpoints.
Step 'model download (mirror fallback / resume / api)' 'app/tests/test_download_models.py'
# scripts/extract_maps.py (.part + atomic rename, byte progress, bad zips) and the
# two /api/library/maps* routes.  Every zip here is a few hundred bytes, so this
# never unpacks the real 943 MB of map geometry.
Step 'map unpacking (.part / progress / api)'            'app/tests/test_extract_maps.py'
# The launcher and `--packaging-selftest`: this one really starts a server on a
# random port and hits /healthz /library /replay /api/maps, so it is ~15 s slower
# than the others -- but it is the only thing that catches "the entry point does
# not come up", which otherwise only a player would ever see.
Step 'desktop launcher + packaging selftest'             'app/tests/test_desktop.py'

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host ("FAILED: " + ($failed -join ', ')) -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "All demo-library checks passed." -ForegroundColor Green
Pop-Location
exit 0

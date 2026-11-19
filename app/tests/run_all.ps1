# Run every demo-library check in one go.
#
# Usage:  pwsh -File app/tests/run_all.ps1
# Exit code is non-zero if any check fails.
#
# Note on stderr: demoparser2 is Rust, and `test_library.py` deliberately feeds
# it malformed files to prove the panic handling holds.  Those panics print a
# backtrace to stderr, which PowerShell turns into a NativeCommandError and an
# exit code of 1 even though the script succeeded -- so each step's output is
# redirected to its own file and the real exit code is what decides.  That file
# is also the only way a CI failure is ever diagnosable: before, stderr went to
# `$null`, so the v5.0.0 release build failed with nothing but
# "app tests failed" in the log (see the tail printed on failure).

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $root

$python = if ($env:CSNET_PYTHON) { $env:CSNET_PYTHON } else { 'python' }
$failed = @()

# A step that never ran anything (`& $null tool`, python not on PATH) exits 0 and
# would read as "passed" while nothing was checked -- the bug the visualizer
# runner documents.  Resolve the interpreter once, up front.  Do NOT probe this
# with `$Error.Count` inside Step: with `*>` the native stderr of test_library
# (demoparser2 panics on purpose) lands in the error stream too, so counting
# errors marks passing steps as failed.
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    Write-Host "python 找不到：$python（用 CSNET_PYTHON 指定解释器）" -ForegroundColor Red
    Pop-Location
    exit 1
}

# One log per step, kept for the whole run and picked up by the CI artifact step
# (`build/test-logs/**`).  Kept on success too -- a green step whose warnings
# mention a deprecation is exactly the thing someone looks for later.
$logDir = Join-Path $root 'build/test-logs/app'
New-Item -ItemType Directory -Force $logDir | Out-Null

function Step($name, $script) {
    Write-Host ""
    Write-Host "=== $name " -NoNewline
    Write-Host ("=" * [Math]::Max(0, 58 - $name.Length))
    $log = Join-Path $logDir ((Split-Path -Leaf $script) -replace '\.py$', '.log')
    & $python $script *> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> $name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "  ---- tail of $log ----" -ForegroundColor Red
        Get-Content $log -Tail 60 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "  $_" }
        $script:failed += $name
        return
    }
    # Everything else went to the log, so echo the one line that matters -- CI
    # logs stay readable without losing the detail.
    $summary = Get-Content $log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match 'checks passed|passed, \d+ failed' } |
        Select-Object -Last 1
    if ($summary) { Write-Host ("  " + $summary.Trim()) }
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
    Write-Host "step logs: $logDir" -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "All demo-library checks passed." -ForegroundColor Green
Pop-Location
exit 0

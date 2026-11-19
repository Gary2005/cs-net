# Run every visualizer check in one go, with the flags each one needs.
#
# Why this exists: `check_syntax.mjs` needs `--experimental-vm-modules`, and
# forgetting it used to print "0 ok, 11 failed" -- indistinguishable from a
# repo-wide syntax catastrophe. Encoding the flags here means they cannot be
# forgotten. (`check_syntax.mjs` now also fails loudly on its own.)
#
# Usage:  pwsh -File visualizer/tests/run_all.ps1
# Exit code is non-zero if any check fails.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $root
$failed = @()

# NOTE FOR EDITORS: keep this file ASCII-only.  Windows PowerShell 5.1 reads a
# BOM-less .ps1 as the system ANSI codepage (GBK here), and a UTF-8 line ending
# in a 3-byte character (e.g. the ideographic full stop) decodes so that the
# newline is swallowed -- the next line silently becomes part of the comment.
# That is how a "$python = ..." assignment once disappeared: the step then ran
# `& $python ...` with $python null, printed nothing, and exited 0, so the suite
# reported "all checks passed" without running the check at all.
#
# Python is only used by the two steps below (both import numpy / demo_extract).
# This used to be a hardcoded conda path; on another machine that is just
# "python is not recognized" (exit 9009).  CSNET_PYTHON overrides it, same as
# app/tests/run_all.ps1.
$python = if ($env:CSNET_PYTHON) { $env:CSNET_PYTHON } else { 'D:\conda\envs\cs-net-v5\python.exe' }

function Step($name, $block) {
    Write-Host ""
    Write-Host "=== $name " -NoNewline
    Write-Host ("=" * [Math]::Max(0, 58 - $name.Length))
    $errorsBefore = $Error.Count
    & $block
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> $name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        $script:failed += $name
        return
    }
    # A block that never invoked anything (the `& $null tool` class of bug) prints
    # nothing and leaves $LASTEXITCODE at the previous step's 0, so the step reads
    # as "passed" while the check never ran.  PowerShell does record a
    # non-terminating error for it, and that is all we have to go on.
    if ($Error.Count -gt $errorsBefore) {
        Write-Host "  -> $name FAILED (raised a PowerShell error; nothing ran)" -ForegroundColor Red
        $script:failed += $name
    }
}

Step 'ESM syntax (needs --experimental-vm-modules)' {
    node --experimental-vm-modules visualizer/tests/check_syntax.mjs
}

Step 'cross-module imports/exports' {
    node visualizer/tests/check_imports.mjs
}

Step 'demo weapon -> model coverage' {
    node visualizer/tests/check_weapon_coverage.mjs
}

Step 'grenade projectile -> model (mapping + long axis)' {
    node visualizer/tests/check_nade_models.mjs
}

Step 'index.html <-> main.js element ids' {
    node visualizer/tests/check_dom_ids.mjs
}

Step 'geometric visibility (brute-force parity + perf)' {
    node visualizer/tests/test_visibility.mjs
}

Step 'yaw interpolation across +-180 (shortest arc)' {
    node visualizer/tests/test_angle_wrap.mjs
}

Step 'player x-ray (camera occlusion parity + perf)' {
    node visualizer/tests/test_player_xray.mjs
}

Step 'match stats (aggregation + /api/stats)' {
    # synthetic cases always; the real-.csz half cross-checks app/metadata.py's
    # scoreboard when a .csz is available.
    $env:PYTHONIOENCODING = 'utf-8'
    & $python visualizer/tests/test_stats.py
}

Step 'baked map colours match their OBJ' {
    # <map>.colors.bin is stored per OBJ *face*, so the face count is the only
    # thing that makes it safe to apply. Skips when nothing has been baked.
    $env:PYTHONIOENCODING = 'utf-8'
    & $python visualizer/tests/test_map_colors.py
}

Step 'startup banner box alignment' {
    # both servers' banners: every line must have the same display width (CJK = 2
    # columns).  A hand-padded line is off by a few cells and nothing else notices.
    $env:PYTHONIOENCODING = 'utf-8'
    & $python visualizer/tests/test_banner.py
}

Pop-Location
Write-Host ""
if ($failed.Count) {
    Write-Host ("FAILED: " + ($failed -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
exit 0

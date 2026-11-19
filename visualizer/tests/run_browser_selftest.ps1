# Run the browser asset self-test in headless Chrome, driven in real time.
#
# Why not `chrome --dump-dom`: that needs --virtual-time-budget, and the virtual
# clock jumps to the end of the budget as soon as the page looks idle. Loading a
# 3 MB GLB plus nine 1024^2 textures is genuinely async, so it got truncated
# every time (sprites passed, weapons always timed out). Here Chrome runs in
# real time and the page POSTs its result to /api/_selftest, which this script
# polls -- so the run ends when the tests actually finish.
#
# Usage: pwsh -File visualizer/tests/run_browser_selftest.ps1 [-Port 5099] [-TimeoutSec 300]

param(
    [int]$Port = 5099,
    [int]$TimeoutSec = 300,
    [string]$Url = "",
    [switch]$KeepScreenshot
)

$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
if (-not $Url) { $Url = "http://127.0.0.1:$Port/static/selftest.html" }
$api = "http://127.0.0.1:$Port/api/_selftest"

$chrome = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $chrome) {
    Write-Error "No Chrome/Edge found. Open the page manually: $Url"
}

# Clear any previous result so we cannot read a stale one.
try { Invoke-RestMethod -Uri $api -Method Post -Body '{"done":false}' -ContentType 'application/json' | Out-Null } catch {}

$profile = Join-Path $root ".build\chrome-selftest"
New-Item -ItemType Directory -Force -Path $profile | Out-Null
$shot = Join-Path $root ".build\selftest_page.png"

$chromeArgs = @(
    "--headless=new",
    "--disable-gpu",
    # headless has no real GPU; without this WebGL is simply unavailable
    "--enable-unsafe-swiftshader",
    "--use-gl=angle", "--use-angle=swiftshader",
    "--no-sandbox", "--disable-dev-shm-usage",
    "--window-size=1400,1200",
    "--user-data-dir=$profile",
    # Without a termination flag (--dump-dom / --screenshot) or a debugging port,
    # headless Chrome starts but never navigates, so the page never runs and no
    # result is ever posted. An explicit debugging port keeps it alive and
    # navigating for as long as we need.
    "--remote-debugging-port=9333",
    # NOTE: no --screenshot / --dump-dom. Both make Chrome exit as soon as it has
    # the frame, which happens while the async tests are still running -- the
    # dumped DOM then just says "running…".
    $Url
)

Write-Host "chrome : $chrome"
Write-Host "url    : $Url"
Write-Host "polling $api (timeout ${TimeoutSec}s)"
Write-Host ""

$proc = Start-Process -FilePath $chrome -ArgumentList $chromeArgs -PassThru -WindowStyle Hidden

$result = $null
# 页面在 module 还没跑完时会先发一个 done:false 的心跳（见 selftest.html 末尾的
# 注释）。留着它：超时的时候能打印出"卡在哪一行"，比一句 "no result" 有用得多。
$partial = $null
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 700
    try {
        $r = Invoke-RestMethod -Uri $api -Method Get -TimeoutSec 5
        if ($r -and $r.done) { $result = $r; break }
        if ($r -and $r.text) { $partial = $r }
    } catch { }
    if ($proc.HasExited) {
        # Chrome died on its own; give a late POST a moment to land, then stop.
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-RestMethod -Uri $api -Method Get -TimeoutSec 5
            if ($r -and $r.done) { $result = $r }
            elseif ($r -and $r.text) { $partial = $r }
        } catch { }
        if (-not $result) { break }
    }
}

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }

if (-not $result) {
    Write-Host "!! no result from the page within ${TimeoutSec}s." -ForegroundColor Red
    if ($partial -and $partial.text) {
        Write-Host "   页面最后停在这里（done:false，module 没跑完）:" -ForegroundColor Yellow
        Write-Host $partial.text
        $partial.text | Set-Content (Join-Path $root ".build\selftest_result.txt") -Encoding utf8
    } else {
        Write-Host "   The page may have failed to load, or the POST was blocked."
    }
    exit 1
}

$result.text | Set-Content (Join-Path $root ".build\selftest_result.txt") -Encoding utf8
Write-Host $result.text

if (Test-Path $shot) {
    Write-Host "`npage screenshot: $shot"
    if (-not $KeepScreenshot) { }
}

Write-Host ""
if ($result.ok) {
    Write-Host "[browser selftest] OK" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[browser selftest] FAILED" -ForegroundColor Red
    exit 1
}

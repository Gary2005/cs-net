# 把 `dist/cs-net` 打成发行包。
#
#   pwsh -File packaging/package.ps1 -Flavor both
#   pwsh -File packaging/package.ps1 -Flavor lite -Version 5.0.0
#
# 两种成品出自**同一棵** `dist/cs-net`：
#
#   cs-net-<ver>-win64-lite.zip    权重不随包（首次运行点横幅下载）
#   cs-net-<ver>-win64-full.zip    权重随包，解压即离线可用
#
# 地图几何在两种包里都是**已解压**的：943 MB 的文本 OBJ 压进 zip 只占 ~250 MB，
# 换玩家一次"点横幅 + 等解压 + 要 943 MB 临时空间"，不划算。
#
# 目录约定（和 app/paths.py 对得上）：
#   cs-net-<ver>-win64\cs-net.exe
#   cs-net-<ver>-win64\_internal\...          资源 + 包
#   cs-net-<ver>-win64\maps\optimized_obj_files\*.obj
#   cs-net-<ver>-win64\models\...             仅 full

[CmdletBinding()]
param(
    [ValidateSet('lite', 'full', 'both')]
    [string]$Flavor = 'both',
    [string]$Version = '',
    [string]$Dist = 'dist\cs-net',
    [string]$Out = 'packaging\out'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Version) {
    $init = Get-Content 'app\__init__.py' -Raw
    if ($init -match '__version__\s*=\s*"([^"]+)"') { $Version = $Matches[1] }
    else { throw 'app/__init__.py 里找不到 __version__' }
}
if (-not (Test-Path $Dist)) { throw "没有 $Dist —— 先跑 pyinstaller packaging/cs-net.spec" }
# 压包时会把工作目录切到 stage 的父目录，所以输出目录必须先转绝对路径。
if (-not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $root $Out }
$Out = [System.IO.Path]::GetFullPath($Out)
New-Item -ItemType Directory -Force $Out | Out-Null

# ── 运行期数据：地图解压好，权重拷到 dist（full 才带）────────────────────────
$mapsSrc = 'maps\optimized_obj_files'
$mapsDst = Join-Path $Dist 'maps\optimized_obj_files'
$objs = @(Get-ChildItem $mapsSrc -Filter '*.obj' -ErrorAction SilentlyContinue)
if ($objs.Count -eq 0) { throw "$mapsSrc 里没有 .obj —— 先跑 python scripts/extract_maps.py" }
New-Item -ItemType Directory -Force $mapsDst | Out-Null
foreach ($f in $objs) { Copy-Item $f.FullName (Join-Path $mapsDst $f.Name) -Force }
Write-Host ("maps  : {0} 个 .obj，{1:N0} MB" -f $objs.Count, (($objs | Measure-Object Length -Sum).Sum / 1MB))

# ── 7z / tar：优先 7z（3 万个小文件快好几倍），没有就退回系统自带 tar ────────
$sevenZip = @(
    'C:\Program Files\7-Zip\7z.exe',
    'C:\Program Files (x86)\7-Zip\7z.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1

function Compress-Dir([string]$Path, [string]$Zip) {
    # 7-Zip 把**命令行上写的相对路径**原样存进压缩包。在仓库根直接 `7z a out.zip
    # packaging\out\cs-net-...`，解压出来就会多一层 `packaging\out\`。所以先
    # 进到父目录，再用裸文件夹名打包。
    $parent = Split-Path -Parent $Path
    $leaf = Split-Path -Leaf $Path
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    Push-Location $parent
    try {
        if ($sevenZip) {
            # -mx=5：2 GB 的目录 -mx=9 要多花好几分钟，省下的那点体积不值（CI 计费）。
            & $sevenZip a -tzip -mx=5 -mmt=on $Zip $leaf | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "7z 失败（exit $LASTEXITCODE）" }
        } else {
            Write-Host '  （没有 7-Zip，用 tar 打包，会慢一些）'
            & tar -a -c -f $Zip $leaf
            if ($LASTEXITCODE -ne 0) { throw "tar 失败（exit $LASTEXITCODE）" }
        }
    } finally {
        Pop-Location
    }
}

function Write-BuildInfo([string]$Dir, [string]$FlavorName, [string]$ZipName) {
    $sha = if ($env:GITHUB_SHA) { $env:GITHUB_SHA } else { '' }
    # 版本号只用来写 build-info；`python` 不在 PATH 上（或者这是个没装 Python 的
    # 构建机）不该让整包流程失败 —— 之前就因为清了 PATH 而在这里炸过。
    # 查找顺序：CSNET_PYTHON > 仓库里的打包用 venv > PATH 上的 python。
    $py = $env:CSNET_PYTHON
    if (-not $py) {
        $venvPy = Join-Path $root '.build\packenv\Scripts\python.exe'
        $py = if (Test-Path $venvPy) { $venvPy } else { 'python' }
    }
    $pyver = try { (& $py -c 'import sys; print(sys.version.split()[0])' 2>$null) } catch { '' }
    $torchver = try { (& $py -c 'import torch; print(torch.__version__)' 2>$null) } catch { '' }
    $models = @{}
    foreach ($p in Get-ChildItem (Join-Path $Dir 'models') -Recurse -File -ErrorAction SilentlyContinue) {
        $models[$p.Name] = (Get-FileHash $p.FullName -Algorithm SHA256).Hash.ToLower()
    }
    $info = [ordered]@{
        version      = $Version
        flavor       = $FlavorName
        zip          = $ZipName
        commit       = $sha
        built_at     = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        python       = "$pyver"
        torch        = "$torchver"
        weights      = $models
        maps         = $objs.Count
    }
    $info | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Dir 'build-info.json') -Encoding UTF8
}

# ── 组包 ────────────────────────────────────────────────────────────────────
$flavors = if ($Flavor -eq 'both') { @('lite', 'full') } else { @($Flavor) }
$built = @()

foreach ($f in $flavors) {
    $name = "cs-net-$Version-win64-$f"
    $stage = Join-Path $Out $name
    Write-Host ""
    Write-Host "== $name" -ForegroundColor Cyan

    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    # robocopy：3 万个文件比 Copy-Item 快一个数量级；退出码 0-7 都算成功。
    & robocopy $Dist $stage /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败（exit $LASTEXITCODE）" }

    $modelsDir = Join-Path $stage 'models'
    if ($f -eq 'full') {
        $weights = @(Get-ChildItem 'models' -Recurse -File -Include '*.pt', '*.yaml' -ErrorAction SilentlyContinue)
        if ($weights.Count -eq 0) { throw 'models/ 里没有权重 —— full 版需要它（先跑 scripts/download_models.py）' }
        foreach ($w in $weights) {
            $rel = $w.FullName.Substring((Resolve-Path 'models').Path.Length + 1)
            $dst = Join-Path $modelsDir $rel
            New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
            Copy-Item $w.FullName $dst -Force
        }
        Write-Host ("weights: {0} 个文件，{1:N0} MB" -f $weights.Count, (($weights | Measure-Object Length -Sum).Sum / 1MB))
    } elseif (Test-Path $modelsDir) {
        # lite：权重不随包，但留个空目录（下载器会往里写；玩家也能手动丢 zip 进去）
        Get-ChildItem $modelsDir -Recurse -File | Remove-Item -Force
    }

    $zipName = "$name.zip"
    # 玩家解压后第一眼看到的东西：先说 "_internal 不能删" 和 "库是空的，去设置里加目录"。
    Copy-Item (Join-Path $PSScriptRoot 'USAGE.txt') (Join-Path $stage 'README.txt') -Force
    Write-BuildInfo $stage $f $zipName
    Write-Host "zip   : $zipName（压缩中…）"
    Compress-Dir $stage (Join-Path $Out $zipName)
    $size = (Get-Item (Join-Path $Out $zipName)).Length / 1MB
    Write-Host ("        {0:N0} MB" -f $size) -ForegroundColor Green
    $built += $zipName

    Remove-Item $stage -Recurse -Force
}

# ── 校验和 ──────────────────────────────────────────────────────────────────
$sums = Join-Path $Out 'sha256sums.txt'
$lines = foreach ($z in $built) {
    $h = (Get-FileHash (Join-Path $Out $z) -Algorithm SHA256).Hash.ToLower()
    "$h  $z"
}
$lines | Set-Content $sums -Encoding ASCII
Write-Host ""
Write-Host "产物：" -NoNewline
Write-Host ($built -join ', ') -ForegroundColor Green
Write-Host "校验：$sums"

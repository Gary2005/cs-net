param(
    [string]$Src,
    [string]$Dst,
    [int]$MaxDim = 1024,
    [string]$Format = 'png'   # png | jpeg
)
# Downscale extracted CS2 textures for web delivery.
#
# Source2Viewer writes textures at full source resolution -- 2048^2 for agent
# materials and up to 4096^2 for weapon materials -- as PNG. That is 3-18 MB per
# texture, i.e. 49 MB for one agent and 143 MB for six weapons. VRF exposes no
# resize flag, so this post-process is the only lever.
#
# Colour maps tolerate JPEG well; normal/ORM/mask maps do not (JPEG artefacts in
# a normal map read as shading noise), so keep those as PNG.
Add-Type -AssemblyName System.Drawing

$srcPath = (Resolve-Path $Src).Path
New-Item -ItemType Directory -Force -Path $Dst | Out-Null
$beforeTotal = 0; $afterTotal = 0; $n = 0

foreach ($f in Get-ChildItem -Path $srcPath -Recurse -File -Filter *.png) {
    $img = [System.Drawing.Image]::FromFile($f.FullName)
    $scale = [math]::Min(1.0, $MaxDim / [math]::Max($img.Width, $img.Height))
    $w = [int][math]::Max(1, [math]::Round($img.Width * $scale))
    $h = [int][math]::Max(1, [math]::Round($img.Height * $scale))

    $bmp = New-Object System.Drawing.Bitmap $w, $h, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $bmp.SetResolution(96, 96)
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $gfx.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $gfx.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $gfx.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $gfx.DrawImage($img, 0, 0, $w, $h)
    $gfx.Dispose()

    $rel = $f.FullName.Substring($srcPath.Length).TrimStart('\')
    $outDir = Join-Path $Dst (Split-Path $rel -Parent)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $base = [IO.Path]::GetFileNameWithoutExtension($f.Name)

    if ($Format -eq 'jpeg') {
        $outFile = Join-Path $outDir "$base.jpg"
        $enc = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' }
        $ps = New-Object System.Drawing.Imaging.EncoderParameters 1
        $ps.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter ([System.Drawing.Imaging.Encoder]::Quality), 82L
        $bmp.Save($outFile, $enc, $ps)
    } else {
        $outFile = Join-Path $outDir "$base.png"
        $bmp.Save($outFile, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    $bmp.Dispose(); $img.Dispose()

    $beforeTotal += $f.Length
    $afterTotal += (Get-Item $outFile).Length
    $n++
}

"resized {0} textures to max {1}px as {2}" -f $n, $MaxDim, $Format
"  before: {0:N1} MB" -f ($beforeTotal / 1MB)
"  after : {0:N1} MB" -f ($afterTotal / 1MB)
if ($beforeTotal -gt 0) { "  ratio : {0:P1}" -f ($afterTotal / $beforeTotal) }

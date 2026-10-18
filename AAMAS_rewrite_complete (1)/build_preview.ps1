$ErrorActionPreference = 'Stop'
$paperPath = $PSScriptRoot
$env:TECTONIC_CACHE_DIR = Join-Path (Split-Path -Parent $paperPath) '.tools-paper\tectonic-cache'
$env:TEMP = Join-Path (Split-Path -Parent $paperPath) '.tmp-paper'
$env:TMP = $env:TEMP
$toolPath = Join-Path (Split-Path -Parent $paperPath) '.tools-paper\tectonic\tectonic.exe'
if (-not (Test-Path -LiteralPath $toolPath)) { throw 'Portable Tectonic is missing from .tools-paper/tectonic.' }
Push-Location -LiteralPath $paperPath
try {
    & $toolPath --keep-logs preview.tex
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX build failed; inspect preview.log.' }
    Copy-Item -LiteralPath 'preview.pdf' -Destination 'paper_progress.pdf' -Force
} finally {
    Pop-Location
}

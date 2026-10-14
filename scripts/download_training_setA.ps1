# PowerShell script to download PhysioNet Challenge 2019 training_setA files
# Destination directory (relative to repository root)
$repoRoot = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $repoRoot "data\raw\training_setA"

# Ensure destination exists
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$baseUrl = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/"

# Download index page to parse file links
$indexPath = Join-Path $dest "index.html"
Invoke-WebRequest -Uri $baseUrl -OutFile $indexPath -UseBasicParsing

# Read index content and extract .psv links
$indexContent = Get-Content $indexPath -Raw
$matches = [regex]::Matches($indexContent, 'href="([^"]+\.psv)"')

foreach ($match in $matches) {
    $fileName = $match.Groups[1].Value
    $fileUrl = $baseUrl + $fileName
    $outPath = Join-Path $dest $fileName
    Write-Host "Downloading $fileName..."
    Invoke-WebRequest -Uri $fileUrl -OutFile $outPath -UseBasicParsing
}

Write-Host "All files downloaded to $dest"

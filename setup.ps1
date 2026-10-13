# pocket-zero : installation autonome pour Windows / self-contained setup for Windows.
# Tout reste dans ce dossier / everything stays in this folder: Python environment (.venv),
# engine binaries (bin\), engine Python module, published network (weights\).
# Rien n'est installé dans le système / nothing is installed system-wide.
#
#   setup.bat            (ou / or: powershell -ExecutionPolicy Bypass -File setup.ps1)

param([switch]$SkipTorch)  # -SkipTorch : test automatique (CI), sans GPU ni réseau / CI test, no GPU, no network download

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # downloads are much faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12  # GitHub API (Windows PowerShell 5.1)
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# 1. Python
Step 'Recherche de Python 3.10+ / looking for Python 3.10+'
$py = $null
foreach ($v in '3.12', '3.13', '3.11', '3.10') {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('py', "-$v"); break } } catch {}
}
if (-not $py) {
    try { & python -c "import sys; assert sys.version_info >= (3, 10)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('python') } } catch {}
}
if (-not $py) { throw "Python 3.10+ introuvable / not found: python.org (coche / tick 'py launcher')." }
Write-Host "Python : $($py -join ' ')"

# 2. Environnement virtuel local / local virtual environment
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Step 'Création de .venv / creating .venv'
    if ($py.Length -gt 1) { & $py[0] $py[1] -m venv .venv } else { & $py[0] -m venv .venv }
}
$vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $vpy -m pip install --upgrade pip --quiet

# 3. PyTorch avec CUDA 12.8 (RTX 20xx à 50xx / RTX 20xx to 50xx, Blackwell included)
Step 'PyTorch (CUDA 12.8) + numpy'
if (-not $SkipTorch) {
    & $vpy -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "Échec de l'installation de PyTorch / PyTorch install failed" }
}
& $vpy -m pip install --quiet numpy

# 4. Binaires du moteur compilés par GitHub Actions (branche windows-bin)
#    Engine binaries built by GitHub Actions (windows-bin branch): no Rust or Visual Studio needed.
Step 'Binaires du moteur / engine binaries'
git fetch origin windows-bin --depth 1
if ($LASTEXITCODE -ne 0) { throw "Branche windows-bin absente / missing: attends le workflow 'Windows build' / wait for the 'Windows build' workflow." }
$zip = Join-Path $env:TEMP 'pocket-zero-bin.zip'
git archive --format=zip -o $zip FETCH_HEAD
$tmp = Join-Path $env:TEMP 'pocket-zero-bin'
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive $zip $tmp
New-Item -ItemType Directory -Force -Path bin | Out-Null
Copy-Item "$tmp\bin\*" bin\ -Force
Write-Host "Version : $(Get-Content $tmp\VERSION)"
& $vpy -m pip install --quiet --force-reinstall (Get-ChildItem "$tmp\wheels\*.whl").FullName

# 5. Réseau pré-entraîné publié (GitHub Releases, fichier pocket-zero.bin), s'il y en a un.
#    Published pretrained network (GitHub Releases, asset pocket-zero.bin), if there is one:
#    this repository's releases first, then the original repository's.
if (-not $SkipTorch) {
    Step 'Réseau pré-entraîné / pretrained network'
    $repos = @()
    try { $origin = git remote get-url origin } catch { $origin = '' }
    if ($origin -match 'github\.com[:/](.+?)(\.git)?$') { $repos += $Matches[1] }
    if ($repos -notcontains 'Undi95/pocket-zero') { $repos += 'Undi95/pocket-zero' }
    $got = $false
    foreach ($r in $repos) {
        try { $rel = Invoke-RestMethod "https://api.github.com/repos/$r/releases/latest" -Headers @{ 'User-Agent' = 'pocket-zero-setup' } }
        catch { continue }
        $asset = $rel.assets | Where-Object { $_.name -eq 'pocket-zero.bin' } | Select-Object -First 1
        if (-not $asset) { continue }
        New-Item -ItemType Directory -Force -Path weights | Out-Null
        $want = "$r $($rel.tag_name)"
        $have = if (Test-Path weights\VERSION) { (Get-Content weights\VERSION) } else { '' }
        if ($have -ne $want -or -not (Test-Path weights\pocket-zero.bin)) {
            Write-Host "Téléchargement / downloading $want ..."
            Invoke-WebRequest $asset.browser_download_url -OutFile weights\pocket-zero.bin -UseBasicParsing
            Set-Content weights\VERSION $want
        }
        Write-Host "Réseau / network: $want -> weights\pocket-zero.bin"
        $got = $true
        break
    }
    if (-not $got) { Write-Host "Aucun réseau publié pour l'instant / no published network yet: run.bat train" }
}

# 6. Vérifications / checks
Step 'Vérification / checks'
& .\bin\pocket-zero.exe info | Out-Null
& $vpy -c "import deckgym.deckgym as d; print('moteur Python / Python engine: OK')"
if ($LASTEXITCODE -ne 0) { throw "Le module Python du moteur ne se charge pas / the engine Python module does not load" }
if (-not $SkipTorch) {
    & $vpy -c "import torch; n = torch.cuda.device_count(); print(f'GPU : {n} -', ', '.join(torch.cuda.get_device_name(i) for i in range(n)) or 'aucun / none (CPU)')"
}

Write-Host "`nInstallation terminée / setup complete.  run.bat help" -ForegroundColor Green

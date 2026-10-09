# pocket-zero : installation autonome pour Windows.
# Tout reste dans ce dossier : environnement Python (.venv), binaires (bin\), module du moteur.
# Rien n'est installé dans le système, tes autres installations Python/CUDA ne sont pas touchées.
#
#   setup.bat            (ou : powershell -ExecutionPolicy Bypass -File setup.ps1)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# 1. Python
Step 'Recherche de Python 3.10+'
$py = $null
foreach ($v in '3.12', '3.13', '3.11', '3.10') {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('py', "-$v"); break } } catch {}
}
if (-not $py) {
    try { & python -c "import sys; assert sys.version_info >= (3, 10)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('python') } } catch {}
}
if (-not $py) { throw "Python 3.10 ou plus est introuvable. Installe-le depuis python.org (coche 'py launcher')." }
Write-Host "Python : $($py -join ' ')"

# 2. Environnement virtuel local
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Step 'Création de l''environnement virtuel .venv'
    if ($py.Length -gt 1) { & $py[0] $py[1] -m venv .venv } else { & $py[0] -m venv .venv }
}
$vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $vpy -m pip install --upgrade pip --quiet

# 3. PyTorch avec CUDA 12.8 (nécessaire pour les RTX 50xx / Blackwell)
Step 'Installation de PyTorch (CUDA 12.8) et numpy dans .venv'
& $vpy -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
& $vpy -m pip install --quiet numpy

# 4. Binaires du moteur compilés par GitHub Actions (branche windows-bin)
Step 'Téléchargement des binaires Windows du moteur'
git fetch origin windows-bin --depth 1
if ($LASTEXITCODE -ne 0) { throw "La branche windows-bin n'existe pas encore : attends la fin du workflow 'Windows build' sur GitHub." }
$zip = Join-Path $env:TEMP 'pocket-zero-bin.zip'
git archive --format=zip -o $zip FETCH_HEAD
$tmp = Join-Path $env:TEMP 'pocket-zero-bin'
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive $zip $tmp
New-Item -ItemType Directory -Force -Path bin | Out-Null
Copy-Item "$tmp\bin\*" bin\ -Force
Write-Host "Version des binaires : $(Get-Content $tmp\VERSION)"
& $vpy -m pip install --quiet --force-reinstall (Get-ChildItem "$tmp\wheels\*.whl").FullName

# 5. Vérifications
Step 'Vérification'
& .\bin\pocket-zero.exe info | Out-Null
& $vpy -c "import torch, deckgym.deckgym as d; n = torch.cuda.device_count(); print(f'GPU : {n} -', ', '.join(torch.cuda.get_device_name(i) for i in range(n)) or 'aucun (le CPU sera utilisé)'); print('moteur Python : OK')"

Write-Host "`nInstallation terminée. Lance l'entraînement avec :  run.bat train" -ForegroundColor Green

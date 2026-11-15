# setup.ps1 - one-time environment setup for the factor-strategy-midcap project (DMF-US-02)
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File "$HOME\Documents\factor-strategy-midcap\setup.ps1"
# The -ExecutionPolicy Bypass flag applies only to this one run; it does not change your system settings.
# Safe to run more than once: anything already installed is skipped.

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Project

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

function Ensure-Package($id, $name) {
    $found = winget list --id $id --exact --source winget 2>$null | Select-String $id
    if ($found) { Ok "$name already installed"; return }
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        Write-Host "  Installing $name (attempt $attempt of 4; winget may ask you to accept its terms)..."
        winget install --id $id --exact --source winget
        if ($LASTEXITCODE -eq 0) { Ok "$name installed"; return }
        $busy = Get-Process msiexec -ErrorAction SilentlyContinue
        if ($attempt -lt 4) {
            if ($busy) { Warn "Another installation is running (Windows Update or an app update). Waiting 2 minutes..." }
            else { Warn "Install failed (exit code $LASTEXITCODE). Retrying in 2 minutes..." }
            Start-Sleep -Seconds 120
        }
    }
    throw "winget could not install $name after 4 attempts. Restart the computer, then run this script again."
}

# 1. Tools -----------------------------------------------------------------
Step "1/7  Installing Python 3.12, Git and VS Code"
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget not found. Install 'App Installer' from the Microsoft Store, then run this script again."
}
Ensure-Package "Python.Python.3.12"        "Python 3.12"
Ensure-Package "Git.Git"                   "Git"
Ensure-Package "Microsoft.VisualStudioCode" "VS Code"
Refresh-Path

# 2. Check versions --------------------------------------------------------
Step "2/7  Checking versions"
$py = "py"
& $py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found via the 'py' launcher. Close PowerShell, open a new window and run the script again." }
git --version
$code = Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\bin\code.cmd"
if (-not (Test-Path $code)) { $code = "code" }
& $code --version | Select-Object -First 1

# 3. Git identity ----------------------------------------------------------
Step "3/7  Git identity"
if (-not (git config --global user.name)) { git config --global user.name "Arnau Mir" }
if (-not (git config --global user.email)) {
    $email = Read-Host "  Email for Git commits (use your GitHub email)"
    git config --global user.email $email
}
Ok ("Git user: " + (git config --global user.name) + " <" + (git config --global user.email) + ">")

# 4. Virtual environment ---------------------------------------------------
Step "4/7  Creating the Python environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py -3.12 -m venv .venv }
$venvPy = Join-Path $Project ".venv\Scripts\python.exe"
& $venvPy --version
Ok ".venv ready (no activation needed for this script)"

# 5. Packages --------------------------------------------------------------
Step "5/7  Installing Python packages"
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install pandas numpy statsmodels pyarrow matplotlib pytest jupyter ipykernel requests
if ($LASTEXITCODE -ne 0) { throw "Core package installation failed." }
& $venvPy -m pip install vectorbt
if ($LASTEXITCODE -ne 0) { Warn "vectorbt did not install. Not needed yet; we'll fix it before Stage 05." }
& $venvPy -m pip freeze | Out-File -Encoding utf8 requirements.txt
Ok "Packages installed; exact versions saved to requirements.txt"

# 6. VS Code extensions ----------------------------------------------------
Step "6/7  VS Code extensions"
& $code --install-extension ms-python.python --force | Out-Null
& $code --install-extension ms-toolsai.jupyter --force | Out-Null
Ok "Python and Jupyter extensions installed"
New-Item -ItemType Directory -Force -Path ".vscode" | Out-Null
@'
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "files.exclude": { "**/__pycache__": true, ".pytest_cache": true }
}
'@ | Out-File -Encoding utf8 ".vscode\settings.json"
Ok "VS Code set to use .venv in this project"

# 7. Git repository --------------------------------------------------------
Step "7/7  Git repository"
if (-not (Test-Path ".git")) { git init -b main | Out-Null }
git add -A
$pending = git status --porcelain
if ($pending) { git commit -m "Initial project structure and environment" | Out-Null; Ok "First commit created" }
else { Ok "Nothing new to commit" }

$hasRemote = git remote 2>$null | Select-String "origin"
if (-not $hasRemote) {
    Write-Host "  Create a PRIVATE, empty repository called factor-strategy-midcap on github.com first."
    $url = Read-Host "  Paste its URL (or press Enter to skip for now)"
    if ($url) {
        git remote add origin $url
        git push -u origin main
        if ($LASTEXITCODE -eq 0) { Ok "Pushed to GitHub" } else { Warn "Push failed. Check the URL and sign-in, then run: git push -u origin main" }
    } else { Warn "Skipped GitHub. Run this script again when the repository exists." }
} else { Ok "GitHub remote already set" }

Write-Host "`nAll done. Open the project with:" -ForegroundColor Cyan
Write-Host "  & `"$code`" `"$Project`""
Write-Host "In VS Code, choose the .venv interpreter if asked."

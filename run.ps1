<#
.SYNOPSIS
    Windows PowerShell wrapper around `python -m sd_compress`.

.DESCRIPTION
    Mirrors run.sh: creates a virtualenv, installs requirements, then dispatches
    to the sd_compress CLI. Run from PowerShell (not cmd.exe).

.EXAMPLE
    .\run.ps1                                # Full pipeline + Gradio server
    .\run.ps1 --no-serve                     # Full pipeline without server
    .\run.ps1 distill-progressive            # Single stage
    .\run.ps1 evaluate --stage distilled --model-dir .\output\distilled

.NOTES
    Configuration is driven by environment variables; see sd_compress/config.py.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { "venv" }
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

if (-not (Test-Path -Path $VenvDir)) {
    Write-Host "[run.ps1] Creating virtual environment in $VenvDir"
    & $PythonBin -m venv $VenvDir
}

$Activate = Join-Path -Path $VenvDir -ChildPath "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    throw "[run.ps1] Could not find $Activate. Recreate the virtualenv?"
}
. $Activate

python -m pip install --upgrade pip | Out-Null

$DepsMarker = Join-Path -Path $VenvDir -ChildPath ".deps_installed"
if (-not (Test-Path $DepsMarker)) {
    Write-Host "[run.ps1] Installing Python requirements (this may take a while)"
    pip install -r requirements.txt
    try {
        # Pin CLIP to a commit so upstream `main` cannot silently change.
        $ClipGitRef = if ($env:CLIP_GIT_REF) { $env:CLIP_GIT_REF } else { "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6" }
        pip install --quiet "git+https://github.com/openai/CLIP.git@$ClipGitRef"
    } catch {
        Write-Warning "[run.ps1] CLIP install failed; CLIP score will be skipped"
    }
    try {
        pip install --quiet xformers
    } catch {
        Write-Host "[run.ps1] NOTE: xformers unavailable on this platform/CUDA combination"
    }
    New-Item -ItemType File -Path $DepsMarker | Out-Null
}

if ($args.Count -eq 0) {
    python -m sd_compress run --serve
    exit $LASTEXITCODE
}

if ($args[0] -eq "--no-serve") {
    python -m sd_compress run
    exit $LASTEXITCODE
}

python -m sd_compress @args
exit $LASTEXITCODE

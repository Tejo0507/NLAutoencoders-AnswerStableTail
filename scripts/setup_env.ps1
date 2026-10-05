# Environment setup for Windows + NVIDIA.
#
# PyTorch's CUDA build is not on PyPI, so it is installed from the PyTorch
# index first and the rest of the project follows from pyproject.toml.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -CudaTag cu126

param(
    # cu128 is what this study ran on. The driver reported CUDA 13.1 and is
    # backward compatible; cu126 and cu130 wheels also exist for cp314.
    [string]$CudaTag = "cu128",
    [string]$TorchVersion = "2.9.1",
    [string]$VenvPath = ".venv",
    # The download cache needs ~22 GB free and is usually best kept off the
    # repo volume: the three bf16 checkpoints total 41 GB. Empty means "beside
    # the repository"; see PROJECT_PLAN.md s2.
    [string]$ModelCache = "",
    [string]$QuantDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== NLA Answer-Stable Tail: environment setup ===" -ForegroundColor Cyan

if (-not (Test-Path $VenvPath)) {
    Write-Host "creating virtual environment at $VenvPath"
    python -m venv $VenvPath
}
$py = Join-Path $root "$VenvPath/Scripts/python.exe"
if (-not (Test-Path $py)) { throw "no interpreter at $py" }

& $py -c "import sys; print('python', sys.version)"
& $py -m pip install --upgrade pip setuptools wheel

# The torch wheel is ~2.9 GB and the PyTorch index dropped the connection
# often enough that pip's own retries were not sufficient; fetch it to disk
# first with curl, which resumes.
$wheelDir = Join-Path $env:TEMP "nlaast_wheels"
New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null
$wheelName = "torch-$TorchVersion+$CudaTag-cp314-cp314-win_amd64.whl"
$wheelPath = Join-Path $wheelDir $wheelName
$wheelUrl = "https://download.pytorch.org/whl/$CudaTag/torch-$TorchVersion%2B$CudaTag-cp314-cp314-win_amd64.whl"

& $py -c "import torch, sys; sys.exit(0)" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (-not (Test-Path $wheelPath)) {
        Write-Host "downloading $wheelName (resumable)" -ForegroundColor Yellow
        $attempt = 0
        while ($attempt -lt 8) {
            $attempt++
            curl.exe -sL -C - -o $wheelPath $wheelUrl
            if ($LASTEXITCODE -eq 0) { break }
            Write-Host "  retry $attempt"
        }
    }
    & $py -m pip install $wheelPath
} else {
    Write-Host "torch already installed, skipping"
}

Write-Host "installing project dependencies" -ForegroundColor Cyan
& $py -m pip install --retries 10 --timeout 120 -e ".[dev]"

Write-Host "verifying" -ForegroundColor Cyan
& $py -c @"
import torch, transformers, bitsandbytes
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu', p.name, round(p.total_memory/1e9, 2), 'GB', f'sm_{p.major}{p.minor}')
print('transformers', transformers.__version__, '| bitsandbytes', bitsandbytes.__version__)
from math_verify import parse, verify
print('math_verify 1/2 == 0.5 ->', verify(parse(r'\$\frac{1}{2}\$', parsing_timeout=None),
                                          parse(r'\$0.5\$', parsing_timeout=None),
                                          timeout_seconds=None))
"@

$siblings = Join-Path (Split-Path -Parent $root) "nla_models"
if (-not $QuantDir) { $QuantDir = Join-Path $siblings "nf4" }
if (-not $ModelCache) { $ModelCache = Join-Path $siblings "hf_cache" }
Write-Host ""
Write-Host "Set these before running the pipeline (point them at whichever" -ForegroundColor Green
Write-Host "volumes have the room - the defaults below sit beside the repo):" -ForegroundColor Green
Write-Host "  `$env:HF_HOME = '$ModelCache'              # bf16 download staging (~22 GB free)"
Write-Host "  `$env:NLAAST_QUANT_DIR = '$QuantDir'       # 4-bit checkpoints (~15 GB)"
Write-Host ""
Write-Host "Then:" -ForegroundColor Green
Write-Host "  $VenvPath/Scripts/python.exe -m pytest"
Write-Host "  $VenvPath/Scripts/python.exe scripts/run_pipeline.py --config smoke"
Write-Host "  $VenvPath/Scripts/python.exe scripts/run_pipeline.py --config pilot"
Write-Host "  $VenvPath/Scripts/python.exe scripts/run_pipeline.py --config main"

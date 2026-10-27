# ============================================================================
# photo2video — Setup GPU Windows
# ============================================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=== photo2video :: Setup GPU ===" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# Encontrar Python 3.12
# -----------------------------------------------------------------------------

function Find-Python312 {

    try {
        py -3.12 -c "import sys" *> $null
        if ($LASTEXITCODE -eq 0) {
            return "py -3.12"
        }
    } catch {}

    try {
        $v = python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"

        if ($v -eq "3.12") {
            return "python"
        }
    } catch {}

    return $null
}

$PythonCmd = Find-Python312

if ($null -eq $PythonCmd) {

    Write-Host ""
    Write-Host "Python 3.12 não encontrado." -ForegroundColor Red
    Write-Host ""
    Write-Host "Instale o Python 3.12 antes de continuar."
    exit 1
}

Write-Host "Python encontrado: $PythonCmd" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Criar ambiente virtual
# -----------------------------------------------------------------------------

$venv = Join-Path $PSScriptRoot "venv_wan"

if (!(Test-Path $venv))
{
    Write-Host ""
    Write-Host "Criando ambiente virtual..."
    Write-Host ""

    Invoke-Expression "$PythonCmd -m venv `"$venv`""
}

$vpy = Join-Path $venv "Scripts\python.exe"

if (!(Test-Path $vpy))
{
    throw "Não foi possível localizar $vpy"
}

# -----------------------------------------------------------------------------
# Atualizar pip
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Atualizando pip..."
Write-Host ""

& $vpy -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Instalar PyTorch CUDA 12.8
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Instalando PyTorch CUDA 12.8..."
Write-Host ""

& $vpy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# -----------------------------------------------------------------------------
# Instalar requirements
# -----------------------------------------------------------------------------

if (Test-Path ".\requirements_cuda.txt")
{
    Write-Host ""
    Write-Host "Instalando requirements_cuda.txt..."
    Write-Host ""

    & $vpy -m pip install -r ".\requirements_cuda.txt"
}

# -----------------------------------------------------------------------------
# Teste CUDA
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Testando CUDA..."
Write-Host ""

$code = @"
import torch

print("Torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Capability:", torch.cuda.get_device_capability(0))
    print("Arquiteturas:", torch.cuda.get_arch_list())
else:
    print("CUDA não disponível.")
"@

& $vpy -c $code

# -----------------------------------------------------------------------------
# Doctor
# -----------------------------------------------------------------------------

if (Test-Path ".\doctor.py")
{
    Write-Host ""
    Write-Host "Executando doctor..."
    Write-Host ""

    & $vpy ".\doctor.py"
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Setup concluído." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
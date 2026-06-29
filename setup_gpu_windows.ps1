# ============================================================================
#  photo2video — setup para Windows + GPU NVIDIA (RTX 5070 Ti / Blackwell)
# ----------------------------------------------------------------------------
#  Cria o ambiente (venv), instala o PyTorch CUDA 12.8 (obrigatório p/ RTX 50),
#  as demais dependências e (opcional) baixa o checkpoint do modelo ti2v-5B.
#
#  Como rodar (PowerShell, na pasta do projeto):
#      powershell -ExecutionPolicy Bypass -File .\setup_gpu_windows.ps1
#
#  Pré-requisitos:
#    - Driver NVIDIA recente (Game Ready/Studio) que suporte CUDA 12.8.
#    - Python 3.12 de 64 bits instalado (PyTorch ainda não suporta 3.13/3.14).
#      Baixe em https://www.python.org/downloads/release/python-3120/ e marque
#      "Add python.exe to PATH" durante a instalação.
# ============================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== photo2video :: setup GPU (Windows) ===" -ForegroundColor Cyan

# --- 1) Encontrar Python 3.12 -------------------------------------------------
function Find-Python312 {
    # Tenta o launcher 'py -3.12' primeiro (mais confiável no Windows).
    try {
        $v = & py -3.12 -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v.Trim() -eq "3.12") { return @("py", "-3.12") }
    } catch {}
    # Fallback: 'python' no PATH, se for 3.12.
    try {
        $v = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v.Trim() -eq "3.12") { return @("python") }
    } catch {}
    return $null
}

$py = Find-Python312
if ($null -eq $py) {
    Write-Host "ERRO: Python 3.12 (64-bit) não encontrado." -ForegroundColor Red
    Write-Host "Instale a partir de https://www.python.org/downloads/release/python-3120/" -ForegroundColor Yellow
    Write-Host "e marque 'Add python.exe to PATH'. Depois rode este script de novo." -ForegroundColor Yellow
    exit 1
}
Write-Host "Python 3.12 encontrado: $($py -join ' ')" -ForegroundColor Green

# --- 2) Criar/ativar o venv ---------------------------------------------------
$venv = Join-Path $PSScriptRoot "venv_wan"
if (-not (Test-Path $venv)) {
    Write-Host "Criando venv em $venv ..." -ForegroundColor Cyan
    & $py[0] $py[1..($py.Length-1)] -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "ERRO: venv não criou python.exe." -ForegroundColor Red; exit 1 }

Write-Host "Atualizando pip ..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip setuptools wheel

# --- 3) PyTorch CUDA 12.8 (obrigatório p/ RTX 50 / Blackwell) -----------------
Write-Host "Instalando PyTorch (CUDA 12.8 / cu128) — obrigatório para RTX 50 ..." -ForegroundColor Cyan
& $vpy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- 4) Demais dependências ---------------------------------------------------
Write-Host "Instalando dependências (requirements_cuda.txt) ..." -ForegroundColor Cyan
& $vpy -m pip install -r (Join-Path $PSScriptRoot "requirements_cuda.txt")

# --- 5) Verificar a GPU -------------------------------------------------------
Write-Host "Verificando suporte à GPU ..." -ForegroundColor Cyan
$check = @"
import torch
ok = torch.cuda.is_available()
print('CUDA disponivel:', ok)
if ok:
    print('GPU:', torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print('Compute capability: sm_%d%d' % cap)
    print('Torch:', torch.__version__, '| build CUDA:', torch.version.cuda)
    # sm_120 = Blackwell (RTX 50). Avisa se o torch nao tiver kernels p/ ela.
    archs = torch.cuda.get_arch_list()
    print('Arquiteturas no build:', archs)
    tag = 'sm_%d%d' % cap
    if tag not in archs and ('sm_120' not in archs):
        print('AVISO: o build do torch pode nao ter kernels para', tag,
              '- reinstale com --index-url https://download.pytorch.org/whl/cu128')
else:
    print('AVISO: CUDA indisponivel. Confira o driver NVIDIA e reinstale o torch cu128.')
"@
& $vpy -c $check

# --- 6) Checkpoint do modelo (opcional) --------------------------------------
$ckpt = Join-Path $PSScriptRoot "Wan2.2-TI2V-5B"
if (Test-Path (Join-Path $ckpt "Wan2.2_VAE.pth")) {
    Write-Host "Checkpoint ti2v-5B já presente em $ckpt." -ForegroundColor Green
} else {
    $r = Read-Host "Baixar agora o checkpoint ti2v-5B (~16GB)? [s/N]"
    if ($r -eq "s" -or $r -eq "S") {
        & $vpy (Join-Path $PSScriptRoot "photo2video.py") --model ti2v-5B --download-only
    } else {
        Write-Host "Pulei o download. Depois rode:" -ForegroundColor Yellow
        Write-Host "  venv_wan\Scripts\python photo2video.py --model ti2v-5B --download-only" -ForegroundColor Yellow
    }
}

# --- 7) Modelo de edição de foto (photo2photo, opcional) ---------------------
$r2 = Read-Host "Baixar agora o modelo de EDIÇÃO de foto Qwen-Image-Edit (~20GB)? [s/N]"
if ($r2 -eq "s" -or $r2 -eq "S") {
    & $vpy (Join-Path $PSScriptRoot "photo2photo.py") --model qwen-edit --download-only
} else {
    Write-Host "Pulei. O modelo de edição baixa sozinho no 1º uso da aba 'Editar foto'," -ForegroundColor Yellow
    Write-Host "ou rode: venv_wan\Scripts\python photo2photo.py --model qwen-edit --download-only" -ForegroundColor Yellow
}

# --- 8) Recursos de qualidade/fidelidade (opcional) -------------------------
$r3 = Read-Host "Instalar recursos de QUALIDADE (upscale + restaurar rosto + checar identidade)? [s/N]"
if ($r3 -eq "s" -or $r3 -eq "S") {
    Write-Host "Instalando requirements_enhance.txt ..." -ForegroundColor Cyan
    & $vpy -m pip install -r (Join-Path $PSScriptRoot "requirements_enhance.txt")
} else {
    Write-Host "Pulei. Depois: venv_wan\Scripts\python -m pip install -r requirements_enhance.txt" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup concluído ===" -ForegroundColor Green
Write-Host "Abrir a interface web:  .\run_ui_windows.bat" -ForegroundColor Green
Write-Host "Ou pela CLI:            venv_wan\Scripts\python app.py" -ForegroundColor Green
Write-Host "Edição por linha de comando: venv_wan\Scripts\python photo2photo.py --help" -ForegroundColor Green

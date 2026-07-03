@echo off
REM ==========================================================================
REM  photo2video  ::  INICIAR (clique duas vezes)
REM --------------------------------------------------------------------------
REM  Faz tudo sozinho:
REM    1) Se o ambiente (venv_wan) nao existir -> roda a instalacao completa
REM       (PyTorch CUDA 12.8 + dependencias) via setup_gpu_windows.ps1.
REM    2) Confere as dependencias-chave e instala o que estiver faltando.
REM    3) Abre http://127.0.0.1:7860 no navegador e sobe a interface web.
REM
REM  Pre-requisito unico: Python 3.12 (64-bit) instalado com "Add to PATH".
REM  (Se faltar, o passo 1 avisa e explica onde baixar.)
REM ==========================================================================
chcp 65001 >nul
cd /d "%~dp0"
title photo2video - iniciar

echo ==========================================================
echo   photo2video  ::  iniciando
echo ==========================================================
echo.

REM --- 1) Ambiente existe? Se nao, instala tudo ---------------------------
if not exist "venv_wan\Scripts\python.exe" (
    echo [1/3] Ambiente nao encontrado. Rodando a instalacao completa...
    echo       ^(PyTorch CUDA 12.8 + dependencias - pode demorar^)
    echo.
    powershell -ExecutionPolicy Bypass -File "%~dp0setup_gpu_windows.ps1"
    if errorlevel 1 (
        echo.
        echo [ERRO] A instalacao falhou. Veja as mensagens acima.
        pause
        exit /b 1
    )
    echo.
) else (
    echo [1/3] Ambiente encontrado ^(venv_wan^).
)

set "VPY=venv_wan\Scripts\python.exe"

REM --- 2) Conferir dependencias-chave; instalar as que faltarem ----------
echo [2/3] Verificando dependencias...
"%VPY%" -c "import torch, gradio, diffusers, transformers, cv2, PIL, imageio, accelerate" 2>nul
if errorlevel 1 (
    echo       Faltam dependencias. Instalando ^(pode demorar^)...
    "%VPY%" -m pip install --upgrade pip >nul
    REM torch precisa do indice cu128 - obrigatorio p/ RTX 50 / Blackwell.
    "%VPY%" -c "import torch" 2>nul || "%VPY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    "%VPY%" -m pip install -r requirements_cuda.txt
    if errorlevel 1 (
        echo.
        echo [ERRO] Falha ao instalar as dependencias. Veja acima.
        pause
        exit /b 1
    )
) else (
    echo       Tudo instalado.
)
echo.

REM --- 3) Abrir o navegador e subir a interface -------------------------
echo [3/3] Subindo a interface em http://127.0.0.1:7860
echo       O navegador abre em alguns segundos. NAO feche esta janela
echo       ^(fecha-la encerra o servidor^).
echo.

REM Abre o navegador depois de ~10s, quando o servidor ja estiver de pe.
start "" cmd /c "timeout /t 10 /nobreak >nul & start "" http://127.0.0.1:7860"

"%VPY%" app.py

echo.
echo Servidor encerrado.
pause

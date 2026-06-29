@echo off
REM ==========================================================================
REM  photo2video - abre a interface web (Gradio) no Windows.
REM  Pre-requisito: ter rodado setup_gpu_windows.ps1 (cria o venv_wan).
REM ==========================================================================
cd /d "%~dp0"

if not exist "venv_wan\Scripts\python.exe" (
    echo [ERRO] venv_wan nao encontrado. Rode primeiro:
    echo     powershell -ExecutionPolicy Bypass -File .\setup_gpu_windows.ps1
    pause
    exit /b 1
)

echo Iniciando a interface em http://127.0.0.1:7860 ...
echo (deixe esta janela aberta; feche-a para encerrar)
start "" http://127.0.0.1:7860
"venv_wan\Scripts\python.exe" app.py
pause

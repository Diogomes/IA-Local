@echo off
REM Diagnostico do ambiente (PyTorch/CUDA/modelos/libs). Use --smoke p/ gerar um
REM video minimo de teste. Pre-requisito: ter rodado setup_gpu_windows.ps1.
cd /d "%~dp0"
if not exist "venv_wan\Scripts\python.exe" (
    echo [ERRO] venv_wan nao encontrado. Rode primeiro setup_gpu_windows.ps1
    pause
    exit /b 1
)
"venv_wan\Scripts\python.exe" doctor.py %*
pause

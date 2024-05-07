@echo off
setlocal
title GPT-SoVITS-Fork

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

.env\python.exe .\tools\download_models.py
.env\python.exe webui.py --open
echo.
pause
@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
::  Video Pipeline — Windows 一键配置脚本
:: ============================================================

title Video Pipeline Setup

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"

echo ============================================
echo   Video Pipeline — One-Click Setup
echo ============================================
echo.
echo   Project : %PROJECT_ROOT%
echo.

:: ============================================================
:: Step 1 — Check Python
:: ============================================================
echo [1/8] Checking Python installation...

set "GLOBAL_PYTHON="
for %%p in (python3 python) do (
    where %%p >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "delims=" %%v in ('%%p --version 2^>^&1') do set "PYVER=%%v"
        set "GLOBAL_PYTHON=%%p"
        goto :python_found
    )
)

echo [ERROR] Python not found on PATH.
echo         Please install Python 3.10+ from https://python.org
echo         Make sure to check "Add Python to PATH" during install.
pause
exit /b 1

:python_found
echo         Found: !PYVER!  (!GLOBAL_PYTHON!)

:: ============================================================
:: Step 2 — Create virtual environment
:: ============================================================
echo.
echo [2/8] Setting up virtual environment...

set "VENV_DIR=%PROJECT_ROOT%\venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

if exist "%PYTHON%" (
    echo         venv already exists, skipping creation.
) else (
    echo         Creating venv at %VENV_DIR% ...
    "!GLOBAL_PYTHON!" -m venv "%VENV_DIR%" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo         venv created.
)

:: Upgrade pip
"%PYTHON%" -m pip install --upgrade pip >nul 2>&1

:: ============================================================
:: Step 3 — Create data directories
:: ============================================================
echo.
echo [3/8] Creating data directories...

for %%d in (
    "%PROJECT_ROOT%\data\input"
    "%PROJECT_ROOT%\data\output"
    "%PROJECT_ROOT%\data\temp"
    "%PROJECT_ROOT%\data\logs"
) do (
    if not exist %%d mkdir %%d >nul 2>&1
)
echo         All data directories ready.

:: ============================================================
:: Step 4 — Detect CUDA and install PyTorch
:: ============================================================
echo.
echo [4/8] Detecting GPU and installing PyTorch...

:: Check for NVIDIA GPU via nvidia-smi
set "CUDA_VER="
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo         NVIDIA GPU detected.
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        echo         GPU: %%a
    )
    echo         Installing PyTorch with CUDA 12.6 support...
    "%PIP%" install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    if !errorlevel! neq 0 (
        echo [WARNING] PyTorch CUDA install failed. Trying CPU version...
        "%PIP%" install torch torchaudio
    )
) else (
    echo         No NVIDIA GPU detected (or nvidia-smi not found).
    echo         Installing PyTorch CPU version...
    "%PIP%" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: ============================================================
:: Step 5 — Install Python dependencies
:: ============================================================
echo.
echo [5/8] Installing Python dependencies...

"%PIP%" install -r "%PROJECT_ROOT%\requirements.txt"
if !errorlevel! neq 0 (
    echo [WARNING] Some packages failed to install.
    echo           Check the output above for details.
    pause
)
echo         Dependencies installed.

:: ============================================================
:: Step 6 — Verify ffmpeg
:: ============================================================
echo.
echo [6/8] Checking ffmpeg...

where ffmpeg >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%a in ('ffmpeg -version 2^>^&1 ^| findstr /i "ffmpeg version"') do (
        echo         ffmpeg found: %%a
    )
) else (
    echo.
    echo   [WARNING] ffmpeg not found on PATH!
    echo.
    echo   ffmpeg is REQUIRED for audio/video processing.
    echo   Download from: https://ffmpeg.org/download.html
    echo.
    echo   After installation, either:
    echo     1. Add ffmpeg to system PATH
    echo     2. Or set the full path in config.yaml: tools.ffmpeg
    echo.
    echo   The script will continue but processing will fail without ffmpeg.
    pause
)

:: ============================================================
:: Step 7 — Copy Demucs model from original project (if exists)
:: ============================================================
echo.
echo [7/8] Checking Demucs model files...

set "ORIGIN_CHECKPOINTS=%SCRIPT_DIR%\..\anime-pipeline\checkpoints\torch_hub"
set "LOCAL_CHECKPOINTS=%PROJECT_ROOT%\checkpoints\hub\checkpoints"

:: Count .th files in local checkpoints
set "LOCAL_COUNT=0"
if exist "%LOCAL_CHECKPOINTS%\*.th" (
    for %%f in ("%LOCAL_CHECKPOINTS%\*.th") do set /a LOCAL_COUNT+=1 2>nul
)

if !LOCAL_COUNT! gtr 0 (
    echo         Model files already present ^(!LOCAL_COUNT! files^).
    goto :models_done
)

:: Try to copy from original project
if exist "%ORIGIN_CHECKPOINTS%\*.th" (
    set "ORIGIN_COUNT=0"
    for %%f in ("%ORIGIN_CHECKPOINTS%\*.th") do set /a ORIGIN_COUNT+=1 2>nul
    echo         Copying !ORIGIN_COUNT! model files from original project...
    mkdir "%LOCAL_CHECKPOINTS%" 2>nul
    copy "%ORIGIN_CHECKPOINTS%\*.th" "%LOCAL_CHECKPOINTS%\" >nul
    echo         Models copied successfully.
    goto :models_done
)

:: Also check if original project already migrated to new structure
set "ORIGIN_CHECKPOINTS2=%SCRIPT_DIR%\..\anime-pipeline\checkpoints\hub\checkpoints"
if exist "%ORIGIN_CHECKPOINTS2%\*.th" (
    echo         Copying model files from original project ^(new structure^)...
    mkdir "%LOCAL_CHECKPOINTS%" 2>nul
    copy "%ORIGIN_CHECKPOINTS2%\*.th" "%LOCAL_CHECKPOINTS%\" >nul
    echo         Models copied successfully.
    goto :models_done
)

:: Not found — inform user
echo.
echo   [INFO] Demucs htdemucs_ft model not found locally.
echo.
echo   On first run, Demucs will automatically download the model
echo   from PyTorch Hub (~330 MB). This requires internet access.
echo.
echo   To skip this, copy .th files from the original project:
echo     %ORIGIN_CHECKPOINTS%
echo   into:
echo     %LOCAL_CHECKPOINTS%
echo.

:models_done

:: ============================================================
:: Step 8 — Done
:: ============================================================
echo.
echo [8/8] Setup complete!
echo.
echo ============================================
echo   Video Pipeline setup finished!
echo.
echo   To start processing:
echo     - Double-click start.bat
echo     - Or run: venv\Scripts\python main.py
echo.
echo   Edit config.yaml to customize settings.
echo ============================================
echo.
pause

endlocal

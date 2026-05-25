@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
::  Anime Pipeline — One-Click Setup & Launch
::  First run installs everything; subsequent runs just start.
:: ============================================================

title Anime Pipeline Setup

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"
set "COMICUT_ROOT=%SCRIPT_DIR%\.."
pushd "%PROJECT_ROOT%" 2>nul || (
    echo [ERROR] Cannot access project directory: %PROJECT_ROOT%
    pause
    exit /b 1
)

:: Resolve COMICUT_ROOT to absolute
for %%i in ("%COMICUT_ROOT%") do set "COMICUT_ROOT=%%~fi"
for %%i in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fi"

set "VENV_DIR=%PROJECT_ROOT%\venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"
set "SETUP_DONE_FILE=%PROJECT_ROOT%\.setup_done"

echo ============================================
echo   Anime Pipeline — One-Click Setup
echo ============================================
echo.
echo   Project : %PROJECT_ROOT%
echo   Parent  : %COMICUT_ROOT%
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
:: Step 2 — Create / activate virtual environment
:: ============================================================
echo.
echo [2/8] Setting up virtual environment...

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
:: Step 3 — Install Python dependencies
:: ============================================================
echo.
echo [3/8] Installing Python dependencies...

if exist "%SETUP_DONE_FILE%" (
    echo         Setup already completed — skipping to launch.
    echo         (Delete .setup_done to force full reinstall.)
    goto :launch
)

echo         Installing from requirements.txt ...
"%PIP%" install -r "%PROJECT_ROOT%\requirements.txt"
if !errorlevel! neq 0 (
    echo.
    echo [WARNING] Some packages failed to install.
    echo           The server may still work for basic features.
    echo           Check the output above for details.
    pause
)
echo         Core dependencies installed.

:: ============================================================
:: Step 4 — Optional ASR dependencies
:: ============================================================
echo.
echo [4/8] Optional ASR dependencies...

set "ASR_REQ=%PROJECT_ROOT%\requirements-asr.txt"
if exist "%ASR_REQ%" (
    echo.
    echo         ASR (speech recognition) packages are optional but large ^(~4GB^).
    echo         Required for: auto-subtitle generation, ASR model comparison.
    echo.
    set /p ASR_INSTALL="         Install ASR dependencies? [y/N]: "
    if /i "!ASR_INSTALL!"=="y" (
        echo         Installing ASR dependencies...
        "%PIP%" install -r "%ASR_REQ%"
        if !errorlevel! neq 0 (
            echo [WARNING] Some ASR packages failed to install.
            echo           You can retry later: venv\Scripts\pip install -r requirements-asr.txt
        )
    ) else (
        echo         Skipped. Run later with: venv\Scripts\pip install -r requirements-asr.txt
    )
)

echo.
echo [5/8] Checking ClearerVoice-Studio...

set "CV_DIR=%COMICUT_ROOT%\ClearerVoice-Studio-main\ClearerVoice-Studio-main"
set "CV_CLEARVOICE=%CV_DIR%\clearvoice"

if exist "%CV_CLEARVOICE%\clearvoice.py" (
    echo         Found at: %CV_DIR%
    goto :clearvoice_install
)

:: Not found — attempt to clone
echo         ClearerVoice-Studio not found.
echo.
echo         It will be cloned from GitHub (~200MB).
echo         This is required for audio denoising features.
echo.
set /p CV_CLONE="         Clone now? [Y/n]: "
if /i "!CV_CLONE!"=="n" (
    echo         Skipped. Denoising features will not be available.
    goto :skip_clearvoice
)

echo         Cloning ClearerVoice-Studio...
set "CV_TEMP=%COMICUT_ROOT%\ClearerVoice-Studio-temp"
set "CV_PARENT=%COMICUT_ROOT%\ClearerVoice-Studio-main"

if exist "%CV_TEMP%" rmdir /s /q "%CV_TEMP%"
git clone --depth 1 https://github.com/modelscope/ClearerVoice-Studio.git "%CV_TEMP%" 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] Git clone failed.
    echo           You can manually download from:
    echo           https://github.com/modelscope/ClearerVoice-Studio
    echo           Extract to: %CV_PARENT%\ClearerVoice-Studio-main\
    goto :skip_clearvoice
)

:: Create the nested structure expected by config.py
if not exist "%CV_PARENT%" mkdir "%CV_PARENT%"
move "%CV_TEMP%" "%CV_DIR%" >nul 2>&1
rmdir /s /q "%CV_TEMP%" 2>nul
echo         Cloned successfully.

:clearvoice_install
:: Install ClearVoice package
echo         Installing ClearVoice package...
"%PIP%" install -e "%CV_DIR%" >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] ClearVoice installation failed.
    echo           pip install -e "%CV_DIR%"
)
echo         ClearVoice ready.

:skip_clearvoice

:: ============================================================
:: Step 5 — External tools (ffmpeg, mkvtoolnix)
:: ============================================================
echo.
echo [6/8] Checking external tools...

set "QUICKCUT_DIR=%COMICUT_ROOT%\QuickCut"
set "MKV_DIR=%COMICUT_ROOT%\mkvtoolnix"
set "TOOLS_OK=1"

:: ffmpeg / ffprobe
if exist "%QUICKCUT_DIR%\ffmpeg.exe" (
    echo         ffmpeg   : found ^(%QUICKCUT_DIR%^)
) else (
    echo         ffmpeg   : MISSING  ^(expected at %QUICKCUT_DIR%\ffmpeg.exe^)
    echo                   Download from https://ffmpeg.org/download.html
    echo                   Place ffmpeg.exe and ffprobe.exe in %QUICKCUT_DIR%\
    set "TOOLS_OK=0"
)

:: mkvtoolnix
if exist "%MKV_DIR%\mkvextract.exe" (
    echo         mkvtoolnix: found ^(%MKV_DIR%^)
) else (
    echo         mkvtoolnix: MISSING  ^(expected at %MKV_DIR%\^)
    echo                    Download from https://mkvtoolnix.download/
    echo                    Place mkvextract.exe, mkvmerge.exe, mkvinfo.exe in %MKV_DIR%\
    set "TOOLS_OK=0"
)

if "!TOOLS_OK!"=="0" (
    echo.
    echo   [NOTE] Some external tools are missing.
    echo          Video splitting and subtitle extraction will not work.
    echo          The server can still start — install tools later and restart.
)

:: ============================================================
:: Step 6 — Clean up stale settings.json
:: ============================================================
echo.
echo [7/8] Checking configuration...

set "SETTINGS_FILE=%PROJECT_ROOT%\data\settings.json"
if exist "%SETTINGS_FILE%" (
    :: Check if settings.json has hardcoded absolute paths (from another machine)
    findstr /c:"E:\\ComicCut" "%SETTINGS_FILE%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo         Removing old settings.json with hardcoded paths...
        del "%SETTINGS_FILE%"
        echo         Fresh settings will be generated on first save.
    ) else (
        echo         settings.json looks OK, keeping it.
    )
) else (
    echo         No existing settings.json — defaults will be used.
)

:: ============================================================
:: Step 8 — Launch server
:: ============================================================
:launch
echo.
echo [8/8] Starting Anime Pipeline Server...
echo.
echo ============================================
echo   Frontend : http://localhost:5800
echo   API docs : http://localhost:5800/docs
echo ============================================
echo.

:: Mark setup as done
echo %DATE% %TIME% > "%SETUP_DONE_FILE%"

:: Check if port 5800 is in use
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5800.*LISTENING"') do (
    echo Port 5800 is in use by PID %%a. Killing...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

:: Launch server with venv python
"%PYTHON%" "%PROJECT_ROOT%\scripts\server.py"
pause

endlocal

@echo off
chcp 65001 >nul
setlocal

:: ============================================================
::  Download all Python wheels for Linux offline install.
::  Run on a machine WITH internet access.
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "WHEELS_DIR=%SCRIPT_DIR%\offline_wheels"
set "REQ=%SCRIPT_DIR%\requirements.txt"
set "ASR_REQ=%SCRIPT_DIR%\requirements-asr.txt"

:: Use Tsinghua mirror (faster in China, bypasses GFW)
set "PIP_TRUST=-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo ============================================
echo   Offline Wheel Downloader
echo ============================================
echo.
echo   Output: %WHEELS_DIR%
echo.

if not exist "%WHEELS_DIR%" mkdir "%WHEELS_DIR%"

:: Quick connectivity test
echo Testing PyPI connectivity...
python -m pip install --upgrade pip %PIP_TRUST% -q 2>&1 | findstr /c:"Successfully" >nul
if %errorlevel% neq 0 (
    echo [WARNING] PyPI reachable but slow/unstable — continuing...
)

:: ============================================================
:: Step 1: Download pip + setuptools (must work for bootstrap)
:: ============================================================
echo.
echo [1/4] Downloading pip + setuptools...
python -m pip download pip setuptools wheel -d "%WHEELS_DIR%" %PIP_TRUST% --no-deps
if %errorlevel% neq 0 (
    echo [ERROR] Cannot reach PyPI at all.
    echo.
    echo   Possible fixes:
    echo   1. Use a VPN or different network
    echo   2. Set proxy: set HTTP_PROXY=http://proxy:port
    echo   3. Try a mirror:
    echo      pip download -i https://pypi.tuna.tsinghua.edu.cn/simple pip setuptools -d offline_wheels
    echo.
    pause
    exit /b 1
)

:: ============================================================
:: Step 2: Core dependencies — download Linux wheels
:: Two-pass: pure Python (any platform) + Linux binaries
:: ============================================================
echo.
echo [2/4] Core dependencies...

:: Pass A: download everything for current platform (deps resolved correctly)
python -m pip download -r "%REQ%" -d "%WHEELS_DIR%" %PIP_TRUST% 2>&1
if %errorlevel% neq 0 echo [WARNING] Some packages failed.

:: Pass B: overwrite with Linux binary wheels for compiled packages
python -m pip download ^
    -r "%REQ%" ^
    -d "%WHEELS_DIR%" ^
    --no-deps ^
    --platform manylinux_2_17_x86_64 ^
    --platform manylinux2014_x86_64 ^
    --only-binary=:all: ^
    --python-version 310 ^
    %PIP_TRUST% 2>&1
if %errorlevel% neq 0 echo [WARNING] Some Linux wheels failed — may need manual fix.

:: ============================================================
:: Step 3: ASR (optional)
:: ============================================================
echo.
echo [3/4] ASR dependencies (optional)...
python -m pip download -r "%ASR_REQ%" -d "%WHEELS_DIR%" %PIP_TRUST% 2>&1
if %errorlevel% neq 0 echo [WARNING] Some ASR packages failed.

echo.
echo ============================================
echo   Done.
echo   Wheels in: %WHEELS_DIR%
echo   Then on server: ./setup.sh
echo ============================================
pause
endlocal

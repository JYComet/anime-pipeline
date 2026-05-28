@echo off
chcp 65001 >nul
setlocal

:: ============================================================
::  Offline Wheel Downloader (Windows)
::  Run on a machine WITH internet access.
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo ============================================
echo   Offline Wheel Downloader
echo ============================================
echo.
echo   Output: %SCRIPT_DIR%\offline_wheels
echo.

:: Quick connectivity test
echo Testing PyPI connectivity...
python -m pip install --upgrade pip -q 2>&1 | findstr /c:"Successfully" >nul
if %errorlevel% neq 0 (
    echo [WARNING] PyPI reachable but slow/unstable — continuing...
)

echo.
echo [1/3] Downloading pip + setuptools + wheel (bootstrap)...
python -m pip download pip setuptools wheel -d "%SCRIPT_DIR%\offline_wheels" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-deps
if %errorlevel% neq 0 (
    echo [ERROR] Cannot reach PyPI at all. Check network/proxy settings.
    pause
    exit /b 1
)

echo.
echo [2/3] Downloading all dependencies...
python "%SCRIPT_DIR%\tools\download_wheels.py" %*

echo.
echo [3/3] Done!
echo.
echo   To also download Linux wheels:
echo     python tools\download_wheels.py --linux
echo.
echo   Then on Linux server: ./setup.sh
echo ============================================
pause
endlocal

@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
::  Video Pipeline — Windows 一键启动脚本
:: ============================================================

title Video Pipeline — 处理中...

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"

set "VENV_DIR=%PROJECT_ROOT%\venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

:: Check if venv exists
if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found.
    echo         Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

:: Check if config exists
if not exist "%PROJECT_ROOT%\config.yaml" (
    echo [ERROR] config.yaml not found at %PROJECT_ROOT%
    echo         Please ensure config.yaml exists.
    pause
    exit /b 1
)

:: Fix PyTorch GBK encoding issue on Chinese Windows
set PYTHONUTF8=1

:: Set Torch Hub cache directory for offline model loading
set TORCH_HOME=%PROJECT_ROOT%\checkpoints

:: ============================================================
:: Launch pipeline
:: ============================================================
echo.
echo ============================================
echo   Video Pipeline — Starting...
echo ============================================
echo.
echo   输入目录:  (from config.yaml)
echo   输出目录:  (from config.yaml)
echo   日志目录:  (from config.yaml)
echo.
echo   Press Ctrl+C at any time to stop gracefully.
echo ============================================
echo.

:: Pass all arguments through to main.py
"%PYTHON%" "%PROJECT_ROOT%\main.py" %*

if !errorlevel! neq 0 (
    echo.
    echo ============================================
    echo   Processing finished with errors.
    echo   Check logs in: data\logs\
    echo ============================================
)

echo.
echo Press any key to close this window...
pause >nul

endlocal

@echo off
chcp 65001 >nul
setlocal

:: ============================================================
::  Anime Pipeline — Quick Start
::  Run setup.bat first if this is a fresh install.
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"
set "VENV_PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"

:: If venv doesn't exist, redirect to setup
if not exist "%VENV_PYTHON%" (
    echo Virtual environment not found. Running setup first...
    echo.
    call "%PROJECT_ROOT%\setup.bat"
    exit /b
)

title Anime Pipeline Server

echo ============================================
echo   Anime Pipeline Server
echo ============================================
echo.

:: Check / free port 5800
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5800.*LISTENING"') do (
    echo Port 5800 is in use by PID %%a. Killing...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

echo Starting server...
echo Frontend: http://localhost:5800
echo.

:: Fix PyTorch GBK encoding issue on Chinese Windows
set PYTHONUTF8=1

"%VENV_PYTHON%" "%PROJECT_ROOT%\scripts\server.py"
pause

endlocal

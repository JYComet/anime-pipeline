@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"
set "VENV_PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"

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

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5800.*LISTENING"') do (
    echo Port 5800 in use by PID %%a. Killing...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

echo Starting server...
echo Frontend: http://localhost:5800
echo.

cd /d "%PROJECT_ROOT%\scripts"
"%VENV_PYTHON%" server.py
pause

endlocal

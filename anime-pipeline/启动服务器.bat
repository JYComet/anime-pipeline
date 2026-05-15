@echo off
echo ============================================
echo   Anime Pipeline Server
echo ============================================
echo.
echo Checking port 5800...

:check_port
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5800.*LISTENING"') do (
    set PID=%%a
    goto :kill
)
goto :port_free

:kill
echo Killing PID %PID% on port 5800...
taskkill /F /PID %PID% >nul 2>&1
echo Waiting for port release...
timeout /t 3 /nobreak >nul
goto :check_port

:port_free
echo Port 5800 is free.
echo Starting server...
echo Frontend: http://localhost:5800
echo.
cd /d "%~dp0scripts"
python server.py
pause

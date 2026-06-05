@echo off
chcp 65001 >nul
title ASR Compare Tool

:: =============================================================================
:: ASR Compare Tool — Windows One-Click Launcher
:: =============================================================================
:: Double-click this file to auto-deploy and run.
:: The window stays open after completion to review results.
::
:: Advanced usage:
::   run.bat --no-delete        — keep all results
::   run.bat --match-threshold 85
::   run.bat --language ja
:: =============================================================================

cd /d "%~dp0"

echo ========================================
echo   ASR Compare Tool
echo   FireRed ASR2 ^<^> Qwen3-ASR (API)
echo ========================================
echo.
echo [1/2] Checking environment and installing dependencies...
echo ========================================

:: Auto-deploy: check env + install deps if needed
python deploy.py
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Deploy failed! Check errors above.
    echo You can also run: python deploy.py --force-cpu
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo [2/2] Starting ASR comparison...
echo ========================================
echo.
echo Config: config.yaml
echo Log:    asr_compare.log
echo Press Ctrl+C to cancel during processing.
echo ========================================
echo.

python asr_compare.py %*

echo.
echo ========================================
echo Task completed. Press any key to exit...
echo ========================================
pause >nul

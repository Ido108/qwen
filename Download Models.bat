@echo off
setlocal
cd /d "%~dp0"
title Qwen - Download and verify models
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\download_models.ps1" %*
if errorlevel 1 goto failed
echo.
echo Q8 and Q4_K_M with their matching files are verified. More models: Settings.
pause
exit /b 0
:failed
echo.
echo Download did not finish. Run this BAT again to resume.
pause
exit /b 1

@echo off
setlocal
cd /d "%~dp0"
title Qwen - Qwen-Image-2.1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1"
if errorlevel 1 (
  echo.
  echo Qwen could not start. See the message above.
  pause
)

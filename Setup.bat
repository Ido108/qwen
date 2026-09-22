@echo off
setlocal
cd /d "%~dp0"
title Qwen - Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
if errorlevel 1 (
  echo Setup did not finish. See the message above.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1"
if errorlevel 1 (
  echo Qwen could not start. See the message above.
  pause
  exit /b 1
)

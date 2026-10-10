@echo off
setlocal
cd /d "%~dp0.."
title Qwen - Restart
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart.ps1"
if errorlevel 1 pause

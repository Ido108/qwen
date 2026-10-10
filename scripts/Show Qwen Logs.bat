@echo off
setlocal
cd /d "%~dp0.."
title Qwen - Activity
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_logs.ps1"
if errorlevel 1 pause

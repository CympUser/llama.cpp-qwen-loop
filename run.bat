@echo off
title Qwen3.8-9B Recurrent-Loop (1..6 Loops)
powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Press any key to close...
    pause >nul
)

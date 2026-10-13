@echo off
title Suzy - Siri AI Assistant
cd /d "%~dp0\suzy"
if exist "%USERPROFILE%\Suzy\.venv\Scripts\python.exe" (
    "%USERPROFILE%\Suzy\.venv\Scripts\python.exe" main.py --ui %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py --ui %*
) else (
    python main.py --ui %*
)
pause

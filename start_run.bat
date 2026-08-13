@echo off
REM Start recording a SHIELD run. Edit the "EDIT PER RUN" section of
REM record_run.py first, then double-click this file. Results are written
REM under results\ in this repo. Press Ctrl+C in this window to stop the run.

setlocal

REM Run from the repo root so results\ lands here regardless of how the
REM script was launched.
cd /d "%~dp0"

REM Prefer the python inside this repo's venv, fall back to the PATH.
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto run

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=python"
    goto run
)

echo Could not find python.
echo Install SHIELD_DAS into a Python environment (pip install -e .) or edit
echo this file so PYTHON points at the environment where it is installed.
pause
exit /b 1

:run
"%PYTHON%" "%~dp0record_run.py"
echo.
pause

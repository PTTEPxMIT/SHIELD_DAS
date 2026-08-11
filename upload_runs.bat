@echo off
REM Upload completed SHIELD runs to SHIELD-Data (one pull request per run).
REM Double-click this file after a run finishes. See docs/auto_upload.md for
REM the one-time setup (GitHub token + config file).

setlocal

REM Prefer the uploader inside this repo's venv, fall back to the PATH.
set "UPLOADER=%~dp0.venv\Scripts\shield-das-upload.exe"
if exist "%UPLOADER%" goto run

where shield-das-upload >nul 2>nul
if %errorlevel%==0 (
    set "UPLOADER=shield-das-upload"
    goto run
)

echo Could not find shield-das-upload.
echo Install SHIELD_DAS into a Python environment (pip install -e .) or edit
echo this file so UPLOADER points at the environment where it is installed.
pause
exit /b 1

:run
"%UPLOADER%" %*
echo.
pause

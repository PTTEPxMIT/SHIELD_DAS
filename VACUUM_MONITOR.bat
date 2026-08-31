@echo off
REM Publish the rig's present vacuum level so it can be checked from anywhere,
REM with no run recording and nothing written to disk. Double-click this file.
REM See docs/live_supabase.md ("The beacon").
REM
REM It stays out of the way: main.py and leak_test.py stop the beacon when a
REM run starts (the Windows LabJack driver does not let two processes share
REM the device) and restart it when the run ends. You do not need to stop it.

setlocal
set "SITE=https://pttepxmit.github.io/SHIELD_DAS/"

tasklist /FI "IMAGENAME eq shield-das-beacon.exe" 2>nul | "%SystemRoot%\System32\find.exe" /I "shield-das-beacon.exe" >nul
if not errorlevel 1 (
    echo.
    echo The vacuum monitor is already running.
    goto show
)

REM Prefer the beacon inside this repo's venv, fall back to the PATH.
set "BEACON=%~dp0.venv\Scripts\shield-das-beacon.exe"
if exist "%BEACON%" goto start

for /f "delims=" %%I in ('"%SystemRoot%\System32\where.exe" shield-das-beacon 2^>nul') do set "BEACON=%%I"
if defined BEACON if exist "%BEACON%" goto start

echo Could not find shield-das-beacon.
echo Install SHIELD_DAS into a Python environment (pip install -e .) or edit
echo this file so BEACON points at the environment where it is installed.
pause
exit /b 1

:start
echo.
echo Starting the vacuum monitor...
start "" wscript.exe "%~dp0_beacon_watcher.vbs" "%BEACON%"

REM Poll rather than sleeping a fixed time: a cold start can take a few
REM seconds, and reporting failure while it is still coming up is worse than
REM waiting a moment longer.
set tries=0
:wait
tasklist /FI "IMAGENAME eq shield-das-beacon.exe" 2>nul | "%SystemRoot%\System32\find.exe" /I "shield-das-beacon.exe" >nul
if not errorlevel 1 goto show
set /a tries+=1
if %tries% geq 10 goto failed
ping -n 2 127.0.0.1 >nul
goto wait

:failed
echo.
echo It did not start. The log usually says why:
echo   %TEMP%\shield_beacon.log
echo.
echo Most likely causes: the Supabase schema has not been applied yet
echo (docs/live_supabase.md), or another program is holding the LabJack.
echo.
pause
exit /b 1

:show
echo.
echo View the vacuum level here (works on your phone, no VPN):
echo   %SITE%
echo.
echo To stop it:  STOP_VACUUM_MONITOR.bat
echo Log file:    %TEMP%\shield_beacon.log
echo.
pause
exit /b 0

@echo off
REM Stop publishing the rig's present vacuum level. Double-click this file.
REM
REM You do NOT need this before starting a run: the beacon releases the
REM LabJack between readings and pauses on its own while a run is recording.

tasklist /FI "IMAGENAME eq shield-das-beacon.exe" 2>nul | "%SystemRoot%\System32\find.exe" /I "shield-das-beacon.exe" >nul
if errorlevel 1 (
    echo.
    echo The vacuum monitor is not running.
    echo.
    pause
    exit /b 0
)

taskkill /IM shield-das-beacon.exe /F >nul 2>&1
echo.
echo Vacuum monitor stopped. The live site will show its last reading as
echo STALE until you start it again with VACUUM_MONITOR.bat.
echo.
pause
exit /b 0

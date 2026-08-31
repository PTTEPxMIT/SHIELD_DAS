@echo off
REM Stop publishing the rig's present vacuum level. Double-click this file.
REM
REM You do NOT need this before starting a run: main.py and leak_test.py
REM stop the beacon themselves and restart it when the run ends (the Windows
REM LabJack driver does not let two processes share the device).

tasklist /FI "IMAGENAME eq shield-das-beacon.exe" 2>nul | "%SystemRoot%\System32\find.exe" /I "shield-das-beacon.exe" >nul
if errorlevel 1 (
    echo.
    echo The vacuum monitor is not running.
    echo.
    pause
    exit /b 0
)

taskkill /F /T /IM shield-das-beacon.exe >nul 2>&1
echo.
echo Vacuum monitor stopped. The live site will show its last reading as
echo STALE until you start it again with VACUUM_MONITOR.bat.
echo.
pause
exit /b 0

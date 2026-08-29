@echo off
rem =====================================================================
rem  SHIELD LIVE DASHBOARD
rem  Double-click me: if a run is currently being recorded, this opens
rem  the live dashboard in your browser. If not, it tells you and exits.
rem =====================================================================
title SHIELD Live Dashboard
set "RESULTS=C:\Users\remidm\Documents\JD\SHIELD_DAS\results"
set "PY=C:\Program Files\Python311\python.exe"

"%PY%" -c "import sys; from shield_das.live_dashboard import find_active_run; sys.exit(0 if find_active_run(r'%RESULTS%', include_test_runs=True) else 1)"
if errorlevel 1 (
    echo.
    echo   No run is currently being recorded - nothing to show.
    echo.
    pause
    exit /b 1
)

echo A run is live - connecting to the dashboard...

rem Start the hidden dashboard server if it is not already answering on
rem http://localhost:8051 (it attaches to the run within ~10 s), then wait
rem for the page to answer before opening the browser.
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8051', timeout=2)" 2>nul
if errorlevel 1 start "" wscript.exe "C:\Users\remidm\Documents\JD\SHIELD_DAS\_live_dashboard_watcher.vbs"

set tries=0
:waitloop
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8051', timeout=2)" 2>nul
if not errorlevel 1 goto open
set /a tries+=1
if %tries% geq 20 (
    echo Dashboard server did not come up after 40 s - check the log at
    echo %%TEMP%%\shield_live_dashboard.log
    pause
    exit /b 1
)
ping -n 3 127.0.0.1 >nul
goto waitloop

:open
start "" http://127.0.0.1:8051
exit /b 0

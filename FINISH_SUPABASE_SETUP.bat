@echo off
rem =====================================================================
rem  FINISH SHIELD SUPABASE SETUP
rem  Double-click me AFTER creating the Supabase project (see
rem  docs/live_supabase.md, "One-time setup": create the project, run
rem  supabase/schema.sql in the SQL editor).
rem
rem  Paste the three values from Project Settings -> API and this script
rem  wires up the rig: SHIELD_SUPABASE_KEY env var, the publisher config,
rem  site/config.js, and starts the publisher in the background.
rem =====================================================================
title Finish SHIELD Supabase setup
setlocal
set "REPO=C:\Users\remidm\Documents\JD\SHIELD_DAS"
set "PY=C:\Program Files\Python311\python.exe"

echo Paste values from the Supabase dashboard: Project Settings ^> API
echo.
set /p SB_URL="Project URL (https://<ref>.supabase.co): "
set /p SB_SERVICE="service_role key (secret): "
set /p SB_ANON="anon public key: "
echo.

"%PY%" "%REPO%\_finish_supabase_setup.py"
if errorlevel 1 (
    echo.
    echo Setup NOT completed - fix the input above and run me again.
    pause
    exit /b 1
)

rem Persist the service key for future logons (the config-file fallback
rem written above covers processes started before this takes effect).
setx SHIELD_SUPABASE_KEY "%SB_SERVICE%" >nul
echo   set SHIELD_SUPABASE_KEY user environment variable

rem Start the publisher now (hidden; logs to %%TEMP%%\shield_publisher.log).
set "SHIELD_SUPABASE_KEY=%SB_SERVICE%"
start "" wscript.exe "%REPO%\_publisher_watcher.vbs"
echo   started the background publisher

echo.
echo Done. Remaining step (once): commit and push site\config.js so the
echo GitHub Pages viewer site can reach the project - or ask Claude to.
echo.
pause
exit /b 0

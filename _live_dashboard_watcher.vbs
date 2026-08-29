' Internal helper for the "SHIELD live dashboard" scheduled task.
' Runs the dashboard watcher invisibly (no console window), logging to
' %TEMP%\shield_live_dashboard.log.
'
' You normally never run this yourself - double-click LIVE_DASHBOARD.bat
' to open the dashboard when a run is being recorded.
Set sh = CreateObject("WScript.Shell")
q = Chr(34)
py = q & "C:\Program Files\Python311\python.exe" & q
res = q & "C:\Users\remidm\Documents\JD\SHIELD_DAS\results" & q
logf = q & sh.ExpandEnvironmentStrings("%TEMP%") & "\shield_live_dashboard.log" & q
cmdline = "cmd /s /c " & q & py & " -m shield_das.live_dashboard --watch --no-browser --include-test-runs --results-dir " & res & " > " & logf & " 2>&1" & q
sh.Run cmdline, 0, True

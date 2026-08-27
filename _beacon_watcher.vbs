' Internal helper for VACUUM_MONITOR.bat (and the optional logon task).
' Runs the vacuum beacon invisibly (no console window), logging to
' %TEMP%\shield_beacon.log.
'
' The beacon publishes the rig's present vacuum level to Supabase whenever no
' run is being recorded, so the viewer site shows a live number instead of
' "waiting for a run" (see docs/live_supabase.md). You normally never run this
' yourself - double-click VACUUM_MONITOR.bat instead.
'
' Usage: wscript.exe _beacon_watcher.vbs ["<path to shield-das-beacon.exe>"]
' With no argument it falls back to whatever is on the PATH.
Set sh = CreateObject("WScript.Shell")
q = Chr(34)

If WScript.Arguments.Count > 0 Then
    exe = q & WScript.Arguments(0) & q
Else
    exe = "shield-das-beacon"
End If

logf = q & sh.ExpandEnvironmentStrings("%TEMP%") & "\shield_beacon.log" & q
cmdline = "cmd /s /c " & q & exe & " > " & logf & " 2>&1" & q
sh.Run cmdline, 0, True

' Internal helper for the "SHIELD publisher" scheduled task.
' Runs shield-das-publish invisibly (no console window), logging to
' %TEMP%\shield_publisher.log.
'
' The publisher mirrors live run data to Supabase for the remote viewer
' site (see docs/live_supabase.md). You normally never run this yourself.
Set sh = CreateObject("WScript.Shell")
q = Chr(34)
exe = q & "C:\Users\remidm\AppData\Roaming\Python\Python311\Scripts\shield-das-publish.exe" & q
logf = q & sh.ExpandEnvironmentStrings("%TEMP%") & "\shield_publisher.log" & q
cmdline = "cmd /s /c " & q & exe & " > " & logf & " 2>&1" & q
sh.Run cmdline, 0, True

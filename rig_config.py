"""Shared hardware configuration for the SHIELD rig.

This is the single place where the rig's gauges and thermocouples are
defined. Every entry-point script (main.py for permeation runs,
leak_test.py for leak tests) builds its hardware from here, so a gauge
swap or AIN channel change only needs to be made once.
"""

from shield_das import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    Thermocouple,
    WGM701_Gauge,
)


def build_gauges():
    """Return the rig's pressure gauges, as currently plumbed."""
    return [
        WGM701_Gauge(
            gauge_location="downstream",
        ),
        CVM211_Gauge(
            gauge_location="upstream",
        ),
        Baratron626D_Gauge(
            name="Baratron626D_1KT",
            gauge_location="upstream",
            full_scale_Torr=1000,
            ain_channel=6,
        ),
        Baratron626D_Gauge(
            name="Baratron626D_1T",
            gauge_location="downstream",
            full_scale_Torr=1,
            ain_channel=4,
        ),
    ]


def build_thermocouples():
    """Return the rig's thermocouples."""
    return [Thermocouple(name="furnace_thermocouple")]


# ---------------------------------------------------------------------------
# Vacuum-beacon coordination.
#
# On this rig's Windows LabJack UD driver, the beacon's open/read/close cycle
# does not reliably release the USB claim to OTHER processes while the beacon
# process lives, so the recorder cannot open the LabJack while the beacon is
# running. The entry-point scripts therefore stop the beacon for the duration
# of a run and restart it afterwards.

import os as _os
import shutil as _shutil
import subprocess as _subprocess

_REPO_DIR = _os.path.dirname(_os.path.abspath(__file__))


def stop_vacuum_beacon() -> bool:
    """Stop the standby vacuum beacon if it is running.

    Returns:
        True if a beacon was running (so the caller should restart it later).
    """
    probe = _subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq shield-das-beacon.exe"],
        capture_output=True,
        text=True,
    )
    if "shield-das-beacon.exe" not in probe.stdout:
        return False
    # /T also kills the python child the console-script shim spawns -- the
    # child is the process actually holding the LabJack.
    _subprocess.run(
        ["taskkill", "/F", "/T", "/IM", "shield-das-beacon.exe"],
        capture_output=True,
    )
    print("Vacuum monitor stopped for this run (will restart when it ends)")
    return True


def start_vacuum_beacon() -> None:
    """Restart the standby vacuum beacon (same launch path as VACUUM_MONITOR.bat)."""
    beacon = _os.path.join(_REPO_DIR, ".venv", "Scripts", "shield-das-beacon.exe")
    if not _os.path.exists(beacon):
        beacon = _shutil.which("shield-das-beacon")
    if not beacon:
        print(
            "Could not find shield-das-beacon to restart the vacuum monitor; "
            "start it manually with VACUUM_MONITOR.bat"
        )
        return
    watcher = _os.path.join(_REPO_DIR, "_beacon_watcher.vbs")
    _subprocess.Popen(
        ["wscript.exe", watcher, beacon],
        creationflags=_subprocess.DETACHED_PROCESS
        | _subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    print("Vacuum monitor restarted")

"""System tools: master volume and a quick health report of the PC."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import psutil

from tools import winutils
from tools.registry import ToolResult, tool

log = logging.getLogger("suzy.tools.system")

# Virtual-key codes used when pycaw is unavailable (each key press = 2% volume).
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
KEYEVENTF_KEYUP = 0x0002


# ============================================================================ volume
class VolumeController:
    """Controls the Windows master volume.

    Uses the Core Audio API through pycaw for exact levels. If pycaw cannot be
    used, it falls back to simulating the keyboard's volume keys, which still
    works but cannot read the current level.
    """

    def _endpoint(self) -> Any:
        """Return an IAudioEndpointVolume for the default speakers."""
        winutils.require_windows("Volume control")
        import comtypes  # Windows-only dependency
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        try:
            comtypes.CoInitialize()  # required once per thread; harmless if repeated
        except OSError:
            pass
        speakers = AudioUtilities.GetSpeakers()
        if speakers is None:
            raise RuntimeError("no active speaker or headphone device was found")
        endpoint = getattr(speakers, "EndpointVolume", None)  # pycaw >= 2024
        if endpoint is not None:
            return endpoint
        from ctypes import POINTER, cast

        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    def get_level(self) -> int | None:
        """Current volume 0-100, or None if it cannot be read."""
        try:
            return int(round(self._endpoint().GetMasterVolumeLevelScalar() * 100))
        except winutils.PlatformNotSupportedError:
            raise
        except Exception as exc:  # noqa: BLE001 - COM errors come in many shapes
            log.warning("Could not read volume via pycaw: %s", exc)
            return None

    def set_level(self, level: int) -> bool:
        """Set the volume. Returns True if exact (pycaw), False if approximated by keys."""
        level = max(0, min(100, int(level)))
        try:
            endpoint = self._endpoint()
            endpoint.SetMasterVolumeLevelScalar(level / 100.0, None)
            if level > 0 and endpoint.GetMute():
                endpoint.SetMute(0, None)
            return True
        except winutils.PlatformNotSupportedError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("pycaw volume failed (%s); falling back to volume keys", exc)
        self._press_key(VK_VOLUME_DOWN, 50)            # drive to 0%
        self._press_key(VK_VOLUME_UP, round(level / 2))  # then up in 2% steps
        return False

    def change_level(self, delta: int) -> int | None:
        """Raise/lower the volume by ``delta`` points. Returns the new level if known."""
        current = self.get_level()
        if current is None:
            key = VK_VOLUME_UP if delta > 0 else VK_VOLUME_DOWN
            self._press_key(key, max(1, round(abs(delta) / 2)))
            return None
        new_level = max(0, min(100, current + delta))
        self.set_level(new_level)
        return new_level

    @staticmethod
    def _press_key(vk_code: int, times: int) -> None:
        winutils.require_windows("Volume control")
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        for _ in range(times):
            user32.keybd_event(vk_code, 0, 0, 0)
            user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


VOLUME = VolumeController()


@tool(
    description=(
        "Set the PC's master volume. Use 'level' for an exact value (0-100), or "
        "'change' to raise/lower relative to now (e.g. +10 for 'a bit louder', -20 for 'quieter')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "level": {"type": "integer", "description": "Exact volume 0-100."},
            "change": {"type": "integer", "description": "Relative change, -100 to 100."},
        },
    },
    category="system",
)
def set_volume(level: int | None = None, change: int | None = None) -> ToolResult:
    if level is None and change is None:
        return ToolResult.fail("Tell me a volume level between 0 and 100, or how much to change it.")

    if level is not None:
        level = max(0, min(100, level))
        exact = VOLUME.set_level(level)
        prefix = "Volume set to" if exact else "Volume set to about"
        return ToolResult.ok(f"{prefix} {level}%.", level=level)

    assert change is not None
    change = max(-100, min(100, change))
    if change == 0:
        return ToolResult.ok("Volume left unchanged.")
    new_level = VOLUME.change_level(change)
    direction = "up" if change > 0 else "down"
    if new_level is None:
        return ToolResult.ok(f"Turned the volume {direction}.")
    return ToolResult.ok(f"Turned the volume {direction} to {new_level}%.", level=new_level)


# ============================================================================ system info
def _gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.1f} GB"


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or not parts:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts)


def _system_drive() -> str:
    if winutils.IS_WINDOWS:
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"


@tool(
    description="Get a quick health report of the PC: CPU usage, RAM, battery, disk space and uptime.",
    category="system",
)
def get_system_info() -> ToolResult:
    data: dict[str, Any] = {}
    sentences: list[str] = []

    # Each section is independent so one failing sensor doesn't hide the rest.
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        data["cpu_percent"] = cpu
        sentences.append(f"CPU is at {cpu:.0f}%")
    except Exception as exc:  # noqa: BLE001
        log.warning("CPU info failed: %s", exc)

    try:
        mem = psutil.virtual_memory()
        data["ram_percent"] = mem.percent
        data["ram_used_gb"] = round(mem.used / 1024 ** 3, 1)
        data["ram_total_gb"] = round(mem.total / 1024 ** 3, 1)
        sentences.append(f"RAM is {mem.percent:.0f}% used ({_gb(mem.used)} of {_gb(mem.total)})")
    except Exception as exc:  # noqa: BLE001
        log.warning("RAM info failed: %s", exc)

    try:
        battery = psutil.sensors_battery()
    except Exception as exc:  # noqa: BLE001
        log.warning("Battery info failed: %s", exc)
        battery = None
    if battery is not None:
        data["battery_percent"] = round(battery.percent)
        data["plugged_in"] = bool(battery.power_plugged)
        state = "charging" if battery.power_plugged else "on battery"
        sentences.append(f"battery is at {battery.percent:.0f}% and {state}")
    else:
        data["battery_percent"] = None

    drive = _system_drive()
    try:
        disk = psutil.disk_usage(drive)
        data["disk_drive"] = drive
        data["disk_free_gb"] = round(disk.free / 1024 ** 3, 1)
        data["disk_total_gb"] = round(disk.total / 1024 ** 3, 1)
        label = drive.rstrip("\\") + " drive" if winutils.IS_WINDOWS else "the main disk"
        sentences.append(f"{label} has {_gb(disk.free)} free of {_gb(disk.total)}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Disk info failed: %s", exc)

    try:
        uptime = time.time() - psutil.boot_time()
        data["uptime_seconds"] = int(uptime)
        sentences.append(f"the PC has been on for {_format_duration(uptime)}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Uptime info failed: %s", exc)

    if not sentences:
        return ToolResult.fail("I couldn't read any system information right now.")
    message = "; ".join(sentences) + "."
    return ToolResult.ok(message[0].upper() + message[1:], **data)

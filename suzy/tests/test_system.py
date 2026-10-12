"""Tests for volume control and system info (all Windows APIs mocked)."""

from __future__ import annotations

from collections import namedtuple

import pytest

from tools import system, winutils
from tools.system import VolumeController, _format_duration


class FakeEndpoint:
    """Mimics pycaw's IAudioEndpointVolume."""

    def __init__(self, level: float = 0.5, muted: bool = False):
        self.level, self.muted = level, muted

    def GetMasterVolumeLevelScalar(self):  # noqa: N802 - COM naming
        return self.level

    def SetMasterVolumeLevelScalar(self, value, _ctx):  # noqa: N802
        self.level = value

    def GetMute(self):  # noqa: N802
        return int(self.muted)

    def SetMute(self, value, _ctx):  # noqa: N802
        self.muted = bool(value)


@pytest.fixture()
def endpoint(monkeypatch):
    fake = FakeEndpoint()
    monkeypatch.setattr(VolumeController, "_endpoint", lambda self: fake)
    return fake


@pytest.fixture()
def keys(monkeypatch):
    """Record simulated volume-key presses."""
    pressed: list[tuple[int, int]] = []
    monkeypatch.setattr(VolumeController, "_press_key", staticmethod(lambda vk, n: pressed.append((vk, n))))
    return pressed


def test_set_exact_volume(endpoint):
    result = system.set_volume(level=30)
    assert result.success and result.message == "Volume set to 30%."
    assert endpoint.level == pytest.approx(0.30)


def test_set_volume_clamps_and_unmutes(endpoint):
    endpoint.muted = True
    assert system.set_volume(level=150).data["level"] == 100
    assert endpoint.level == pytest.approx(1.0) and not endpoint.muted
    assert system.set_volume(level=-5).data["level"] == 0


def test_relative_change(endpoint):
    endpoint.level = 0.40
    result = system.set_volume(change=15)
    assert result.message == "Turned the volume up to 55%."
    assert system.set_volume(change=-80).data["level"] == 0
    assert system.set_volume(change=0).message == "Volume left unchanged."


def test_volume_requires_an_argument(endpoint):
    assert not system.set_volume().success


def test_falls_back_to_volume_keys(monkeypatch, keys):
    def broken(self):
        raise OSError("COM error")

    monkeypatch.setattr(VolumeController, "_endpoint", broken)
    result = system.set_volume(level=40)
    assert result.success and result.message == "Volume set to about 40%."
    assert keys == [(system.VK_VOLUME_DOWN, 50), (system.VK_VOLUME_UP, 20)]

    keys.clear()
    result = system.set_volume(change=-10)
    assert result.message == "Turned the volume down."
    assert keys == [(system.VK_VOLUME_DOWN, 5)]


def test_volume_off_windows_is_reported(registry, monkeypatch):
    monkeypatch.setattr(winutils, "IS_WINDOWS", False)
    result = registry.execute("set_volume", {"level": 20})
    assert not result.success and "only works on Windows" in result.message


# ----------------------------------------------------------------- system info
Mem = namedtuple("Mem", "percent used total")
Disk = namedtuple("Disk", "free total")
Battery = namedtuple("Battery", "percent power_plugged")
GB = 1024 ** 3


@pytest.fixture()
def fake_psutil(monkeypatch):
    monkeypatch.setattr(system.psutil, "cpu_percent", lambda interval=None: 12.0)
    monkeypatch.setattr(system.psutil, "virtual_memory", lambda: Mem(45.0, 7.2 * GB, 16 * GB))
    monkeypatch.setattr(system.psutil, "disk_usage", lambda _p: Disk(120 * GB, 476 * GB))
    monkeypatch.setattr(system.psutil, "sensors_battery", lambda: Battery(80.0, True))
    monkeypatch.setattr(system.psutil, "boot_time", lambda: system.time.time() - 3 * 3600 - 5 * 60 - 30)
    monkeypatch.setattr(winutils, "IS_WINDOWS", True)
    monkeypatch.setenv("SystemDrive", "C:")


def test_system_info_full(fake_psutil):
    result = system.get_system_info()
    assert result.success
    assert result.message == (
        "CPU is at 12%; RAM is 45% used (7.2 GB of 16.0 GB); battery is at 80% and charging; "
        "C: drive has 120.0 GB free of 476.0 GB; the PC has been on for 3 hours 5 minutes."
    )
    assert result.data["battery_percent"] == 80 and result.data["plugged_in"] is True


def test_system_info_desktop_without_battery(fake_psutil, monkeypatch):
    monkeypatch.setattr(system.psutil, "sensors_battery", lambda: None)
    result = system.get_system_info()
    assert result.success and "battery" not in result.message
    assert result.data["battery_percent"] is None


def test_system_info_partial_failure(fake_psutil, monkeypatch):
    def broken(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr(system.psutil, "disk_usage", broken)
    monkeypatch.setattr(system.psutil, "sensors_battery", broken)
    result = system.get_system_info()
    assert result.success and "CPU is at 12%" in result.message and "drive" not in result.message


def test_system_info_total_failure(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("no sensors")

    for name in ("cpu_percent", "virtual_memory", "disk_usage", "sensors_battery", "boot_time"):
        monkeypatch.setattr(system.psutil, name, broken)
    assert not system.get_system_info().success


def test_system_info_real_psutil_smoke():
    """Runs against the real machine: must succeed on any OS."""
    assert system.get_system_info().success


@pytest.mark.parametrize(
    "seconds, text",
    [(0, "0 minutes"), (60, "1 minute"), (3660, "1 hour 1 minute"), (90000, "1 day 1 hour"),
     (2 * 86400 + 120, "2 days 2 minutes")],
)
def test_format_duration(seconds, text):
    assert _format_duration(seconds) == text

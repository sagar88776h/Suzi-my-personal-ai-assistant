"""Tests for app discovery, fuzzy matching and the open_app tool."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import apps, winutils
from tools.apps import AppEntry, AppIndex, normalize_name


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


@pytest.fixture()
def fake_pc(tmp_path):
    """A fake Start Menu and Program Files tree."""
    start = tmp_path / "Start Menu" / "Programs"
    _touch(start / "Google Chrome.lnk")
    _touch(start / "Visual Studio Code" / "Visual Studio Code.lnk")
    _touch(start / "VLC" / "VLC media player.lnk")
    _touch(start / "VLC" / "Uninstall VLC.lnk")
    _touch(start / "Spotify.url")
    _touch(start / "Microsoft Word.lnk")
    program_files = tmp_path / "Program Files"
    _touch(program_files / "Audacity" / "Audacity.exe")
    _touch(program_files / "Audacity" / "unins000.exe")
    _touch(program_files / "Google" / "Chrome" / "Application" / "chrome.exe")  # too deep
    _touch(program_files / "Tool" / "readme.txt")
    return AppIndex(start_menu_dirs=[start], program_dirs=[program_files], include_start_apps=False)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Google Chrome", "google chrome"),
        ("the Chrome app", "chrome"),
        ("notepad.exe", "notepad"),
        ("VS_Code", "vs code"),
        ("  Spotify!  ", "spotify"),
        ("my calculator application", "calculator"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_index_scans_sources_and_skips_noise(fake_pc):
    names = {e.name for e in fake_pc.build()}
    assert {"Google Chrome", "Visual Studio Code", "VLC media player", "Spotify", "Audacity"} <= names
    assert "Uninstall VLC" not in names
    assert "unins000" not in names
    assert "chrome" not in names  # deeper than two levels in Program Files
    assert "Notepad" in names     # built-in Windows program


def test_shortcut_beats_builtin_for_same_name(tmp_path):
    start = tmp_path / "Programs"
    _touch(start / "Notepad.lnk")
    index = AppIndex(start_menu_dirs=[start], program_dirs=[], include_start_apps=False)
    entry = index.find("notepad").entry
    assert entry.kind == "shortcut"


@pytest.mark.parametrize(
    "query, expected",
    [
        ("chrome", "Google Chrome"),
        ("google chrome", "Google Chrome"),
        ("vs code", "Visual Studio Code"),
        ("visual studio", "Visual Studio Code"),
        ("vlc", "VLC media player"),
        ("spotify", "Spotify"),
        ("audacity", "Audacity"),
        ("word", "Microsoft Word"),
        ("calculator", "Calculator"),
        ("task manager", "Task Manager"),
    ],
)
def test_find(fake_pc, query, expected):
    match = fake_pc.find(query)
    assert match.entry is not None, f"{query} -> {match}"
    assert match.entry.name == expected


def test_find_suggests_close_names(fake_pc):
    fake_pc.fuzzy_threshold = 95
    match = fake_pc.find("spotfy")
    assert match.entry is None
    assert match.suggestion is not None and match.suggestion.name == "Spotify"


def test_find_nothing(fake_pc):
    match = fake_pc.find("zzqqxx")
    assert match.entry is None and match.suggestion is None
    assert fake_pc.find("   ").entry is None


def test_start_apps_parsing(monkeypatch):
    payload = [{"Name": "WhatsApp", "AppID": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"},
               {"Name": "Uninstall Foo", "AppID": "foo"}]
    completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(winutils, "run_powershell", lambda *a, **k: completed)
    index = AppIndex(start_menu_dirs=[], program_dirs=[], include_start_apps=True, include_builtins=False)
    entries = index.build()
    assert entries == [AppEntry("WhatsApp", "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App", "startapp")]


def test_start_apps_single_object_and_failures(monkeypatch):
    single = subprocess.CompletedProcess([], 0, stdout='{"Name": "Calculator", "AppID": "calc!App"}', stderr="")
    monkeypatch.setattr(winutils, "run_powershell", lambda *a, **k: single)
    index = AppIndex(start_menu_dirs=[], program_dirs=[], include_start_apps=True, include_builtins=False)
    assert [e.name for e in index.build()] == ["Calculator"]

    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired("powershell", 20)

    monkeypatch.setattr(winutils, "run_powershell", timeout)
    assert index.build() == []

    broken = subprocess.CompletedProcess([], 0, stdout="not json", stderr="")
    monkeypatch.setattr(winutils, "run_powershell", lambda *a, **k: broken)
    assert index.build() == []


# ----------------------------------------------------------------- open_app tool
@pytest.fixture()
def launched(monkeypatch, fake_pc):
    """Swap in the fake index and record launches instead of starting programs."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(apps, "APP_INDEX", fake_pc)
    monkeypatch.setattr(winutils, "startfile", lambda target: calls.append(("startfile", target)))
    monkeypatch.setattr(winutils, "launch_shell_app", lambda app_id: calls.append(("shell", app_id)))
    return calls


def test_open_app_launches_shortcut(launched):
    result = apps.open_app("chrome")
    assert result.success and result.message == "Opening Google Chrome."
    assert launched == [("startfile", str(apps.APP_INDEX.find("chrome").entry.target))]


def test_open_app_start_app_uses_shell(launched, monkeypatch):
    entry = AppEntry("WhatsApp", "WhatsApp!App", "startapp")
    monkeypatch.setattr(apps.APP_INDEX, "entries", lambda force_refresh=False: [entry])
    assert apps.open_app("whatsapp").success
    assert launched == [("shell", "WhatsApp!App")]


def test_open_app_did_you_mean(launched):
    apps.APP_INDEX.fuzzy_threshold = 95
    result = apps.open_app("spotfy")
    assert not result.success
    assert result.message == "I couldn't find an app called spotfy. Did you mean Spotify?"
    assert launched == []


def test_open_app_not_found(launched):
    result = apps.open_app("zzqqxx")
    assert not result.success and "couldn't find an app called zzqqxx" in result.message


def test_open_app_refreshes_stale_index(launched, monkeypatch, tmp_path):
    apps.APP_INDEX.build()
    # Pretend the cache is old, then "install" a new app.
    monkeypatch.setattr(type(apps.APP_INDEX), "age_seconds", property(lambda self: 120.0))
    _touch(apps.APP_INDEX.start_menu_dirs[0] / "Discord.lnk")
    assert apps.open_app("discord").success


def test_open_app_launch_error(launched, monkeypatch):
    def fail(_target):
        raise OSError(2, "The system cannot find the file specified")

    monkeypatch.setattr(winutils, "startfile", fail)
    result = apps.open_app("vlc")
    assert not result.success
    assert "wouldn't start it" in result.message


def test_open_app_empty_name(launched):
    assert not apps.open_app("  ").success


def test_open_app_via_registry_off_windows(registry, monkeypatch, fake_pc):
    """On non-Windows the real launcher refuses politely instead of crashing."""
    monkeypatch.setattr(apps, "APP_INDEX", fake_pc)
    monkeypatch.setattr(winutils, "IS_WINDOWS", False)
    result = registry.execute("open_app", {"name": "chrome"})
    assert not result.success and "only works on Windows" in result.message


def test_configure_adds_extra_dirs(monkeypatch, tmp_path):
    index = AppIndex(start_menu_dirs=[], program_dirs=[], include_start_apps=False)
    monkeypatch.setattr(apps, "APP_INDEX", index)
    apps.configure(fuzzy_threshold=85, suggest_threshold=55, extra_search_dirs=[str(tmp_path)])
    assert index.fuzzy_threshold == 85 and index.suggest_threshold == 55
    assert tmp_path in index.program_dirs

"""Tests for the web tools (browser mocked)."""

from __future__ import annotations

import webbrowser

import pytest

from tools import web


@pytest.fixture()
def opened(monkeypatch):
    urls: list[str] = []

    def fake_open(url, new=0):
        urls.append(url)
        return True

    monkeypatch.setattr(web.webbrowser, "open", fake_open)
    return urls


def test_google_search_opens_encoded_url(opened):
    result = web.google_search("weather in Delhi & Mumbai")
    assert result.success and result.message == "Searching Google for weather in Delhi & Mumbai."
    assert opened == ["https://www.google.com/search?q=weather+in+Delhi+%26+Mumbai"]


def test_google_search_hinglish_unicode(opened):
    assert web.google_search("आज का मौसम").success
    assert opened[0].startswith("https://www.google.com/search?q=%E0%A4")


def test_google_search_empty(opened):
    assert not web.google_search("   ").success
    assert opened == []


def test_google_search_no_browser(monkeypatch):
    monkeypatch.setattr(web.webbrowser, "open", lambda url, new=0: False)
    assert not web.google_search("python").success

    def broken(url, new=0):
        raise webbrowser.Error("no browser")

    monkeypatch.setattr(web.webbrowser, "open", broken)
    result = web.google_search("python")
    assert not result.success and "browser" in result.message

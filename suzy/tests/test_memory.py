"""Unit tests for Suzy's memory store and tools."""

from __future__ import annotations

import pytest

from core.memory import MemorySecurityError, MemoryStore
from tools import memory_tools


@pytest.fixture
def temp_memory_store(tmp_path):
    return MemoryStore(db_path=tmp_path / "test_memory.db")


def test_remember_and_recall(temp_memory_store):
    rec = temp_memory_store.remember("Prefers dark mode in all apps", category="preference", key="theme")
    assert rec.id > 0
    assert rec.category == "preference"

    # Search
    results = temp_memory_store.recall("dark mode")
    assert len(results) == 1
    assert "dark mode" in results[0].content

    # Update with same key
    updated = temp_memory_store.remember("Prefers deep space dark mode", category="preference", key="theme")
    assert updated.id == rec.id
    assert "deep space" in updated.content


def test_memory_security_rejection(temp_memory_store):
    # Passwords must be rejected
    with pytest.raises(MemorySecurityError):
        temp_memory_store.remember("My password is SuperSecret123!")

    # API keys must be rejected
    with pytest.raises(MemorySecurityError):
        temp_memory_store.remember("My Groq key is gsk_abcdefghijklmnopqrstuvwxyz123456")

    with pytest.raises(MemorySecurityError):
        temp_memory_store.remember("Here is the api_key: AIzaSyD1234567890abcdefghijklmnopqrstu")


def test_forget_and_list(temp_memory_store):
    m1 = temp_memory_store.remember("Fact one", category="fact")
    m2 = temp_memory_store.remember("Fact two", category="fact")

    all_mems = temp_memory_store.list_all(active_only=True)
    assert len(all_mems) == 2

    assert temp_memory_store.forget(m1.id)
    active_mems = temp_memory_store.list_all(active_only=True)
    assert len(active_mems) == 1
    assert active_mems[0].id == m2.id


def test_memory_tools(tmp_path, monkeypatch):
    store = MemoryStore(db_path=tmp_path / "tools_mem.db")
    monkeypatch.setattr(memory_tools, "get_memory_store", lambda: store)

    res = memory_tools.remember_fact("Wife's birthday is June 4", category="fact")
    assert res.success
    assert "I've remembered that" in res.message

    # Test security failure in tool
    sec_res = memory_tools.remember_fact("password = password123")
    assert not sec_res.success
    assert "Security safeguard" in sec_res.message

    # Recall
    recall_res = memory_tools.recall_memories("birthday")
    assert recall_res.success
    assert "June 4" in recall_res.message

    # List
    list_res = memory_tools.list_saved_memories()
    assert list_res.success
    assert "1 saved memories" in list_res.message

    # Forget
    mid = recall_res.data["memories"][0]["id"]
    forget_res = memory_tools.forget_memory(mid)
    assert forget_res.success

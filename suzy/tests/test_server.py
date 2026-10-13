"""Integration tests for Suzy's web server API endpoints."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from core.assistant import Assistant, AssistantReply
from core.config import Config
from core.llm import LLMRouter
from tools.registry import ToolRegistry, ToolResult
from web_server import SuzyWebServer


class DummyRouter:
    providers = []
    configured_providers = []

    def status(self):
        return {}


class DummyRegistry:
    def __init__(self):
        self._executed = []

    def execute(self, name, args):
        self._executed.append((name, args))
        return ToolResult.ok(f"Executed {name}")


class DummyAssistant:
    def __init__(self):
        self.config = Config({})

    def process(self, text):
        return AssistantReply(f"Echo: {text}", [])


class TestSuzyWebServer(AioHTTPTestCase):
    async def get_application(self):
        self.assistant = DummyAssistant()
        self.router = DummyRouter()
        self.registry = DummyRegistry()
        self.config = Config({"user": {"name": "Sagar"}})
        self.server = SuzyWebServer(self.assistant, self.router, self.registry, self.config)
        return self.server.app

    async def test_status_endpoint(self):
        resp = await self.client.request("GET", "/api/status")
        assert resp.status == 200
        data = await resp.json()
        assert data["user_name"] == "Sagar"
        assert "mood" in data

    async def test_chat_endpoint(self):
        resp = await self.client.request(
            "POST",
            "/api/chat",
            json={"message": "hello", "mood": "natural"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert "Echo: hello" in data["text"]
        assert "latency_ms" in data

    async def test_mood_endpoints(self):
        resp = await self.client.request(
            "POST",
            "/api/mood",
            json={"mood": "focus"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["mood"] == "focus"

        get_resp = await self.client.request("GET", "/api/mood")
        assert get_resp.status == 200
        get_data = await get_resp.json()
        assert get_data["mood"] == "focus"

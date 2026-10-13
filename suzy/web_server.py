"""Asynchronous Web & API server for Suzy's Siri-grade UI.

Provides:
  - Static file serving for the Siri-style frontend
  - REST API for chat, email, memory, moods, and settings
  - Audio streaming endpoint for Edge-TTS synthesis
  - Real-time WebSocket connection for streaming events and orb states
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from aiohttp import web, WSMsgType

from core.assistant import Assistant, AssistantReply
from core.config import Config
from core.email_service import get_email_service
from core.llm import LLMRouter
from core.memory import get_memory_store
from core.moods import Mood, get_current_mood, get_mood_instructions, set_current_mood
from core.paths import APP_DIR
from core.permissions import get_permission_manager
from tools.registry import ToolRegistry, load_all_tools

log = logging.getLogger("suzy.server")

STATIC_DIR = Path(__file__).parent / "ui"


class SuzyWebServer:
    def __init__(self, assistant: Assistant, router: LLMRouter, registry: ToolRegistry,
                 config: Config, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.assistant = assistant
        self.router = router
        self.registry = registry
        self.config = config
        self.host = host
        self.port = port
        self.app = web.Application()
        self.ws_clients: set[web.WebSocketResponse] = set()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/ws", self.handle_websocket)
        self.app.router.add_post("/api/chat", self.handle_chat)
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/email", self.handle_email_latest)
        self.app.router.add_get("/api/email/search", self.handle_email_search)
        self.app.router.add_post("/api/email/draft", self.handle_email_draft)
        self.app.router.add_get("/api/memory", self.handle_memory_list)
        self.app.router.add_post("/api/memory", self.handle_memory_add)
        self.app.router.add_delete("/api/memory/{id}", self.handle_memory_delete)
        self.app.router.add_post("/api/mood", self.handle_mood_set)
        self.app.router.add_get("/api/mood", self.handle_mood_get)
        self.app.router.add_post("/api/tts", self.handle_tts)
        self.app.router.add_post("/api/action/confirm", self.handle_action_confirm)

        if STATIC_DIR.exists():
            self.app.router.add_static("/static/", path=STATIC_DIR, name="static")

    async def broadcast_ws(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast state event to all connected WebSocket clients."""
        payload = json.dumps({"type": event_type, "data": data})
        stale = set()
        for ws in self.ws_clients:
            try:
                await ws.send_str(payload)
            except Exception:
                stale.add(ws)
        self.ws_clients.difference_update(stale)

    async def handle_index(self, request: web.Request) -> web.Response:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return web.Response(text="Suzy UI files not found in ui/.", status=404)
        return web.FileResponse(index_file)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)
        log.info("WebSocket client connected (%d active)", len(self.ws_clients))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        req_data = json.loads(msg.data)
                        action = req_data.get("action")
                        if action == "state":
                            # Client sending state ping (listening / speaking / idle)
                            state = req_data.get("state", "idle")
                            await self.broadcast_ws("state_change", {"state": state})
                    except Exception as err:
                        log.debug("WS message parsing error: %s", err)
                elif msg.type == WSMsgType.ERROR:
                    log.debug("WS connection closed with exception: %s", ws.exception())
        finally:
            self.ws_clients.discard(ws)
            log.info("WebSocket client disconnected (%d remaining)", len(self.ws_clients))
        return ws

    async def handle_chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        user_text = (body.get("message") or "").strip()
        mood = body.get("mood")
        if mood:
            set_current_mood(mood)

        if not user_text:
            return web.json_response({"error": "Empty message"}, status=400)

        await self.broadcast_ws("state_change", {"state": "thinking"})
        t0 = time.perf_counter()

        # Run assistant processing in background thread
        loop = asyncio.get_running_loop()
        reply: AssistantReply = await loop.run_in_executor(None, self.assistant.process, user_text)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        actions_data = []
        for a in reply.actions:
            actions_data.append({
                "tool": a.tool,
                "arguments": a.arguments,
                "success": a.result.success,
                "message": a.result.message,
                "data": a.result.data,
            })

        await self.broadcast_ws("state_change", {"state": "speaking"})

        # Check if a pending action needs confirmation
        perm_mgr = get_permission_manager()
        pending = None
        if perm_mgr.has_pending:
            p = perm_mgr.pending_action
            if p:
                pending = {
                    "id": p.id,
                    "tool": p.tool_name,
                    "arguments": p.arguments,
                    "summary": p.summary,
                }

        response_payload = {
            "text": reply.text,
            "actions": actions_data,
            "provider": reply.provider,
            "error": reply.error,
            "detail": reply.detail,
            "latency_ms": elapsed_ms,
            "mood": get_current_mood().value,
            "pending_action": pending,
        }
        return web.json_response(response_payload)

    async def handle_status(self, request: web.Request) -> web.Response:
        email_svc = get_email_service()
        mem_store = get_memory_store()
        mem_count = len(mem_store.list_all(active_only=True))

        providers = [p.name for p in self.router.providers]
        configured = self.router.configured_providers

        return web.json_response({
            "assistant_name": self.config.get("assistant.name", "Suzy"),
            "user_name": self.config.get("user.name", "Sagar"),
            "mood": get_current_mood().value,
            "brains": {
                "configured": configured,
                "all": providers,
                "primary": self.config.get("llm.provider", "groq"),
            },
            "email": email_svc.status(),
            "memories_count": mem_count,
            "tts_voice": self.config.get("tts.voice", "en-IN-NeerjaExpressiveNeural"),
        })

    async def handle_email_latest(self, request: web.Request) -> web.Response:
        email_svc = get_email_service()
        limit = int(request.query.get("limit", 5))
        unread_only = request.query.get("unread", "false").lower() == "true"
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(
            None, email_svc.get_latest_emails, limit, unread_only
        )
        return web.json_response({
            "messages": [m.to_dict() for m in messages],
            "status": email_svc.status(),
        })

    async def handle_email_search(self, request: web.Request) -> web.Response:
        email_svc = get_email_service()
        q = request.query.get("q", "")
        limit = int(request.query.get("limit", 5))
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, email_svc.search_emails, q, "", "", limit
        )
        return web.json_response({
            "results": [m.to_dict() for m in results],
            "query": q,
        })

    async def handle_email_draft(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        email_id = body.get("email_id")
        reply_body = body.get("reply_body", "")
        to = body.get("to", "")
        subject = body.get("subject", "Reply")

        email_svc = get_email_service()
        loop = asyncio.get_running_loop()
        draft = await loop.run_in_executor(
            None, email_svc.draft_reply, to, subject, reply_body, email_id
        )
        return web.json_response(draft)

    async def handle_memory_list(self, request: web.Request) -> web.Response:
        store = get_memory_store()
        mems = store.list_all(active_only=True)
        return web.json_response({"memories": [m.to_dict() for m in mems]})

    async def handle_memory_add(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        fact = (body.get("fact") or "").strip()
        cat = body.get("category", "fact")
        if not fact:
            return web.json_response({"error": "Memory text cannot be empty"}, status=400)

        store = get_memory_store()
        try:
            rec = store.remember(fact, category=cat)
            return web.json_response({"success": True, "memory": rec.to_dict()})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def handle_memory_delete(self, request: web.Request) -> web.Response:
        mid_str = request.match_info.get("id")
        try:
            mid = int(mid_str or "0")
        except ValueError:
            return web.json_response({"error": "Invalid ID"}, status=400)

        store = get_memory_store()
        success = store.forget(mid)
        return web.json_response({"success": success})

    async def handle_mood_set(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        mood = body.get("mood", "natural")
        new_mood = set_current_mood(mood)
        return web.json_response({"mood": new_mood.value, "instructions": get_mood_instructions(new_mood)})

    async def handle_mood_get(self, request: web.Request) -> web.Response:
        current = get_current_mood()
        return web.json_response({"mood": current.value, "instructions": get_mood_instructions(current)})

    async def handle_tts(self, request: web.Request) -> web.Response:
        """Synthesize text via Edge-TTS and stream MP3 bytes directly to client."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        text = (body.get("text") or "").strip()
        voice = body.get("voice") or self.config.get("tts.voice", "en-IN-NeerjaExpressiveNeural")
        rate = body.get("rate") or "+0%"

        if not text:
            return web.json_response({"error": "Empty text"}, status=400)

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            audio_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    audio_bytes.extend(chunk["data"])

            return web.Response(body=bytes(audio_bytes), content_type="audio/mpeg")
        except Exception as exc:
            log.warning("Edge-TTS streaming synthesis error: %s", exc)
            return web.json_response({"error": f"TTS synthesis failed: {exc}"}, status=500)

    async def handle_action_confirm(self, request: web.Request) -> web.Response:
        """Handle Level 3 user confirmation."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        confirmed = bool(body.get("confirmed", False))
        perm_mgr = get_permission_manager()
        pending = perm_mgr.pop_pending() if confirmed else perm_mgr.cancel_pending()

        if not pending:
            return web.json_response({"error": "No pending action found"}, status=404)

        if not confirmed:
            return web.json_response({
                "status": "cancelled",
                "message": f"Cancelled '{pending.summary}'. Nothing was changed.",
            })

        # Execute the confirmed tool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.registry.execute, pending.tool_name, pending.arguments
        )
        return web.json_response({
            "status": "executed",
            "success": result.success,
            "message": result.message,
            "data": result.data,
        })

    def run(self) -> None:
        """Start the web server."""
        print(f"\n✨ Suzy Siri Interface running at: http://{self.host}:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port, print=None)

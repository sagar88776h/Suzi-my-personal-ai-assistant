"""Web tools: searching and opening pages in the default browser."""

from __future__ import annotations

import logging
import webbrowser
from urllib.parse import quote_plus

from tools.registry import ToolResult, tool

log = logging.getLogger("suzy.tools.web")

GOOGLE_SEARCH_URL = "https://www.google.com/search?q={query}"


def open_in_browser(url: str) -> bool:
    """Open ``url`` in the default browser. Returns False if no browser could be used."""
    try:
        return bool(webbrowser.open(url, new=2))
    except webbrowser.Error as exc:
        log.warning("Browser failed to open %s: %s", url, exc)
        return False


@tool(
    description="Search Google for something and show the results in the default web browser.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
    category="web",
)
def google_search(query: str) -> ToolResult:
    query = query.strip()
    if not query:
        return ToolResult.fail("What should I search for?")
    url = GOOGLE_SEARCH_URL.format(query=quote_plus(query))
    if not open_in_browser(url):
        return ToolResult.fail("I couldn't open your web browser.")
    return ToolResult.ok(f"Searching Google for {query}.", url=url)

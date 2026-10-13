"""Tools for managing personal memory and preferences."""

from __future__ import annotations

from core.memory import MemorySecurityError, get_memory_store
from tools.registry import ToolResult, tool
from tools.safety import SafetyLevel


@tool(
    description="Explicitly remember a fact, user preference, project detail, or work context for future conversations.",
    parameters={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact, preference, or detail to remember (e.g. 'Prefers dark mode', 'Working on website redesign').",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "preference", "project", "work_context"],
                "description": "Category for the memory. Defaults to 'fact'.",
            },
        },
        "required": ["fact"],
    },
    safety=SafetyLevel.SAFE,
    category="memory",
)
def remember_fact(fact: str, category: str = "fact") -> ToolResult:
    store = get_memory_store()
    try:
        rec = store.remember(fact, category=category)
        return ToolResult.ok(
            f"I've remembered that: {rec.content}",
            memory_id=rec.id,
            category=rec.category,
        )
    except MemorySecurityError as sec_err:
        return ToolResult.fail(str(sec_err))
    except Exception as exc:
        return ToolResult.fail(f"Couldn't save memory: {exc}")


@tool(
    description="Search or retrieve stored personal memories, preferences, or ongoing project context.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword to search for in memories, or empty string to retrieve recent memories.",
            },
            "category": {
                "type": "string",
                "description": "Filter by category ('fact', 'preference', 'project', 'work_context', or 'all').",
            },
        },
    },
    safety=SafetyLevel.SAFE,
    category="memory",
)
def recall_memories(query: str = "", category: str = "all") -> ToolResult:
    store = get_memory_store()
    try:
        results = store.recall(query=query, category=category, limit=8)
        if not results:
            return ToolResult.ok(
                "No saved memories found matching that query.",
                count=0,
                memories=[],
            )
        summaries = [f"#{m.id} [{m.category}]: {m.content}" for m in results]
        return ToolResult.ok(
            "Found memories:\n" + "\n".join(summaries),
            count=len(results),
            memories=[m.to_dict() for m in results],
        )
    except Exception as exc:
        return ToolResult.fail(f"Couldn't recall memories: {exc}")


@tool(
    description="Forget or delete a previously saved memory by its ID number.",
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "The ID number of the memory to forget.",
            },
        },
        "required": ["memory_id"],
    },
    safety=SafetyLevel.CONFIRM,
    category="memory",
)
def forget_memory(memory_id: int) -> ToolResult:
    store = get_memory_store()
    try:
        success = store.forget(memory_id)
        if success:
            return ToolResult.ok(f"I've forgotten memory #{memory_id}.", memory_id=memory_id)
        return ToolResult.fail(f"Memory #{memory_id} was not found.")
    except Exception as exc:
        return ToolResult.fail(f"Couldn't forget memory: {exc}")


@tool(
    description="List all active saved memories, preferences, and project facts.",
    safety=SafetyLevel.SAFE,
    category="memory",
)
def list_saved_memories() -> ToolResult:
    store = get_memory_store()
    try:
        memories = store.list_all(active_only=True)
        if not memories:
            return ToolResult.ok("No memories saved yet.", count=0, memories=[])
        lines = [f"#{m.id} [{m.category}]: {m.content}" for m in memories]
        return ToolResult.ok(
            f"You have {len(memories)} saved memories:\n" + "\n".join(lines),
            count=len(memories),
            memories=[m.to_dict() for m in memories],
        )
    except Exception as exc:
        return ToolResult.fail(f"Couldn't list memories: {exc}")

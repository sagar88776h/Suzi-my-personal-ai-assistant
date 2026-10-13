"""Email tools for Suzy.

Allows Suzy to:
  - Check email connection status
  - Retrieve and search recent emails
  - Read specific email threads with attachments
  - Summarize email communications with grounded citations
  - Prepare draft replies (Level 2: Prepare) with explicit review
"""

from __future__ import annotations

import logging
from datetime import datetime

from core.email_service import EmailMessage, get_email_service
from tools.registry import ToolResult, tool
from tools.safety import ExecutionLevel, SafetyLevel

log = logging.getLogger("suzy.tools.email")


def _no_email_account_message() -> str:
    return (
        "I'm not connected to an email account yet. "
        "You can connect Outlook on this PC, or add your email credentials to the .env file "
        "(GMAIL_USER and GMAIL_APP_PASSWORD, or IMAP_HOST, IMAP_USER, IMAP_PASSWORD). "
        "Once configured, I can search, summarize, and draft replies for your emails."
    )


@tool(
    description="Check whether an email provider (Outlook or IMAP/Gmail) is connected and authorized.",
    safety=SafetyLevel.SAFE,
    category="email",
)
def check_email_connection() -> ToolResult:
    service = get_email_service()
    st = service.status()
    if st["configured"]:
        active = st["active_provider"]
        return ToolResult.ok(
            f"Email connected via {active.upper()}.",
            **st,
        )
    return ToolResult.ok(
        _no_email_account_message(),
        **st,
    )


@tool(
    description="Retrieve the latest incoming emails from the user's inbox.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of emails to retrieve (default: 5, max: 10).",
            },
            "unread_only": {
                "type": "boolean",
                "description": "Whether to only fetch unread emails.",
            },
            "folder": {
                "type": "string",
                "description": "Folder name (default: 'INBOX').",
            },
        },
    },
    safety=SafetyLevel.SAFE,
    category="email",
)
def get_latest_emails(limit: int = 5, unread_only: bool = False, folder: str = "INBOX") -> ToolResult:
    service = get_email_service()
    limit = max(1, min(limit, 10))
    messages = service.get_latest_emails(limit=limit, unread_only=unread_only, folder=folder)

    if not messages:
        st = service.status()
        if not st["configured"]:
            return ToolResult.ok(_no_email_account_message(), count=0, messages=[])
        qualifier = "unread " if unread_only else ""
        return ToolResult.ok(f"No {qualifier}emails found in {folder}.", count=0, messages=[])

    lines = []
    for m in messages:
        att = " [Attachments: " + ", ".join(m.attachment_names) + "]" if m.has_attachments else ""
        lines.append(f"• {m.format_citation()}{att}\n  Preview: {m.snippet}")

    summary = f"Found {len(messages)} latest emails:\n" + "\n".join(lines)
    return ToolResult.ok(
        summary,
        count=len(messages),
        messages=[m.to_dict() for m in messages],
    )


@tool(
    description="Search emails by keyword, sender, or subject.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text or keyword to search for in message subject and body (e.g. 'invoice', 'website').",
            },
            "sender": {
                "type": "string",
                "description": "Filter by sender name or email address.",
            },
            "subject": {
                "type": "string",
                "description": "Filter by subject line keywords.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
            },
        },
    },
    safety=SafetyLevel.SAFE,
    category="email",
)
def search_emails(query: str = "", sender: str = "", subject: str = "", limit: int = 5) -> ToolResult:
    service = get_email_service()
    limit = max(1, min(limit, 10))
    messages = service.search_emails(query=query, sender=sender, subject=subject, limit=limit)

    if not messages:
        st = service.status()
        if not st["configured"]:
            return ToolResult.ok(_no_email_account_message(), count=0, results=[])
        desc = f"'{query}'" if query else (f"from '{sender}'" if sender else f"subject '{subject}'")
        return ToolResult.ok(f"No emails found matching {desc}.", count=0, results=[])

    lines = []
    for m in messages:
        att = " [Attachments: " + ", ".join(m.attachment_names) + "]" if m.has_attachments else ""
        lines.append(f"• {m.format_citation()}{att}\n  Excerpt: {m.snippet}")

    summary = f"Found {len(messages)} matching email(s):\n" + "\n".join(lines)
    return ToolResult.ok(
        summary,
        count=len(messages),
        results=[m.to_dict() for m in messages],
    )


@tool(
    description="Read the complete body and full thread details of a specific email by its ID.",
    parameters={
        "type": "object",
        "properties": {
            "email_id": {
                "type": "string",
                "description": "The unique ID of the email to inspect.",
            },
        },
        "required": ["email_id"],
    },
    safety=SafetyLevel.SAFE,
    category="email",
)
def read_email_detail(email_id: str) -> ToolResult:
    service = get_email_service()
    msg = service.get_email_detail(email_id)

    if not msg:
        return ToolResult.fail(f"Could not locate email with ID '{email_id}'.")

    att_info = f"\nAttachments: {', '.join(msg.attachment_names)}" if msg.has_attachments else ""
    detail = (
        f"Subject: {msg.subject}\n"
        f"From: {msg.sender_name} <{msg.sender_email}>\n"
        f"Date: {msg.date}{att_info}\n\n"
        f"Body:\n{msg.body}"
    )
    return ToolResult.ok(detail, email=msg.to_dict())


@tool(
    description="Summarize recent or unread emails, highlighting key sender requests, invoices, and deadlines.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search term to filter emails before summarizing (e.g. 'project', 'client').",
            },
            "unread_only": {
                "type": "boolean",
                "description": "Whether to only summarize unread emails.",
            },
            "today_only": {
                "type": "boolean",
                "description": "Whether to only summarize emails received today.",
            },
        },
    },
    safety=SafetyLevel.SAFE,
    category="email",
)
def summarize_emails(query: str = "", unread_only: bool = False, today_only: bool = False) -> ToolResult:
    service = get_email_service()
    if query:
        messages = service.search_emails(query=query, limit=5)
    else:
        messages = service.get_latest_emails(limit=5, unread_only=unread_only)

    if not messages:
        st = service.status()
        if not st["configured"]:
            return ToolResult.ok(_no_email_account_message(), count=0)
        return ToolResult.ok("No emails found to summarize.", count=0)

    # Filter for today if requested
    if today_only:
        today_str = datetime.now().strftime("%Y-%m-%d")
        messages = [m for m in messages if today_str in m.date]
        if not messages:
            return ToolResult.ok("No emails received today were found.", count=0)

    summary_items = []
    for m in messages:
        summary_items.append(
            f"1. {m.format_citation()}\n"
            f"   Key content: {m.snippet}\n"
            f"   Attachments: {', '.join(m.attachment_names) if m.has_attachments else 'None'}"
        )

    out = f"Summary of {len(messages)} email(s):\n" + "\n\n".join(summary_items)
    return ToolResult.ok(out, count=len(messages), emails=[m.to_dict() for m in messages])


@tool(
    description="Draft a response to an email. Creates a draft (Level 2: Prepare) for user review without sending.",
    parameters={
        "type": "object",
        "properties": {
            "email_id": {
                "type": "string",
                "description": "The ID of the email to reply to.",
            },
            "reply_body": {
                "type": "string",
                "description": "The drafted response text.",
            },
        },
        "required": ["email_id", "reply_body"],
    },
    safety=SafetyLevel.PREPARE,
    category="email",
)
def draft_email_reply(email_id: str, reply_body: str) -> ToolResult:
    service = get_email_service()
    original = service.get_email_detail(email_id)
    if not original:
        return ToolResult.fail(f"Could not find email with ID '{email_id}' to draft a reply.")

    recipient = original.sender_email or original.sender_name
    subj = f"Re: {original.subject}" if not original.subject.lower().startswith("re:") else original.subject

    draft_info = service.draft_reply(to=recipient, subject=subj, body=reply_body, email_id=email_id)

    msg = (
        f"I've drafted a reply for your review (not sent):\n"
        f"To: {recipient}\n"
        f"Subject: {subj}\n"
        f"Body:\n{reply_body}\n\n"
        f"(Saved in Drafts: {draft_info['saved_in_drafts']})"
    )
    return ToolResult.ok(msg, draft=draft_info)

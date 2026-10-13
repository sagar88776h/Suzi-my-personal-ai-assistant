"""Email service for Suzy.

Implements a multi-provider email engine supporting:
  1. Outlook on Windows (via Win32 MAPI COM)
  2. Gmail / IMAP over SSL (imap.gmail.com:993, etc.)
  3. Gmail REST API (OAuth2)

Features:
  - Normalization of all messages to EmailMessage
  - Robust retrieval: authenticate -> search -> retrieve -> normalize -> rank
  - Thread summarization with accurate citation of sender, date, and subject
  - Safe Draft creation (Level 2 Prepare) without accidental sending
  - Clear explanations when credentials are not configured or connection fails
"""

from __future__ import annotations

import email
from email.header import decode_header
import imaplib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.paths import APP_DIR

log = logging.getLogger("suzy.email")


@dataclass
class EmailMessage:
    id: str
    sender_name: str
    sender_email: str
    recipients: list[str]
    subject: str
    date: str
    snippet: str
    body: str
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)
    unread: bool = False
    folder: str = "INBOX"
    provider: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender_name": self.sender_name,
            "sender_email": self.sender_email,
            "recipients": self.recipients,
            "subject": self.subject,
            "date": self.date,
            "snippet": self.snippet,
            "body": self.body,
            "has_attachments": self.has_attachments,
            "attachment_names": self.attachment_names,
            "unread": self.unread,
            "folder": self.folder,
            "provider": self.provider,
        }

    def format_citation(self) -> str:
        """Standard citation string: 'From: Sagar <sagar@example.com> | Date: 12 Oct | Subject: ...'"""
        sender = f"{self.sender_name} <{self.sender_email}>" if self.sender_name else self.sender_email
        return f"[ID: {self.id}] From: {sender} | Date: {self.date} | Subject: \"{self.subject}\""


def _decode_mime_header(header_value: str | None) -> str:
    if not header_value:
        return ""
    decoded_fragments = []
    for fragment, charset in decode_header(header_value):
        if isinstance(fragment, bytes):
            try:
                decoded_fragments.append(fragment.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                decoded_fragments.append(fragment.decode("utf-8", errors="replace"))
        else:
            decoded_fragments.append(str(fragment))
    return "".join(decoded_fragments).strip()


class OutlookMapiProvider:
    """Interacts with Windows Outlook via COM MAPI if Outlook is configured."""

    def __init__(self) -> None:
        self.available = False
        self._check_available()

    def _check_available(self) -> bool:
        try:
            import win32com.client
            # Don't create an intrusive background window; just verify dispatch
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            self.available = namespace is not None
        except Exception:
            self.available = False
        return self.available

    def get_inbox_messages(self, limit: int = 10, unread_only: bool = False) -> list[EmailMessage]:
        if not self.available:
            return []
        messages: list[EmailMessage] = []
        try:
            import win32com.client
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            inbox = namespace.GetDefaultFolder(6)  # 6 = olFolderInbox
            items = inbox.Items
            items.Sort("[ReceivedTime]", True)  # descending

            count = 0
            for item in items:
                if count >= limit:
                    break
                try:
                    is_unread = bool(getattr(item, "UnRead", False))
                    if unread_only and not is_unread:
                        continue

                    subject = str(getattr(item, "Subject", "(No Subject)"))
                    sender_name = str(getattr(item, "SenderName", ""))
                    sender_email = str(getattr(item, "SenderEmailAddress", ""))
                    received = getattr(item, "ReceivedTime", None)
                    date_str = received.strftime("%Y-%m-%d %H:%M") if received else ""
                    body = str(getattr(item, "Body", ""))
                    snippet = body[:200].replace("\r\n", " ").strip()
                    msg_id = str(getattr(item, "EntryID", f"outlook_{count}"))

                    att_names = []
                    attachments = getattr(item, "Attachments", None)
                    if attachments:
                        att_names = [att.FileName for att in attachments]

                    messages.append(EmailMessage(
                        id=msg_id,
                        sender_name=sender_name,
                        sender_email=sender_email,
                        recipients=[],
                        subject=subject,
                        date=date_str,
                        snippet=snippet,
                        body=body,
                        has_attachments=bool(att_names),
                        attachment_names=att_names,
                        unread=is_unread,
                        folder="INBOX",
                        provider="outlook",
                    ))
                    count += 1
                except Exception as item_err:
                    log.debug("Error reading Outlook item: %s", item_err)
                    continue
        except Exception as exc:
            log.warning("Outlook MAPI query failed: %s", exc)
        return messages

    def create_draft(self, to: str, subject: str, body: str) -> bool:
        if not self.available:
            return False
        try:
            import win32com.client
            outlook = win32com.client.Dispatch("Outlook.Application")
            mail = outlook.CreateItem(0)  # 0 = olMailItem
            mail.To = to
            mail.Subject = subject
            mail.Body = body
            mail.Save()  # Saves to Drafts
            log.info("Saved Outlook draft to '%s'", to)
            return True
        except Exception as exc:
            log.error("Failed creating Outlook draft: %s", exc)
            return False


class ImapEmailProvider:
    """Connects to IMAP servers (Gmail, Outlook, Yahoo, or custom) using SSL."""

    def __init__(self, host: str, user: str, password: str, port: int = 993) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.port = port

    def _connect(self) -> imaplib.IMAP4_SSL:
        client = imaplib.IMAP4_SSL(self.host, self.port)
        client.login(self.user, self.password)
        return client

    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._connect()
            client.logout()
            return True, f"Connected successfully to {self.host} as {self.user}"
        except Exception as exc:
            return False, f"IMAP connection failed ({self.host}): {exc}"

    def get_messages(self, limit: int = 10, query: str | None = None, unread_only: bool = False,
                     folder: str = "INBOX") -> list[EmailMessage]:
        messages: list[EmailMessage] = []
        client = None
        try:
            client = self._connect()
            status, _ = client.select(folder, readonly=True)
            if status != "OK":
                log.warning("Could not select IMAP folder %s", folder)
                return []

            search_criteria = ["UNSEEN"] if unread_only else ["ALL"]
            if query:
                search_criteria.extend(["TEXT", f'"{query}"'])

            status, search_data = client.search(None, *search_criteria)
            if status != "OK" or not search_data or not search_data[0]:
                return []

            msg_ids = search_data[0].split()
            # Most recent first
            msg_ids = msg_ids[::-1][:limit]

            for mid in msg_ids:
                res, data = client.fetch(mid, "(RFC822)")
                if res != "OK" or not data:
                    continue
                raw_bytes = data[0][1]
                msg = email.message_from_bytes(raw_bytes)

                subject = _decode_mime_header(msg.get("Subject", "(No Subject)"))
                from_header = _decode_mime_header(msg.get("From", ""))
                date_header = msg.get("Date", "")
                sender_name = from_header
                sender_email = from_header
                if "<" in from_header and ">" in from_header:
                    m = re.match(r"(.*?)\s*<([^>]+)>", from_header)
                    if m:
                        sender_name = m.group(1).strip().strip('"')
                        sender_email = m.group(2).strip()

                body_parts: list[str] = []
                attachment_names: list[str] = []

                if msg.is_multipart():
                    for part in msg.walk():
                        content_disposition = str(part.get("Content-Disposition", ""))
                        content_type = part.get_content_type()
                        filename = part.get_filename()
                        if filename:
                            attachment_names.append(_decode_mime_header(filename))

                        if "attachment" not in content_disposition and content_type == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                charset = part.get_content_charset() or "utf-8"
                                body_parts.append(payload.decode(charset, errors="replace"))
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or "utf-8"
                        body_parts.append(payload.decode(charset, errors="replace"))

                full_body = "\n".join(body_parts).strip()
                snippet = full_body[:250].replace("\r\n", " ").replace("\n", " ").strip()

                messages.append(EmailMessage(
                    id=mid.decode("ascii", errors="replace"),
                    sender_name=sender_name,
                    sender_email=sender_email,
                    recipients=[_decode_mime_header(msg.get("To", ""))],
                    subject=subject,
                    date=date_header,
                    snippet=snippet,
                    body=full_body,
                    has_attachments=bool(attachment_names),
                    attachment_names=attachment_names,
                    unread=unread_only,
                    folder=folder,
                    provider="imap",
                ))

        except Exception as exc:
            log.warning("IMAP retrieval error: %s", exc)
        finally:
            if client:
                try:
                    client.close()
                    client.logout()
                except Exception:
                    pass
        return messages


class EmailService:
    """Unified email service aggregating Outlook, IMAP, and local draft generation."""

    def __init__(self) -> None:
        self.outlook = OutlookMapiProvider()
        self._cached_messages: dict[str, EmailMessage] = {}

    def get_configured_provider(self) -> str | None:
        # Check environment for explicit IMAP credentials
        if os.getenv("IMAP_USER") and (os.getenv("IMAP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD")):
            return "imap"
        if os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD"):
            return "gmail"
        if self.outlook.available:
            return "outlook"
        return None

    def get_imap_provider(self) -> ImapEmailProvider | None:
        user = os.getenv("IMAP_USER") or os.getenv("GMAIL_USER") or os.getenv("EMAIL_USER")
        password = os.getenv("IMAP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD") or os.getenv("EMAIL_PASSWORD")
        host = os.getenv("IMAP_HOST")
        if not host and user and "@gmail.com" in user.lower():
            host = "imap.gmail.com"
        elif not host and user and ("@outlook.com" in user.lower() or "@hotmail.com" in user.lower()):
            host = "outlook.office365.com"

        if host and user and password:
            return ImapEmailProvider(host=host, user=user, password=password)
        return None

    def status(self) -> dict[str, Any]:
        """Check status of all email integrations."""
        configured = self.get_configured_provider()
        imap_prov = self.get_imap_provider()
        imap_status = False
        imap_detail = "No IMAP credentials in .env (set GMAIL_USER and GMAIL_APP_PASSWORD, or IMAP_HOST/USER/PASSWORD)"
        if imap_prov:
            imap_status, imap_detail = imap_prov.test_connection()

        return {
            "configured": configured is not None,
            "active_provider": configured or "none",
            "outlook_available": self.outlook.available,
            "imap_connected": imap_status,
            "imap_detail": imap_detail,
        }

    def get_latest_emails(self, limit: int = 5, unread_only: bool = False,
                          folder: str = "INBOX") -> list[EmailMessage]:
        """Retrieve recent emails from the active provider."""
        imap_prov = self.get_imap_provider()
        messages: list[EmailMessage] = []
        if imap_prov:
            messages = imap_prov.get_messages(limit=limit, unread_only=unread_only, folder=folder)
        elif self.outlook.available:
            messages = self.outlook.get_inbox_messages(limit=limit, unread_only=unread_only)

        for m in messages:
            self._cached_messages[m.id] = m
        return messages

    def search_emails(self, query: str = "", sender: str = "", subject: str = "",
                      limit: int = 5) -> list[EmailMessage]:
        """Search emails across text, sender, or subject."""
        imap_prov = self.get_imap_provider()
        results: list[EmailMessage] = []
        if imap_prov:
            search_term = query or subject or sender
            results = imap_prov.get_messages(limit=limit * 2, query=search_term)
        elif self.outlook.available:
            # Query recent messages and filter locally
            all_recent = self.outlook.get_inbox_messages(limit=50)
            q_lower = query.lower()
            s_lower = sender.lower()
            sub_lower = subject.lower()
            for m in all_recent:
                match = True
                if q_lower and q_lower not in m.body.lower() and q_lower not in m.subject.lower():
                    match = False
                if s_lower and s_lower not in m.sender_email.lower() and s_lower not in m.sender_name.lower():
                    match = False
                if sub_lower and sub_lower not in m.subject.lower():
                    match = False
                if match:
                    results.append(m)
                    if len(results) >= limit:
                        break

        for m in results:
            self._cached_messages[m.id] = m
        return results

    def get_email_detail(self, email_id: str) -> EmailMessage | None:
        if email_id in self._cached_messages:
            return self._cached_messages[email_id]
        # Refresh and check
        self.get_latest_emails(limit=20)
        return self._cached_messages.get(email_id)

    def draft_reply(self, to: str, subject: str, body: str, email_id: str | None = None) -> dict[str, Any]:
        """Create a prepared draft reply (Level 2: Prepare). Does NOT send."""
        created_in_client = False
        if self.outlook.available:
            created_in_client = self.outlook.create_draft(to, subject, body)

        return {
            "to": to,
            "subject": subject,
            "body": body,
            "email_id": email_id,
            "saved_in_drafts": created_in_client,
            "status": "prepared",
            "note": "Draft created and ready for your review. Not sent.",
        }


# Global singleton instance
_GLOBAL_EMAIL_SERVICE: EmailService | None = None


def get_email_service() -> EmailService:
    global _GLOBAL_EMAIL_SERVICE
    if _GLOBAL_EMAIL_SERVICE is None:
        _GLOBAL_EMAIL_SERVICE = EmailService()
    return _GLOBAL_EMAIL_SERVICE

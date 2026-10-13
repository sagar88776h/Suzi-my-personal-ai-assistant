"""Unit tests for Suzy's email service and tools."""

from __future__ import annotations

import pytest

from core.email_service import EmailMessage, EmailService
from tools import email as email_tools
from tools.safety import SafetyLevel


def sample_message(mid="m1", subject="Website Proposal", sender="client@example.com", body="Hi Sagar, please see proposal."):
    return EmailMessage(
        id=mid,
        sender_name="Client Partner",
        sender_email=sender,
        recipients=["sagar@example.com"],
        subject=subject,
        date="2026-10-12 10:30",
        snippet=body[:50],
        body=body,
        has_attachments=False,
        unread=True,
        provider="test",
    )


def test_email_message_formatting():
    msg = sample_message()
    citation = msg.format_citation()
    assert "[ID: m1]" in citation
    assert "Client Partner <client@example.com>" in citation
    assert "Website Proposal" in citation
    d = msg.to_dict()
    assert d["id"] == "m1"
    assert d["subject"] == "Website Proposal"


def test_email_service_fallback(monkeypatch):
    monkeypatch.setenv("IMAP_USER", "")
    monkeypatch.setenv("GMAIL_USER", "")
    svc = EmailService()
    svc.outlook.available = False

    st = svc.status()
    assert st["configured"] is False
    assert svc.get_latest_emails() == []
    assert svc.search_emails("invoice") == []


def test_email_tools_without_connection(monkeypatch):
    monkeypatch.setenv("IMAP_USER", "")
    monkeypatch.setenv("GMAIL_USER", "")
    svc = email_tools.get_email_service()
    svc.outlook.available = False

    res = email_tools.check_email_connection()
    assert "not connected to an email account" in res.message

    res_latest = email_tools.get_latest_emails(limit=5)
    assert res_latest.success
    assert "not connected" in res_latest.message

    res_search = email_tools.search_emails(query="tax")
    assert res_search.success
    assert "not connected" in res_search.message


def test_email_tools_with_messages(monkeypatch):
    svc = email_tools.get_email_service()
    m1 = sample_message("101", "Project Invoice #402", "accounts@vendor.com", "Attached invoice for October.")
    m2 = sample_message("102", "Website feedback", "dev@client.com", "Please change header color.")

    monkeypatch.setattr(svc, "get_latest_emails", lambda limit=5, unread_only=False, folder="INBOX": [m1, m2])
    monkeypatch.setattr(svc, "search_emails", lambda query="", sender="", subject="", limit=5: [m1])
    monkeypatch.setattr(svc, "get_email_detail", lambda eid: m1 if eid == "101" else None)

    latest = email_tools.get_latest_emails(limit=5)
    assert latest.success
    assert "Project Invoice #402" in latest.message
    assert "Website feedback" in latest.message

    search = email_tools.search_emails(query="invoice")
    assert search.success
    assert "Project Invoice #402" in search.message

    detail = email_tools.read_email_detail("101")
    assert detail.success
    assert "Attached invoice for October" in detail.message

    missing = email_tools.read_email_detail("999")
    assert not missing.success

    summary = email_tools.summarize_emails()
    assert summary.success
    assert "Summary of 2 email(s)" in summary.message


def test_draft_reply(monkeypatch):
    svc = email_tools.get_email_service()
    m = sample_message("202", "Client request", "client@biz.com", "Can you send the contract?")
    monkeypatch.setattr(svc, "get_email_detail", lambda eid: m if eid == "202" else None)

    res = email_tools.draft_email_reply("202", "Sure, I have attached the contract.")
    assert res.success
    assert "drafted a reply for your review" in res.message
    assert "client@biz.com" in res.message
    assert "Re: Client request" in res.message

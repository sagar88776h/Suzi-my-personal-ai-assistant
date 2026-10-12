"""What the user just said, for tools that must only act on an explicit "yes".

The assistant records the latest user message before running tools. A tool
such as ``apply_my_update`` checks it, so an action that changes Suzy herself
can never be triggered by text from a web page or a file, only by Sagar
clearly agreeing.
"""

from __future__ import annotations

import re
import threading

_local = threading.local()

AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|haan|han|ha|haa|ji|ok|okay|sure|confirm|confirmed|go ahead|do it|"
    r"install|apply|kar do|karo|chalo|theek hai|thik hai)\b",
    re.IGNORECASE,
)
NEGATIVE = re.compile(r"\b(no|not|nahi|nahin|mat|don't|dont|cancel|stop|wait)\b", re.IGNORECASE)


def set_user_message(text: str) -> None:
    _local.user_message = text


def user_message() -> str:
    return getattr(_local, "user_message", "")


def user_confirmed() -> bool:
    """True if the latest user message clearly says yes (and not no)."""
    text = user_message()
    return bool(AFFIRMATIVE.search(text)) and not NEGATIVE.search(text)

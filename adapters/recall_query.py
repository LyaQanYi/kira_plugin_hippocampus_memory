"""Pure helper for deriving the memory-recall query from an inbound turn.

Kept free of any ``core.*`` import so the query-sanitisation logic can be
unit-tested without booting (or stubbing) KiraAI's provider stack.
"""

from __future__ import annotations

from typing import Any


def query_from_event(event: Any) -> str:
    """Join the raw per-message text of a ``KiraMessageBatchEvent``.

    ``message_str`` is filled in KiraAI's ``core/message_manager.py`` before any
    ``llm_request`` hook runs and carries only the message body — it never
    includes the ``[date] [message_id: ...] [group_name: ... group_id: ...
    user_nickname: ..., user_id: ...] | <body>`` envelope that the built-in
    ``kira-ai`` plugin splices into ``req.user_prompt`` at SYS_HIGH priority.
    Reading it here yields a clean recall query, independent of hook ordering and
    free of the generic envelope words (``group``/``name``/``user``/``id``) that
    would otherwise flood the FTS query and false-match stored facts.

    Returns an empty string when the event carries no usable message text, so the
    caller can fall back to the request-based extractor.
    """
    messages = getattr(event, "messages", None) or []
    return " ".join(
        (getattr(m, "message_str", "") or "") for m in messages
    ).strip()

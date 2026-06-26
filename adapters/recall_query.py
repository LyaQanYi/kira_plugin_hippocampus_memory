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


def sender_users(event):
    """Distinct ``[(entity_id, nickname)]`` for the users who spoke in this batch.

    ``entity_id`` is ``f"{adapter}:{user_id}"``. Order-preserving, mirroring
    KiraAI-lightning's ``seen_senders`` set built from ``event.messages`` — used to
    inject the always-on user/group profile context into the live turn (the
    biggest end-to-end gap vs lightning, which passes ``user_profile=`` into
    ``get_agent_prompt`` every turn).

    Pure (no ``core.*`` import) so it stays unit-testable. ``event`` is
    duck-typed: ``.adapter.name`` and ``.messages[*].sender.{user_id, nickname}``.
    """
    adapter = ""
    adapter_obj = getattr(event, "adapter", None)
    if adapter_obj is not None:
        adapter = getattr(adapter_obj, "name", "") or ""

    out: list = []
    seen: set = set()
    messages = getattr(event, "messages", None) or []
    for msg in messages:
        sender = getattr(msg, "sender", None)
        if sender is None:
            continue
        uid = getattr(sender, "user_id", "") or ""
        if not uid or not adapter:
            continue
        eid = f"{adapter}:{uid}"
        if eid in seen:
            continue
        seen.add(eid)
        nick = getattr(sender, "nickname", "") or ""
        out.append((eid, nick))
    return out


def recall_targets(event, session_entity_id, session_entity_type):
    """Decide which ``(entity_id, entity_type)`` scopes to recall for a turn.

    Mirrors KiraAI-lightning's ``message_manager`` recall logic:
      - always recall the **speaking user's own** entity, so personal memories
        (learned in DM, or routed to the user entity inside a group) surface in
        any context — this is what fixes the "cross-session amnesia" where a
        group turn used to recall only the group entity;
      - additionally recall the **group** entity when the turn is a group
        message;
      - dedup while preserving order (user first).

    Falls back to the session entity when the speaker can't be resolved (e.g. a
    DM where ``messages`` carry no sender, or a group with an unresolved sender),
    so behaviour never regresses below the old single-entity recall.

    Pure function (no ``core.*`` import) so it stays unit-testable. ``event`` is
    duck-typed: it only needs ``.adapter.name`` and ``.messages[*].sender.user_id``.
    """
    targets: list = []
    seen: set = set()

    def _add(eid, etype):
        if eid and (eid, etype) not in seen:
            seen.add((eid, etype))
            targets.append((eid, etype))

    adapter = ""
    adapter_obj = getattr(event, "adapter", None)
    if adapter_obj is not None:
        adapter = getattr(adapter_obj, "name", "") or ""

    # Speaker = the most recent message's sender (matches lightning's
    # ``event.messages[-1].sender.user_id``); scan back for the first usable id.
    user_id = ""
    messages = getattr(event, "messages", None) or []
    for msg in reversed(messages):
        sender = getattr(msg, "sender", None)
        if sender is not None:
            uid = getattr(sender, "user_id", "") or ""
            if uid:
                user_id = uid
                break

    if adapter and user_id:
        _add(f"{adapter}:{user_id}", "user")

    if session_entity_type == "group":
        _add(session_entity_id, "group")

    if not targets:
        # Speaker unresolved → preserve the legacy single-entity behaviour.
        _add(session_entity_id, session_entity_type)

    return targets

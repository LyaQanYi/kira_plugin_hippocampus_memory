"""Cross-user memory search — ported from KiraAI-lightning's ``memory_search``.

The built-in ``memory_search`` tool used to recall only the current session
entity (a single user, or a whole group). Lightning's tool is far richer: it
resolves a name / alias / QQ number to an entity, searches **multiple users in
parallel**, auto-detects which users a query is about from conversation context
(via the fast LLM), and summarises the merged result. This module reproduces
that, adapted to the plugin's primitives (``HippocampusManager`` + the
``EntityProfileStore.resolve_entity_by_name`` it already ships).

Dependencies are passed in explicitly (no module-level globals like lightning's)
so the orchestrator is unit-testable with a fake LLM + a real manager.
"""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Tuple

from core.logging_manager import get_logger

from .llm import chat_text
from ..memory.manager import _TYPE_LABELS, _opaque_label  # single source of truth

logger = get_logger("hippocampus.entity_search", "cyan")


# How many user entities to fold into the fast-LLM "known users" hint. Bounds
# both the sequential profile reads and the prompt size at scale.
_MAX_HINT_USERS = 50


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def looks_like_entity_id(value: str) -> bool:
    """True for a canonical ``adapter:id`` entity id (vs a bare nickname).

    Both halves must be non-empty — ``":12345"`` (empty adapter) is malformed and
    must NOT pass, or it would skip name resolution and reach recall as-is."""
    if not value or ":" not in value:
        return False
    adapter, _, ident = value.partition(":")
    return bool(adapter) and bool(ident)


def looks_like_group_id(value: str) -> bool:
    """Detect an entity_id the LLM mistakenly filled with a group reference.

    We can't reliably tell a personal QQ number from a group number by digits
    alone, so only reject the obvious markers (``group`` / ``群``)."""
    if not value:
        return False
    low = value.strip().lower()
    return "group" in low or "群" in value


def _dedup(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Resolution / extraction (need the manager + fast LLM)
# ---------------------------------------------------------------------------

async def resolve_name(profile_store, name: str, entity_type: str) -> str:
    """Resolve a nickname / alias / QQ number to an entity_id (or '')."""
    name = (name or "").strip()
    if not name:
        return ""
    if looks_like_entity_id(name):
        return name
    try:
        resolved = await profile_store.resolve_entity_by_name(name, entity_type)
    except Exception as e:
        logger.debug(f"resolve_entity_by_name failed for {name!r}: {e}")
        return ""
    if resolved:
        # debug, not info: this maps a raw name/QQ to a canonical id and we
        # don't want those identifiers persisted in business logs by default.
        logger.debug(f"Nickname resolved -> {entity_type} entity")
        return resolved
    return ""


def conversation_context(sender_cache, sid: str, *, limit: int = 10) -> str:
    """Recent ``nickname: text`` lines for the session, for entity extraction.

    Sourced from the SenderCache the plugin already maintains (the plugin does
    not keep lightning's chat_memory.json; short-term history lives in KiraAI's
    SessionManager)."""
    if sender_cache is None or not sid:
        return ""
    try:
        recents = sender_cache.get_recent(sid, max_age_sec=24 * 3600)
    except Exception:
        return ""
    lines = []
    for item in recents[-limit:]:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if len(text) > 200:  # cap per message so one huge message can't dominate
            text = text[:200] + "…"
        who = item.get("nickname") or item.get("user_id") or "User"
        lines.append(f"{who}: {text}")
    # Join whole lines (already capped by `limit` and per-message length) — do
    # not char-slice the joined string, which would cut a line mid-word.
    return "\n".join(lines)


async def known_users_hint(profile_store, list_entities_fn) -> str:
    """A `entity_id -> names` listing to help the fast LLM map names to people.

    Capped at ``_MAX_HINT_USERS`` and fetched concurrently: an unbounded
    sequential scan would add one I/O round-trip per user before the first
    search result and could push the hint past a small LLM's context window."""
    try:
        entities = list(list_entities_fn("user"))[:_MAX_HINT_USERS]
        if not entities:
            return ""
        profiles = await asyncio.gather(
            *(profile_store.get_profile(eid, etype) for eid, etype in entities),
            return_exceptions=True,
        )
        users = []
        for (eid, _etype), profile in zip(entities, profiles):
            if isinstance(profile, Exception):
                continue
            names = []
            if profile.name:
                names.append(profile.name)
            if profile.nickname and profile.nickname != profile.name:
                names.append(profile.nickname)
            for alias in profile.aliases:
                if alias and alias not in names:
                    names.append(alias)
            if names:
                users.append(f"  {eid} -> {'/'.join(names)}")
        if users:
            return "\n已知用户：\n" + "\n".join(users)
    except Exception as e:
        logger.debug(f"known_users_hint failed: {e}")
    return ""


async def extract_subjects(
    fast_llm, query: str, context: str, hint: str
) -> List[str]:
    """Use the fast LLM to extract which people a query is about.

    Returns a list of nicknames / QQ numbers; empty when it's about the speaker
    themselves (SELF) or undeterminable (NONE)."""
    if fast_llm is None or not query:
        return []
    prompt = (
        "从以下查询和对话上下文中，提取所有被提及的人物标识（昵称或QQ号）。\n"
        "规则：\n"
        '- 如果查询是关于当前发言者自己的（如"我喜欢…"、"记住我…"），返回 SELF\n'
        "- 如果涉及其他用户，返回他们的昵称或QQ号，每行一个\n"
        "- 如果无法确定具体人物，返回 NONE\n"
        "- 不要输出任何解释，只输出标识\n"
        f"{hint}\n\n"
        f"查询：{query}\n"
        f"对话上下文：{context[-800:] if context else 'N/A'}\n\n"
        "提取的人物标识（每行一个）："
    )
    try:
        raw = (await chat_text(fast_llm, prompt)).strip()
    except Exception as e:
        logger.debug(f"extract_subjects failed: {e}")
        return []
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    return [ln for ln in lines if ln not in ("SELF", "NONE", "UNKNOWN", "无", "")]


def _format(memories, source_label: str, multi: bool, seen_ids: set) -> List[str]:
    lines = []
    for mem in memories:
        if mem.id in seen_ids:
            continue
        seen_ids.add(mem.id)
        label = _TYPE_LABELS.get(mem.type, mem.type)
        tags = f" [{', '.join(mem.tags)}]" if mem.tags else ""
        # In multi-entity mode each line is annotated with its source — by a
        # resolved display name (or an opaque label), NEVER the raw entity_id,
        # which must not reach the agent LLM via the tool result.
        prefix = f"[{source_label}] " if multi else ""
        lines.append(f"{prefix}[{label}]{tags} {mem.raw_text}")
    return lines


async def _resolve_source_labels(profile_store, resolved) -> dict:
    """Map each resolved entity_id → a source label for the merged result.

    Prefers the profile's display name (name / nickname / first alias) and falls
    back to an opaque per-turn label ("用户A", …) so the canonical entity_id is
    never surfaced in the memory_search tool result. Two entities that resolve
    to the SAME display name are disambiguated with the opaque token (e.g.
    "小明(用户B)") so each source stays distinguishable — the raw id is never
    surfaced either way."""
    labels: dict = {}
    used: set = set()
    for i, (eid, etype) in enumerate(resolved):
        name = ""
        if profile_store is not None:
            try:
                p = await profile_store.get_profile(eid, etype)
                name = p.name or p.nickname or (p.aliases[0] if p.aliases else "")
            except Exception as e:
                logger.debug(f"source label resolve failed: {type(e).__name__}")
        # Display names are user-controlled (nickname / alias). Fold control
        # chars and the bracket delimiters so a crafted name can't break the
        # "[source] [type] [tags] text" line contract or forge a source line in
        # the tool result the agent reads.
        name = re.sub(r"[\[\]\r\n]+", " ", name).strip()
        token = _opaque_label(i)
        if not name:
            # ``token`` is unique per index, but a real display name above could
            # be literally "用户B" (== _opaque_label(1)); if so, disambiguate so
            # two sources never share the same prefix.
            label = token if token not in used else f"{token}#{i}"
        elif name in used:
            label = f"{name}({token})"
        else:
            label = name
        used.add(label)
        labels[eid] = label
    return labels


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def search_memories(
    *,
    manager,
    fast_llm,
    sender_cache,
    sid: str,
    query: str,
    entity_id: str = "",
    entity_type: str = "user",
    k: int = 5,
    fallback_targets: Optional[List[Tuple[str, str]]] = None,
    list_entities_fn=None,
    allow_auto_extract: bool = False,
) -> str:
    """Cross-user memory search (lightning parity) with a dual-recall fallback.

    Resolution order:
      1. explicit ``entity_id`` (comma-separated names / QQ / aliases) → resolve each;
      2. (only when ``allow_auto_extract``) fast-LLM extract the subjects from
         context → resolve each — off by default: it's an extra LLM round-trip
         and silently falls through to the speaker when it mis-resolves;
      3. otherwise ``fallback_targets`` (the speaking user + group, from
         ``recall_query.recall_targets``) so a query still finds the caller's
         own memories instead of nothing.

    Multiple entities are searched in parallel and merged (dedup by id, labelled
    by entity). No second "summarise" LLM call — the agent reads the block.
    """
    query = (query or "").strip()
    if not query or manager is None:
        return ""
    try:
        k = max(1, int(k))
    except (TypeError, ValueError):
        k = 5

    profile_store = getattr(manager, "profile_store", None)

    resolved: List[Tuple[str, str]] = []

    # 1. explicit entity_id(s) — comma-separated names / QQ / ids. The group
    #    guard is applied PER TOKEN and only to a token that ALREADY looks like a
    #    canonical id (e.g. "platform:群123"); a bare nickname that merely
    #    contains "群" (like "阿群") falls through to resolve_name and stays
    #    searchable. So "小明,阿群" still resolves 小明 and tries 阿群 as a name.
    if entity_id and profile_store is not None:
        for name in (n.strip() for n in entity_id.split(",") if n.strip()):
            if looks_like_entity_id(name) and looks_like_group_id(name):
                logger.debug("Skipping group-like token in memory_search")
                continue
            rid = await resolve_name(profile_store, name, entity_type)
            if looks_like_entity_id(rid):
                resolved.append((rid, entity_type))

    # 2. auto-extract subjects from conversation context — OFF by default. This
    #    is the expensive, fragile path (a fast-LLM call + name resolution); when
    #    it mis-resolves it silently falls through to the speaker. Gated behind a
    #    config flag so the default search stays fast and deterministic.
    if (
        not resolved
        and allow_auto_extract
        and fast_llm is not None
        and profile_store is not None
    ):
        hint = ""
        if list_entities_fn is not None:
            hint = await known_users_hint(profile_store, list_entities_fn)
        context = conversation_context(sender_cache, sid)
        for name in await extract_subjects(fast_llm, query, context, hint):
            rid = await resolve_name(profile_store, name, entity_type)
            if looks_like_entity_id(rid):
                resolved.append((rid, entity_type))

    # 3. dual-recall fallback (speaker user + group)
    if not resolved and fallback_targets:
        resolved = [t for t in fallback_targets if looks_like_entity_id(t[0])]

    resolved = _dedup(resolved)
    if not resolved:
        return ""

    search_results = await asyncio.gather(
        *(manager.recall(query, entity_id=eid, entity_type=etype, k=k)
          for eid, etype in resolved),
        return_exceptions=True,
    )

    multi = len(resolved) > 1
    # Resolve a display label per source only when we actually annotate lines
    # (multi-entity mode) — keeps the single-entity path free of profile reads.
    source_labels = await _resolve_source_labels(profile_store, resolved) if multi else {}
    seen_ids: set = set()
    parts: List[str] = []
    for (eid, _etype), res in zip(resolved, search_results):
        if isinstance(res, Exception):
            logger.warning(f"Search failed for {eid}: {res}")
            continue
        parts.extend(_format(res, source_labels.get(eid, ""), multi, seen_ids))

    # Results are already labelled by entity when multi; the main agent LLM
    # reads them directly. We deliberately do NOT run a second "summarise"
    # LLM round-trip here — it doubled latency for marginal benefit.
    return "\n".join(parts)

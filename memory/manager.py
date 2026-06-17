"""Hippocampus memory coordinator.

Owned by the plugin instance. Wires together TomlTreeStore + MemoryIndex +
MemoryExtractor + SenderCache to provide:

- Fast loop: `recall()` for prompt injection.
- Slow loop: `_buffer_for_hippocampus()` accumulates chunks, then a background
  task runs `_hippocampus_process()` for extraction, dual-path routing,
  dedup/merge, elevation, and profile updates.

Short-term conversation memory is NOT managed here — KiraAI's SessionManager
already owns that (`core/chat/session_manager.py`).
"""

from __future__ import annotations

import asyncio
import math
import time
from threading import Lock
from typing import List, Optional, Tuple

from core.logging_manager import get_logger

from .memory_index import MemoryIndex
from .toml_tree_store import TomlTreeStore, Memory
from .memory_extractor import MemoryExtractor
from .memory_decay import MemoryDecayEngine
from .entity_profile import EntityProfileStore, EntityProfile
from .persona_evolution import PersonaEvolutionEngine
from .paths import (
    get_index_db_path,
    get_global_facts_dir,
    ensure_directory_structure,
    ENTITY_USER,
    ENTITY_GROUP,
)

logger = get_logger("hippocampus.manager", "green")


_TYPE_LABELS = {
    "fact": "事实",
    "reflection": "洞察",
    "episodic": "事件",
    "skill": "技能",
    "summary": "摘要",
}


def _as_int(value, default: int) -> int:
    """Coerce a config value to int, tolerating None / blank / non-numeric."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(f"Expected integer config, got {value!r}; using {default}")
        return default


class HippocampusManager:
    """Top-level coordinator owned by the plugin."""

    def __init__(self, plugin_cfg: dict):
        self.plugin_cfg = plugin_cfg

        ensure_directory_structure()

        self.memory_index = MemoryIndex(db_path=get_index_db_path())
        self.tree_store = TomlTreeStore(index=self.memory_index)
        self.extractor = MemoryExtractor(self.tree_store, llm_client=None)
        self.extractor.reflection_threshold = _as_int(
            plugin_cfg.get("reflection_threshold"), 5
        )
        self.profile_store = EntityProfileStore()
        self.decay_engine = MemoryDecayEngine(self.tree_store)
        self.persona_engine = PersonaEvolutionEngine(self.tree_store, persona_manager=None)

        # Wired by the plugin after construction.
        self.sender_cache = None  # adapters.sender_cache.SenderCache
        self.router = None  # filled by stage C router if used

        # Background hippocampus state.
        self._pending_conversations: dict[str, list] = {}
        self._hippocampus_threshold = _as_int(
            plugin_cfg.get("hippocampus_chunk_threshold"), 3
        )
        self._hippocampus_lock = Lock()
        self._background_tasks: set[asyncio.Task] = set()
        self._background_tasks_lock = Lock()

        logger.info("HippocampusManager initialized")

    def set_clients(self, llm_client, fast_llm_client=None) -> None:
        """Inject LLM clients after PluginContext is available."""
        self.extractor.set_llm_client(llm_client)
        if fast_llm_client is not None:
            self.extractor.set_fast_llm_client(fast_llm_client)

    def get_fast_llm(self):
        """The lightweight client for cheap LLM tasks (entity extraction,
        result summarisation). Falls back to the main client, None if neither."""
        if self.extractor is None:
            return None
        return self.extractor.get_fast_client()

    def set_sender_cache(self, sender_cache) -> None:
        self.sender_cache = sender_cache

    def set_persona_manager(self, persona_manager) -> None:
        self.persona_engine.set_persona_manager(persona_manager)

    def set_persona_brief(self, persona_brief: str) -> None:
        """Give the extractor the bot's persona so subjective extractions
        (group atmosphere, reflections, self-awareness) are judged in-character
        rather than from a neutral observer's view (issue #4). Read-only."""
        self.extractor.set_persona_brief(persona_brief)

    async def async_init(self) -> None:
        """Rebuild the SQLite index from TOML files on disk (disaster recovery)."""
        try:
            await self.tree_store.rebuild_index()
        except Exception as e:
            logger.error(f"Index rebuild failed (non-fatal): {e}")

    async def close(self) -> None:
        # Cancel any in-flight hippocampus tasks.
        with self._background_tasks_lock:
            tasks = list(self._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with self._background_tasks_lock:
            self._background_tasks.clear()

        try:
            self.memory_index.close()
        except Exception as e:
            logger.warning(f"MemoryIndex close failed: {e}")

    # ==================================================================
    # Recall (fast loop)
    # ==================================================================

    async def recall(
        self,
        query: str,
        entity_id: str = "",
        entity_type: str = "user",
        k: int = 5,
    ) -> List[Memory]:
        """Cross-folder recall (facts + reflections), composite-scored."""
        try:
            k = max(1, int(k))
        except (TypeError, ValueError):
            k = 5

        if not entity_id or not query:
            return []

        try:
            results: List[Memory] = await self.tree_store.search_across_folders(
                query=query,
                entity_id=entity_id,
                entity_type=entity_type,
                folders=["facts", "reflections"],
                k=k,
            )
        except Exception as e:
            logger.error(f"Recall error: {e}")
            return []

        # Also fold in global facts (low priority, only when there's room).
        if len(results) < k:
            try:
                global_hits = await self.tree_store.search(
                    query=query,
                    base_dir=get_global_facts_dir(),
                    folder="",
                    k=k - len(results),
                )
                results.extend(global_hits)
            except Exception as e:
                logger.debug(f"Global facts recall skipped: {e}")

        now = time.time()
        results.sort(
            key=lambda m: m.importance * 0.6
            + math.exp(-(now - m.last_accessed) / 86400 / 30.0) * 0.4,
            reverse=True,
        )
        return results[:k]

    def format_recalled_memories(self, memories: List[Memory]) -> str:
        if not memories:
            return ""
        parts = []
        for mem in memories:
            label = _TYPE_LABELS.get(mem.type, mem.type)
            tags_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            parts.append(f"[{label}]{tags_str} {mem.raw_text}")
        return "\n".join(parts)

    # ==================================================================
    # Manual fact CRUD (powering memory_add/update/remove tools)
    # ==================================================================

    async def add_fact(
        self,
        text: str,
        entity_id: str = "",
        entity_type: str = "user",
        importance: int = 7,
        tags: Optional[list] = None,
        source: Optional[dict] = None,
    ) -> Memory:
        """Add a fact. Falls back to global/facts/ when entity_id is empty."""
        if entity_id:
            return await self.tree_store.add_memory(
                content_text=text,
                memory_type="fact",
                importance=importance,
                tags=tags or [],
                source=source or {},
                entity_id=entity_id,
                entity_type=entity_type,
                folder="facts",
            )
        return await self.tree_store.add_memory(
            content_text=text,
            memory_type="fact",
            importance=importance,
            tags=tags or [],
            source=source or {},
            base_dir=get_global_facts_dir(),
            folder="",
        )

    async def list_facts_for_session(
        self, session: str, k: int = 50
    ) -> List[Memory]:
        """List facts for the entity inferred from a session id.

        Used by memory_update / memory_remove tools to resolve `index` → memory.
        """
        try:
            entity_id, entity_type = self._parse_entity_from_session(session)
        except ValueError:
            return []
        try:
            return await self.tree_store.get_all_memories(
                entity_id=entity_id,
                entity_type=entity_type,
                folder="facts",
            )
        except Exception as e:
            logger.warning(f"list_facts_for_session failed for {session}: {e}")
            return []

    async def update_fact_by_index(
        self, session: str, index: int, text: str
    ) -> Optional[Memory]:
        facts = await self.list_facts_for_session(session)
        if index < 0 or index >= len(facts):
            return None
        mem = facts[index]
        mem.text = text
        ok = await self.tree_store.update_memory(mem)
        return mem if ok else None

    async def delete_fact_by_index(self, session: str, index: int) -> bool:
        facts = await self.list_facts_for_session(session)
        if index < 0 or index >= len(facts):
            return False
        mem = facts[index]
        return await self.tree_store.delete_memory(
            mem.id,
            entity_id=mem._entity_id,
            entity_type=mem._entity_type,
            folder=mem._folder,
            base_dir=mem._base_dir,
        )

    # ==================================================================
    # Hippocampus (slow loop)
    # ==================================================================

    def submit_chunk(self, session: str, user_text: str, assistant_text: str) -> None:
        """Feed an (user, assistant) exchange to the hippocampus buffer.

        Sender info for the user side is pulled from the SenderCache.
        Triggers a background task when the buffer reaches the threshold.
        """
        if not session or not (user_text or assistant_text):
            return
        if self.extractor is None or self.extractor._llm_client is None:
            return

        chunk: list[dict] = []
        # Look up the most recent matching user record in sender cache.
        sender_id = ""
        sender_name = ""
        if self.sender_cache is not None and user_text:
            for item in reversed(self.sender_cache.get_recent(session, max_age_sec=600)):
                if item.get("text") == user_text:
                    sender_id = item.get("user_id", "")
                    sender_name = item.get("nickname", "")
                    break
            if not sender_id:
                # Fallback: take the latest user message of this session.
                recents = self.sender_cache.get_recent(session, max_age_sec=600)
                if recents:
                    sender_id = recents[-1].get("user_id", "")
                    sender_name = recents[-1].get("nickname", "")

        if user_text:
            msg = {"role": "user", "content": user_text}
            if sender_id:
                msg["sender_id"] = sender_id
            if sender_name:
                msg["sender_name"] = sender_name
            chunk.append(msg)
        if assistant_text:
            chunk.append({"role": "assistant", "content": assistant_text})

        if not chunk:
            return

        self._buffer_for_hippocampus(session, chunk)

    def _buffer_for_hippocampus(self, session: str, new_chunk: list[dict]) -> None:
        chunks_to_process = None
        with self._hippocampus_lock:
            self._pending_conversations.setdefault(session, []).append(new_chunk)
            if len(self._pending_conversations[session]) >= self._hippocampus_threshold:
                chunks_to_process = self._pending_conversations[session][:]
                self._pending_conversations[session] = []

        if chunks_to_process is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop — put them back and skip.
            with self._hippocampus_lock:
                self._pending_conversations[session] = (
                    chunks_to_process + self._pending_conversations.get(session, [])
                )
            logger.debug("No running event loop, skipping hippocampus processing")
            return

        task = loop.create_task(
            self._hippocampus_process(session, chunks_to_process)
        )
        with self._background_tasks_lock:
            self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        with self._background_tasks_lock:
            self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(
                "Background hippocampus task failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _hippocampus_process(self, session: str, chunks: list) -> None:
        try:
            session_entity_id, session_entity_type = self._parse_entity_from_session(session)
        except ValueError:
            # Non-conversational session (e.g. system messages) — nothing to learn.
            logger.debug(f"Skipping hippocampus for non-conversational session: {session}")
            return
        try:
            sender_map = self._build_sender_map(session, chunks)
            unique_senders = self._unique_senders(chunks)
            logger.debug(
                f"Hippocampus sender_map={sender_map}, unique_senders={unique_senders}"
            )

            adapter = session.split(":", maxsplit=1)[0]
            is_group = session_entity_type == ENTITY_GROUP

            conversation_text = self._chunks_to_text(chunks, sender_map)

            if is_group:
                personal_facts, group_facts = await asyncio.gather(
                    self.extractor.extract_personal_facts(conversation_text),
                    self.extractor.extract_group_facts(conversation_text),
                )
                logger.info(
                    f"Hippocampus dual-path: {len(personal_facts)} personal, "
                    f"{len(group_facts)} group facts"
                )
            else:
                personal_facts = await self.extractor.extract_facts(conversation_text)
                group_facts = []

            if not personal_facts and not group_facts:
                return

            routed: set[tuple[str, str]] = set()

            for fact in personal_facts:
                eid, etype = self._resolve_fact_entity(
                    fact, adapter, sender_map, unique_senders,
                    session_entity_id, session_entity_type,
                )
                logger.debug(
                    f"Personal fact routed: '{fact.get('content', '')[:40]}...' "
                    f"→ {etype}:{eid}"
                )
                await self.extractor.deduplicate_and_store(fact, eid, etype)
                routed.add((eid, etype))

                # Charter rule #3: high-importance facts also seed the profile.
                if etype == ENTITY_USER:
                    await self._update_profile_from_fact(eid, etype, fact)

            for fact in group_facts:
                logger.debug(
                    f"Group fact stored: '{fact.get('content', '')[:40]}...' "
                    f"→ {ENTITY_GROUP}:{session_entity_id}"
                )
                await self.extractor.deduplicate_and_store(
                    fact, session_entity_id, ENTITY_GROUP
                )
                routed.add((session_entity_id, ENTITY_GROUP))

            # Elevation (charter rule #2).
            for eid, etype in routed:
                try:
                    if await self.extractor.check_elevation_trigger(eid, etype):
                        await self.extractor.generate_reflections(eid, etype)
                except Exception as e:
                    logger.warning(f"Elevation failed for {etype}:{eid}: {e}")

            # Phase 1: self-awareness capture (write-only).
            if self.plugin_cfg.get("enable_self_awareness", True):
                try:
                    await self._collect_self_awareness(conversation_text)
                except Exception as e:
                    logger.debug(f"Self-awareness skipped: {e}")

            total = len(personal_facts) + len(group_facts)
            logger.info(
                f"Hippocampus completed for session {session}: {total} facts "
                f"({len(personal_facts)} personal + {len(group_facts)} group), "
                f"senders={unique_senders}"
            )
        except Exception as e:
            logger.error(f"Hippocampus processing error: {e}", exc_info=True)

    async def _update_profile_from_fact(
        self, entity_id: str, entity_type: str, fact: dict
    ) -> None:
        content = fact.get("content", "")
        importance = fact.get("importance", 5)
        if importance >= 7 and content:
            try:
                await self.profile_store.add_fact(entity_id, content, entity_type)
            except Exception as e:
                logger.debug(f"Profile add_fact failed for {entity_type}:{entity_id}: {e}")

    async def _collect_self_awareness(self, conversation_text: str) -> None:
        if self.extractor is None or self.extractor._llm_client is None:
            return
        try:
            insights = await self.extractor.extract_self_awareness(conversation_text)
        except Exception as e:
            logger.debug(f"Self-awareness extraction failed: {e}")
            return
        for insight in insights:
            try:
                await self.persona_engine.record_self_awareness(
                    content=insight,
                    importance=3,
                    tags=["auto-extracted", "phase1"],
                )
                logger.info(f"[Phase1] Self-awareness recorded: {insight[:60]}...")
            except Exception as e:
                logger.warning(f"[Phase1] record_self_awareness failed: {e}")

    # ==================================================================
    # Profiles
    # ==================================================================

    async def get_profile(
        self, entity_id: str, entity_type: str = ENTITY_USER
    ) -> EntityProfile:
        return await self.profile_store.get_profile(entity_id, entity_type)

    async def get_profile_prompt(
        self, entity_id: str, entity_type: str = ENTITY_USER
    ) -> str:
        return await self.profile_store.get_profile_prompt(entity_id, entity_type)

    async def update_user_interaction(
        self, user_id: str, platform: str = "", nickname: str = ""
    ) -> None:
        updates: dict = {}
        if platform:
            updates["platform"] = platform
        if nickname:
            updates["nickname"] = nickname
            profile = await self.profile_store.get_profile(user_id, ENTITY_USER)
            if not profile.name:
                updates["name"] = nickname
        await self.profile_store.increment_interaction(user_id, ENTITY_USER, **updates)

    # ==================================================================
    # Periodic maintenance
    # ==================================================================

    async def run_forgetting_cycle(self) -> tuple[int, int]:
        try:
            return await self.decay_engine.run_full_cycle()
        except Exception as e:
            logger.error(f"Forgetting cycle failed: {e}")
            return 0, 0

    async def run_evolution_cycle(self) -> None:
        if self.persona_engine.persona_manager is None:
            logger.debug("Persona evolution skipped (no persona_manager wired)")
            return
        try:
            await self.persona_engine.run_evolution_cycle(
                llm_client=self.extractor._llm_client
            )
        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")

    # ==================================================================
    # Helpers
    # ==================================================================

    def _build_sender_map(self, session: str, chunks: list) -> dict[str, str]:
        """Build a bidirectional {speaker_id|nickname.lower(): user_id} map.

        Sources, in order: chunk-attached sender_id/sender_name, then the
        SenderCache.
        """
        sender_map: dict[str, str] = {}
        for chunk in chunks:
            for msg in chunk:
                if msg.get("role") != "user":
                    continue
                sid = msg.get("sender_id", "")
                name = msg.get("sender_name", "")
                if sid:
                    sender_map[sid] = sid
                    if name:
                        sender_map[name.lower()] = sid

        if self.sender_cache is not None:
            for item in self.sender_cache.get_recent(session, max_age_sec=24 * 3600):
                uid = item.get("user_id", "")
                if not uid:
                    continue
                sender_map.setdefault(uid, uid)
                nick = item.get("nickname", "")
                if nick:
                    sender_map.setdefault(nick.lower(), uid)
        return sender_map

    @staticmethod
    def _unique_senders(chunks: list) -> list[str]:
        seen = set()
        result = []
        for chunk in chunks:
            for msg in chunk:
                if msg.get("role") != "user":
                    continue
                sid = msg.get("sender_id", "")
                if sid and sid not in seen:
                    seen.add(sid)
                    result.append(sid)
        return result

    def _resolve_fact_entity(
        self,
        fact: dict,
        adapter: str,
        sender_map: dict[str, str],
        unique_senders: list[str],
        session_entity_id: str,
        session_entity_type: str,
    ) -> Tuple[str, str]:
        """Five-tier routing: speaker_id → subject → "group" → single-user → fallback."""
        speaker_id = (fact.get("speaker_id", "") or "").strip()
        subject = (fact.get("subject", "") or "").strip()

        if speaker_id and speaker_id in sender_map:
            return f"{adapter}:{sender_map[speaker_id]}", ENTITY_USER
        if subject and subject.lower() in sender_map:
            return f"{adapter}:{sender_map[subject.lower()]}", ENTITY_USER
        if subject.lower() == "group":
            return session_entity_id, ENTITY_GROUP
        if len(unique_senders) == 1:
            return f"{adapter}:{unique_senders[0]}", ENTITY_USER
        return session_entity_id, session_entity_type

    @staticmethod
    def _chunks_to_text(chunks: list, sender_map: dict[str, str]) -> str:
        # Reverse the sender_map (id → preferred display nickname).
        id_to_name: dict[str, str] = {}
        for k, v in sender_map.items():
            # Skip the id→id self entries; only keep the name→id entries.
            if k != v and v not in id_to_name:
                id_to_name[v] = k

        lines = []
        for chunk in chunks:
            for msg in chunk:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    sender_name = msg.get("sender_name") or id_to_name.get(
                        msg.get("sender_id", ""), "User"
                    )
                    sender_id = msg.get("sender_id", "")
                    label = f"{sender_name}({sender_id})" if sender_id else (sender_name or "User")
                    lines.append(f"{label}: {content}")
                elif role == "assistant":
                    lines.append(f"Bot: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_entity_from_session(session: str) -> Tuple[str, str]:
        """Parse a KiraAI session sid `adapter:type:id` to (entity_id, entity_type).

        KiraAI uses `gm` for group, `dm` for direct messages.
        """
        parts = session.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"Invalid session id: {session}")

        adapter, session_type, session_id = parts
        # Whitelist conversational types only. KiraAI also emits `sm` (system
        # messages) which must NOT spawn a user/group entity — fail loud so the
        # caller skips it rather than silently polluting entities/.
        if session_type == "gm":
            return f"{adapter}:{session_id}", ENTITY_GROUP
        if session_type == "dm":
            return f"{adapter}:{session_id}", ENTITY_USER
        raise ValueError(f"Non-conversational session type {session_type!r}: {session}")

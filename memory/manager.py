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


def _opaque_label(index: int) -> str:
    """Opaque, per-turn participant label ("用户A", "用户B", … "用户Z", "用户AA").

    Used wherever a raw platform/canonical id would otherwise be shown to an
    LLM (or returned by a tool): the model can still tell participants apart,
    but the real identifier never reaches the prompt. Bijective base-26 so the
    labels stay distinct for arbitrarily many participants."""
    n = index
    letters = ""
    while True:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return f"用户{letters}"


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

    async def add_fact_curated(
        self,
        text: str,
        entity_id: str = "",
        entity_type: str = "user",
        importance: int = 7,
        tags: Optional[list] = None,
        source: Optional[dict] = None,
    ) -> str:
        """Manual add that runs through the hippocampus dedup/merge pipeline.

        Mirrors KiraAI-lightning's MemoryAddTool: an entity-scoped add goes
        through SHA-256 + FTS5 (+ LLM) dedup so a manual add can't create a
        duplicate the background extractor would have collapsed — and can MERGE
        into an existing memory. Exact-hash dedup works even with no LLM
        configured (the conflict-check just degrades to "new").

        Falls back to a direct write when there's no entity scope (global facts,
        which the dedup path doesn't cover) or no extractor.

        Returns "duplicate" | "update" | "new" | "stored" | "skip".
        """
        text = (text or "").strip()
        if not text:
            return "skip"
        tags = tags or ["explicit"]

        if not entity_id or self.extractor is None:
            await self.add_fact(text, entity_id, entity_type, importance, tags, source)
            return "stored"

        fact = {
            "content": text,
            "subject": "",
            "speaker_id": "",
            "importance": importance,
            "tags": tags,
            "semantic_id": "",
        }
        try:
            return await self.extractor.deduplicate_and_store(
                fact, entity_id, entity_type
            )
        except Exception as e:
            # Log only the exception type — the message can embed the entity id
            # or profile path, which this plugin keeps out of logs/prompts.
            logger.warning(f"Curated add failed, writing directly: {type(e).__name__}")
            await self.add_fact(text, entity_id, entity_type, importance, tags, source)
            return "stored"

    async def list_editable_memories(
        self, session: str, k: int = 50
    ) -> List[Memory]:
        """Facts AND reflections for the session entity, newest first.

        The editable set surfaced to memory_update / memory_remove. Lightning's
        edit tools were memory_id + entity-resolved across folders; this broadens
        the plugin's previously facts-only, single-folder view to include elevated
        reflections so the model can also correct a wrong insight.
        """
        try:
            entity_id, entity_type = self._parse_entity_from_session(session)
        except ValueError:
            return []
        out: List[Memory] = []
        for folder in ("facts", "reflections"):
            try:
                out.extend(
                    await self.tree_store.get_all_memories(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        folder=folder,
                    )
                )
            except Exception as e:
                logger.debug(f"list_editable_memories {folder} failed: {e}")
        out.sort(key=lambda m: m.last_accessed, reverse=True)
        return out[:k]

    @staticmethod
    def format_editable_list(memories: List[Memory]) -> str:
        """Numbered listing for the memory_update / memory_remove `index` arg."""
        lines = []
        for i, m in enumerate(memories):
            label = _TYPE_LABELS.get(m.type, m.type)
            lines.append(f"{i}. [{label}] {m.raw_text}")
        return "\n".join(lines)

    async def update_memory_at(
        self, memories: List[Memory], index: int, text: str
    ) -> Optional[Memory]:
        if index < 0 or index >= len(memories):
            return None
        # Refuse to blank an existing memory: a single bad model call passing an
        # empty string must not wipe stored content.
        text = (text or "").strip()
        if not text:
            return None
        mem = memories[index]
        mem.text = text
        ok = await self.tree_store.update_memory(mem)
        return mem if ok else None

    async def delete_memory_at(
        self, memories: List[Memory], index: int
    ) -> bool:
        if index < 0 or index >= len(memories):
            return False
        mem = memories[index]
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

    def submit_exchange(
        self, session: str, user_msgs: list, assistant_text: str
    ) -> None:
        """Feed a completed turn to the hippocampus with EXACT sender identity.

        ``user_msgs`` is a list of SenderCache records
        (``{user_id, nickname, text, ...}``) — typically every user message
        recorded since the previous turn (``SenderCache.take_unconsumed``). Each
        keeps its precise ``sender_id``/``sender_name``, so fact routing no longer
        depends on matching the assistant turn back to a user message by text,
        which mis-routed facts under duplicate text or racy multi-speaker bursts.

        Preferred over ``submit_chunk`` for the live @on.step_result path; the
        single-user ``submit_chunk`` remains for callers that only have a text.
        """
        if not session or not (user_msgs or assistant_text):
            return
        if self.extractor is None or self.extractor._llm_client is None:
            return

        chunk: list[dict] = []
        for um in user_msgs or []:
            text = (um.get("text") or "").strip()
            if not text:
                continue
            msg = {"role": "user", "content": text}
            uid = um.get("user_id", "")
            nick = um.get("nickname", "")
            if uid:
                msg["sender_id"] = uid
            if nick:
                msg["sender_name"] = nick
            chunk.append(msg)
        if assistant_text:
            chunk.append({"role": "assistant", "content": assistant_text})

        if not chunk:
            return
        self._buffer_for_hippocampus(session, chunk)

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

            # Assign every distinct sender an opaque per-turn token ("用户A", …)
            # so the raw platform id never reaches the extraction prompt; the
            # inverse map routes extracted facts back to the real sender.
            token_by_sid = self._participant_tokens(unique_senders)
            label_to_sid: dict[str, str] = {}
            for sid, tok in token_by_sid.items():
                label_to_sid[tok] = sid
                label_to_sid[tok.lower()] = sid

            conversation_text = self._chunks_to_text(chunks, sender_map, token_by_sid)

            # 构建 sender profile 上下文，辅助 LLM 更准确地提取和路由事实，
            # 避免重复记录已知信息（精确复刻 lightning memory_manager 的提取增强）。
            profile_context = await self._build_sender_profiles_context(
                adapter, unique_senders, token_by_sid
            )
            if profile_context:
                conversation_text = f"{profile_context}\n\n{conversation_text}"

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
                    session_entity_id, session_entity_type, label_to_sid,
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

    async def build_turn_profile_prompt(
        self,
        sender_entities: list,
        session_entity_id: str,
        session_entity_type: str,
        is_group: bool,
    ) -> str:
        """Always-on profile context for a single turn (parity with lightning).

        KiraAI-lightning's ``message_manager`` passes ``user_profile=`` into the
        agent prompt every turn:
          - in a group with >1 distinct speakers → aggregate **every** speaker's
            profile, each labelled by display name (never the raw entity_id);
          - otherwise → the single speaker's profile (in a DM that's the session
            entity itself).

        ``sender_entities`` is ``[(entity_id, nickname), ...]`` (from
        ``recall_query.sender_users``). Returns ``""`` when there is no usable
        profile text, so the caller injects nothing rather than a placeholder.
        """
        try:
            distinct: list = []
            seen: set = set()
            for eid, nick in sender_entities or []:
                if eid and eid not in seen:
                    seen.add(eid)
                    distinct.append((eid, nick))

            if is_group and len(distinct) > 1:
                parts: list = []
                for i, (eid, nick) in enumerate(distinct):
                    try:
                        profile = await self.get_profile(eid, ENTITY_USER)
                    except Exception:
                        continue
                    p = profile.to_prompt()
                    if not p or p == "暂无画像信息":
                        continue
                    # Label by display name only. Never fall back to the raw id
                    # tail — that's the bare platform/QQ number, which the plugin
                    # must keep out of prompts. Use an opaque ordinal so the model
                    # can still tell speakers apart.
                    label = profile.name or profile.nickname or nick or _opaque_label(i)
                    parts.append(f"【{label}】\n{p}")
                if not parts:
                    return ""
                return (
                    "以下是本次群聊中参与对话的用户的画像信息，"
                    "帮助你了解每个人的背景和偏好：\n\n" + "\n\n".join(parts)
                )

            # Single speaker (or DM): the speaker's profile, falling back to the
            # session entity when the speaker can't be resolved.
            if distinct:
                target_id, target_type = distinct[0][0], ENTITY_USER
            else:
                target_id, target_type = session_entity_id, session_entity_type
            prompt = await self.get_profile_prompt(target_id, target_type)
            if not prompt or prompt == "暂无画像信息":
                return ""
            return prompt
        except Exception as e:
            logger.debug(f"build_turn_profile_prompt failed: {type(e).__name__}")
            return ""

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

    async def _build_sender_profiles_context(
        self, adapter: str, unique_senders: list, token_by_sid: dict | None = None
    ) -> str:
        """为海马体提取构建 sender profile 摘要

        让 LLM 在提取事实时知道每个 sender 的已知信息（昵称、曾用名、特征等），
        避免重复提取已有事实，并辅助 entity 路由。
        """
        if not unique_senders:
            return ""
        if token_by_sid is None:
            token_by_sid = self._participant_tokens(unique_senders)

        parts = []
        for i, sid in enumerate(unique_senders):
            entity_id = f"{adapter}:{sid}"
            try:
                profile = await self.profile_store.get_profile(entity_id, ENTITY_USER)
                # Label by a resolved display name (name / nickname / alias);
                # fall back to the opaque per-turn token, never str(sid) — that's
                # the bare platform id the plugin must keep out of prompts.
                label = (
                    profile.name
                    or profile.nickname
                    or (profile.aliases[0] if profile.aliases else "")
                    or token_by_sid.get(sid)
                    or _opaque_label(i)
                )
                info = []
                if profile.name:
                    info.append(f"名字: {profile.name}")
                if profile.nickname and profile.nickname != profile.name:
                    info.append(f"当前昵称: {profile.nickname}")
                if profile.aliases:
                    info.append(f"曾用名: {', '.join(profile.aliases)}")
                if profile.traits:
                    info.append(f"特征: {', '.join(profile.traits)}")
                if profile.facts:
                    info.append(f"已知事实: {'; '.join(profile.facts[:5])}")
                if info:
                    parts.append(f"【{label}】 {' | '.join(info)}")
            except Exception as e:
                # Tolerate a single bad profile read, but don't swallow it
                # silently — a quiet failure here just degrades extraction
                # quality with no trace. Log the exception *type* only: its
                # str() can embed the profile path (which contains the entity
                # id), and the plugin keeps identifiers out of business logs.
                logger.debug(
                    f"Skipping a sender profile in extraction context: "
                    f"{type(e).__name__}"
                )
                continue

        if not parts:
            return ""
        return (
            "## 参与者已知信息（以下是对话中提到的用户的已有画像，"
            "提取事实时请避免重复记录这些已有内容）\n"
            + "\n".join(parts)
        )

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

    @staticmethod
    def _participant_tokens(unique_senders: list) -> dict[str, str]:
        """Map each distinct sender id → an opaque per-turn token ("用户A", …).

        The token stands in for the raw id everywhere the conversation is shown
        to the LLM; its inverse (``label_to_sid``) routes extracted facts home,
        so a canonical id never has to appear in the prompt to preserve routing."""
        return {sid: _opaque_label(i) for i, sid in enumerate(unique_senders)}

    def _resolve_fact_entity(
        self,
        fact: dict,
        adapter: str,
        sender_map: dict[str, str],
        unique_senders: list[str],
        session_entity_id: str,
        session_entity_type: str,
        label_to_sid: Optional[dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Five-tier routing: speaker_id → subject → "group" → single-user → fallback.

        ``label_to_sid`` maps the opaque per-turn tokens we showed the LLM
        ("用户A", …) back to the real sender id. Since the extraction prompt now
        carries tokens rather than raw ids, the ``speaker_id`` / ``subject`` the
        model echoes back is a token; we translate it first, then fall through
        to the legacy raw-id lookups so a directly-fed chunk still routes."""
        label_to_sid = label_to_sid or {}
        speaker_id = (fact.get("speaker_id", "") or "").strip()
        subject = (fact.get("subject", "") or "").strip()

        def _token_sid(value: str) -> str:
            """Resolve a token back to a sender id, tolerating the full rendered
            label form ``昵称(用户A)`` — the prompt shows the model that composite,
            so a non-compliant model may echo it back whole instead of the bare
            token. Strip a trailing ``(token)`` and retry before giving up."""
            value = (value or "").strip()
            if not value:
                return ""
            sid = label_to_sid.get(value) or label_to_sid.get(value.lower())
            if sid:
                return sid
            if value.endswith(")") and "(" in value:
                tok = value.rsplit("(", 1)[-1][:-1].strip()
                return label_to_sid.get(tok) or label_to_sid.get(tok.lower()) or ""
            return ""

        tok_sid = _token_sid(speaker_id)
        if tok_sid:
            return f"{adapter}:{tok_sid}", ENTITY_USER
        if speaker_id and speaker_id in sender_map:
            return f"{adapter}:{sender_map[speaker_id]}", ENTITY_USER

        sub_sid = _token_sid(subject)
        if sub_sid:
            return f"{adapter}:{sub_sid}", ENTITY_USER
        if subject and subject.lower() in sender_map:
            return f"{adapter}:{sender_map[subject.lower()]}", ENTITY_USER

        if subject.lower() == "group":
            return session_entity_id, ENTITY_GROUP
        if len(unique_senders) == 1:
            return f"{adapter}:{unique_senders[0]}", ENTITY_USER
        return session_entity_id, session_entity_type

    @staticmethod
    def _chunks_to_text(
        chunks: list,
        sender_map: dict[str, str],
        token_by_sid: dict[str, str],
    ) -> str:
        # Reverse the sender_map (id → preferred display nickname) for messages
        # that didn't carry a sender_name inline.
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
                    sender_id = msg.get("sender_id", "")
                    sender_name = msg.get("sender_name") or id_to_name.get(sender_id, "")
                    # Opaque per-turn token in place of the raw sender id: the
                    # canonical id must never reach the extraction prompt. The
                    # parenthetical stays the LLM's routing handle, decoded back
                    # to the real sender via label_to_sid in _resolve_fact_entity.
                    token = token_by_sid.get(sender_id, "")
                    if sender_name and token:
                        label = f"{sender_name}({token})"
                    elif token:
                        label = token
                    else:
                        label = sender_name or "User"
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

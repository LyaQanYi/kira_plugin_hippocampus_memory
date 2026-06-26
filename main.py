"""KiraAI plugin entrypoint — Hippocampus Memory.

Stage A: scaffolding + auto-disable simple_memory + recall injection +
memory_add/update/remove tools + simple_memory data migration.

Later stages will attach hippocampus background extraction (B),
decay/profile/router/persona evolution (C), and tests/docs (D).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from core.plugin import BasePlugin, logger, on, Priority, register
from core.provider import LLMRequest

from .adapters.llm import append_to_prompt_section
from .adapters.migration import migrate_simple_memory_if_needed
from .adapters.recall_query import query_from_event, recall_targets, sender_users
from .adapters.entity_search import (
    search_memories,
    looks_like_entity_id,
    looks_like_group_id,
)
from .adapters.sender_cache import SenderCache
from .memory.manager import HippocampusManager
from .memory.paths import list_all_entities, set_memory_root


# Copied verbatim from kira_plugin_simple_memory so the LLM still sees the
# memory tool guidance after simple_memory is disabled.
MEM_RULE_PROMPT = """
### 隐私与安全约束
- 绝对不要在回复中直接暴露原始记忆内容。
- 不要主动提及用户的敏感个人信息（如QQ号，群号，电话号码、地址、身份证号等），即使记忆中包含这些内容。
- 引用记忆内容时，使用自然的转述方式，而非逐字复述。
- 如果用户明确要求查看自己的数据，可以简要概述，但仍应避免暴露系统内部结构。
"""

MEM_TOOL_FEW_SHOT = """
### 记忆工具说明（Memory Tools）
你拥有一套完整的核心记忆系统。

#### 核心记忆工具
核心记忆用于记录你**主动认为重要的信息**，包括用户分享的重要信息和你自己的相关信息。

* `memory_add`: 添加一条记忆到核心记忆和长期记忆（自动去重/合并）
* `memory_update`: 修改特定的核心记忆（通过索引号；编号无效时会返回当前列表）
* `memory_remove`: 删除一条核心记忆（通过索引号；编号无效时会返回当前列表）
* `memory_search`: 主动检索长期记忆（按语义/关键词）
* `memory_profile`: 查看某个用户/群的画像（名字、曾用名、特征、偏好、已知事实）

#### 使用原则
* 你需要 **主动调用记忆工具** 记录重要信息
* 在**无有效信息的闲聊**中，不要记录杂乱或无价值的信息
* 系统会在后台自动从对话中抽取事实，无需为日常对话手动调用 memory_add

#### 示例说明

**示例 1**
user：你喜欢吃什么呀？
* 若人格信息中没有相关内容
* 需要调用 `memory_add` 工具写入相关信息

**示例 2**
user：我最近改过自新努力不熬夜了！
* 调用 `memory_update` 工具修改原有记忆

**示例 3**
user：我之前骗你的，其实我 xxx
* 调用 `memory_remove` 工具删除错误记忆
"""


class HippocampusMemoryPlugin(BasePlugin):
    """Dual-brain long-term memory plugin for KiraAI."""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._manager: Optional[HippocampusManager] = None
        self._sender_cache = SenderCache()
        self._bg_tasks: set[asyncio.Task] = set()
        self._plugin_data_dir = None  # captured in initialize()
        # Dedup key for step_result (sid → last raw_output hash) so multi-step
        # agent loops don't keep re-feeding the hippocampus. Bounded LRU so a
        # long-running process with many sessions doesn't leak memory.
        self._last_step_hash: "OrderedDict[str, str]" = OrderedDict()
        self._STEP_HASH_CAP = 512

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def initialize(self) -> None:
        # 1. Resolve and set up the memory root under our plugin_data dir.
        self._plugin_data_dir = self.ctx.get_plugin_data_dir()
        if self._plugin_data_dir is None:
            logger.error(
                "Could not resolve plugin_data_dir; refusing to initialize."
            )
            return

        memory_root = self._plugin_data_dir / "memory"
        set_memory_root(memory_root)

        # 2. Create the manager (initializes SQLite + TOML store).
        self._manager = HippocampusManager(self.plugin_cfg)
        await self._manager.async_init()

        # 2b. Wire LLM clients + sender cache. Fetching the default LLM raises
        # when no model is configured yet — the plugin must still load. Recall
        # (FTS), manual memory tools, migration and disabling simple_memory all
        # work without an LLM; only background hippocampus extraction needs one,
        # and it stays dormant until a model is available.
        extraction_uuid = (self.plugin_cfg.get("extraction_model") or "").strip()
        reflection_uuid = (self.plugin_cfg.get("reflection_model") or "").strip()
        default_llm = self._safe_get_llm(fast=False)
        fast_llm = self._safe_get_llm(fast=True) or default_llm

        primary = default_llm
        if reflection_uuid:
            picked = self.ctx.get_llm_client(model_uuid=reflection_uuid)
            if picked is not None:
                primary = picked
        secondary = fast_llm
        if extraction_uuid:
            picked = self.ctx.get_llm_client(model_uuid=extraction_uuid)
            if picked is not None:
                secondary = picked

        # Extractor uses primary for heavy tasks (reflection) and secondary
        # for cheap ones (dedup checks, semantic ids). Lightning's MemoryExtractor
        # treats _llm_client as the main and _fast_llm_client as the fast one.
        self._manager.set_clients(llm_client=primary, fast_llm_client=secondary)
        self._manager.set_sender_cache(self._sender_cache)
        if primary is None:
            logger.warning(
                "No default LLM configured — hippocampus extraction is dormant. "
                "Recall and manual memory tools still work. Configure a default "
                "LLM in WebUI and restart to enable background fact extraction."
            )
        # Wire persona manager so Tier-3 leap can write back into KiraAI persona.
        persona_mgr = getattr(self.ctx, "persona_mgr", None)
        if persona_mgr is not None and self.plugin_cfg.get("enable_persona_evolution", False):
            self._manager.set_persona_manager(persona_mgr)

        # Persona perspective for subjective extraction (issue #4): read the
        # bot's persona (read-only, never written) so atmosphere/reflection/
        # self-awareness are judged in-character instead of from a neutral
        # observer's view. Independent of — and far lighter than — Tier-3
        # evolution, so it has its own switch.
        if persona_mgr is not None and self.plugin_cfg.get("enable_persona_perspective", False):
            try:
                persona = await persona_mgr.get_persona()
                brief = (getattr(persona, "content", "") or "").strip()
                if brief:
                    self._manager.set_persona_brief(brief)
                    logger.info(
                        "Persona perspective enabled for subjective extraction "
                        "(group atmosphere / reflections / self-awareness)."
                    )
                else:
                    logger.debug("Persona perspective on, but persona text is empty.")
            except Exception as e:
                logger.warning(f"Could not load persona for perspective injection: {e}")

        # 3. One-shot migration from simple_memory's core.txt before we
        #    disable it (otherwise the file might be gone next run).
        if self.plugin_cfg.get("migrate_simple_memory_on_first_run", True):
            try:
                count = await migrate_simple_memory_if_needed(
                    self._manager.tree_store, self._plugin_data_dir
                )
                if count:
                    logger.info(f"Migrated {count} legacy memories")
            except Exception as e:
                logger.error(f"Migration step failed (non-fatal): {e}")

        # 4. Auto-disable simple_memory after migration.
        if self.plugin_cfg.get("auto_disable_simple_memory", True):
            await self._disable_simple_memory()

        # 5. Warm up jieba in the background (first lcut is ~1s).
        self._spawn_bg(self._warm_jieba(), name="jieba-warmup")

        # 6. Periodic maintenance loops.
        decay_h = self._int_cfg("decay_interval_hours", 24)
        if decay_h > 0:
            self._spawn_bg(self._decay_loop(decay_h * 3600), name="decay-loop")
        evo_h = self._int_cfg("persona_evolution_interval_hours", 168)
        if evo_h > 0 and self.plugin_cfg.get("enable_persona_evolution", False):
            self._spawn_bg(self._evolution_loop(evo_h * 3600), name="evolution-loop")

        logger.info("kira_plugin_hippocampus_memory initialized")

    async def terminate(self) -> None:
        # Cancel background tasks.
        for task in list(self._bg_tasks):
            if not task.done():
                task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()

        if self._manager is not None:
            await self._manager.close()
            self._manager = None

        self._sender_cache.clear()
        self._last_step_hash.clear()

        logger.info("kira_plugin_hippocampus_memory terminated")

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _int_cfg(self, key: str, default: int) -> int:
        """Read an integer config value, tolerating bad WebUI input.

        schema declares these as integers, but a user can still save a blank or
        non-numeric value; a bare int() would abort initialize(). Falls back to
        `default` on missing/invalid input."""
        raw = self.plugin_cfg.get(key, default)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(f"Config {key}={raw!r} is not an integer; using {default}")
            return default

    def _safe_get_llm(self, *, fast: bool):
        """Best-effort default LLM client. Returns None (instead of raising)
        when no default model is configured, so initialize() can proceed."""
        try:
            if fast:
                return self.ctx.get_default_fast_llm_client()
            return self.ctx.get_default_llm_client()
        except Exception as e:
            logger.debug(f"Default {'fast ' if fast else ''}LLM unavailable: {e}")
            return None

    def _spawn_bg(self, coro, *, name: str = "") -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(f"No running loop for bg task {name}")
            coro.close()
            return
        task = loop.create_task(coro, name=name or "hippocampus-bg")
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    @staticmethod
    async def _warm_jieba() -> None:
        try:
            await asyncio.to_thread(lambda: __import__("jieba").initialize())
        except Exception as e:
            logger.warning(f"jieba warmup failed: {e}")

    async def _decay_loop(self, interval_sec: float) -> None:
        try:
            # Stagger the first run so it doesn't fire on every startup.
            await asyncio.sleep(min(interval_sec, 60.0))
            while True:
                if self._manager is not None:
                    try:
                        deleted, downgraded = await self._manager.run_forgetting_cycle()
                        if deleted or downgraded:
                            logger.info(
                                f"Forgetting cycle: deleted={deleted}, downgraded={downgraded}"
                            )
                    except Exception as e:
                        logger.warning(f"Decay loop iteration failed: {e}")
                await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            return

    async def _evolution_loop(self, interval_sec: float) -> None:
        try:
            await asyncio.sleep(min(interval_sec, 60.0))
            while True:
                if self._manager is not None:
                    try:
                        await self._manager.run_evolution_cycle()
                    except Exception as e:
                        logger.warning(f"Evolution loop iteration failed: {e}")
                await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            return

    async def _disable_simple_memory(self) -> None:
        pm = getattr(self.ctx, "plugin_mgr", None)
        if pm is None:
            logger.warning(
                "plugin_mgr unavailable; cannot auto-disable simple_memory"
            )
            return
        target = "kira_plugin_simple_memory"
        try:
            if not pm.is_plugin_enabled(target):
                return
        except Exception:
            return
        try:
            await pm.set_plugin_enabled(target, False)
            logger.warning(
                f"Auto-disabled {target} because hippocampus memory is active. "
                f"You can re-enable it in WebUI > Plugin Manager."
            )
        except Exception as e:
            logger.error(f"Failed to auto-disable {target}: {e}")

    @staticmethod
    def _extract_query(req: LLMRequest) -> str:
        """Pick a recall query from the latest user prompt segment."""
        for p in reversed(req.user_prompt):
            content = getattr(p, "content", "") or ""
            if content.strip():
                return content.strip()
        # Fallback: scan messages list for the last user role entry.
        for msg in reversed(req.messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
        return ""

    def _resolve_session(self, event) -> str:
        """Best-effort session id from a KiraAI event."""
        sid = getattr(event, "sid", None)
        if sid:
            return sid
        session = getattr(event, "session", None)
        if session is not None and getattr(session, "sid", None):
            return session.sid
        return ""

    @staticmethod
    def _coerce_index(value) -> Optional[int]:
        """Parse a tool ``index`` arg to an int, REJECTING non-integral values.

        A bare ``int(1.9)`` silently truncates to 1, so memory_update/remove could
        edit or delete the wrong memory. Returns None on anything non-integral or
        unparseable so the caller surfaces the numbered list instead of acting on
        a guessed index."""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if not f.is_integer():
            return None
        return int(f)

    # ==================================================================
    # Hooks
    # ==================================================================

    @on.im_batch_message(priority=Priority.HIGH)
    async def cache_senders(self, event, *_):
        """Stash sender info per session so the hippocampus can route facts."""
        if self._manager is None:
            return
        sid = self._resolve_session(event)
        if not sid:
            return
        adapter_name = ""
        if getattr(event, "adapter", None) is not None:
            adapter_name = getattr(event.adapter, "name", "") or ""

        messages = getattr(event, "messages", None) or []
        seen_users: dict[str, str] = {}
        for msg in messages:
            sender = getattr(msg, "sender", None)
            if sender is None:
                continue
            user_id = getattr(sender, "user_id", "") or ""
            nickname = getattr(sender, "nickname", "") or ""
            text = getattr(msg, "message_str", "") or ""
            if not user_id and not text:
                continue
            self._sender_cache.record(sid, user_id, nickname, text)
            if user_id and user_id not in seen_users:
                seen_users[user_id] = nickname

        # Maintain the profile's interaction_count and alias history.
        if seen_users and adapter_name:
            for uid, nick in seen_users.items():
                try:
                    await self._manager.update_user_interaction(
                        user_id=f"{adapter_name}:{uid}",
                        platform=adapter_name,
                        nickname=nick,
                    )
                except Exception as e:
                    logger.debug(f"update_user_interaction failed for {uid}: {e}")

    @on.step_result(priority=Priority.LOW)
    async def feed_hippocampus(self, event, step_result, *_):
        """Feed completed (user, assistant) exchanges to the hippocampus."""
        if self._manager is None or self._manager.extractor is None:
            return
        sid = self._resolve_session(event)
        if not sid:
            return
        raw_output = (getattr(step_result, "raw_output", "") or "").strip()
        if not raw_output:
            return

        # Dedup: skip if the same assistant text was just submitted for this sid.
        from hashlib import sha256
        h = sha256(raw_output.encode("utf-8", errors="replace")).hexdigest()[:16]
        if self._last_step_hash.get(sid) == h:
            self._last_step_hash.move_to_end(sid)
            return
        self._last_step_hash[sid] = h
        self._last_step_hash.move_to_end(sid)
        while len(self._last_step_hash) > self._STEP_HASH_CAP:
            self._last_step_hash.popitem(last=False)

        # Pair this assistant turn with the EXACT user message(s) recorded since
        # the previous turn. Each carries its own user_id/nickname (from the
        # @on.im_batch_message cache), so facts route to the right entity even in
        # busy multi-speaker group bursts — no matching the reply back to a user
        # message by text. take_unconsumed advances a per-session watermark, so a
        # multi-step agent loop consumes the user turn once and later steps feed
        # only the assistant text.
        user_msgs = self._sender_cache.take_unconsumed(sid, max_age_sec=600)
        self._manager.submit_exchange(sid, user_msgs, raw_output)

    @on.llm_request(priority=Priority.MEDIUM)
    async def inject_memory(self, event, req: LLMRequest, *_):
        """Inject recalled memories + user profile + tool guidance into the prompt."""
        if self._manager is None:
            return

        # Always append the rule/few-shot guidance (simple_memory is disabled).
        append_to_prompt_section(req.system_prompt, "tools", MEM_TOOL_FEW_SHOT)

        sid = self._resolve_session(event)
        if not sid:
            return
        try:
            entity_id, entity_type = self._manager._parse_entity_from_session(sid)
        except ValueError:
            return
        is_group = entity_type == "group"

        # --- Always-on profile context (parity with lightning's per-turn
        # user_profile injection). In a group this aggregates every participant's
        # profile; in a DM it's the single user's. Independent of recall — the bot
        # should know *who it's talking to* even when nothing is recalled and even
        # when there is no usable recall query (e.g. a sticker-only turn).
        profile_block = ""
        if self.plugin_cfg.get("enable_profile_injection", True):
            try:
                profile_block = await self._manager.build_turn_profile_prompt(
                    sender_users(event), entity_id, entity_type, is_group
                )
            except Exception as e:
                logger.debug(f"Profile injection skipped: {e}")
            # Only truncate on a positive cap; a 0/negative value (bad WebUI
            # input) means "don't truncate" rather than a broken negative slice.
            max_profile = self._int_cfg("max_profile_chars", 800)
            if profile_block and max_profile > 0 and len(profile_block) > max_profile:
                profile_block = profile_block[:max_profile] + "\n…(truncated)"

        # --- Recalled memories (RAG).
        block = ""
        if self.plugin_cfg.get("enable_recall", True):
            # Prefer the clean per-message text from the event over req.user_prompt.
            # By the time this MEDIUM hook runs, kira-ai's SYS_HIGH hook has spliced
            # a message envelope ([date] [message_id: ...] [group_name: ...
            # group_id: ... user_nickname: ..., user_id: ...] | <body>) into
            # req.user_prompt, whose generic words flood the FTS query and
            # false-match stored facts. event.messages[*].message_str holds the
            # envelope-free body, filled before any llm_request hook. Fall back to
            # _extract_query(req) when the event has no usable message text.
            query = query_from_event(event) or self._extract_query(req)
            if query:
                k = self._int_cfg("recall_top_k", 5)
                # Dual-path recall (mirrors lightning's message_manager): always
                # recall the speaking user's own memories, and in a group
                # additionally the group entity, then merge + dedup by id. Without
                # this a group turn recalled ONLY the group entity, so a user's
                # personal memories never surfaced — the "cross-session amnesia"
                # users hit.
                memories: list = []
                seen_ids: set = set()
                for tgt_id, tgt_type in recall_targets(event, entity_id, entity_type):
                    try:
                        hits = await self._manager.recall(
                            query=query, entity_id=tgt_id, entity_type=tgt_type, k=k
                        )
                    except Exception as e:
                        logger.debug(f"Recall failed for {tgt_type}:{tgt_id}: {e}")
                        continue
                    for m in hits:
                        if m.id not in seen_ids:
                            seen_ids.add(m.id)
                            memories.append(m)

                block = self._manager.format_recalled_memories(memories)
                # A 0/negative value (bad WebUI input) means "don't truncate"
                # rather than a broken empty/negative slice — same guard as the
                # profile block above.
                max_chars = self._int_cfg("max_recall_chars", 1500)
                if max_chars > 0 and len(block) > max_chars:
                    block = block[:max_chars] + "\n…(truncated)"

        addition_parts: list[str] = []
        if profile_block:
            addition_parts.append("\n## 用户画像\n" + profile_block)
        if block:
            addition_parts.append("\n## 长期记忆召回\n" + block)
        addition_parts.append("\n" + MEM_RULE_PROMPT)
        addition = "".join(addition_parts)

        if not append_to_prompt_section(req.system_prompt, "memory", addition):
            # No memory section in this prompt — silently skip.
            logger.debug("No 'memory' section in system_prompt; skip inject.")

    # ==================================================================
    # Tools (LLM-callable)
    # ==================================================================

    @register.tool(
        name="memory_add",
        description="Add a memory to long term memory",
        params={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要记录的记忆文本",
                },
            },
            "required": ["text"],
        },
    )
    async def memory_add(self, event, *_, text: str) -> str:
        if self._manager is None:
            return "Memory plugin not initialized"
        text = (text or "").strip()
        if not text:
            return "Empty memory text"
        sid = self._resolve_session(event) if event is not None else ""
        entity_id, entity_type = ("", "user")
        if sid:
            try:
                entity_id, entity_type = self._manager._parse_entity_from_session(sid)
            except ValueError:
                pass

        # Route through the dedup/merge pipeline (like the background hippocampus)
        # so a manual add can't create a duplicate the extractor would collapse,
        # and can merge into an existing memory instead.
        decision = await self._manager.add_fact_curated(
            text=text,
            entity_id=entity_id,
            entity_type=entity_type,
            importance=7,
            tags=["explicit"],
            source={"session": sid, "origin": "memory_add"},
        )
        return {
            "duplicate": "已存在相同记忆，无需重复记录",
            "update": "已合并进已有记忆",
            "new": "已写入长期记忆",
            "stored": "已写入长期记忆",
            "skip": "Empty memory text",
        }.get(decision, "已写入长期记忆")

    @register.tool(
        name="memory_update",
        description="修改特定核心记忆",
        params={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "要修改的记忆编号",
                },
                "text": {
                    "type": "string",
                    "description": "要更新成的记忆文本",
                },
            },
            "required": ["index", "text"],
        },
    )
    async def memory_update(self, event, *_, index: int, text: str) -> str:
        if self._manager is None:
            return "Memory plugin not initialized"
        sid = self._resolve_session(event) if event is not None else ""
        if not sid:
            return "Cannot resolve session"
        text = (text or "").strip()
        if not text:
            return "Empty memory text"
        idx = self._coerce_index(index)
        mems = await self._manager.list_editable_memories(sid)
        if idx is None or idx < 0 or idx >= len(mems):
            listing = self._manager.format_editable_list(mems)
            return (
                f"编号 {index} 无效或超出范围。当前可编辑记忆：\n{listing}"
                if listing
                else "当前没有可修改的记忆"
            )
        mem = await self._manager.update_memory_at(mems, idx, text)
        return "Core memory updated" if mem is not None else "Update failed"

    @register.tool(
        name="memory_remove",
        description="删除一条核心记忆",
        params={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "要删除的记忆编号",
                },
            },
            "required": ["index"],
        },
    )
    async def memory_remove(self, event, *_, index: int) -> str:
        if self._manager is None:
            return "Memory plugin not initialized"
        sid = self._resolve_session(event) if event is not None else ""
        if not sid:
            return "Cannot resolve session"
        idx = self._coerce_index(index)
        mems = await self._manager.list_editable_memories(sid)
        if idx is None or idx < 0 or idx >= len(mems):
            listing = self._manager.format_editable_list(mems)
            return (
                f"编号 {index} 无效或超出范围。当前可删除记忆：\n{listing}"
                if listing
                else "当前没有可删除的记忆"
            )
        ok = await self._manager.delete_memory_at(mems, idx)
        return "Core memory removed" if ok else "Remove failed"

    @register.tool(
        name="memory_search",
        description=(
            "搜索长期记忆（fact 与 reflection）。**搜某个特定用户时，请把该用户的"
            "昵称 / 曾用名 / QQ 号传入 entity_id**（逗号分隔可同时搜多个）。"
            "省略 entity_id 则只搜当前对话的用户与所在群。"
        ),
        params={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询关键词或自然语言问题",
                },
                "entity_id": {
                    "type": "string",
                    "description": (
                        "目标用户：传昵称、曾用名或 QQ 号，系统自动匹配；"
                        "逗号分隔可同时搜多个用户（如 '小明,小红'）。"
                        "想查某个具体的人时务必填这个；省略则只搜当前对话用户与所在群。"
                    ),
                },
                "entity_type": {
                    "type": "string",
                    "description": "实体类型（默认 user）",
                    "enum": ["user", "group", "channel"],
                },
                "k": {
                    "type": "number",
                    "description": "每个实体返回的最大记忆数（默认 5）",
                },
            },
            "required": ["query"],
        },
    )
    async def memory_search(
        self,
        event,
        *_,
        query: str,
        entity_id: str = "",
        entity_type: str = "user",
        k: int = 5,
    ) -> str:
        if self._manager is None:
            return "Memory plugin not initialized"
        q = (query or "").strip()
        if not q:
            return "Empty query"

        sid = self._resolve_session(event) if event is not None else ""
        # Dual-recall fallback (speaking user + group) for when no specific
        # subject can be resolved — keeps a group query from finding nothing.
        fallback: list = []
        if sid:
            try:
                seid, setype = self._manager._parse_entity_from_session(sid)
                fallback = recall_targets(event, seid, setype)
            except ValueError:
                pass

        block = await search_memories(
            manager=self._manager,
            fast_llm=self._manager.get_fast_llm(),
            sender_cache=self._sender_cache,
            sid=sid,
            query=q,
            entity_id=(entity_id or "").strip(),
            entity_type=entity_type or "user",
            k=k or 5,  # search_memories normalises; don't int() here (would raise)
            fallback_targets=fallback,
            list_entities_fn=list_all_entities,
            # LLM-driven "guess which users this query is about" — extra LLM
            # round-trip + fragile; opt-in, default off.
            allow_auto_extract=bool(
                self.plugin_cfg.get("enable_search_auto_detect", False)
            ),
        )
        return block or "暂无相关长期记忆"

    @register.tool(
        name="memory_profile",
        description=(
            "查看某个用户/群的画像（名字、昵称、曾用名、特征、偏好、已知事实）。"
            "entity_id 可传昵称、曾用名或 QQ 号，系统自动匹配；省略时查看当前对话对象。"
        ),
        params={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "目标：昵称/曾用名/QQ号；留空 = 当前对话对象",
                },
                "entity_type": {
                    "type": "string",
                    "description": "实体类型（默认 user）",
                    "enum": ["user", "group", "channel"],
                },
            },
            "required": [],
        },
    )
    async def memory_profile(
        self, event, *_, entity_id: str = "", entity_type: str = "user"
    ) -> str:
        if self._manager is None:
            return "Memory plugin not initialized"
        entity_id = (entity_id or "").strip()
        etype = entity_type or "user"

        resolved = ""
        if entity_id:
            if looks_like_entity_id(entity_id):
                resolved = entity_id
                # A canonical id carries no reliable type marker. If it's
                # obviously group-like ("group"/"群"), prefer the group profile so
                # a group id passed with the default entity_type=user isn't read
                # as a non-existent user profile. (Numeric group ids are
                # indistinguishable from user ids — the caller must set
                # entity_type for those.)
                if looks_like_group_id(entity_id):
                    etype = "group"
            else:
                # Resolve a nickname / alias / QQ number to a canonical id.
                try:
                    resolved = await self._manager.profile_store.resolve_entity_by_name(
                        entity_id, etype
                    ) or ""
                except Exception:
                    resolved = ""
                if not resolved:
                    return f"找不到名为「{entity_id}」的{etype}"

        # No (or unresolved-but-empty) target → current conversation entity.
        if not resolved:
            sid = self._resolve_session(event) if event is not None else ""
            if sid:
                try:
                    resolved, etype = self._manager._parse_entity_from_session(sid)
                except ValueError:
                    resolved = ""
        if not resolved:
            return "无法确定要查看的对象"

        prompt = await self._manager.get_profile_prompt(resolved, etype)
        return prompt or "暂无画像信息"

    # ==================================================================
    # HTTP APIs (debug / WebUI)
    # ==================================================================

    @register.api(method="GET", path="/health", auth=False)
    async def api_health(self) -> Dict[str, Any]:
        ok = self._manager is not None
        count = 0
        if ok:
            try:
                count = self._manager.memory_index.count_memories()
            except Exception:
                pass
        return {"ok": ok, "total_memories": count}

    @register.api(method="POST", path="/recall", auth=True)
    async def api_recall(self, body: dict) -> Dict[str, Any]:
        if self._manager is None:
            return {"error": "not initialized"}
        query = (body or {}).get("query", "")
        entity_id = (body or {}).get("entity_id", "")
        entity_type = (body or {}).get("entity_type", "user")
        k = int((body or {}).get("k", 5))
        memories = await self._manager.recall(query, entity_id, entity_type, k=k)
        return {
            "results": [
                {
                    "id": m.id,
                    "type": m.type,
                    "text": m.text,
                    "importance": m.importance,
                    "tags": m.tags,
                    "entity_id": m._entity_id,
                    "entity_type": m._entity_type,
                    "folder": m._folder,
                }
                for m in memories
            ]
        }

    @register.api(method="GET", path="/entities", auth=True)
    async def api_list_entities(self) -> Dict[str, Any]:
        from .memory.paths import list_all_entities
        try:
            entities = list_all_entities()
        except Exception as e:
            return {"error": str(e), "entities": []}
        return {
            "entities": [
                {"entity_id": eid, "entity_type": etype}
                for eid, etype in entities
            ]
        }

    @register.api(method="GET", path="/profile/{entity_id}", auth=True)
    async def api_get_profile(
        self, entity_id: str, entity_type: str = "user"
    ) -> Dict[str, Any]:
        if self._manager is None:
            return {"error": "not initialized"}
        try:
            profile = await self._manager.get_profile(entity_id, entity_type)
        except Exception as e:
            return {"error": str(e)}
        return profile.to_dict()

    @register.api(method="POST", path="/decay/run", auth=True)
    async def api_run_decay(self) -> Dict[str, Any]:
        if self._manager is None:
            return {"error": "not initialized"}
        deleted, downgraded = await self._manager.run_forgetting_cycle()
        return {"deleted": deleted, "downgraded": downgraded}

    @register.api(method="POST", path="/evolution/run", auth=True)
    async def api_run_evolution(self) -> Dict[str, Any]:
        if self._manager is None:
            return {"error": "not initialized"}
        if self._manager.persona_engine.persona_manager is None:
            return {"error": "persona evolution disabled"}
        await self._manager.run_evolution_cycle()
        return {"ok": True}

    @register.api(method="DELETE", path="/memory/{mem_id}", auth=True)
    async def api_delete_memory(self, mem_id: str) -> Dict[str, Any]:
        if self._manager is None:
            return {"error": "not initialized"}
        meta = self._manager.memory_index.get_meta(mem_id)
        if meta is None:
            return {"error": "not found"}
        ok = await self._manager.tree_store.delete_memory(
            mem_id,
            entity_id=meta.get("entity_id", ""),
            entity_type=meta.get("entity_type", "") or "user",
            folder=meta.get("folder", "facts") or "facts",
            base_dir=meta.get("base_dir", "") or "",
        )
        return {"deleted": bool(ok), "id": mem_id}

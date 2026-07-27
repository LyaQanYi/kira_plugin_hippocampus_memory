"""
实体画像存储层

支持 user / group / channel 三种实体类型。
画像以 profile.json 存储，完全取代旧的 profile.md + YAML frontmatter 方案。
"""

import json
import os
import re
import tempfile
import time
import asyncio
from dataclasses import dataclass, field, fields, asdict
from typing import Awaitable, Callable, Optional

from core.logging_manager import get_logger
from .paths import (
    get_entity_profile_path,
    ensure_entity_dirs,
    ENTITY_USER,
    ENTITY_GROUP,
    ENTITY_CHANNEL,
)

logger = get_logger("entity_profile", "green")

# Backstop cap on a single profile fact — the merge/compaction prompts already
# ask the LLM for short text, but a model that ignores instructions must not
# be able to write an unbounded paragraph into the profile.
_FACT_MAX_CHARS = 200
_UPSERT_MAX_RETRIES = 3

# Matches a "[标签] 内容" prefix left by profile compaction, so to_prompt() can
# group facts by tag instead of dumping an unbounded flat bullet list.
_FACT_TAG_RE = re.compile(r"^\[(.+?)\]\s*(.*)$")

ConflictCheckFn = Callable[[str, str], Awaitable[str]]
MergeFn = Callable[[str, str], Awaitable[str]]


def _clip_fact(text: str) -> str:
    """Hard length backstop, independent of prompt wording (defense in depth)."""
    text = (text or "").strip()
    if len(text) > _FACT_MAX_CHARS:
        text = text[: _FACT_MAX_CHARS - 1].rstrip() + "…"
    return text


def _char_bigrams(text: str) -> set:
    t = re.sub(r"\s+", "", text or "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _similarity(a: str, b: str) -> float:
    """Cheap Jaccard similarity over character bigrams — good enough to rank
    which existing facts are worth an LLM conflict-check call against, without
    pulling in a segmentation dependency here."""
    sa, sb = _char_bigrams(a), _char_bigrams(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _rank_candidates(new_fact: str, facts: list, k: int) -> list:
    """Top-k most similar existing facts (index, text), most similar first.

    Narrows the LLM conflict-check to plausible near-duplicates instead of
    every fact in the list. Zero-score facts still fill unused slots after
    positive matches: a weak literal match must not hide a zero-overlap
    semantic duplicate such as "程序员" vs "从事软件开发"."""
    scored = [(i, f, _similarity(new_fact, f)) for i, f in enumerate(facts)]
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(i, f) for i, f, _s in scored[:k]]


@dataclass
class EntityProfile:
    """通用实体画像数据类

    适用于 user、group、channel 三种实体。
    序列化为 profile.json。
    """

    entity_id: str
    entity_type: str = ENTITY_USER  # user / group / channel

    name: str = ""
    nickname: str = ""
    description: str = ""
    platform: str = ""

    # 特征标签（["耐心", "技术导向", ...]）
    traits: list = field(default_factory=list)
    # 偏好字典（{"theme": "dark", "language": "zh"}）
    preferences: dict = field(default_factory=dict)
    # 关系图（{"user_456": "好友", "group_123": "管理员"}）
    relationships: dict = field(default_factory=dict)
    # 已知核心事实（高度浓缩的关键信息）
    facts: list = field(default_factory=list)
    # 曾用名/历史昵称（用于昵称反查，自动维护）
    aliases: list = field(default_factory=list)

    interaction_count: int = 0
    last_interaction: float = 0.0

    # 自由扩展区
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EntityProfile":
        """从字典反序列化（忽略多余字段）"""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_prompt(self) -> str:
        """格式化为 LLM Prompt 文本"""
        parts = []
        if self.name:
            parts.append(f"名字: {self.name}")
        if self.nickname and self.nickname != self.name:
            parts.append(f"昵称: {self.nickname}")
        if self.aliases:
            parts.append(f"曾用名: {', '.join(self.aliases)}")
        if self.platform:
            parts.append(f"平台: {self.platform}")
        if self.description:
            parts.append(f"描述: {self.description}")
        if self.traits:
            parts.append(f"特征: {', '.join(self.traits)}")
        if self.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.preferences.items())
            parts.append(f"偏好: {prefs}")
        if self.relationships:
            rels = ", ".join(f"{k}: {v}" for k, v in self.relationships.items())
            parts.append(f"关系: {rels}")
        if self.facts:
            # Group by a "[标签]" prefix (left by profile compaction) so the
            # rendered block reads as organized sections rather than an
            # unbounded flat list; untagged (legacy) facts fall into "其他".
            grouped: dict[str, list[str]] = {}
            for f in self.facts:
                m = _FACT_TAG_RE.match(f)
                tag, body = (m.group(1), m.group(2)) if m else ("其他", f)
                grouped.setdefault(tag, []).append(body or f)
            fact_lines = [
                f"- [{tag}] {item}"
                for tag, items in grouped.items()
                for item in items
            ]
            facts_str = "\n  ".join(fact_lines)
            parts.append(f"已知事实:\n  {facts_str}")
        if self.interaction_count:
            parts.append(f"互动次数: {self.interaction_count}")
        return "\n".join(parts) if parts else "暂无画像信息"


class EntityProfileStore:
    """实体画像 CRUD 管理器

    所有操作均为异步。画像存储为:
    data/memory/entities/{type}_{id}/profile.json
    """

    def __init__(self):
        # Serializes the actual file write. Per-entity locks below coordinate
        # the wider read-modify-write sequences (upsert / compact merge) so two
        # mutations for the same entity can't interleave between re-read and
        # save (Greptile P1: concurrent fact still lost after naive re-merge).
        self._lock = asyncio.Lock()
        self._entity_locks: dict[str, asyncio.Lock] = {}
        logger.info("EntityProfileStore initialized")

    def _get_entity_lock(self, entity_id: str, entity_type: str) -> asyncio.Lock:
        key = f"{entity_type}:{entity_id}"
        lock = self._entity_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._entity_locks[key] = lock
        return lock

    async def _read_existing_profile(
        self, entity_id: str, entity_type: str
    ) -> Optional[EntityProfile]:
        """Read an existing profile without creating or rewriting it."""
        fpath = get_entity_profile_path(entity_id, entity_type)
        if not os.path.exists(fpath):
            return None
        try:
            data = await asyncio.to_thread(self._sync_read, fpath)
            return EntityProfile.from_dict(data)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to read profile {fpath}: {e}")
            raise

    async def get_profile(
        self, entity_id: str, entity_type: str = ENTITY_USER
    ) -> EntityProfile:
        """获取实体画像，不存在则创建默认画像"""
        async with self._get_entity_lock(entity_id, entity_type):
            return await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )

    async def _get_or_create_profile_unlocked(
        self, entity_id: str, entity_type: str
    ) -> EntityProfile:
        """Get or create a profile while the caller holds its entity lock."""
        profile = await self._read_existing_profile(entity_id, entity_type)
        if profile is not None:
            return profile

        # 创建默认画像
        profile = EntityProfile(entity_id=entity_id, entity_type=entity_type)
        await self.save_profile(profile)
        return profile

    async def save_profile(self, profile: EntityProfile) -> bool:
        """保存画像到文件"""
        ensure_entity_dirs(profile.entity_id, profile.entity_type)
        fpath = get_entity_profile_path(profile.entity_id, profile.entity_type)

        async with self._lock:
            try:
                await asyncio.to_thread(self._sync_write, fpath, profile.to_dict())
                return True
            except Exception as e:
                logger.error(f"Failed to save profile {fpath}: {e}")
                return False

    async def update_profile(
        self, entity_id: str, entity_type: str = ENTITY_USER, **kwargs
    ) -> Optional[EntityProfile]:
        """部分更新画像字段。

        Returns: 更新后的画像；持久化失败时返回 ``None``——调用方（例如
        ``HippocampusManager.compact_profile``）不能想当然认为字段已经落盘，
        必须据此判断更新是否真的生效，而不是无视 ``save_profile`` 的布尔
        返回值直接报告成功。

        持有 per-entity 锁，避免与 ``apply_compacted_facts`` / ``upsert_fact``
        交叉覆盖 ``facts``。
        """
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            allowed = {f.name for f in fields(EntityProfile)} - {
                "entity_id",
                "entity_type",
            }

            for key, value in kwargs.items():
                if key in allowed:
                    setattr(profile, key, value)

            profile.last_interaction = time.time()
            ok = await self.save_profile(profile)
            return profile if ok else None

    # === 便捷方法 ===

    async def add_trait(self, entity_id: str, trait: str, entity_type: str = ENTITY_USER):
        """添加特征标签（大小写/首尾空白归一化后去重，同款 bug 与 add_fact 一并修复）"""
        trait = (trait or "").strip()
        if not trait:
            return
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            normalized = trait.lower()
            if not any(t.strip().lower() == normalized for t in profile.traits):
                profile.traits.append(trait)
                await self.save_profile(profile)

    async def remove_trait(self, entity_id: str, trait: str, entity_type: str = ENTITY_USER):
        """移除特征标签"""
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            if trait in profile.traits:
                profile.traits.remove(trait)
                await self.save_profile(profile)

    async def add_fact(self, entity_id: str, fact: str, entity_type: str = ENTITY_USER):
        """添加核心事实到画像（精确字符串去重；语义去重见 upsert_fact）"""
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            if fact not in profile.facts:
                profile.facts.append(fact)
                await self.save_profile(profile)

    async def apply_compacted_facts(
        self,
        entity_id: str,
        entity_type: str,
        snapshot: list,
        compacted: list[str],
    ) -> Optional[EntityProfile]:
        """原子地写入压实结果，并保留 snapshot 之后并发追加的 facts。

        整个 re-read → merge → save 持有该实体的锁，与 ``upsert_fact`` /
        ``add_fact`` 互斥，堵住「re-read 之后、save 之前又有写入」的 TOCTOU
        空窗（Greptile P1 follow-up on PR #13）。

        若 snapshot 中任一条在当前 facts 里已缺失（并发删除或原位替换），
        放弃本次压实并返回 ``None``，避免旧摘要复活过期事实（CodeRabbit）。
        """
        async with self._get_entity_lock(entity_id, entity_type):
            # Compaction is a compare-and-swap against an existing snapshot.
            # A missing file is a failed precondition, not a reason to create
            # and persist an empty profile before returning failure. Other read
            # errors propagate so callers cannot mistake corruption for absence.
            profile = await self._read_existing_profile(entity_id, entity_type)
            if profile is None:
                return None
            if any(f not in profile.facts for f in snapshot):
                return None
            appended = [f for f in profile.facts if f not in snapshot]
            final_facts = list(compacted) + [f for f in appended if f not in compacted]
            profile.facts = final_facts
            profile.last_interaction = time.time()
            ok = await self.save_profile(profile)
            return profile if ok else None

    async def upsert_fact(
        self,
        entity_id: str,
        fact: str,
        entity_type: str = ENTITY_USER,
        *,
        conflict_check: Optional[ConflictCheckFn] = None,
        merge: Optional[MergeFn] = None,
        candidate_k: int = 3,
        replace_on_duplicate: bool = False,
    ) -> str:
        """语义去重/合并后写入画像事实，取代裸 append（画像去重精简 #2）。

        1. 精确匹配（大小写/空白归一化）→ 直接跳过，零 LLM 调用
        2. 用字符 bigram 相似度挑出最相近的 ``candidate_k`` 条已有事实
        3. 若提供 ``conflict_check``（复用 MemoryExtractor._check_conflict），
           逐条判断 duplicate/update/new
        4. ``update`` 时用 ``merge``（复用 MemoryExtractor.merge_facts）合并后
           原位替换；``new`` 才追加

        ``replace_on_duplicate`` is reserved for an authoritative upstream
        merge result. When enabled, a semantic duplicate is replaced with the
        supplied canonical text instead of leaving stale profile wording.

        未提供 ``conflict_check``（如未配置 LLM）时退化为精确去重 + 直接追加，
        保证在没有 LLM 的情况下也能正常工作。

        LLM conflict/merge calls run outside the per-entity lock. Each attempt
        snapshots facts under the lock, performs model work without it, then
        uses a short compare-and-swap section to preserve concurrent mutations.

        Returns: "duplicate" | "update" | "new" | "skip" | "error"（"error" =
        决策已判定但落盘失败——调用方不能把它当成功处理，见 ``save_profile``
        的布尔返回值检查）
        """
        # Normalize to the persisted representation before exact matching.
        # Otherwise a long fact is compared untrimmed on every call while the
        # stored copy is clipped, so identical inputs can accumulate forever.
        fact = _clip_fact(fact)
        if not fact:
            return "skip"

        normalized = fact.lower()
        entity_lock = self._get_entity_lock(entity_id, entity_type)

        for _attempt in range(_UPSERT_MAX_RETRIES):
            async with entity_lock:
                profile = await self._get_or_create_profile_unlocked(
                    entity_id, entity_type
                )
                if any(
                    existing.strip().lower() == normalized
                    for existing in profile.facts
                ):
                    return "duplicate"

                snapshot = list(profile.facts)
                if conflict_check is None or not snapshot:
                    profile.facts.append(fact)
                    return (
                        "new"
                        if await self.save_profile(profile)
                        else "error"
                    )

            # Model calls can take seconds and must never hold the entity lock:
            # interaction updates for the live user share the same lock.
            candidates = _rank_candidates(fact, snapshot, candidate_k)
            action = "new"
            target_idx = -1
            target_existing = ""
            replacement = ""

            for idx, existing in candidates:
                try:
                    decision = await conflict_check(fact, existing)
                except Exception as e:
                    logger.debug(
                        f"Profile upsert conflict_check failed: {type(e).__name__}"
                    )
                    continue
                if decision not in ("duplicate", "update"):
                    continue

                target_idx = idx
                target_existing = existing
                if replace_on_duplicate and decision in ("duplicate", "update"):
                    action = "update"
                    replacement = fact
                elif decision == "duplicate":
                    action = "duplicate"
                else:
                    action = "update"
                    if merge is not None:
                        try:
                            replacement = _clip_fact(
                                await merge(existing, fact)
                            )
                        except Exception as e:
                            logger.debug(
                                f"Profile upsert merge failed: {type(e).__name__}"
                            )
                            return "error"
                    else:
                        replacement = _clip_fact(f"{existing}；{fact}")
                    if not replacement:
                        return "error"
                break

            async with entity_lock:
                current = await self._get_or_create_profile_unlocked(
                    entity_id, entity_type
                )
                if any(
                    existing.strip().lower() == normalized
                    for existing in current.facts
                ):
                    return "duplicate"

                if action == "new":
                    if current.facts != snapshot:
                        continue
                    current.facts.append(fact)
                    return (
                        "new"
                        if await self.save_profile(current)
                        else "error"
                    )

                if (
                    target_idx >= len(current.facts)
                    or current.facts[target_idx] != target_existing
                ):
                    continue
                if action == "duplicate":
                    return "duplicate"

                current.facts[target_idx] = replacement
                return (
                    "update"
                    if await self.save_profile(current)
                    else "error"
                )

        logger.debug(
            f"Profile upsert CAS retries exhausted for {entity_type}:{entity_id}"
        )
        return "error"

    async def update_fact(
        self, entity_id: str, old_fact: str, new_fact: str, entity_type: str = ENTITY_USER
    ):
        """更新画像中的事实"""
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            if old_fact in profile.facts:
                idx = profile.facts.index(old_fact)
                profile.facts[idx] = new_fact
                await self.save_profile(profile)

    async def remove_fact(self, entity_id: str, fact: str, entity_type: str = ENTITY_USER):
        """移除画像中的事实"""
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            if fact in profile.facts:
                profile.facts.remove(fact)
                await self.save_profile(profile)

    async def set_relationship(
        self,
        entity_id: str,
        target: str,
        relation: str,
        entity_type: str = ENTITY_USER,
    ):
        """设置关系"""
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            profile.relationships[target] = relation
            await self.save_profile(profile)

    async def increment_interaction(
        self, entity_id: str, entity_type: str = ENTITY_USER, **extra_updates
    ):
        """递增交互计数并可选更新其他字段

        特别处理 nickname 变更：旧昵称自动归档到 aliases。

        整段 RMW 持有 per-entity 锁，避免与压实/upsert 交叉时用陈旧
        整对象覆盖掉新写入的 facts（Greptile P1 / CodeRabbit on PR #13）。
        """
        async with self._get_entity_lock(entity_id, entity_type):
            profile = await self._get_or_create_profile_unlocked(
                entity_id, entity_type
            )
            profile.interaction_count += 1
            profile.last_interaction = time.time()

            # 昵称变更时，将旧昵称归档到 aliases
            new_nickname = extra_updates.get("nickname", "")
            if new_nickname and profile.nickname and new_nickname != profile.nickname:
                if profile.nickname not in profile.aliases:
                    profile.aliases.append(profile.nickname)

            allowed = {f.name for f in fields(EntityProfile)} - {"entity_id", "entity_type"}
            for key, value in extra_updates.items():
                if key in allowed:
                    setattr(profile, key, value)

            await self.save_profile(profile)

    async def resolve_entity_by_name(
        self, name_query: str, entity_type: str = ENTITY_USER
    ) -> Optional[str]:
        """通过昵称、QQ号或名字反查 entity_id

        扫描所有指定类型的实体画像，匹配优先级：
        1. entity_id 尾部精确匹配（纯数字 QQ 号）
        2. name/nickname 精确匹配
        3. name/nickname 包含匹配
        返回最佳匹配的 entity_id，找不到则返回 None。
        """
        from .paths import list_all_entities

        if not name_query or not name_query.strip():
            return None

        query = name_query.strip()
        query_lower = query.lower()
        is_numeric = query.isdigit()
        candidates = []

        entities = list(list_all_entities(entity_type))

        # 优先级 1：entity_id 尾部匹配（adapter:qq_number 格式），无需读 profile
        # 用户传入纯 QQ 号 "123456789"，匹配 "onebot:123456789"
        if is_numeric:
            for eid, _etype in entities:
                if eid.endswith(f":{query}"):
                    return eid

        # 并发读取所有 profile（顺序逐个 await 在用户多时会很慢）
        profiles = await asyncio.gather(
            *(self.get_profile(eid, etype) for eid, etype in entities),
            return_exceptions=True,
        )

        for (eid, _etype), profile in zip(entities, profiles):
            if isinstance(profile, Exception):
                continue

            # 优先级 2：name/nickname/aliases 精确匹配（按目录顺序，第一个命中即返回）
            if profile.name and profile.name.lower() == query_lower:
                return eid
            if profile.nickname and profile.nickname.lower() == query_lower:
                return eid
            for alias in profile.aliases:
                if alias and alias.lower() == query_lower:
                    return eid

            # 优先级 3：包含匹配（兜底）
            if profile.name and query_lower in profile.name.lower():
                candidates.append((eid, len(profile.name)))
            elif profile.nickname and query_lower in profile.nickname.lower():
                candidates.append((eid, len(profile.nickname)))
            else:
                for alias in profile.aliases:
                    if alias and query_lower in alias.lower():
                        candidates.append((eid, len(alias)))
                        break

        # 返回名字最短的（最精确匹配）
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    async def get_profile_prompt(
        self, entity_id: str, entity_type: str = ENTITY_USER
    ) -> str:
        """获取画像的 Prompt 格式文本"""
        profile = await self.get_profile(entity_id, entity_type)
        return profile.to_prompt()

    async def delete_profile(
        self, entity_id: str, entity_type: str = ENTITY_USER
    ) -> bool:
        """删除画像文件"""
        async with self._get_entity_lock(entity_id, entity_type):
            fpath = get_entity_profile_path(entity_id, entity_type)
            try:
                if os.path.exists(fpath):
                    await asyncio.to_thread(os.remove, fpath)
                    return True
            except Exception as e:
                logger.error(f"Failed to delete profile {fpath}: {e}")
        return False

    # === 内部同步 IO ===

    @staticmethod
    def _sync_read(fpath: str) -> dict:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _sync_write(fpath: str, data: dict):
        directory = os.path.dirname(fpath)
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".profile-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, fpath)
        except Exception:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            raise

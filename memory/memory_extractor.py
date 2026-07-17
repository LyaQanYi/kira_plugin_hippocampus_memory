"""
海马体核心逻辑 — 记忆提取、去重、合并、升维

负责从对话中提取事实 → 去重审查 → 合并同义记忆 → 触发升维反思。
遵循宪章 Agent 行为守则的四条铁律。

去重流程（两级）:
1. SHA-256 内容哈希精确去重（零 LLM 调用）
2. FTS5 语义搜索 + LLM 判断（duplicate/update/new）

语义 ID 生成:
- LLM 从事实内容生成简短的 snake_case slug（如 "hates_css"）
- 回退：从文本前缀 + hash 生成
"""

import ast
import json
import re
import time
from typing import Optional

from core.logging_manager import get_logger
from .toml_tree_store import TomlTreeStore, Memory
from .memory_index import MemoryIndex
from ..adapters.llm import chat_text

logger = get_logger("memory_extractor", "green")

# Cap the persona brief injected into subjective extractions so the extra
# system prompt stays cheap and can't dominate the context (issue #4).
_PERSONA_BRIEF_MAX = 800

# Backstop cap on a merged fact/reflection/profile-summary line. The merge
# and compaction prompts already ask for short output, but a model that
# ignores instructions must not be able to write an unbounded paragraph.
_MERGE_MAX_CHARS = 200

# Profile compaction: cap how many bullets a compacted profile keeps, and the
# tags the LLM is asked to sort facts into.
_PROFILE_COMPACT_MAX_FACTS = 12
_PROFILE_FACT_TAGS = ("身份", "性格", "技能兴趣", "互动习惯", "关系", "其他")


def _clip_merged(text: str) -> str:
    text = (text or "").strip()
    if len(text) > _MERGE_MAX_CHARS:
        text = text[:_MERGE_MAX_CHARS].rstrip() + "…"
    return text


class MemoryExtractor:
    """海马体：事实提取 → 去重 → 合并 → 升维"""

    def __init__(self, tree_store: TomlTreeStore, llm_client=None):
        self.tree_store = tree_store
        self.index: MemoryIndex = tree_store.index
        self._llm_client = llm_client
        self._fast_llm_client = None  # 轻量模型，用于去重/合并等低复杂度任务
        # 角色视角简介（issue #4）：仅注入到"主观类"提取（群氛围、reflection、
        # 自我觉察），让这些内容站在角色本人视角而非中立旁观者视角。空 = 关闭。
        self._persona_brief = ""

        # 升维阈值：facts 积累达到此数量时触发反思
        self.reflection_threshold = 5

    def set_llm_client(self, llm_client):
        self._llm_client = llm_client

    def set_fast_llm_client(self, fast_llm_client):
        """设置轻量 LLM 客户端，用于去重/合并（回退到 _llm_client）"""
        self._fast_llm_client = fast_llm_client

    def get_fast_client(self):
        """Public accessor for the fast (or fallback) LLM client."""
        return self._fast_or_default

    def set_persona_brief(self, persona_brief: str) -> None:
        """设置角色人设简介，用于给主观类提取注入角色视角（issue #4）。

        只读人设、绝不回写。空字符串则关闭（提取退回中立旁观视角）。
        超长会截断以控制提取 token 成本。
        """
        text = (persona_brief or "").strip()
        if len(text) > _PERSONA_BRIEF_MAX:
            text = text[:_PERSONA_BRIEF_MAX].rstrip() + "…"
        self._persona_brief = text

    def _persona_system(self) -> Optional[str]:
        """构造"角色视角"system prompt，仅用于主观类提取。

        未配置人设简介时返回 None —— 此时提取保持中立旁观视角（客观事实提取
        以及本特性关闭时都走这条路）。提示里明确"人设只是背景、不得当作对话事实"，
        以免模型把人设设定误当成抽取到的记忆。
        """
        brief = self._persona_brief
        if not brief:
            return None
        return (
            "你正在以一个特定角色的身份回顾刚刚的对话。下面是你的人设简介，"
            "它**只用来**让你站在角色本人的主观视角去体会感受（例如对群聊氛围的感觉、"
            "对自己表现的评价），而不是让你扮演中立的旁观者。\n"
            "铁律：人设简介只是你的背景，**绝不能**把简介里描写的设定当成这段对话里"
            "真实发生、可被记录的事实；你的输出必须完全基于实际对话内容来推断。\n\n"
            f"【你的人设简介】\n{brief}"
        )

    @property
    def _fast_or_default(self):
        """获取快速 LLM 客户端，未设置则回退到主 LLM"""
        return self._fast_llm_client or self._llm_client

    # ==========================================
    # 事实提取（双路径）
    # ==========================================

    async def extract_personal_facts(self, conversation_text: str) -> list[dict]:
        """从对话中提取个人事实（用户级）

        专注于每位用户的偏好、身份、经历、观点、习惯等。
        结果将路由到各用户的 entity 目录下。

        Returns:
            [{"content": "...", "importance": 7, "tags": [...],
              "speaker_id": "12345", "subject": "昵称", "semantic_id": "..."}, ...]
        """
        if not self._llm_client:
            return []

        prompt = f"""分析以下对话片段，提取每位用户的**个人事实**。忽略寒暄和无意义内容。
对话中每位用户的格式为 "昵称(ID): 内容"，请注意区分不同用户。如果对话前面附带
"## 参与者已知信息" 或 "## 已有相关记忆" 部分，那是该用户已经记录过的内容——
**不要重复提取其中已经覆盖的信息**；只有出现实质性新变化或补充时才提取，且只
写变化点本身，不要重复旧信息的细节。

只关注个人层面的信息，包括：
- 用户的偏好、喜好、厌恶
- 身份信息（职业、年龄、所在地等）
- 个人经历、故事
- 观点、立场
- 习惯、性格特征

**格式要求**：
- 每条事实是一句**原子短句**（≤40 字），一事实一事，不要用"并且/而且/还"
  把多件事拼进一条
- 反复出现的互动习惯（比如"喜欢拍/摸 bot 的头"这类动作）只记录一次稳定
  模式，不要因为对话里发生了好几次就写好几条相似的事实——这几轮如果已经
  记过同类习惯，这轮就不用再写

对话:
{conversation_text}

请以 JSON 数组格式输出，每条事实包含：
- "speaker_id": 该事实所属用户的 ID（从对话中括号内提取，如 "12345"）
- "subject": 该用户的昵称
- "content": 事实描述，用该用户昵称作主语，写成简短陈述句（≤40字）。例如：✅ "小明喜欢用Python" ✅ "阿花是一名大三学生" ❌ "该用户喜欢Python"（禁止使用"该用户"）
- "importance": 重要性评分(1-10)
- "tags": 相关标签数组
- "semantic_id": 简短 snake_case 标识符（如 "xiaoming_likes_python"）

**严禁使用"该用户""该成员""此人"等模糊代词，必须用具体昵称。**

只输出 JSON 数组，不要有其他内容。如果没有值得记录的个人事实，输出空数组 []。"""

        try:
            text = await chat_text(self._llm_client, prompt)
            if text:
                return self._parse_json_array(text)
        except Exception as e:
            logger.error(f"Personal fact extraction error: {e}")
        return []

    async def extract_group_facts(self, conversation_text: str) -> list[dict]:
        """从对话中提取群组事实（群级）

        专注于群聊整体的信息：氛围、话题、成员关系、群体特征。
        结果将路由到群组 entity 目录下。

        Returns:
            [{"content": "...", "importance": 7, "tags": [...],
              "subject": "group", "semantic_id": "..."}, ...]
        """
        if not self._llm_client:
            return []

        persona_system = self._persona_system()
        persona_clause = (
            "\n**主观视角要求**：对于「群体氛围、群文化特征」这类带主观评价的内容，"
            "请站在你（角色本人）的视角如实写下你的真实感受，而不是中立旁观者的口吻——"
            "例如一个怕吵的角色面对满屏刷屏，应记成「群里太吵，让你觉得烦」，"
            "而不是「群里氛围热闹轻松」。对「话题方向、成员关系、群内事件」等客观信息仍保持中立客观。"
            "只依据本段对话判断，不要照搬人设设定。\n"
        ) if persona_system else ""

        prompt = f"""分析以下群聊对话片段，提取**群组级别**的信息。忽略寒暄和无意义内容。
对话中每位用户的格式为 "昵称(ID): 内容"。如果对话前面附带 "## 参与者已知信息"
或 "## 已有相关记忆" 部分，那是已经记录过的内容——**不要重复提取其中已经覆盖的
信息**，只有出现实质性新变化时才提取，且只写变化点本身。

只关注群聊层面的信息，包括：
- 群聊的常见话题和讨论方向
- 群体氛围、文化特征
- 成员之间的互动关系和社交动态（如"小明和阿花经常互怼"）
- 群内的共识、群规、惯例
- 群内事件（如群友组织活动、群聊里发生的趣事）

**格式要求**：
- 每条事实是一句**原子短句**（≤40 字），一事实一事
- 反复出现的互动模式（比如某人反复刷屏、某两人反复互怼）只记录一次稳定
  模式，不要为每次发生单独写一条
{persona_clause}
对话:
{conversation_text}

请以 JSON 数组格式输出，每条事实包含：
- "speaker_id": 留空 ""
- "subject": "group"
- "content": 事实描述，写成关于群聊的简短陈述句（≤40字）。涉及具体成员时必须用昵称，例如：✅ "群里最近在讨论AI绘画" ✅ "小明和阿花经常在群里互怼" ✅ "群友们普遍偏好深夜聊天" ❌ "该用户经常发言"（禁止使用"该用户"，且这不是群级信息）
- "importance": 重要性评分(1-10)
- "tags": 相关标签数组
- "semantic_id": 简短 snake_case 标识符（如 "group_discusses_ai_art"）

**严禁使用"该用户""该成员""此人"等模糊代词，涉及具体人时用昵称。**
**不要提取个人偏好/身份等个人事实，那些由另一个流程处理。**

只输出 JSON 数组，不要有其他内容。如果没有值得记录的群组事实，输出空数组 []。"""

        try:
            text = await chat_text(self._llm_client, prompt, system=persona_system)
            if text:
                return self._parse_json_array(text)
        except Exception as e:
            logger.error(f"Group fact extraction error: {e}")
        return []

    async def extract_facts(self, conversation_text: str) -> list[dict]:
        """从对话中提取事实（私聊兼容接口）

        私聊场景只有一个用户，不需要双路径，走单次提取即可。
        """
        if not self._llm_client:
            return []

        prompt = f"""分析以下对话片段，提取关键事实。忽略寒暄和无意义内容。
对话中用户的格式为 "昵称(ID): 内容"。如果对话前面附带 "## 参与者已知信息" 或
"## 已有相关记忆" 部分，那是已经记录过的内容——**不要重复提取其中已经覆盖的
信息**，只有出现实质性新变化或补充时才提取，且只写变化点本身。

**格式要求**：
- 每条事实是一句**原子短句**（≤40 字），一事实一事
- 反复出现的互动习惯只记录一次稳定模式，不要为每次发生单独写一条

对话:
{conversation_text}

请以 JSON 数组格式输出，每条事实包含：
- "speaker_id": 该事实所属用户的 ID（从对话中括号内提取，如 "12345"）
- "subject": 该用户的昵称
- "content": 事实描述，用昵称作主语，写成简短陈述句（≤40字）。例如：✅ "小明喜欢吃辣" ❌ "该用户喜欢吃辣"
- "importance": 重要性评分(1-10)
- "tags": 相关标签数组
- "semantic_id": 简短 snake_case 标识符（如 "xiaoming_likes_spicy"）

**严禁使用"该用户"等模糊代词，必须用具体昵称。**

只输出 JSON 数组，不要有其他内容。如果没有值得记录的事实，输出空数组 []。"""

        try:
            text = await chat_text(self._llm_client, prompt)
            if text:
                return self._parse_json_array(text)
        except Exception as e:
            logger.error(f"Fact extraction error: {e}")
        return []

    # ==========================================
    # 自我觉察提取（Phase 1: 只存不读）
    # ==========================================

    async def extract_self_awareness(
        self, conversation_text: str, ai_response_text: str = ""
    ) -> list[str]:
        """从对话中提取 AI 关于自身行为的觉察

        Phase 1 只存不读：觉察写入 global/self/facts/，不影响召回。
        大部分对话不应产出觉察（返回空列表）。只有当 AI 在这次互动中
        表现出明显的行为模式时才记录。

        Args:
            conversation_text: 本轮对话全文
            ai_response_text: AI 在这轮对话中的回复文本（可选）

        Returns:
            觉察文本列表（通常 0-2 条，大部分情况为空）
        """
        if not self._llm_client:
            return []

        response_section = ""
        if ai_response_text:
            response_section = f"\n\n你的回复:\n{ai_response_text}"

        persona_system = self._persona_system()
        persona_clause = (
            "\n（请结合你的角色设定来审视自己的表现是否贴合人设，"
            "但只依据这次对话里你的实际行为来判断，不要照搬设定本身。）\n"
        ) if persona_system else ""

        prompt = f"""你刚刚参与了一段对话。请回顾这次互动，思考你自己在这次对话中的**行为表现**。

对话内容:
{conversation_text}{response_section}

请思考：
- 你的回复风格有什么特点？（比如偏啰嗦/偏简短、语气偏冷/偏热情）
- 你处理这类话题/这类用户时有什么倾向？
- 有没有什么做得不好的地方，或者做得特别好的地方？
- 你注意到自己的什么习惯或模式？
{persona_clause}
**输出要求**：
- 只关注你自己的行为模式，不要总结对话内容
- 每条觉察必须以"我"开头（例如："我在回答技术问题时倾向于给出过于详细的解释"）
- 只输出有价值的觉察，不要为了输出而输出
- 如果这次对话没有值得记录的行为觉察，直接输出 NONE
- 如果有，每条一行，最多2条

直接输出觉察内容或 NONE，不要有其他内容。"""

        try:
            text = (await chat_text(self._llm_client, prompt, system=persona_system)).strip()
            if not text or text.upper() == "NONE":
                return []
            insights = [
                line.strip()
                for line in text.split("\n")
                if line.strip() and line.strip().upper() != "NONE"
            ]
            # 过滤：必须以"我"开头，且长度合理
            insights = [
                s for s in insights
                if s.startswith("我") and 5 < len(s) < 200
            ]
            return insights[:2]  # 最多 2 条
        except Exception as e:
            logger.error(f"Self-awareness extraction error: {e}")
        return []

    # ==========================================
    # 语义 ID 生成
    # ==========================================

    async def generate_semantic_id(self, content: str) -> str:
        """让 LLM 生成语义化 slug ID

        回退策略：文本前缀 + hash
        """
        if not self._llm_client:
            return ""

        prompt = f"""为以下记忆内容生成一个简短的 snake_case 文件名标识符（英文，无空格，不超过 30 字符）。
例如：hates_css, loves_python, pet_cat_xiaoju, prefers_dark_mode

内容: {content}

只输出标识符，不要有其他内容。"""

        try:
            slug = (await chat_text(self._llm_client, prompt)).strip().lower()
            # 清理非法字符
            slug = re.sub(r"[^a-z0-9_]", "_", slug)
            slug = re.sub(r"_+", "_", slug).strip("_")
            if slug and len(slug) <= 40:
                return slug
        except Exception as e:
            logger.debug(f"Semantic ID generation failed: {e}")
        return ""

    # ==========================================
    # 去重审查（宪章铁律 #1）
    # ==========================================

    async def deduplicate(
        self,
        new_content: str,
        entity_id: str,
        entity_type: str = "user",
        folder: str = "facts",
    ) -> tuple[str, Optional[Memory]]:
        """两级去重：SHA-256 精确匹配 → FTS5 语义搜索 + LLM 判断

        Returns:
            (decision, matched_memory)
            decision: "duplicate" | "update" | "new"
            matched_memory: 匹配到的旧记忆（仅 duplicate/update 时非 None）
        """
        # === 第一级：SHA-256 精确去重（零 LLM 调用） ===
        content_hash = MemoryIndex.content_hash(new_content)
        exact_match = self.index.find_by_hash(
            content_hash, entity_id, entity_type, folder
        )
        if exact_match:
            logger.debug(f"Exact hash match: {new_content[:50]}...")
            return "duplicate", None

        # === 第二级：FTS5 语义搜索 + LLM 判断（多候选） ===
        existing = await self.tree_store.search(
            query=new_content,
            entity_id=entity_id,
            entity_type=entity_type,
            folder=folder,
            k=3,
            update_access=False,
        )

        if not existing:
            return "new", None

        # 逐条检查，命中即返回（按相似度排序，最相似的先检查）
        for candidate in existing:
            decision = await self._check_conflict(new_content, candidate.text)
            if decision in ("duplicate", "update"):
                return decision, candidate

        return "new", None

    async def _check_conflict(self, new_content: str, existing_content: str) -> str:
        """用 LLM 判断新旧记忆的关系（使用快速模型）"""
        client = self._fast_or_default
        if not client:
            return "new"

        prompt = f"""比较以下两条信息，判断它们的关系：

已有信息: {existing_content}
新信息: {new_content}

只输出以下三个选项之一：
- "duplicate"：新信息与已有信息基本相同，或只是同一习惯/事实的不同措辞，
  无需记录。例如「喜欢被摸头」和「喜欢别人拍他的头」是同一习惯的不同说法，
  应判定为 duplicate；「今天又被摸头了」和「喜欢被摸头」也是同一习惯的
  不同表述，同样是 duplicate
- "update"：新信息是对已有信息的更新、补充或程度变化，需要合并
- "new"：新信息与已有信息无关，是全新信息

只输出选项文本，不要有其他内容。"""

        try:
            result = (await chat_text(client, prompt)).strip().strip('"').lower()
            if result in ("duplicate", "update", "new"):
                return result
        except Exception as e:
            logger.error(f"Conflict check error: {e}")
        return "new"

    # ==========================================
    # 合并
    # ==========================================

    async def merge_facts(self, existing_text: str, new_text: str) -> str:
        """LLM 合并两条事实为一条（使用快速模型），目标是更短而不是更长。

        代码层加保险性硬截断（``_clip_merged``），不完全依赖提示词自觉——
        模型不听话仍写长文本时，也不会让画像/记忆无限膨胀。
        """
        client = self._fast_or_default
        if not client:
            return _clip_merged(f"{existing_text}；{new_text}")

        prompt = f"""将以下两条信息合并为一条**更短**的陈述，删除同义重复的内容，
禁止堆砌举例或罗列多次发生的细节，只保留最新、最准确的结论：

已有信息: {existing_text}
新信息: {new_text}

直接输出合并后的结果（尽量控制在 40 字以内），不要有其他内容。"""

        try:
            merged = (await chat_text(client, prompt)).strip()
            if merged:
                return _clip_merged(merged)
        except Exception as e:
            logger.error(f"Merge facts error: {e}")
        return _clip_merged(f"{existing_text}；{new_text}")

    # ==========================================
    # 去重并存储（完整流程）
    # ==========================================

    async def deduplicate_and_store(
        self,
        fact: dict,
        entity_id: str,
        entity_type: str = "user",
    ) -> str:
        """铁律 #1 完整实现：去重 → 合并/新增

        Args:
            fact: {"content": "...", "importance": 7, "tags": [...], "semantic_id": "..."}

        Returns:
            决策结果 "skip" | "duplicate" | "update" | "new"，供调用方（如手动
            memory_add 工具）反馈给用户/模型。后台海马体流程忽略此返回值。
        """
        decision, _final_text = await self.deduplicate_and_store_ex(
            fact, entity_id, entity_type
        )
        return decision

    async def deduplicate_and_store_ex(
        self,
        fact: dict,
        entity_id: str,
        entity_type: str = "user",
    ) -> tuple[str, str]:
        """同 ``deduplicate_and_store``，但额外返回最终落地的文本。

        画像 upsert（画像去重精简 #2）需要在 "update" 时用**合并后**的文本
        播种画像，而不是合并前的原始提取——否则画像会绕开 TOML 侧刚做完的
        去重结果，重新写入一份近义原文。

        Returns:
            (decision, final_text)
            decision: "skip" | "duplicate" | "update" | "new"
            final_text: "duplicate" → 命中的旧记忆原文；"update" → 合并后的
            文本；"new"/"skip" → 传入的 content（"skip" 时为空串）
        """
        content = fact.get("content", "")
        importance = fact.get("importance", 5)
        tags = fact.get("tags", [])
        semantic_id = fact.get("semantic_id", "")

        if not content:
            return "skip", ""

        decision, matched = await self.deduplicate(
            content, entity_id, entity_type, "facts"
        )

        if decision == "duplicate":
            logger.debug(f"Duplicate memory skipped: {content[:50]}...")
            return "duplicate", (matched.text if matched else content)

        if decision == "update" and matched:
            # 合并后更新旧记忆
            merged_text = await self.merge_facts(matched.text, content)
            matched.text = merged_text
            matched.importance = max(importance, matched.importance)
            matched.meta["last_accessed"] = time.time()

            # 合并 tags
            existing_tags = set(matched.tags)
            existing_tags.update(tags)
            matched.tags = list(existing_tags)

            if await self.tree_store.update_memory(matched):
                logger.info(f"Memory merged: id={matched.id}")
            else:
                logger.warning(f"Failed to merge memory {matched.id}")
            return "update", merged_text

        # 全新事实 → 写入
        # 尝试获取语义 ID
        if not semantic_id:
            semantic_id = await self.generate_semantic_id(content)

        await self.tree_store.add_memory(
            content_text=content,
            memory_type="fact",
            importance=importance,
            tags=tags,
            semantic_id=semantic_id,
            entity_id=entity_id,
            entity_type=entity_type,
            folder="facts",
        )
        logger.info(f"New fact stored for {entity_type}:{entity_id}")
        return "new", content

    # ==========================================
    # 信息升维（宪章铁律 #2）
    # ==========================================

    async def check_elevation_trigger(
        self,
        entity_id: str,
        entity_type: str = "user",
    ) -> bool:
        """检查 facts 是否积累到升维阈值"""
        facts = await self.tree_store.get_all_memories(
            entity_id=entity_id, entity_type=entity_type, folder="facts"
        )
        return len(facts) >= self.reflection_threshold

    async def generate_reflections(
        self,
        entity_id: str,
        entity_type: str = "user",
    ) -> list[str]:
        """从 facts 群提炼 reflections（升维），并归档被吸收的 facts

        Returns:
            生成的 reflection 文本列表
        """
        if not self._llm_client:
            return []

        facts = await self.tree_store.get_all_memories(
            entity_id=entity_id, entity_type=entity_type, folder="facts"
        )
        if len(facts) < self.reflection_threshold:
            return []

        facts_text = "\n".join(
            f"{i + 1}. {f.text}" for i, f in enumerate(facts)
        )

        # 洞察属于主观推断 —— 注入角色视角（issue #4）。
        persona_system = self._persona_system()
        persona_clause = (
            "（请从你角色本人的主观视角来提炼这些洞察，保持与你人设一致的态度；"
            "但只能基于上面列出的事实推断，不要照搬人设设定。）\n"
        ) if persona_system else ""

        if entity_type == "group":
            prompt = f"""基于以下关于这个群聊的事实，你能推断出什么更高层面的洞察？
比如群体性格、社交动态、群文化特征等。涉及具体成员时用昵称，不要说"该用户"。

事实:
{facts_text}
{persona_clause}
请输出 1-3 条简洁的洞察，每条一行，不需要编号。只输出洞察内容，不要有其他内容。"""
        else:
            prompt = f"""基于以下关于这位用户的事实，你能推断出什么更高层面的洞察？
比如性格特征、兴趣偏好的模式、生活方式等。用该用户的昵称作主语，不要说"该用户"。

事实:
{facts_text}
{persona_clause}
请输出 1-3 条简洁的洞察，每条一行，不需要编号。只输出洞察内容，不要有其他内容。"""

        generated = []
        try:
            text = (await chat_text(self._llm_client, prompt, system=persona_system)).strip()
            if not text:
                return []

            insights = [
                line.strip()
                for line in text.split("\n")
                if line.strip()
            ]

            for insight in insights:
                # 去重检查：是否已有相似 reflection
                existing = await self.tree_store.search(
                    query=insight,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    folder="reflections",
                    k=1,
                    update_access=False,
                )
                if existing:
                    merged = await self.merge_facts(existing[0].text, insight)
                    existing[0].text = merged
                    existing[0].meta["last_accessed"] = time.time()
                    await self.tree_store.update_memory(existing[0])
                    logger.debug(f"Reflection merged with existing: {insight[:50]}...")
                    continue

                # 生成语义 ID
                sem_id = await self.generate_semantic_id(insight)

                await self.tree_store.add_memory(
                    content_text=insight,
                    memory_type="reflection",
                    importance=7,
                    semantic_id=sem_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    folder="reflections",
                )
                generated.append(insight)
                logger.info(f"Reflection stored for {entity_type}:{entity_id}")

            # 归档被吸收的低重要性 facts
            if generated:
                for fact in facts:
                    if fact.importance <= 4:
                        await self.tree_store.archive_memory(
                            memory_id=fact.id,
                            entity_id=entity_id,
                            entity_type=entity_type,
                            folder="facts",
                        )
                        logger.debug(f"Absorbed fact archived: {fact.id}")

        except Exception as e:
            logger.error(f"Reflection generation error: {e}")

        return generated

    # ==========================================
    # 画像压实（画像去重精简 #3）
    # ==========================================

    async def summarize_profile_facts(
        self, facts: list[str], traits: Optional[list[str]] = None,
    ) -> list[str]:
        """把冗长/重复的画像 facts 压实为带 "[标签]" 前缀的规范短句列表。

        用于修复已经膨胀的存量画像（例如同一个习惯被反复用不同话描述好几遍）。
        解析失败或空输出时返回 ``[]``——调用方（``HippocampusManager.compact_profile``）
        必须据此保留原值，绝不能用空列表覆盖已有画像。
        """
        client = self._fast_or_default
        if not client or not facts:
            return []

        facts_text = "\n".join(f"- {f}" for f in facts)
        traits_text = f"\n已有特征标签: {', '.join(traits)}" if traits else ""
        tags = "、".join(_PROFILE_FACT_TAGS)

        prompt = f"""以下是关于同一个人的一份画像事实列表，其中可能有重复、同义
或过于细碎的内容（比如同一个习惯被反复用不同话描述了好几次）。请把它们去重、
合并、精简为规范的短句列表。

规则：
- 每条不超过 30 字，一条只说一件事
- 同义/重复内容只保留一条最准确的表述，不要罗列发生了几次
- 每条前面加一个 "[标签]" 前缀，标签从以下几类中选：{tags}
- 按重要性排序，最多输出 {_PROFILE_COMPACT_MAX_FACTS} 条
- 不要编造原始列表里没有的信息

原始事实列表:
{facts_text}{traits_text}

只输出结果，每行一条，格式为 "[标签] 内容"，不要有编号、不要有其他说明文字。"""

        try:
            text = (await chat_text(client, prompt)).strip()
            if not text:
                return []
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            cleaned = []
            for ln in lines:
                # 容错去掉模型可能加的项目符号/编号前缀。
                ln = re.sub(r"^[-*\d.\s]+(?=\[)", "", ln).strip()
                if not ln:
                    continue
                if not re.match(r"^\[.+?\]", ln):
                    ln = f"[其他] {ln}"
                cleaned.append(_clip_merged(ln))
            return cleaned[:_PROFILE_COMPACT_MAX_FACTS]
        except Exception as e:
            logger.error(f"Profile compaction summarize error: {e}")
            return []

    # ==========================================
    # 工具方法
    # ==========================================

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """健壮地解析 LLM 输出的 JSON 数组"""
        text = text.strip()

        # 去除 markdown code fence
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # 提取第一个 JSON 数组
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start: end + 1]

        # 多次尝试解析
        for attempt in range(3):
            try:
                if attempt == 1:
                    # 移除尾随逗号
                    text = re.sub(r",\s*([}\]])", r"\1", text)
                if attempt == 2:
                    # 回退到 ast.literal_eval
                    obj = ast.literal_eval(text)
                    result = json.loads(json.dumps(obj))
                    if isinstance(result, list):
                        return _clean_facts(result)
                    return []

                result = json.loads(text)
                if isinstance(result, list):
                    return _clean_facts(result)
                return []
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

        return []


def _clean_facts(facts: list) -> list[dict]:
    """清理和标准化事实列表"""
    cleaned = []
    for f in facts:
        if not isinstance(f, dict) or "content" not in f:
            continue
        # 标准化 importance
        raw_imp = f.get("importance")
        if raw_imp is None or raw_imp == "":
            f["importance"] = 5
        else:
            try:
                f["importance"] = max(1, min(10, int(float(raw_imp))))
            except (ValueError, TypeError):
                f["importance"] = 5
        # 确保 tags 是 list
        if not isinstance(f.get("tags"), list):
            f["tags"] = []
        # 清理 semantic_id
        sem_id = f.get("semantic_id", "")
        if sem_id:
            sem_id = re.sub(r"[^a-z0-9_]", "_", sem_id.lower())
            sem_id = re.sub(r"_+", "_", sem_id).strip("_")
            f["semantic_id"] = sem_id
        cleaned.append(f)
    return cleaned

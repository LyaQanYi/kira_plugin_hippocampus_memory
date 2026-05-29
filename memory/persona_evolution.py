"""
三级人设演进漏斗

实现宪章 §6 的 Identity Evolution 机制:
- Tier 1: 瞬时自觉 → global/self/facts/ (低 importance，易衰减)
- Tier 2: 刻意练习 → global/self/reflections/ (高 importance，难衰减)
- Tier 3: 核心人设跃迁 → 不可逆合入 persona.txt，销毁原 reflection
"""

import time
from datetime import datetime
from typing import Optional

from core.logging_manager import get_logger
from .toml_tree_store import TomlTreeStore, Memory
from .paths import get_global_self_dir
from ..adapters.llm import chat_text

logger = get_logger("persona_evolution", "green")


class PersonaEvolutionEngine:
    """三级人设漏斗引擎"""

    # Tier 3 跃迁条件（满足任一即触发）
    LEAP_ACCESS_COUNT_THRESHOLD = 10    # access_count >= 10
    LEAP_IMPORTANCE_THRESHOLD = 9       # importance >= 9
    LEAP_SURVIVAL_DAYS = 30             # 存活超过 30 天

    # 跃迁冷却期
    LEAP_COOLDOWN_DAYS = 7              # 每 7 天最多执行一次跃迁

    def __init__(self, tree_store: TomlTreeStore, persona_manager=None):
        self.tree_store = tree_store
        self.persona_manager = persona_manager
        self._last_leap_time: float = 0.0

    def set_persona_manager(self, pm):
        self.persona_manager = pm

    # ==========================================
    # Tier 1: 瞬时自觉
    # ==========================================

    async def record_self_awareness(
        self, content: str, importance: int = 3, tags: list = None
    ) -> Memory:
        """记录一条自我觉察到 global/self/facts/

        Tier 1: 低 importance，易衰减。
        例："我刚刚的代码回答太啰嗦了，下次要精简"
        """
        return await self.tree_store.add_memory(
            content_text=content,
            memory_type="fact",
            importance=max(1, min(5, importance)),  # Tier 1 限制 importance ≤ 5
            tags=tags or ["self-awareness"],
            base_dir=get_global_self_dir(),
            folder="facts",
        )

    # ==========================================
    # Tier 2: 升维为反思
    # ==========================================

    async def elevate_to_reflection(
        self, content: str, importance: int = 8, source_fact_ids: list = None
    ) -> Memory:
        """将自我觉察升维为反思，写入 global/self/reflections/

        Tier 2: 高 importance，难衰减。相当于形成了习惯。
        例："系统倾向于简练专业的对话风格"
        """
        tags = ["self-reflection"]
        if source_fact_ids:
            tags.append(f"from:{','.join(source_fact_ids)}")

        reflection = await self.tree_store.add_memory(
            content_text=content,
            memory_type="reflection",
            importance=max(7, min(10, importance)),  # Tier 2 importance ≥ 7
            tags=tags,
            base_dir=get_global_self_dir(),
            folder="reflections",
        )

        # 删除被吸收的 Tier 1 facts
        if source_fact_ids:
            for fid in source_fact_ids:
                await self.tree_store.delete_memory(
                    memory_id=fid,
                    base_dir=get_global_self_dir(),
                    folder="facts",
                )
            logger.info(
                f"Tier 1→2 elevation: {len(source_fact_ids)} facts absorbed into reflection"
            )

        return reflection

    # ==========================================
    # Tier 3: 核心人设跃迁
    # ==========================================

    async def check_persona_leap(self) -> Optional[str]:
        """扫描 global/self/reflections/，检查是否有 reflection 达到跃迁条件

        Returns:
            跃迁的 reflection 文本，或 None
        """
        if not self.persona_manager:
            return None

        # 冷却期检查
        now = time.time()
        days_since_last = (now - self._last_leap_time) / 86400
        if days_since_last < self.LEAP_COOLDOWN_DAYS and self._last_leap_time > 0:
            return None

        reflections = await self.tree_store.get_all_memories(
            base_dir=get_global_self_dir(),
            folder="reflections",
        )

        for ref in reflections:
            if self._is_leap_candidate(ref, now):
                success = await self._merge_into_persona(ref)
                if success:
                    # 销毁已合入的 reflection（宪章 §6.3）
                    await self.tree_store.delete_memory(
                        memory_id=ref.id,
                        base_dir=get_global_self_dir(),
                        folder="reflections",
                    )
                    self._last_leap_time = now
                    logger.info(
                        f"PERSONA LEAP: reflection {ref.id} merged into persona"
                    )
                    return ref.text

        return None

    async def _merge_into_persona(self, ref: Memory) -> bool:
        """Append the reflection text to KiraAI's persona content (destructive)."""
        try:
            persona = await self.persona_manager.get_persona()
        except Exception as e:
            logger.error(f"Failed to get persona: {e}")
            return False
        if persona is None or persona.content is None:
            return False
        stamp = datetime.now().strftime("%Y-%m-%d")
        insert = f"\n\n# Evolved insight ({ref.id}, {stamp})\n{ref.text}"
        persona.content = persona.content.rstrip() + insert
        try:
            ok = await self.persona_manager.update_persona(persona)
            return bool(ok)
        except Exception as e:
            logger.error(f"Failed to update persona: {e}")
            return False

    def _is_leap_candidate(self, ref: Memory, now: float) -> bool:
        """判断一条 reflection 是否达到跃迁条件"""
        # 条件 1: access_count 突破阈值
        if ref.access_count >= self.LEAP_ACCESS_COUNT_THRESHOLD:
            return True

        # 条件 2: importance 极高
        if ref.importance >= self.LEAP_IMPORTANCE_THRESHOLD:
            return True

        # 条件 3: 存活超过指定天数
        creation_time = ref.timestamp
        if creation_time > 0:
            days_alive = (now - creation_time) / 86400
            if days_alive >= self.LEAP_SURVIVAL_DAYS:
                return True

        return False

    # ==========================================
    # 自动化流程
    # ==========================================

    async def run_evolution_cycle(self, llm_client=None):
        """完整的人设演进周期

        1. 扫描 global/self/facts/，看是否有足够的 Tier 1 事实可以升维
        2. 如果有，用 LLM 提炼为 Tier 2 reflection
        3. 检查是否有 Tier 2 reflection 达到 Tier 3 跃迁条件
        """
        # Step 1: 检查 Tier 1 → Tier 2 升维
        facts = await self.tree_store.get_all_memories(
            base_dir=get_global_self_dir(),
            folder="facts",
        )

        if len(facts) >= 5 and llm_client:
            # 有足够的自我觉察，尝试提炼
            facts_text = "\n".join(
                f"- {f.text}" for f in facts
            )
            prompt = f"""基于以下 Agent 的自我觉察记录，提炼出 1-2 条关于自身行为模式的深层洞察:

{facts_text}

直接输出洞察，每条一行，不要编号，不要多余内容。"""

            try:
                text = (await chat_text(llm_client, prompt)).strip()
                if text:
                    insights = [
                        line.strip()
                        for line in text.split("\n")
                        if line.strip()
                    ]
                    for insight in insights:
                        await self.elevate_to_reflection(
                            content=insight,
                            importance=8,
                            source_fact_ids=[f.id for f in facts if f.importance <= 4],
                        )
            except Exception as e:
                logger.error(f"Tier 1→2 evolution error: {e}")

        # Step 2: 检查 Tier 2 → Tier 3 跃迁
        await self.check_persona_leap()

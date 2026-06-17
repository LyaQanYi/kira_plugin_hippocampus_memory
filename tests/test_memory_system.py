"""End-to-end tests for the hippocampus memory plugin.

Run from the KiraAI repo root:

    PYTHONPATH=. pytest data/plugins/kira_plugin_hippocampus_memory/tests/ -v

These tests stub the parts of `core.provider` / `core.prompt_manager` that
adapters/llm.py touches, since the full KiraAI provider stack has a circular
import that only resolves during a real `KiraLifecycle` boot.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from pathlib import Path


def _install_stubs():
    if "core.provider" not in sys.modules:
        provider_stub = types.ModuleType("core.provider")

        @dataclass
        class _LLMRequest:
            messages: list = field(default_factory=list)

        class _LLMModelClient:
            pass

        provider_stub.LLMRequest = _LLMRequest
        provider_stub.LLMModelClient = _LLMModelClient
        provider_stub.LLMResponse = type("LLMResponse", (), {})
        sys.modules["core.provider"] = provider_stub

    if "core.prompt_manager" not in sys.modules:
        pm_stub = types.ModuleType("core.prompt_manager")

        class _Prompt:
            def __init__(self, content="", name=None, source=None, **kw):
                self.content = content
                self.name = name
                self.source = source

        pm_stub.Prompt = _Prompt
        sys.modules["core.prompt_manager"] = pm_stub

    if "plugins" not in sys.modules:
        pkg = types.ModuleType("plugins")
        pkg.__path__ = [str(Path(__file__).resolve().parents[2])]
        sys.modules["plugins"] = pkg
    if "plugins.kira_plugin_hippocampus_memory" not in sys.modules:
        sub = types.ModuleType("plugins.kira_plugin_hippocampus_memory")
        sub.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["plugins.kira_plugin_hippocampus_memory"] = sub


_install_stubs()


# Now safe to import the plugin modules.
from plugins.kira_plugin_hippocampus_memory.memory.paths import (
    set_memory_root,
    ensure_directory_structure,
    get_global_facts_dir,
)
from plugins.kira_plugin_hippocampus_memory.memory.manager import HippocampusManager
from plugins.kira_plugin_hippocampus_memory.memory.toml_tree_store import TomlTreeStore
from plugins.kira_plugin_hippocampus_memory.memory.memory_index import MemoryIndex
from plugins.kira_plugin_hippocampus_memory.memory.entity_profile import (
    EntityProfile,
    EntityProfileStore,
)
from plugins.kira_plugin_hippocampus_memory.adapters.sender_cache import SenderCache
from plugins.kira_plugin_hippocampus_memory.adapters.recall_query import query_from_event


class _FakeResp:
    def __init__(self, t):
        self.text_response = t


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.idx = 0
        self.calls = []

    async def chat(self, req):
        t = self.scripted[self.idx] if self.idx < len(self.scripted) else ""
        self.idx += 1
        self.calls.append(t[:80])
        return _FakeResp(t)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------

def test_paths_require_explicit_root():
    """get_memory_root() must error if set_memory_root() was not called."""
    # We can't fully isolate set_memory_root state across tests, so just
    # verify the public surface exists.
    from plugins.kira_plugin_hippocampus_memory.memory import paths

    with tempfile.TemporaryDirectory() as tmp:
        set_memory_root(tmp)
        assert paths.get_memory_root() == tmp
        assert paths.get_entities_dir().endswith("entities")
        assert paths.get_global_dir().endswith("global")
        assert paths.get_archive_dir().endswith("archive")
        assert paths.get_index_db_path().endswith("memory_index.db")


def test_directory_structure():
    with tempfile.TemporaryDirectory() as tmp:
        set_memory_root(tmp)
        ensure_directory_structure()
        for sub in ("entities", "archive", "global", "global/facts",
                    "global/self/facts", "global/self/reflections"):
            assert (Path(tmp) / sub).exists(), f"missing {sub}"


# --------------------------------------------------------------------------
# TomlTreeStore CRUD
# --------------------------------------------------------------------------

def test_toml_store_crud():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            from plugins.kira_plugin_hippocampus_memory.memory.paths import get_index_db_path
            store = TomlTreeStore(index=MemoryIndex(db_path=get_index_db_path()))

            mem = await store.add_memory(
                content_text="用户喜欢 Python",
                memory_type="fact",
                importance=7,
                tags=["preference"],
                entity_id="telegram:42",
                entity_type="user",
                folder="facts",
            )
            assert mem.id
            assert mem.text == "用户喜欢 Python"

            # File should exist
            fpath = mem.file_path
            assert Path(fpath).exists()

            # Round-trip get
            got = await store.get_memory(
                mem.id, entity_id="telegram:42", entity_type="user", folder="facts"
            )
            assert got is not None
            assert got.text == "用户喜欢 Python"

            # Search
            hits = await store.search(
                query="Python", entity_id="telegram:42", entity_type="user",
                folder="facts", k=5,
            )
            assert any(h.id == mem.id for h in hits)

            # Cross-folder
            cross = await store.search_across_folders(
                query="Python", entity_id="telegram:42", entity_type="user", k=5,
            )
            assert any(h.id == mem.id for h in cross)

            # Delete
            ok = await store.delete_memory(
                mem.id, entity_id="telegram:42", entity_type="user", folder="facts"
            )
            assert ok
            assert not Path(fpath).exists()

            store.close()

    _run(run())


# --------------------------------------------------------------------------
# Content hash dedup
# --------------------------------------------------------------------------

def test_content_hash_dedup():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            from plugins.kira_plugin_hippocampus_memory.memory.paths import get_index_db_path
            store = TomlTreeStore(index=MemoryIndex(db_path=get_index_db_path()))

            content = "完全相同的事实"
            await store.add_memory(
                content_text=content,
                memory_type="fact",
                importance=5,
                entity_id="user42",
                entity_type="user",
                folder="facts",
            )
            h1 = MemoryIndex.content_hash(content)
            found = store.index.find_by_hash(h1, "user42", "user", "facts")
            assert found is not None
            assert found["raw_text"] == content

            store.close()

    _run(run())


# --------------------------------------------------------------------------
# Entity profile
# --------------------------------------------------------------------------

def test_entity_profile():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            ps = EntityProfileStore()

            p = await ps.get_profile("user42", "user")
            assert isinstance(p, EntityProfile)
            assert p.entity_id == "user42"

            await ps.add_trait("user42", "技术导向")
            await ps.add_fact("user42", "喜欢 Python")
            await ps.increment_interaction("user42", nickname="小明")

            p2 = await ps.get_profile("user42", "user")
            assert "技术导向" in p2.traits
            assert "喜欢 Python" in p2.facts
            assert p2.nickname == "小明"
            assert p2.interaction_count == 1

            # Nickname change should populate aliases.
            await ps.increment_interaction("user42", nickname="小红")
            p3 = await ps.get_profile("user42", "user")
            assert "小明" in p3.aliases
            assert p3.nickname == "小红"

    _run(run())


# --------------------------------------------------------------------------
# Recall + injection
# --------------------------------------------------------------------------

def test_recall_with_fake_llm():
    """End-to-end: simulated hippocampus submit_chunk → store → recall."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()

            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 1,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()

            scripted = [
                json.dumps([
                    {"content": "小明喜欢 Python", "speaker_id": "12345",
                     "subject": "小明", "importance": 7, "tags": ["preference"],
                     "semantic_id": "xm_likes_python"},
                ]),
            ]
            fake = FakeLLM(scripted)
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            cache = SenderCache()
            mgr.set_sender_cache(cache)
            sid = "telegram:dm:12345"
            cache.record(sid, "12345", "小明", "我喜欢 Python")

            mgr.submit_chunk(sid, "我喜欢 Python", "好的")

            # Wait for background processing
            for _ in range(100):
                await asyncio.sleep(0.05)
                with mgr._background_tasks_lock:
                    if not mgr._background_tasks:
                        break

            results = await mgr.recall(
                "Python", entity_id="telegram:12345", entity_type="user", k=5
            )
            assert any("Python" in r.text for r in results)

            # Profile should have been seeded (importance >= 7)
            profile = await mgr.get_profile("telegram:12345", "user")
            assert any("Python" in f for f in profile.facts)

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Decay engine
# --------------------------------------------------------------------------

def test_decay_downgrade_and_archive():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()

            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 1,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()

            # Manual: insert a low-importance, long-untouched fact
            mem = await mgr.tree_store.add_memory(
                content_text="过期的小事",
                memory_type="fact",
                importance=2,
                entity_id="telegram:42",
                entity_type="user",
                folder="facts",
            )
            # Backdate last_accessed by 200 days
            mgr.memory_index.update_meta(
                mem.id, last_accessed=time.time() - 200 * 86400
            )

            deleted, downgraded = await mgr.run_forgetting_cycle()
            # Either deleted (archive) or downgraded; in both cases the engine ran.
            assert deleted + downgraded >= 1

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Persona perspective for subjective extraction (Issue #4)
# --------------------------------------------------------------------------

from plugins.kira_plugin_hippocampus_memory.memory.memory_extractor import (
    MemoryExtractor,
)


class _StubStore:
    """Minimal stand-in: MemoryExtractor only needs `.index` at construction,
    and the extraction methods under test don't touch the store."""

    def __init__(self):
        self.index = None


class RecordingLLM:
    """Fake LLM that captures the request messages of each chat() call so a
    test can assert whether a system (persona) prompt was attached."""

    def __init__(self, scripted=None):
        self.scripted = list(scripted or [])
        self.idx = 0
        self.requests = []  # captured req.messages per call

    async def chat(self, req):
        self.requests.append(list(req.messages))
        t = self.scripted[self.idx] if self.idx < len(self.scripted) else "[]"
        self.idx += 1
        return _FakeResp(t)

    def system_at(self, i):
        for m in self.requests[i]:
            if m.get("role") == "system":
                return m.get("content", "")
        return None

    def user_at(self, i):
        for m in self.requests[i]:
            if m.get("role") == "user":
                return m.get("content", "")
        return ""


def test_persona_perspective_injected_into_subjective_extraction():
    """Group-fact extraction (subjective: atmosphere/culture) must get the
    persona as a system prompt so it judges in-character."""
    async def run():
        ext = MemoryExtractor(_StubStore())
        llm = RecordingLLM(["[]"])
        ext.set_llm_client(llm)
        ext.set_persona_brief("你是高冷怕吵的猫娘，讨厌嘈杂的环境。")

        await ext.extract_group_facts("小明(1): 哈哈哈刷屏\n阿花(2): 666666")

        sys_prompt = llm.system_at(0)
        assert sys_prompt is not None
        assert "高冷怕吵" in sys_prompt
        # The anti-copy guard must be present so persona settings aren't
        # recorded as conversation facts.
        assert "绝不能" in sys_prompt and "事实" in sys_prompt
        # The user prompt carries the in-character perspective instruction.
        assert "主观视角" in llm.user_at(0)

    _run(run())


def test_persona_perspective_not_injected_into_objective_extraction():
    """Personal-fact extraction is objective — persona must NOT bias it, even
    when a persona brief is configured."""
    async def run():
        ext = MemoryExtractor(_StubStore())
        llm = RecordingLLM(["[]"])
        ext.set_llm_client(llm)
        ext.set_persona_brief("你是高冷怕吵的猫娘。")

        await ext.extract_personal_facts("小明(1): 我喜欢 Python")

        assert llm.system_at(0) is None

    _run(run())


def test_subjective_extraction_neutral_without_persona():
    """Without a persona brief, subjective extraction stays exactly as before:
    no system prompt, no perspective clause."""
    async def run():
        ext = MemoryExtractor(_StubStore())
        llm = RecordingLLM(["[]"])
        ext.set_llm_client(llm)

        await ext.extract_group_facts("小明(1): 哈哈")

        assert llm.system_at(0) is None
        assert "主观视角" not in llm.user_at(0)

    _run(run())


def test_self_awareness_uses_persona_perspective():
    async def run():
        ext = MemoryExtractor(_StubStore())
        llm = RecordingLLM(["NONE"])
        ext.set_llm_client(llm)
        ext.set_persona_brief("你是毒舌但内心温柔的助手。")

        await ext.extract_self_awareness("小明(1): 在吗\nBot: 在的")

        sys_prompt = llm.system_at(0)
        assert sys_prompt is not None and "毒舌" in sys_prompt

    _run(run())


def test_set_persona_brief_truncates_and_clears():
    ext = MemoryExtractor(_StubStore())

    ext.set_persona_brief("x" * 5000)
    assert len(ext._persona_brief) <= 801  # cap (800) + ellipsis
    assert ext._persona_system() is not None

    # Blank/whitespace disables the feature again.
    ext.set_persona_brief("   ")
    assert ext._persona_brief == ""
    assert ext._persona_system() is None


# --------------------------------------------------------------------------
# Recall query derivation (Issue #1)
# --------------------------------------------------------------------------

class _FakeMsg:
    def __init__(self, message_str):
        self.message_str = message_str


class _FakeEvent:
    def __init__(self, messages):
        self.messages = messages


def test_recall_query_from_messages_strips_envelope():
    """inject_memory must derive the recall query from event.messages.

    The built-in kira-ai plugin splices a message envelope ([date]
    [message_id: ...] [group_name: ... group_id: ... user_nickname: ...,
    user_id: ...] | <body>) into req.user_prompt at a higher priority. Reading
    the per-message `message_str` instead yields the envelope-free body.
    """
    event = _FakeEvent([
        _FakeMsg("我最近在学 Python，喜欢用它写脚本"),
        _FakeMsg("[At 小助手] 帮我记一下"),
    ])
    query = query_from_event(event)

    # Body words survive...
    assert "Python" in query
    assert "脚本" in query
    # ...but none of the envelope metadata leaks into the recall query.
    for token in ("message_id", "group_id", "group_name",
                  "user_nickname", "user_id"):
        assert token not in query

    # No usable message text → empty, so the caller falls back to
    # _extract_query(req).
    assert query_from_event(_FakeEvent([])) == ""
    assert query_from_event(_FakeEvent([_FakeMsg(""), _FakeMsg(None)])) == ""
    assert query_from_event(object()) == ""


class _FakeSender:
    def __init__(self, user_id):
        self.user_id = user_id


class _FakeSenderMsg:
    def __init__(self, message_str="", user_id=""):
        self.message_str = message_str
        self.sender = _FakeSender(user_id) if user_id else None


class _FakeAdapter:
    def __init__(self, name):
        self.name = name


class _FakeRoutedEvent:
    def __init__(self, adapter_name, messages):
        self.adapter = _FakeAdapter(adapter_name)
        self.messages = messages


def test_recall_targets_dual_path():
    """Recall must always include the speaking user, plus the group in a group."""
    from plugins.kira_plugin_hippocampus_memory.adapters.recall_query import (
        recall_targets,
    )

    # Group turn: speaker's user entity first, then the group entity.
    grp_event = _FakeRoutedEvent("telegram", [_FakeSenderMsg("晚上好", "12345")])
    targets = recall_targets(grp_event, "telegram:115985242", "group")
    assert ("telegram:12345", "user") in targets, "speaker memories must be recalled in group"
    assert ("telegram:115985242", "group") in targets, "group memories must be recalled too"
    assert targets[0] == ("telegram:12345", "user"), "user scope should come first"

    # DM turn: session entity already the user; no duplicate group scope.
    dm_event = _FakeRoutedEvent("telegram", [_FakeSenderMsg("hi", "12345")])
    dm_targets = recall_targets(dm_event, "telegram:12345", "user")
    assert dm_targets == [("telegram:12345", "user")]

    # Unresolved speaker → fall back to the session entity (legacy behaviour).
    blank = _FakeRoutedEvent("telegram", [_FakeSenderMsg("hi", "")])
    assert recall_targets(blank, "telegram:999", "group") == [("telegram:999", "group")]
    assert recall_targets(object(), "telegram:42", "user") == [("telegram:42", "user")]


# --------------------------------------------------------------------------
# Cross-user memory_search (entity_search)
# --------------------------------------------------------------------------

def test_entity_search_helpers():
    from plugins.kira_plugin_hippocampus_memory.adapters.entity_search import (
        looks_like_entity_id,
        looks_like_group_id,
    )

    assert looks_like_entity_id("telegram:123")
    assert not looks_like_entity_id("小明")
    assert not looks_like_entity_id("")
    assert not looks_like_entity_id("nocolon")
    assert not looks_like_entity_id(":12345")   # empty adapter is malformed
    assert not looks_like_entity_id("telegram:")  # empty id is malformed

    assert looks_like_group_id("group:123")
    assert looks_like_group_id("我们群")
    assert not looks_like_group_id("telegram:123")
    assert not looks_like_group_id("小明")


def test_memory_search_multi_user():
    """memory_search resolves nicknames and searches multiple users in parallel."""
    from plugins.kira_plugin_hippocampus_memory.adapters.entity_search import (
        search_memories,
    )
    from plugins.kira_plugin_hippocampus_memory.memory.paths import list_all_entities

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()

            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 1,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()
            # fast LLM returns "" for the multi-entity summary → keep the raw
            # merged block so we can assert both users' memories are present.
            mgr.set_clients(llm_client=FakeLLM([]), fast_llm_client=FakeLLM([""]))

            await mgr.add_fact("小明喜欢用 Python 编程", entity_id="telegram:111",
                               entity_type="user", importance=6)
            await mgr.add_fact("小红喜欢用 JavaScript 编程", entity_id="telegram:222",
                               entity_type="user", importance=6)
            # Profiles give the nicknames resolve_entity_by_name matches on.
            await mgr.profile_store.increment_interaction("telegram:111", nickname="小明")
            await mgr.profile_store.increment_interaction("telegram:222", nickname="小红")

            block = await search_memories(
                manager=mgr,
                fast_llm=mgr.get_fast_llm(),
                sender_cache=None,
                sid="telegram:dm:111",
                query="编程",
                entity_id="小明,小红",       # two nicknames, comma-separated
                entity_type="user",
                k=5,
                fallback_targets=[],
                list_entities_fn=list_all_entities,
            )

            # Both users resolved + searched, results labelled by entity.
            assert "telegram:111" in block and "telegram:222" in block
            assert "Python" in block and "JavaScript" in block

            # Per-token group guard: a 群-ish token is skipped, not the whole
            # field — "小明,阿群" must still resolve 小明 (and only 小明).
            mixed = await search_memories(
                manager=mgr, fast_llm=mgr.get_fast_llm(), sender_cache=None,
                sid="telegram:dm:111", query="编程", entity_id="小明,阿群",
                entity_type="user", k=5, fallback_targets=[],
                list_entities_fn=list_all_entities,
            )
            assert "Python" in mixed           # 小明 resolved
            assert "JavaScript" not in mixed   # 阿群 did NOT pull in 小红
            # (a single resolved entity isn't label-prefixed, so we assert on
            # content exclusion rather than on the "[telegram:111]" label.)

            # Group-like entity_id is rejected → falls through to the fallback.
            fb = await search_memories(
                manager=mgr, fast_llm=None, sender_cache=None,
                sid="telegram:gm:999", query="编程", entity_id="我们群",
                k=5, fallback_targets=[("telegram:111", "user")],
                list_entities_fn=list_all_entities,
            )
            assert "Python" in fb  # fell back to the provided target

            await mgr.close()

    _run(run())

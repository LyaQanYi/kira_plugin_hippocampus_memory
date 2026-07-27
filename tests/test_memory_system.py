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
    get_entity_profile_path,
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


def test_exact_duplicate_returns_persisted_text_and_importance():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            from plugins.kira_plugin_hippocampus_memory.memory.paths import (
                get_index_db_path,
            )
            store = TomlTreeStore(index=MemoryIndex(db_path=get_index_db_path()))
            ext = MemoryExtractor(store)
            content = "完全相同的高重要性事实"

            await store.add_memory(
                content_text=content,
                memory_type="fact",
                importance=8,
                entity_id="user42",
                entity_type="user",
                folder="facts",
            )

            decision, final_text, final_importance = (
                await ext.deduplicate_and_store_ex(
                    {"content": content, "importance": 3},
                    "user42",
                    "user",
                )
            )

            assert decision == "duplicate"
            assert final_text == content
            assert final_importance == 8
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


def test_resolve_source_labels_disambiguates_collisions():
    """Two entities resolving to the SAME display name stay distinguishable via
    the opaque token, and the raw entity_id is never used as a label."""
    from plugins.kira_plugin_hippocampus_memory.adapters.entity_search import (
        _resolve_source_labels,
    )

    class _P:
        def __init__(self, name="", nickname="", aliases=None):
            self.name = name
            self.nickname = nickname
            self.aliases = aliases or []

    class _Store:
        def __init__(self, by_id):
            self._by_id = by_id

        async def get_profile(self, eid, etype):
            return self._by_id.get(eid, _P())

    store = _Store({
        "telegram:111": _P(name="小明"),
        "telegram:222": _P(name="小明"),       # same display name → collision
        "telegram:333": _P(),                   # no name → opaque token
    })
    resolved = [("telegram:111", "user"), ("telegram:222", "user"),
                ("telegram:333", "user")]

    labels = _run(_resolve_source_labels(store, resolved))

    assert labels["telegram:111"] == "小明"
    assert labels["telegram:222"] == "小明(用户B)"   # disambiguated, not the id
    assert labels["telegram:333"] == "用户C"          # no name → opaque
    # No raw entity_id leaks into any label.
    for v in labels.values():
        assert "telegram:" not in v and "111" not in v and "222" not in v


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

            # Both users resolved + searched, results labelled by display name
            # — never the raw canonical entity_id (which must not reach the LLM).
            assert "[小明]" in block and "[小红]" in block
            assert "telegram:111" not in block and "telegram:222" not in block
            assert "111" not in block and "222" not in block
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


# --------------------------------------------------------------------------
# Sender-profile extraction context (ported from lightning memory_manager:
# _build_sender_profiles_context — prepended to the conversation before the
# hippocampus extracts, so the LLM avoids re-recording already-known facts)
# --------------------------------------------------------------------------

def test_build_sender_profiles_context():
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

            # Seed a known profile for telegram:111.
            await mgr.profile_store.update_profile(
                "telegram:111",
                name="小明",
                nickname="小明明",       # differs from name → shown as 当前昵称
                aliases=["阿明"],
            )
            await mgr.profile_store.add_trait("telegram:111", "内向")
            await mgr.profile_store.add_fact("telegram:111", "喜欢 Python")

            ctx = await mgr._build_sender_profiles_context("telegram", ["111"])

            # Header + per-field rendering, all faithful to lightning's strings.
            assert "## 参与者已知信息" in ctx
            assert "【小明】" in ctx            # label prefers name
            assert "名字: 小明" in ctx
            assert "当前昵称: 小明明" in ctx
            assert "曾用名: 阿明" in ctx
            assert "特征: 内向" in ctx
            assert "已知事实: 喜欢 Python" in ctx
            # Never leak the raw system entity_id into the extraction prompt.
            assert "telegram:111" not in ctx

            # No senders → empty string (skip the prepend entirely).
            assert await mgr._build_sender_profiles_context("telegram", []) == ""

            # Unknown sender (no profile, no info) → empty string, not a header.
            assert await mgr._build_sender_profiles_context("telegram", ["999"]) == ""

            # A sender WITH info but NO name/nickname → labelled by an opaque
            # per-turn token, never the bare platform id (site #2 regression).
            await mgr.profile_store.add_trait("telegram:222", "潜水")  # trait only
            anon = await mgr._build_sender_profiles_context("telegram", ["222"])
            assert "特征: 潜水" in anon
            assert "【用户A】" in anon          # opaque token (single sender → 用户A)
            assert "222" not in anon            # bare platform id must not leak
            assert "telegram:222" not in anon

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Site #3: the conversation handed to the extractor must carry an opaque
# per-turn token, never the raw sender id — yet facts must still route home.
# --------------------------------------------------------------------------

def test_chunks_to_text_hides_raw_sender_id_but_routes_home():
    chunks = [[
        {"role": "user", "sender_id": "111", "sender_name": "小明",
         "content": "我喜欢 Python"},
        {"role": "user", "sender_id": "222", "sender_name": "小红",
         "content": "我用 JavaScript"},
        {"role": "assistant", "content": "了解了"},
    ]]
    sender_map = {"111": "111", "小明": "111", "222": "222", "小红": "222"}

    unique = HippocampusManager._unique_senders(chunks)
    assert unique == ["111", "222"]
    token_by_sid = HippocampusManager._participant_tokens(unique)
    text = HippocampusManager._chunks_to_text(chunks, sender_map, token_by_sid)

    # Display nicknames survive; the raw platform ids do NOT appear anywhere.
    assert "小明" in text and "小红" in text
    assert "(111)" not in text and "(222)" not in text
    assert "111" not in text and "222" not in text
    # The opaque tokens are the parenthetical the LLM sees instead of the id.
    assert "小明(用户A)" in text and "小红(用户B)" in text

    # Routing parity: a fact the LLM emits referencing a token routes back to
    # the real sender id, exactly as the old raw-id parenthetical did.
    label_to_sid = {tok: sid for sid, tok in token_by_sid.items()}
    mgr = HippocampusManager.__new__(HippocampusManager)  # no __init__ I/O needed
    eid, etype = mgr._resolve_fact_entity(
        {"speaker_id": "用户A", "subject": "小明"}, "telegram", sender_map,
        unique, "telegram:115", "group", label_to_sid,
    )
    assert (eid, etype) == ("telegram:111", "user")

    # A participant rendered as the token alone (no nickname) still routes when
    # the model puts the token in `subject`.
    eid2, _ = mgr._resolve_fact_entity(
        {"speaker_id": "", "subject": "用户B"}, "telegram", sender_map,
        unique, "telegram:115", "group", label_to_sid,
    )
    assert eid2 == "telegram:222"

    # An un-named participant is shown as the bare token, no raw id.
    anon_chunks = [[
        {"role": "user", "sender_id": "333", "content": "潜水中"},
    ]]
    anon_tokens = HippocampusManager._participant_tokens(
        HippocampusManager._unique_senders(anon_chunks)
    )
    anon_text = HippocampusManager._chunks_to_text(anon_chunks, {}, anon_tokens)
    assert anon_text == "用户A: 潜水中"
    assert "333" not in anon_text


# --------------------------------------------------------------------------
# Fix #1: always-on profile injection into the live turn (parity with
# lightning's per-turn user_profile). build_turn_profile_prompt aggregates in
# groups and uses the single speaker (or session entity) otherwise.
# --------------------------------------------------------------------------

def test_sender_users_extracts_distinct_speakers():
    from plugins.kira_plugin_hippocampus_memory.adapters.recall_query import (
        sender_users,
    )

    event = _FakeRoutedEvent("telegram", [
        _FakeSenderMsg("hi", "111"),
        _FakeSenderMsg("yo", "222"),
        _FakeSenderMsg("again", "111"),   # duplicate speaker collapses
        _FakeSenderMsg("anon", ""),       # no user_id → skipped
    ])
    assert sender_users(event) == [("telegram:111", ""), ("telegram:222", "")]
    # No adapter / no messages → empty.
    assert sender_users(object()) == []
    assert sender_users(_FakeRoutedEvent("telegram", [])) == []


def test_build_turn_profile_prompt_dm_and_group():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()

            await mgr.profile_store.update_profile("telegram:111", name="小明")
            await mgr.profile_store.update_profile("telegram:222", name="小红")

            # DM / single speaker → that user's profile, no aggregate header.
            dm = await mgr.build_turn_profile_prompt(
                [("telegram:111", "小明")], "telegram:111", "user", is_group=False
            )
            assert "名字: 小明" in dm
            assert "参与对话的用户" not in dm

            # Group with >1 speakers → aggregated, each labelled by display name.
            grp = await mgr.build_turn_profile_prompt(
                [("telegram:111", "小明"), ("telegram:222", "小红")],
                "telegram:999", "group", is_group=True,
            )
            assert "本次群聊中参与对话的用户" in grp
            assert "【小明】" in grp and "【小红】" in grp
            assert "名字: 小明" in grp and "名字: 小红" in grp
            # System entity_id must never leak into the prompt.
            assert "telegram:111" not in grp

            # Group with an un-named participant → opaque ordinal label, never
            # the bare id tail (a QQ/platform number the plugin must not leak).
            await mgr.profile_store.add_trait("telegram:333", "潜水")  # trait, no name
            anon_grp = await mgr.build_turn_profile_prompt(
                [("telegram:111", "小明"), ("telegram:333", "")],
                "telegram:999", "group", is_group=True,
            )
            assert "【小明】" in anon_grp
            assert "【用户2】" in anon_grp     # un-named → 用户N, not the id tail
            assert "333" not in anon_grp       # bare id fragment must not leak
            assert "111" not in anon_grp

            # No usable profile (unknown user) → empty string, inject nothing.
            empty = await mgr.build_turn_profile_prompt(
                [("telegram:404", "幽灵")], "telegram:404", "user", is_group=False
            )
            assert empty == ""

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Fix #2: exact sender identity for post-turn extraction. take_unconsumed
# uses a monotonic watermark; submit_exchange carries precise sender_id/name.
# --------------------------------------------------------------------------

def test_sender_cache_take_unconsumed_watermark():
    c = SenderCache()
    sid = "telegram:gm:1"
    c.record(sid, "111", "小明", "a")
    c.record(sid, "222", "小红", "b")

    first = c.take_unconsumed(sid)
    assert [x["text"] for x in first] == ["a", "b"]
    assert [x["user_id"] for x in first] == ["111", "222"]

    # Nothing new since the watermark advanced.
    assert c.take_unconsumed(sid) == []

    c.record(sid, "111", "小明", "c")
    second = c.take_unconsumed(sid)
    assert [x["text"] for x in second] == ["c"]

    # take_unconsumed does NOT delete — the full window is still visible.
    assert len(c.get_recent(sid, max_age_sec=9999)) == 3


def test_sender_cache_bounds_sessions():
    """_data and _consumed_seq must not grow without bound across sessions."""
    c = SenderCache(max_sessions=3)
    for i in range(5):
        c.record(f"sid{i}", str(i), f"u{i}", "hi")
        c.take_unconsumed(f"sid{i}")   # also populates _consumed_seq[sid{i}]

    assert len(c._data) == 3            # oldest evicted FIFO
    assert len(c._consumed_seq) <= 3   # watermark map pruned in lockstep
    assert "sid0" not in c._data and "sid1" not in c._data
    assert "sid0" not in c._consumed_seq and "sid1" not in c._consumed_seq
    assert "sid4" in c._data


def test_submit_exchange_carries_exact_identity():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,   # never auto-flush; inspect buffer
                "enable_self_awareness": False,
            })
            await mgr.async_init()
            mgr.set_clients(llm_client=FakeLLM([]))   # non-None so submit proceeds

            sid = "telegram:gm:999"
            user_msgs = [
                {"user_id": "111", "nickname": "小明", "text": "我喜欢 Python"},
                {"user_id": "222", "nickname": "小红", "text": "我用 JavaScript"},
            ]
            mgr.submit_exchange(sid, user_msgs, "了解了")

            buffered = mgr._pending_conversations.get(sid, [])
            assert len(buffered) == 1
            chunk = buffered[0]
            users = [m for m in chunk if m["role"] == "user"]
            # Every user message keeps its precise sender_id/sender_name — no
            # reconstruction by text match.
            assert {m["sender_id"] for m in users} == {"111", "222"}
            assert {m.get("sender_name") for m in users} == {"小明", "小红"}
            assert chunk[-1] == {"role": "assistant", "content": "了解了"}

            # The sender map derives id + nickname keys straight from the chunk.
            smap = mgr._build_sender_map(sid, [chunk])
            assert smap.get("111") == "111"
            assert smap.get("小明") == "111"

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Fix #3: manual memory_add runs through dedup/merge; update/remove span
# facts + reflections with a numbered listing on an out-of-range index.
# --------------------------------------------------------------------------

def test_add_fact_curated_dedups_exact_duplicate():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()
            # No LLM needed: exact-hash dedup is LLM-free; the conflict check
            # degrades to "new" without one. fast LLM returns "" semantic ids.
            mgr.set_clients(llm_client=FakeLLM([""] * 8), fast_llm_client=FakeLLM([""] * 8))

            d1 = await mgr.add_fact_curated(
                "用户喜欢喝美式咖啡", entity_id="telegram:111", entity_type="user"
            )
            assert d1 == "new"

            d2 = await mgr.add_fact_curated(
                "用户喜欢喝美式咖啡", entity_id="telegram:111", entity_type="user"
            )
            assert d2 == "duplicate"          # exact-hash dedup caught it

            facts = await mgr.tree_store.get_all_memories(
                entity_id="telegram:111", entity_type="user", folder="facts"
            )
            assert len(facts) == 1            # NOT duplicated on disk

            # No entity scope → direct global write ("stored"), not dedup path.
            d3 = await mgr.add_fact_curated("世界是圆的", entity_id="")
            assert d3 == "stored"

            await mgr.close()

    _run(run())


def test_list_editable_memories_spans_facts_and_reflections():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()
            mgr.set_clients(llm_client=FakeLLM([]))

            await mgr.tree_store.add_memory(
                content_text="喜欢 Python", memory_type="fact", importance=6,
                entity_id="telegram:111", entity_type="user", folder="facts",
            )
            await mgr.tree_store.add_memory(
                content_text="技术导向的人", memory_type="reflection", importance=7,
                entity_id="telegram:111", entity_type="user", folder="reflections",
            )

            sid = "telegram:dm:111"
            mems = await mgr.list_editable_memories(sid)
            texts = {m.raw_text for m in mems}
            assert "喜欢 Python" in texts        # fact included
            assert "技术导向的人" in texts        # reflection included too

            listing = mgr.format_editable_list(mems)
            assert listing.startswith("0. ")
            assert "1. " in listing

            # Empty/whitespace update is refused — a bad call must not blank a memory.
            assert await mgr.update_memory_at(mems, 0, "   ") is None
            assert await mgr.update_memory_at(mems, 0, "") is None
            # A real update still works.
            updated = await mgr.update_memory_at(mems, 0, "改成新内容")
            assert updated is not None and updated.raw_text == "改成新内容"

            # Remove by index actually deletes.
            ok = await mgr.delete_memory_at(mems, 0)
            assert ok is True
            remaining = await mgr.list_editable_memories(sid)
            assert len(remaining) == 1

            await mgr.close()

    _run(run())


def test_memory_search_auto_extract_gated():
    """The LLM auto-detect path runs only when allow_auto_extract=True."""
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
            await mgr.add_fact("小明喜欢吃辣", entity_id="telegram:111",
                               entity_type="user", importance=6)
            await mgr.profile_store.increment_interaction("telegram:111", nickname="小明")

            # No entity_id. Flag OFF → fast LLM must NOT be consulted; only the
            # fallback target is searched.
            fake_off = FakeLLM(["小明"])  # would resolve 小明 if it were called
            mgr.set_clients(llm_client=fake_off, fast_llm_client=fake_off)
            off = await search_memories(
                manager=mgr, fast_llm=mgr.get_fast_llm(), sender_cache=None,
                sid="telegram:dm:999", query="谁喜欢吃辣", entity_id="",
                k=5, fallback_targets=[("telegram:999", "user")],
                list_entities_fn=list_all_entities, allow_auto_extract=False,
            )
            assert fake_off.idx == 0, "auto-extract LLM must not run when gated off"
            assert "吃辣" not in off  # 小明 not reached; only the (empty) fallback

            # Flag ON → fast LLM extracts 小明 → resolves → search finds the fact.
            fake_on = FakeLLM(["小明"])
            mgr.set_clients(llm_client=fake_on, fast_llm_client=fake_on)
            on = await search_memories(
                manager=mgr, fast_llm=mgr.get_fast_llm(), sender_cache=None,
                sid="telegram:dm:999", query="谁喜欢吃辣", entity_id="",
                k=5, fallback_targets=[], list_entities_fn=list_all_entities,
                allow_auto_extract=True,
            )
            assert "吃辣" in on

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Profile upsert: semantic dedup/merge instead of a bare exact-string append
# (the reported bug: "拍头/摸头" style near-duplicates piling up in a profile)
# --------------------------------------------------------------------------

def test_profile_upsert_semantic_dedup_and_merge():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            ps = EntityProfileStore()

            async def conflict_check(new, existing):
                keys = ("摸头", "拍头")
                if any(k in new for k in keys) and any(k in existing for k in keys):
                    return "update"
                return "new"

            async def merge(existing, new):
                return "喜欢被摸头（合并版）"

            d1 = await ps.upsert_fact(
                "user1", "喜欢被拍头", conflict_check=conflict_check, merge=merge
            )
            assert d1 == "new"

            d2 = await ps.upsert_fact(
                "user1", "喜欢被摸头", conflict_check=conflict_check, merge=merge
            )
            assert d2 == "update"

            p = await ps.get_profile("user1", "user")
            assert p.facts == ["喜欢被摸头（合并版）"]   # merged in place, not appended

            # Exact-text duplicate short-circuits without even calling conflict_check.
            calls = []

            async def counting_check(new, existing):
                calls.append((new, existing))
                return "new"

            d3 = await ps.upsert_fact(
                "user1", "喜欢被摸头（合并版）",
                conflict_check=counting_check, merge=merge,
            )
            assert d3 == "duplicate"
            assert calls == []

            # Unrelated new fact → appended, not merged.
            d4 = await ps.upsert_fact(
                "user1", "小明是一名大三学生", conflict_check=conflict_check, merge=merge
            )
            assert d4 == "new"
            p2 = await ps.get_profile("user1", "user")
            assert len(p2.facts) == 2

            # No conflict_check (e.g. no LLM configured) → degrades to exact
            # dedup + plain append, never raises.
            d5 = await ps.upsert_fact("user2", "喜欢喝咖啡")
            assert d5 == "new"
            d6 = await ps.upsert_fact("user2", "喜欢喝咖啡")
            assert d6 == "duplicate"

    _run(run())


def test_add_trait_normalizes_case_and_whitespace():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            ps = EntityProfileStore()

            await ps.add_trait("user1", "Nice")
            await ps.add_trait("user1", " nice ")   # same trait, different case/space
            await ps.add_trait("user1", "内向")

            p = await ps.get_profile("user1", "user")
            assert len(p.traits) == 2
            assert "内向" in p.traits

    _run(run())


def test_hippocampus_gates_profile_write_on_dedup_decision():
    """A fact the TOML pipeline judges "duplicate" must not still land in the
    profile as fresh (near-duplicate) text — the bug behind the reported
    "拍头" being recorded several times over."""
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
                    {"content": "小明喜欢被拍头", "speaker_id": "12345",
                     "subject": "小明", "importance": 8, "tags": [],
                     "semantic_id": "xm_pat_head"},
                ]),
            ]
            fake = FakeLLM(scripted)
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            cache = SenderCache()
            mgr.set_sender_cache(cache)
            sid = "telegram:dm:12345"
            cache.record(sid, "12345", "小明", "我喜欢被拍头")
            mgr.submit_chunk(sid, "我喜欢被拍头", "好的")

            for _ in range(100):
                await asyncio.sleep(0.05)
                with mgr._background_tasks_lock:
                    if not mgr._background_tasks:
                        break

            profile = await mgr.get_profile("telegram:12345", "user")
            assert len(profile.facts) == 1
            assert await mgr.profile_store.delete_profile(
                "telegram:12345", "user"
            )

            # Same underlying habit again, worded differently. The tree's own
            # dedup returns "duplicate". A missing profile must be reseeded from
            # the matched TOML text, not from the fresh near-duplicate wording.
            fake.scripted.append(json.dumps([
                {"content": "小明很喜欢别人摸他的头", "speaker_id": "12345",
                 "subject": "小明", "importance": 8, "tags": [],
                 "semantic_id": "xm_pat_head"},
            ]))
            # Conflict-check + merge calls triggered by the second round.
            fake.scripted.append("duplicate")

            cache.record(sid, "12345", "小明", "我很喜欢别人摸我的头")
            mgr.submit_chunk(sid, "我很喜欢别人摸我的头", "好的")

            for _ in range(100):
                await asyncio.sleep(0.05)
                with mgr._background_tasks_lock:
                    if not mgr._background_tasks:
                        break

            profile2 = await mgr.get_profile("telegram:12345", "user")
            assert profile2.facts == ["小明喜欢被拍头"]

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Profile compaction: shrink a bloated facts list into tagged short bullets;
# never destroy data on a bad/empty LLM output.
# --------------------------------------------------------------------------

def test_compact_profile_shrinks_bloated_facts():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 3,
            })
            await mgr.async_init()

            bloated = [
                "小明喜欢被拍头",
                "小明喜欢被摸头",
                "小明很喜欢别人摸他的头",
            ]
            await mgr.profile_store.update_profile("telegram:111", facts=list(bloated))

            summary = "[互动习惯] 喜欢被摸头/拍头"
            mgr.set_clients(
                llm_client=FakeLLM([summary]), fast_llm_client=FakeLLM([summary])
            )

            compacted = await mgr.compact_profile("telegram:111", "user")
            assert compacted is True

            p = await mgr.get_profile("telegram:111", "user")
            assert p.facts == ["[互动习惯] 喜欢被摸头/拍头"]

            # to_prompt() groups by the [标签] prefix rather than dumping a
            # flat bullet list.
            prompt = p.to_prompt()
            assert "[互动习惯] 喜欢被摸头/拍头" in prompt

            await mgr.close()

    _run(run())


def test_compact_profile_keeps_original_on_parse_failure():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 3,
            })
            await mgr.async_init()

            bloated = ["事实一", "事实二", "事实三"]
            await mgr.profile_store.update_profile("telegram:111", facts=list(bloated))

            # Fast LLM returns an empty string → summarize_profile_facts yields [].
            mgr.set_clients(llm_client=FakeLLM([""]), fast_llm_client=FakeLLM([""]))

            compacted = await mgr.compact_profile("telegram:111", "user")
            assert compacted is False

            p = await mgr.get_profile("telegram:111", "user")
            assert p.facts == bloated   # untouched, nothing destroyed

            await mgr.close()

    _run(run())


def test_compact_profile_below_threshold_is_noop_unless_forced():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 12,
            })
            await mgr.async_init()
            fake = FakeLLM(["[其他] 手动压实后的事实"])
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            await mgr.profile_store.update_profile("telegram:111", facts=["单条事实"])
            compacted = await mgr.compact_profile("telegram:111", "user")
            assert compacted is False
            assert fake.idx == 0   # no LLM call spent on a profile below threshold

            compacted = await mgr.compact_profile(
                "telegram:111", "user", force=True
            )
            assert compacted is True
            assert fake.idx == 1
            profile = await mgr.get_profile("telegram:111", "user")
            assert profile.facts == ["[其他] 手动压实后的事实"]

            await mgr.close()

    _run(run())


def test_run_forgetting_cycle_sweeps_bloated_inactive_profiles():
    """Compaction must not depend solely on a NEW hippocampus write — an
    inactive user's already-bloated profile is only ever revisited by the
    periodic sweep piggybacked on run_forgetting_cycle."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 2,
            })
            await mgr.async_init()

            await mgr.profile_store.update_profile(
                "telegram:999", facts=["很水的事实一", "很水的事实二", "很水的事实三"]
            )
            summary = "[其他] 精简后的事实"
            mgr.set_clients(
                llm_client=FakeLLM([summary]), fast_llm_client=FakeLLM([summary])
            )

            await mgr.run_forgetting_cycle()

            p = await mgr.get_profile("telegram:999", "user")
            assert p.facts == ["[其他] 精简后的事实"]

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# Pre-extraction related-memory recall: existing facts most relevant to what
# a sender just said are surfaced as a hard constraint block, not just the
# first-5 profile summary (画像去重精简 #1).
# --------------------------------------------------------------------------

def test_related_memories_context_injects_relevant_facts():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()

            await mgr.tree_store.add_memory(
                content_text="小明喜欢用 Python 写后端",
                memory_type="fact",
                importance=6,
                entity_id="telegram:111",
                entity_type="user",
                folder="facts",
            )
            await mgr.tree_store.add_memory(
                content_text="小明喜欢深夜写代码",
                memory_type="fact",
                importance=5,
                entity_id="telegram:111",
                entity_type="user",
                folder="facts",
            )

            own_texts = {"111": "我现在还是最喜欢用 Python"}
            token_by_sid = {"111": "用户A"}

            ctx = await mgr._build_related_memories_context(
                "telegram", ["111"], own_texts, token_by_sid,
                is_group=False, session_entity_id="telegram:111",
            )
            assert "## 已有相关记忆" in ctx
            assert "Python" in ctx
            assert "【用户A】" in ctx

            # Missing token mappings must fall back to an opaque ordinal label,
            # never the raw platform sender id.
            fallback_ctx = await mgr._build_related_memories_context(
                "telegram", ["111"], own_texts, {},
                is_group=False, session_entity_id="telegram:111",
            )
            assert "【用户A】" in fallback_ctx
            assert "【111】" not in fallback_ctx

            # No senders / no own text → empty string, nothing forced in.
            empty_ctx = await mgr._build_related_memories_context(
                "telegram", [], {}, {}, is_group=False, session_entity_id="telegram:x",
            )
            assert empty_ctx == ""

            await mgr.close()

    _run(run())


# --------------------------------------------------------------------------
# PR #13 review fixes: compaction race, untagged-output rejection, zero-bigram
# fallback, save-failure propagation, importance-gate correctness, and the
# update_memory-failure decision.
# --------------------------------------------------------------------------

def test_rank_candidates_falls_back_when_no_positive_similarity():
    """Zero bigram overlap must not skip the semantic conflict-check
    entirely — it should fall back to the first-k facts instead of an empty
    candidate list (CodeRabbit)."""
    from plugins.kira_plugin_hippocampus_memory.memory.entity_profile import (
        _rank_candidates,
    )

    facts = ["abc", "xyz789"]
    candidates = _rank_candidates("完全不同的中文内容", facts, k=3)
    assert len(candidates) == 2
    assert {f for _, f in candidates} == set(facts)

    # When a positive-similarity match exists, ranking still prefers it.
    facts2 = ["喜欢被拍头", "今天天气不错"]
    ranked = _rank_candidates("喜欢被摸头", facts2, k=3)
    assert ranked[0][1] == "喜欢被拍头"
    assert len(ranked) == 2

    # A positive literal match must not leave the remaining candidate slots
    # empty and hide a zero-overlap semantic duplicate.
    facts3 = ["软件安装指南", "程序员", "喜欢喝咖啡"]
    filled = _rank_candidates("从事软件开发工作", facts3, k=3)
    assert len(filled) == 3
    assert filled[0][1] == "软件安装指南"
    assert "程序员" in {fact for _, fact in filled}

    # No existing facts at all → no candidates, not an error.
    assert _rank_candidates("任意内容", [], k=3) == []


def test_upsert_fact_checks_conflict_even_without_bigram_overlap():
    """A semantic duplicate with zero literal character overlap (e.g.
    "程序员" vs "从事软件开发工作") must still reach conflict_check instead of
    being silently appended as a brand-new fact."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            ps = EntityProfileStore()

            calls = []

            async def conflict_check(new, existing):
                calls.append((new, existing))
                if {new, existing} == {"从事软件开发工作", "程序员"}:
                    return "update"
                return "new"

            async def merge(existing, new):
                return "软件工程师"

            d1 = await ps.upsert_fact(
                "userX", "从事软件开发工作", conflict_check=conflict_check, merge=merge
            )
            assert d1 == "new"

            d2 = await ps.upsert_fact(
                "userX", "程序员", conflict_check=conflict_check, merge=merge
            )
            assert calls, "conflict_check must be consulted even with zero bigram overlap"
            assert d2 == "update"

            p = await ps.get_profile("userX", "user")
            assert p.facts == ["软件工程师"]

    _run(run())


def test_upsert_fact_deduplicates_after_length_clipping():
    """Exact dedup must compare the persisted, clipped representation.

    Comparing an untrimmed input with the previously clipped copy would append
    the same long fact on every call when no conflict-check client is present.
    """
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            long_fact = "很长的画像事实" * 50

            assert await store.upsert_fact("user-long", long_fact) == "new"
            assert await store.upsert_fact("user-long", long_fact) == "duplicate"

            profile = await store.get_profile("user-long", "user")
            assert len(profile.facts) == 1
            assert len(profile.facts[0]) <= 200

    _run(run())


async def _always_fail_save(profile):
    return False


def test_upsert_fact_reports_error_when_save_fails():
    """A persistence failure must not be reported as "new"/"update" — the
    caller would otherwise believe a fact landed when it never did
    (CodeRabbit: ignored save_profile() bool)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            ps = EntityProfileStore()
            ps.save_profile = _always_fail_save

            d1 = await ps.upsert_fact("userY", "喜欢喝咖啡")
            assert d1 == "error"

            # Nothing was actually persisted (save always failed).
            ps_read = EntityProfileStore()
            p = await ps_read.get_profile("userY", "user")
            assert "喜欢喝咖啡" not in p.facts

    _run(run())


def test_update_profile_returns_none_on_save_failure():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            ps = EntityProfileStore()
            ps.save_profile = _always_fail_save

            result = await ps.update_profile("userZ", facts=["a"])
            assert result is None

    _run(run())


def test_profile_read_error_does_not_overwrite_existing_file():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            await store.update_profile("user-corrupt", facts=["事实"])
            profile_path = Path(get_entity_profile_path("user-corrupt", "user"))
            corrupt = "{not valid json"
            profile_path.write_text(corrupt, encoding="utf-8")

            try:
                await store.get_profile("user-corrupt", "user")
            except json.JSONDecodeError:
                pass
            else:
                raise AssertionError("corrupt profile reads must propagate")

            assert profile_path.read_text(encoding="utf-8") == corrupt

    _run(run())


def test_profile_sync_write_is_atomic_on_serialization_failure():
    with tempfile.TemporaryDirectory() as tmp:
        profile_path = Path(tmp) / "profile.json"
        original = '{"facts": ["safe"]}'
        profile_path.write_text(original, encoding="utf-8")

        try:
            EntityProfileStore._sync_write(
                str(profile_path), {"facts": [object()]}
            )
        except TypeError:
            pass
        else:
            raise AssertionError("unserializable profile data must fail")

        assert profile_path.read_text(encoding="utf-8") == original
        assert not list(profile_path.parent.glob(".profile-*.tmp"))


def test_profile_upsert_merge_failure_preserves_existing_fact():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            await store.update_profile("user-merge-fail", facts=["原有事实"])

            async def conflict_check(new, existing):
                return "update"

            async def failing_merge(existing, new):
                raise RuntimeError("merge failed")

            result = await store.upsert_fact(
                "user-merge-fail",
                "更正后的新事实",
                conflict_check=conflict_check,
                merge=failing_merge,
            )

            assert result == "error"
            profile = await store.get_profile("user-merge-fail", "user")
            assert profile.facts == ["原有事实"]

    _run(run())


def test_profile_upsert_authoritative_replacement_skips_second_merge():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()

            for decision in ("duplicate", "update"):
                entity_id = f"user-authoritative-{decision}"
                await store.update_profile(entity_id, facts=["旧画像措辞"])

                async def conflict_check(new, existing, result=decision):
                    return result

                async def merge_must_not_run(existing, new):
                    raise AssertionError("canonical text must not be merged again")

                result = await store.upsert_fact(
                    entity_id,
                    "TOML 最终合并文本",
                    conflict_check=conflict_check,
                    merge=merge_must_not_run,
                    replace_on_duplicate=True,
                )

                assert result == "update"
                profile = await store.get_profile(entity_id, "user")
                assert profile.facts == ["TOML 最终合并文本"]

    _run(run())


def test_profile_upsert_failure_skips_post_write_compaction():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
            })
            await mgr.async_init()
            mgr.profile_store.save_profile = _always_fail_save

            compact_called = False

            async def track_compaction(*args, **kwargs):
                nonlocal compact_called
                compact_called = True
                return False

            mgr.compact_profile = track_compaction
            updated = await mgr._update_profile_from_fact(
                "telegram:save-failure",
                "user",
                {"content": "重要事实", "importance": 8},
            )

            assert updated is False
            assert compact_called is False
            await mgr.close()

    _run(run())


def test_summarize_profile_facts_rejects_malformed_output():
    """Untagged text, refusals, or an unrecognized tag must make the whole
    summary a parse failure (return []) instead of being auto-wrapped into a
    "[其他]" fact that would then replace every existing profile fact
    (Greptile P1 / CodeRabbit)."""
    async def run():
        ext = MemoryExtractor(_StubStore())

        ext.set_fast_llm_client(FakeLLM(["抱歉，我无法完成这个摘要请求。"]))
        assert await ext.summarize_profile_facts(["事实一", "事实二"]) == []

        ext.set_fast_llm_client(FakeLLM(["[乱写标签] 一些内容"]))
        assert await ext.summarize_profile_facts(["事实一", "事实二"]) == []

        ext.set_fast_llm_client(FakeLLM(["[其他]   "]))  # allowed tag, empty body
        assert await ext.summarize_profile_facts(["事实一", "事实二"]) == []

        ext.set_fast_llm_client(FakeLLM(['["some", "json"]']))
        assert await ext.summarize_profile_facts(["事实一", "事实二"]) == []

        # A properly tagged, non-empty line still works.
        ext.set_fast_llm_client(FakeLLM(["[身份] 是一名学生"]))
        assert await ext.summarize_profile_facts(["事实一", "事实二"]) == ["[身份] 是一名学生"]

    _run(run())


def test_summarize_profile_facts_respects_max_facts():
    async def run():
        ext = MemoryExtractor(_StubStore())
        text = "\n".join(f"[身份] 事实{i}" for i in range(5))
        ext.set_fast_llm_client(FakeLLM([text]))

        result = await ext.summarize_profile_facts(["a", "b"], max_facts=2)
        assert len(result) == 2

    _run(run())


def test_deduplicate_and_store_ex_returns_skip_when_merge_persist_fails():
    """If the merge's ``update_memory`` write fails, the decision must not
    still be reported as "update" with the merged text — that would let the
    manager seed/update the profile with a merge that never landed on disk
    (CodeRabbit outside-diff)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            from plugins.kira_plugin_hippocampus_memory.memory.paths import (
                get_index_db_path,
            )
            store = TomlTreeStore(index=MemoryIndex(db_path=get_index_db_path()))
            ext = MemoryExtractor(store)

            await store.add_memory(
                content_text="小明喜欢用 Python 编程",
                memory_type="fact",
                importance=5,
                entity_id="telegram:111",
                entity_type="user",
                folder="facts",
            )

            fake = FakeLLM(["update", "小明改用 JavaScript 编程"])
            ext.set_llm_client(fake)
            ext.set_fast_llm_client(fake)

            async def _failing_update_memory(memory):
                return False

            store.update_memory = _failing_update_memory

            fact = {"content": "小明改用 JavaScript 编程了", "importance": 9, "tags": []}
            decision, final_text, final_importance = await ext.deduplicate_and_store_ex(
                fact, "telegram:111", "user"
            )

            assert decision == "skip"
            assert final_text == "小明喜欢用 Python 编程"
            assert final_importance == 5

            store.close()

    _run(run())


def test_profile_gate_uses_final_merged_importance_not_raw():
    """A low-importance correction (importance=3) that TOML-side merges into
    an existing high-importance memory (upgraded via ``max(...)``) must still
    sync to the profile — the gate must use the FINAL importance, not the
    raw extracted value (CodeRabbit)."""
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
                    {"content": "小明喜欢用 Python 编程", "speaker_id": "12345",
                     "subject": "小明", "importance": 8, "tags": [],
                     "semantic_id": "xm_python"},
                ]),
                json.dumps([
                    {"content": "小明现在改用 JavaScript 编程了", "speaker_id": "12345",
                     "subject": "小明", "importance": 3, "tags": [],
                     "semantic_id": "xm_js"},
                ]),
                "update",                      # TOML-level _check_conflict
                "小明改用 JavaScript 编程",       # merge_facts
                "update",                      # profile-side _check_conflict
            ]
            fake = FakeLLM(scripted)
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            cache = SenderCache()
            mgr.set_sender_cache(cache)
            sid = "telegram:dm:12345"

            cache.record(sid, "12345", "小明", "我喜欢用 Python 编程")
            mgr.submit_chunk(sid, "我喜欢用 Python 编程", "好的")
            for _ in range(100):
                await asyncio.sleep(0.05)
                with mgr._background_tasks_lock:
                    if not mgr._background_tasks:
                        break

            profile1 = await mgr.get_profile("telegram:12345", "user")
            assert any("Python" in f for f in profile1.facts)

            cache.record(sid, "12345", "小明", "我现在改用 JavaScript 编程了")
            mgr.submit_chunk(sid, "我现在改用 JavaScript 编程了", "好的")
            for _ in range(100):
                await asyncio.sleep(0.05)
                with mgr._background_tasks_lock:
                    if not mgr._background_tasks:
                        break

            profile2 = await mgr.get_profile("telegram:12345", "user")
            # Raw importance of the correction (3) is below the profile
            # gate's threshold (7), but it merged into a memory whose final
            # importance was upgraded to 8 — the profile must reflect that.
            assert profile2.facts == ["小明改用 JavaScript 编程"]

            await mgr.close()

    _run(run())


def test_compact_profile_works_with_only_fast_client():
    """compact_profile must not require the main llm_client — it should run
    off the fast client alone, matching summarize_profile_facts's own client
    selection (CodeRabbit: compaction disabled when only fast LLM)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 3,
            })
            await mgr.async_init()

            await mgr.profile_store.update_profile(
                "telegram:222", facts=["事实一", "事实二", "事实三"]
            )
            mgr.extractor.set_fast_llm_client(FakeLLM(["[其他] 精简后的事实"]))
            assert mgr.extractor._llm_client is None

            compacted = await mgr.compact_profile("telegram:222", "user")
            assert compacted is True

            p = await mgr.get_profile("telegram:222", "user")
            assert p.facts == ["[其他] 精简后的事实"]

            await mgr.close()

    _run(run())


def test_compact_profile_preserves_concurrent_write():
    """A hippocampus write that lands while the compaction's LLM call is
    still pending must not be dropped by the replace-all write at the end of
    compact_profile (Greptile/Codex P1: compaction race)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 3,
            })
            await mgr.async_init()

            bloated = ["小明喜欢被拍头", "小明喜欢被摸头", "小明很喜欢别人摸他的头"]
            await mgr.profile_store.update_profile("telegram:111", facts=list(bloated))

            summary = "[互动习惯] 喜欢被摸头/拍头"
            fake = FakeLLM([summary])
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            original_summarize = mgr.extractor.summarize_profile_facts

            async def summarize_and_race(facts, traits=None, max_facts=12):
                # Simulate a concurrent hippocampus write landing while this
                # (slow) LLM call is in flight.
                await mgr.profile_store.add_fact("telegram:111", "小明是一名程序员")
                return await original_summarize(facts, traits, max_facts=max_facts)

            mgr.extractor.summarize_profile_facts = summarize_and_race

            compacted = await mgr.compact_profile("telegram:111", "user")
            assert compacted is True

            p = await mgr.get_profile("telegram:111", "user")
            assert "[互动习惯] 喜欢被摸头/拍头" in p.facts
            # The concurrently-written fact must survive the compaction's
            # replace-all write, not be silently dropped.
            assert "小明是一名程序员" in p.facts

            await mgr.close()

    _run(run())


def test_apply_compacted_facts_serializes_concurrent_add():
    """Greptile P1 follow-up: a write that races between re-read and save must
    still land — per-entity lock serializes apply_compacted_facts with add_fact
    so the new fact is either merged into the compact write or appended after."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            await store.update_profile(
                "telegram:111", facts=["事实一", "事实二", "事实三"]
            )
            snapshot = ["事实一", "事实二", "事实三"]
            compacted = ["[其他] 精简后的事实"]

            lock = store._get_entity_lock("telegram:111", "user")
            await lock.acquire()
            try:
                t_apply = asyncio.create_task(
                    store.apply_compacted_facts(
                        "telegram:111", "user", snapshot, compacted
                    )
                )
                t_add = asyncio.create_task(
                    store.add_fact("telegram:111", "压实空窗期写入的新事实")
                )
                # Both tasks should be blocked on the entity lock.
                await asyncio.sleep(0.05)
                assert not t_apply.done() and not t_add.done()
            finally:
                lock.release()

            updated, _ = await asyncio.gather(t_apply, t_add)
            assert updated is not None

            p = await store.get_profile("telegram:111", "user")
            assert "[其他] 精简后的事实" in p.facts
            assert "压实空窗期写入的新事实" in p.facts

    _run(run())


def test_compact_profile_result_stays_below_threshold():
    """After a successful compaction, the resulting fact count must land
    strictly below the threshold, otherwise the very next check would
    immediately re-trigger compaction (CodeRabbit: re-compaction loop)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 3,
            })
            await mgr.async_init()

            bloated = ["事实一", "事实二", "事实三", "事实四", "事实五"]
            await mgr.profile_store.update_profile("telegram:333", facts=list(bloated))

            # A misbehaving model tries to return more lines than the
            # threshold should allow.
            summary = "\n".join(f"[其他] 精简事实{i}" for i in range(5))
            fake = FakeLLM([summary])
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            compacted = await mgr.compact_profile("telegram:333", "user")
            assert compacted is True

            p = await mgr.get_profile("telegram:333", "user")
            assert len(p.facts) == 2   # threshold(3) - 1
            assert not mgr._profile_needs_compaction(p)

            await mgr.close()

    _run(run())


def test_compact_profile_clipped_line_stays_within_char_trigger():
    """The ellipsis must count toward the 200-character hard limit.

    A 201-character "200 chars + ellipsis" result immediately exceeded the
    minimum char trigger and caused every subsequent check to compact again.
    """
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 99,
                "profile_compact_max_chars": 200,
            })
            await mgr.async_init()

            await mgr.profile_store.update_profile(
                "telegram:char-limit", facts=["甲" * 101, "乙" * 101]
            )
            summary = "[其他] " + "摘要" * 150
            fake = FakeLLM([summary])
            mgr.set_clients(llm_client=fake, fast_llm_client=fake)

            assert await mgr.compact_profile("telegram:char-limit", "user") is True
            profile = await mgr.get_profile("telegram:char-limit", "user")
            assert len(profile.facts) == 1
            assert len(profile.facts[0]) <= mgr._compact_max_chars()
            assert not mgr._profile_needs_compaction(profile)

            await mgr.close()

    _run(run())


def test_compact_threshold_clamped_to_at_least_two():
    """A misconfigured 0/1/negative threshold must clamp to 2 so a single
    retained fact after compaction does not immediately re-trigger
    (CodeRabbit: re-compaction loop)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            mgr = HippocampusManager({
                "hippocampus_chunk_threshold": 99,
                "reflection_threshold": 100,
                "enable_self_awareness": False,
                "profile_compact_threshold": 0,
            })
            await mgr.async_init()
            assert mgr._compact_threshold() == 2

            await mgr.profile_store.update_profile(
                "telegram:1", facts=["唯一一条事实"]
            )
            p = await mgr.get_profile("telegram:1", "user")
            assert not mgr._profile_needs_compaction(p)
            await mgr.close()

    _run(run())


def test_apply_compacted_facts_aborts_on_snapshot_conflict():
    """If a snapshot fact was deleted/replaced during the LLM call, discard
    the stale summary instead of resurrecting it (CodeRabbit)."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            await store.update_profile(
                "telegram:111", facts=["事实一", "事实二", "事实三"]
            )
            snapshot = ["事实一", "事实二", "事实三"]
            await store.remove_fact("telegram:111", "事实二")
            updated = await store.apply_compacted_facts(
                "telegram:111",
                "user",
                snapshot,
                ["[其他] 过期摘要"],
            )
            assert updated is None
            p = await store.get_profile("telegram:111", "user")
            assert p.facts == ["事实一", "事实三"]
            assert "[其他] 过期摘要" not in p.facts

    _run(run())


def test_apply_compacted_facts_does_not_recreate_missing_profile():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            snapshot = ["事实一", "事实二"]
            await store.update_profile("telegram:missing", facts=list(snapshot))

            profile_path = Path(
                get_entity_profile_path("telegram:missing", "user")
            )
            assert await store.delete_profile("telegram:missing", "user")
            assert not profile_path.exists()

            updated = await store.apply_compacted_facts(
                "telegram:missing",
                "user",
                snapshot,
                ["[其他] 旧摘要"],
            )

            assert updated is None
            assert not profile_path.exists()

    _run(run())


def test_increment_interaction_does_not_clobber_compacted_facts():
    """Greptile P1: a stale interaction RMW must not overwrite facts that
    landed via compaction while the interaction update was in flight."""
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            set_memory_root(tmp)
            ensure_directory_structure()
            store = EntityProfileStore()
            await store.update_profile(
                "telegram:111",
                facts=["事实一", "事实二", "事实三"],
                nickname="旧昵称",
            )

            lock = store._get_entity_lock("telegram:111", "user")
            await lock.acquire()
            try:
                t_inc = asyncio.create_task(
                    store.increment_interaction(
                        "telegram:111", nickname="新昵称"
                    )
                )
                t_apply = asyncio.create_task(
                    store.apply_compacted_facts(
                        "telegram:111",
                        "user",
                        ["事实一", "事实二", "事实三"],
                        ["[其他] 压实后的事实"],
                    )
                )
                await asyncio.sleep(0.05)
                assert not t_inc.done() and not t_apply.done()
            finally:
                lock.release()

            await asyncio.gather(t_inc, t_apply)
            p = await store.get_profile("telegram:111", "user")
            assert "[其他] 压实后的事实" in p.facts
            assert p.nickname == "新昵称"
            assert p.interaction_count >= 1

    _run(run())

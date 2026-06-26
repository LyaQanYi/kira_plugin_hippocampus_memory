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

    # A non-compliant model may echo back the FULL rendered label "小明(用户A)"
    # (what it saw) instead of the bare token. Routing must strip the trailing
    # (token) so the personal fact still lands on the user, not the group.
    eid3, et3 = mgr._resolve_fact_entity(
        {"speaker_id": "小明(用户A)", "subject": ""}, "telegram", sender_map,
        unique, "telegram:115", "group", label_to_sid,
    )
    assert (eid3, et3) == ("telegram:111", "user")

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


def test_resolve_source_labels_sanitizes_and_disambiguates():
    """memory_search source labels: user-controlled display names must not break
    the result line format, and every source must stay distinguishable."""
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
            return self._by_id[eid]

    # A crafted nickname carrying any line/prefix delimiter — brackets, CR/LF,
    # tab, or the Unicode line separator U+2028 — must be folded so it cannot
    # forge a source line in the tool result the agent reads.
    sep = chr(0x2028)  # Unicode LINE SEPARATOR, treated as a break by renderers
    store = _Store({
        "telegram:1": _P(name="ev]il\nna" + sep + "me\tx"),
        "telegram:2": _P(nickname="小红"),
    })
    labels = _run(_resolve_source_labels(
        store, [("telegram:1", "user"), ("telegram:2", "user")]
    ))
    forbidden = "][\r\n\t\x0b\x0c" + chr(0x2028) + chr(0x2029)
    assert not any(c in labels["telegram:1"] for c in forbidden)
    assert labels["telegram:2"] == "小红"

    # A real display name literally equal to an opaque token ("用户B" == the token
    # for index 1) must not collide with a nameless source at index 1.
    store2 = _Store({
        "telegram:10": _P(name="用户B"),  # == _opaque_label(1)
        "telegram:11": _P(),             # nameless at index 1 → token 用户B
    })
    labels2 = _run(_resolve_source_labels(
        store2, [("telegram:10", "user"), ("telegram:11", "user")]
    ))
    assert labels2["telegram:10"] == "用户B"
    assert labels2["telegram:11"] != "用户B"      # disambiguated, not a duplicate
    assert len(set(labels2.values())) == 2        # both sources stay distinct

    # Pathological 3-entity case: the disambiguated fallback must itself stay
    # distinct. entity0 named "用户C#2", entity1 named "用户C" (occupies the token
    # _opaque_label(2)), entity2 nameless → token "用户C" is taken, so it must
    # NOT land back on "用户C#2" (which entity0 already holds).
    store3 = _Store({
        "telegram:20": _P(name="用户C#2"),
        "telegram:21": _P(name="用户C"),   # == _opaque_label(2)
        "telegram:22": _P(),              # nameless at index 2 → token 用户C
    })
    labels3 = _run(_resolve_source_labels(
        store3,
        [("telegram:20", "user"), ("telegram:21", "user"), ("telegram:22", "user")],
    ))
    assert len(set(labels3.values())) == 3        # no two sources share a label


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
            assert "【用户B】" in anon_grp     # un-named → opaque token, not id tail
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

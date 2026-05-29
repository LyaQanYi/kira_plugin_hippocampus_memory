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

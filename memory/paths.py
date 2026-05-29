"""
Memory path management — root is injected at plugin init time.

Ported from KiraAI-lightning core/chat/memory_paths.py.

Key change from upstream: the root directory is no longer a hardcoded
"data/memory" module constant. Call `set_memory_root(path)` in the plugin's
initialize() before any memory module is used.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

from core.logging_manager import get_logger

logger = get_logger("memory_paths", "green")

# === Entity types ===
ENTITY_USER = "user"
ENTITY_GROUP = "group"
ENTITY_CHANNEL = "channel"
VALID_ENTITY_TYPES = {ENTITY_USER, ENTITY_GROUP, ENTITY_CHANNEL}

# === Memory subfolders ===
MEMORY_FOLDERS = ("facts", "reflections", "skills")

# === ID validation ===
_SAFE_ID_RE = re.compile(r"^[\w\-.:]+$")

# === Injected root ===
_memory_root: Optional[Path] = None


def set_memory_root(root: Path | str) -> None:
    global _memory_root
    _memory_root = Path(root)
    _memory_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Memory root set to {_memory_root}")


def get_memory_root() -> str:
    if _memory_root is None:
        raise RuntimeError(
            "Memory root has not been set. Call set_memory_root() before using memory paths."
        )
    return str(_memory_root)


def get_entities_dir() -> str:
    return os.path.join(get_memory_root(), "entities")


def get_global_dir() -> str:
    return os.path.join(get_memory_root(), "global")


def get_archive_dir() -> str:
    return os.path.join(get_memory_root(), "archive")


def get_index_db_path() -> str:
    return os.path.join(get_memory_root(), "memory_index.db")


def _validate_id(entity_id: str) -> str:
    if not entity_id or not _SAFE_ID_RE.match(entity_id):
        raise ValueError(f"Invalid entity id: {entity_id!r}")
    return entity_id


def encode_id(entity_id: str) -> str:
    """Make an entity id safe to use as a directory name on all platforms.

    Entity ids look like "telegram:12345" — the colon is a reserved character
    on Windows (NTFS), so a literal dir name like "user_telegram:12345" cannot
    be created. We percent-encode reserved characters; alphanumerics and the
    unreserved set (``_.-~``) stay readable. Reversible via ``decode_id``.
    The original (un-encoded) id is still what gets stored in the SQLite index.
    """
    return quote(entity_id, safe="")


def decode_id(safe_id: str) -> str:
    """Inverse of ``encode_id`` — recover the original entity id from a dir name."""
    return unquote(safe_id)


# === Entity paths ===

def get_entity_dir(entity_id: str, entity_type: str) -> str:
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(f"Unknown entity type: {entity_type!r}, valid: {VALID_ENTITY_TYPES}")
    _validate_id(entity_id)
    return os.path.join(get_entities_dir(), f"{entity_type}_{encode_id(entity_id)}")


def get_entity_folder(entity_id: str, entity_type: str, folder: str) -> str:
    return os.path.join(get_entity_dir(entity_id, entity_type), folder)


def get_entity_profile_path(entity_id: str, entity_type: str) -> str:
    return os.path.join(get_entity_dir(entity_id, entity_type), "profile.json")


# === Global paths ===

def get_global_self_dir() -> str:
    return os.path.join(get_global_dir(), "self")


def get_global_facts_dir() -> str:
    return os.path.join(get_global_dir(), "facts")


def get_global_skills_dir() -> str:
    return os.path.join(get_global_dir(), "skills")


# === Shortcuts ===

def get_user_dir(user_id: str) -> str:
    return get_entity_dir(user_id, ENTITY_USER)


def get_user_folder(user_id: str, folder: str) -> str:
    return get_entity_folder(user_id, ENTITY_USER, folder)


def get_group_dir(group_id: str) -> str:
    return get_entity_dir(group_id, ENTITY_GROUP)


def get_group_folder(group_id: str, folder: str) -> str:
    return get_entity_folder(group_id, ENTITY_GROUP, folder)


def get_channel_dir(channel_id: str) -> str:
    return get_entity_dir(channel_id, ENTITY_CHANNEL)


def get_channel_folder(channel_id: str, folder: str) -> str:
    return get_entity_folder(channel_id, ENTITY_CHANNEL, folder)


# === Bootstrap ===

def ensure_directory_structure() -> None:
    """Create the full memory directory skeleton (call once at init)."""
    root = get_memory_root()
    global_dir = get_global_dir()
    dirs_to_create = [
        root,
        get_entities_dir(),
        get_archive_dir(),
        global_dir,
        os.path.join(global_dir, "facts"),
        os.path.join(global_dir, "skills"),
        os.path.join(global_dir, "self"),
        os.path.join(global_dir, "self", "facts"),
        os.path.join(global_dir, "self", "reflections"),
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    logger.info("Memory directory structure initialized")


def ensure_entity_dirs(entity_id: str, entity_type: str) -> None:
    """Create subfolders for a specific entity (lazy create on first write)."""
    base = get_entity_dir(entity_id, entity_type)
    os.makedirs(base, exist_ok=True)

    if entity_type == ENTITY_USER:
        folders = ("facts", "reflections")
    elif entity_type == ENTITY_GROUP:
        folders = ("facts", "reflections")
    elif entity_type == ENTITY_CHANNEL:
        folders = ("facts",)
    else:
        folders = ("facts",)

    for folder in folders:
        os.makedirs(os.path.join(base, folder), exist_ok=True)


# === Scan tools ===

def list_all_entities(entity_type: Optional[str] = None) -> list[tuple[str, str]]:
    """Scan entities/ and return all (entity_id, entity_type) pairs."""
    results = []
    entities_dir = get_entities_dir()
    if not os.path.exists(entities_dir):
        return results

    for dirname in os.listdir(entities_dir):
        for et in VALID_ENTITY_TYPES:
            prefix = f"{et}_"
            if dirname.startswith(prefix):
                eid = decode_id(dirname[len(prefix):])
                if entity_type is None or et == entity_type:
                    results.append((eid, et))
                break

    return results

"""One-shot migration from simple_memory's core.txt into the new TOML store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.utils.path_utils import get_data_path
from core.logging_manager import get_logger

from ..memory.paths import get_global_facts_dir

logger = get_logger("hippocampus.migration", "cyan")

_MARKER_NAME = ".simple_memory_migrated"


async def migrate_simple_memory_if_needed(tree_store, plugin_data_dir: Path) -> int:
    """If simple_memory's core.txt exists and we haven't migrated yet, import
    each non-empty line as an importance=5 fact under global/facts/.

    Returns the number of entries imported (0 if skipped or empty).
    """
    marker = Path(plugin_data_dir) / _MARKER_NAME
    if marker.exists():
        return 0

    legacy_path = Path(str(get_data_path())) / "memory" / "core.txt"
    if not legacy_path.exists():
        # Nothing to migrate; still drop the marker so we don't keep checking.
        try:
            marker.touch()
        except Exception:
            pass
        return 0

    try:
        raw = legacy_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read legacy core.txt: {e}")
        return 0

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        try:
            marker.touch()
        except Exception:
            pass
        return 0

    count = 0
    base_dir = get_global_facts_dir()
    os.makedirs(base_dir, exist_ok=True)

    for line in lines:
        try:
            await tree_store.add_memory(
                content_text=line,
                memory_type="fact",
                importance=5,
                tags=["migrated", "from-simple-memory"],
                source={"origin": "simple_memory_core_txt"},
                base_dir=base_dir,
                folder="",
            )
            count += 1
        except Exception as e:
            logger.warning(f"Migration failed for line {line!r}: {e}")

    try:
        marker.write_text(str(count), encoding="utf-8")
    except Exception:
        pass

    if count:
        logger.warning(
            f"Migrated {count} entries from simple_memory's core.txt into "
            f"global/facts/. You can disable kira_plugin_simple_memory now."
        )
    return count

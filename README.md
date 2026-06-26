# Hippocampus Memory

A dual-brain long-term memory plugin for KiraAI, ported from the
`KiraAI-lightning` `core/chat/` memory subsystem.

## What it does

Replaces the built-in `kira_plugin_simple_memory` (linear `core.txt` lines, no
recall) with a production-grade memory system:

- **TOML + SQLite dual storage**: human-readable TOML files (you can edit them
  by hand) backed by a SQLite index with FTS5 full-text search and optional
  `sqlite-vec` vectors.
- **Hippocampus**: background LLM-driven fact extraction, two-level dedup
  (SHA-256 + FTS5 + LLM), automatic merge/update.
- **Dimensional reflection**: when an entity accumulates ≥ N facts, the system
  generates higher-level "reflections" (e.g., "tech-oriented") and archives
  the absorbed low-importance facts.
- **Entity profiles** for `user` / `group` / `channel`, with alias tracking,
  injected into the prompt every turn (in a group, all participants') so the bot
  always knows *who it's talking to* — not just what it recalled.
- **Decay-forgetting**: dynamic retention scoring with archive on low scores.
- **Three-tier persona evolution** (Tier 1 self-awareness → Tier 2 reflections
  → Tier 3 persona, last leap off by default).
- **Group-chat dual routing**: personal facts go to user entities, group facts
  go to the group entity.
- **Chinese-aware**: FTS5 uses `jieba` segmentation for proper Chinese full-text
  search.

## Install

```bash
cd data/plugins/kira_plugin_hippocampus_memory
pip install -r requirements.txt
```

Restart KiraAI. The plugin is auto-discovered from `data/plugins/`.

> **Dependencies**: installing via the WebUI (zip upload / GitHub URL) auto-runs
> `requirements.txt`. If you **manually** copy the folder into `data/plugins/`,
> install the requirements yourself into the Python environment KiraAI actually
> uses (e.g. its `venv`), or loading fails with `No module named 'tomli_w'`.
>
> **Loads without an LLM**: the plugin initializes even when no default LLM is
> configured — recall (FTS), manual memory tools, migration and auto-disabling
> simple_memory all still work; only background hippocampus extraction stays
> dormant until an LLM is configured and KiraAI is restarted.

## Relationship with simple_memory

When this plugin starts, it (by default) auto-disables
`kira_plugin_simple_memory` via the official `PluginManager` API so the two
don't double-inject into the `memory` system prompt section.

Before disabling, it imports each non-empty line of simple_memory's
`data/memory/core.txt` as an `importance=5` fact under `global/facts/`. The
import is one-shot and tracked via `.simple_memory_migrated`.

To opt out of either behavior, toggle the corresponding switch in WebUI →
Plugin Manager → Hippocampus Memory.

## Data layout

```
<plugin_data>/kira_plugin_hippocampus_memory/memory/
├── memory_index.db          # SQLite + FTS5 (and optional sqlite-vec)
├── entities/
│   ├── user_<adapter>%3A<uid>/      # ':' in ids is percent-encoded for Windows/NTFS
│   │   ├── facts/*.toml
│   │   └── reflections/*.toml
│   └── group_<adapter>%3A<gid>/
│       ├── facts/
│       └── reflections/
├── global/
│   ├── self/{facts,reflections}/    # Bot self-awareness (Tier 1/2)
│   └── facts/                       # World knowledge / migrated entries
└── archive/                         # Decayed memories (TOML w/ full meta)
```

TOML files are the source of truth. If the SQLite index is lost, the plugin
rebuilds it from the TOML files on next start.

Entity ids such as `telegram:12345` contain a colon, which is reserved on
Windows/NTFS and cannot appear in a directory name, so directory names
percent-encode it (`telegram%3A12345`). The SQLite index still stores the
original (un-encoded) id and decodes it back on rebuild, so behavior is
identical across platforms.

## Configuration

See WebUI → Plugin Manager → Hippocampus Memory for the full list, or
[`schema.json`](./schema.json) for defaults.

Key switches:

- `enable_recall`: turn off recalled-memory prompt injection if you only want
  background extraction.
- `enable_profile_injection`: inject the speaking user's (or, in a group, every
  participant's) profile into the prompt each turn, alongside recall. On by
  default; `max_profile_chars` caps the injected block.
- `auto_disable_simple_memory` / `migrate_simple_memory_on_first_run`: control
  the takeover behavior.
- `enable_persona_evolution`: opt in to Tier-3 persona leap (destructive).
- `enable_search_auto_detect`: let `memory_search` use the fast LLM to guess
  which users a query is about when no `entity_id` is given. Off by default
  (extra LLM call + can mis-resolve); when off, `memory_search` uses the
  explicit `entity_id` or the current user/group only.
- `enable_persona_perspective`: feed the bot's persona (read-only) into
  **subjective** extractions only — group atmosphere, reflections and
  self-awareness — so they read in-character (a noise-averse persona records
  "too loud" rather than a neutral "lively vibe") instead of from a neutral
  observer's view. Objective fact extraction stays persona-blind so the
  character's bias can't pollute hard facts. Off by default; costs extra tokens.

## HTTP API (debug)

All routes are prefixed `/api/plugin/kira_plugin_hippocampus_memory/`.

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | no | Index status + memory count |
| GET | `/entities` | yes | List all entity dirs |
| POST | `/recall` | yes | Body `{query, entity_id, entity_type, k}` |
| GET | `/profile/{entity_id}?entity_type=user` | yes | Inspect an entity profile |
| POST | `/decay/run` | yes | Manually run a forgetting cycle |
| POST | `/evolution/run` | yes | Manually run a persona-evolution cycle |
| DELETE | `/memory/{mem_id}` | yes | Delete a single memory |

## Implementation stages

All four stages are implemented and the test suite passes.

- **Stage A**: scaffolding, auto-disable, recall injection,
  `memory_add/update/remove/search` tools, simple_memory data migration.
- **Stage B**: hippocampus background extraction (sender cache, dual routing).
- **Stage C**: decay engine, entity profiles, persona evolution.
- **Stage D**: docs polish, full test suite ported from
  `test_memory_system.py`.
- **Stage E** (integration parity with `KiraAI-lightning`): always-on profile
  injection into the live turn; exact per-message sender identity for the
  post-turn hippocampus feed (monotonic-watermark `SenderCache`, no text-match
  guessing); `memory_add` routed through the dedup/merge pipeline;
  `memory_update`/`memory_remove` span facts + reflections; read-only
  `memory_profile` tool.

## Testing

```bash
PYTHONPATH=. pytest data/plugins/kira_plugin_hippocampus_memory/tests/ -v
```

The suite (23 tests) covers: path management, directory structure, TOML CRUD,
SHA-256 dedup, entity profile + alias archiving, the recall pipeline, the decay
engine, recall-query envelope sanitization, persona-perspective extraction, and
the Stage-E integration fixes (per-turn profile injection, the monotonic
sender-cache watermark, exact-identity `submit_exchange`, curated `memory_add`
dedup, and the facts+reflections editable listing). Tests run with a Python that
has `pytest` + `jieba` + `tomli_w` installed (the repo's runtime `venv` may not
include `pytest`).

## Credits

All core algorithms ported from KiraAI-lightning. Adapter layer
(`adapters/llm.py`, `adapters/migration.py`, `adapters/sender_cache.py`) is
original work to bridge the lightning-standalone code into KiraAI's plugin host.

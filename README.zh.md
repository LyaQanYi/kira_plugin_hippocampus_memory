# 海马体记忆 (Hippocampus Memory)

KiraAI 的双脑长期记忆插件，移植自 `KiraAI-lightning` 的 `core/chat/` 记忆子系统。

## 功能

替换内置的 `kira_plugin_simple_memory`（纯文本 `core.txt` 行式存储，无召回检索），提供生产级记忆系统：

- **TOML + SQLite 双层存储**：人类可读的 TOML 文件（你可以手工编辑），背后是 SQLite 索引，含 FTS5 全文检索和可选的 `sqlite-vec` 向量。
- **海马体**：后台 LLM 驱动的事实提取，两级去重（SHA-256 + FTS5 + LLM），自动合并/更新。
- **升维反思**：实体积累 ≥ N 条 facts 时，系统生成更高层的"洞察"（如"技术导向"），并归档被吸收的低重要性 facts。
- **实体画像**：`user` / `group` / `channel` 三类，含昵称变更归档（aliases）。
- **衰减遗忘**：动态保留分评分，低分自动归档。
- **三级人设演进**（Tier 1 自我觉察 → Tier 2 反思 → Tier 3 人设，最后一级默认关闭）。
- **群聊双路径路由**：个人事实归入用户实体，群组事实归入群组实体。
- **中文友好**：FTS5 使用 `jieba` 分词，中文全文检索精确。

## 安装

```bash
cd data/plugins/kira_plugin_hippocampus_memory
pip install -r requirements.txt
```

重启 KiraAI。插件会从 `data/plugins/` 自动发现。

> **依赖安装**：通过 WebUI 上传 zip 安装时，KiraAI 会自动执行 `requirements.txt`。
> 如果你是**手动**把目录拷进 `data/plugins/`，则需自己 `pip install -r requirements.txt`
> 到 KiraAI 实际使用的 Python 环境（如 `venv`），否则会报 `No module named 'tomli_w'`。
>
> **无需 LLM 也能加载**：即使尚未配置任何默认 LLM，插件也会正常加载——召回（FTS）、
> 手动记忆工具、迁移、自动禁用 simple_memory 都照常工作，只有后台海马体提取会休眠，
> 待配置好 LLM 并重启后自动启用。

## 与 simple_memory 的关系

插件启动时（默认）会通过官方 `PluginManager` API 自动禁用 `kira_plugin_simple_memory`，避免两个插件同时向系统 prompt 的 `memory` 段重复注入。

禁用前会一次性把 simple_memory 的 `data/memory/core.txt` 每一行非空内容作为 `importance=5` 的 fact 导入 `global/facts/`，导入标记落地为 `.simple_memory_migrated` 文件。

如果想关掉这两步，去 WebUI → 插件管理 → 海马体记忆 → 改对应开关。

## 数据布局

```
<plugin_data>/kira_plugin_hippocampus_memory/memory/
├── memory_index.db          # SQLite + FTS5（以及可选的 sqlite-vec）
├── entities/
│   ├── user_<adapter>%3A<uid>/      # 冒号在 Windows/NTFS 是保留字符，按 %3A 编码
│   │   ├── facts/*.toml
│   │   └── reflections/*.toml
│   └── group_<adapter>%3A<gid>/
│       ├── facts/
│       └── reflections/
├── global/
│   ├── self/{facts,reflections}/    # Bot 自我觉察（Tier 1/2）
│   └── facts/                       # 全局知识 / 迁移过来的旧数据
└── archive/                         # 衰减归档（TOML 含完整 meta）
```

TOML 文件是真相源。SQLite 索引丢失时，下次启动会从 TOML 文件重建。

实体 ID（如 `telegram:12345`）含冒号，而冒号在 Windows/NTFS 上是保留字符，无法作目录名，
因此目录名对其做百分号编码（`telegram%3A12345`）；SQLite 索引里仍存原始未编码的 ID，
重建索引时会自动解码还原，跨平台一致。

## 配置

完整字段见 WebUI → 插件管理 → 海马体记忆，或 [`schema.json`](./schema.json) 的默认值。

关键开关：

- `enable_recall`：关闭后只跑后台提取，不向 prompt 注入。
- `auto_disable_simple_memory` / `migrate_simple_memory_on_first_run`：控制接管 simple_memory 的行为。
- `enable_persona_evolution`：启用 Tier-3 人设跃迁（破坏性，默认关闭）。
- `enable_persona_perspective`：只给**主观类**提取（群氛围、反思、自我觉察）喂入 Bot 人设（只读），
  让其以角色视角判断——怕吵的人设会记成「太吵」而非中立的「氛围轻松」，避免主观感受错位。
  **客观**事实提取保持中立，防止角色偏见污染硬信息。默认关闭，会增加 token 开销。
- `decay_interval_hours`：衰减遗忘周期（小时），0 = 不跑。
- `hippocampus_chunk_threshold` / `reflection_threshold`：海马体触发与升维阈值。

## 调试 API

所有路由都在 `/api/plugin/kira_plugin_hippocampus_memory/` 下。

| 方法 | 路径 | 认证 | 描述 |
|---|---|---|---|
| GET | `/health` | 否 | 索引状态 + 记忆总数 |
| GET | `/entities` | 是 | 列出所有实体目录 |
| POST | `/recall` | 是 | body `{query, entity_id, entity_type, k}` |
| GET | `/profile/{entity_id}?entity_type=user` | 是 | 查实体画像 |
| POST | `/decay/run` | 是 | 手动触发一次衰减周期 |
| POST | `/evolution/run` | 是 | 手动触发一次人设演进 |
| DELETE | `/memory/{mem_id}` | 是 | 删除单条记忆 |

## 实施阶段

本插件按四个阶段实施。

- **阶段 A**：脚手架、自动禁用、recall 注入、`memory_add/update/remove/search` 工具、simple_memory 数据迁移。
- **阶段 B**：海马体后台提取（sender 缓存、双路径路由）。
- **阶段 C**：衰减引擎、实体画像、人设演进。
- **阶段 D**：文档与测试套件（基于 lightning `test_memory_system.py` 改写）。

## 测试

```bash
PYTHONPATH=. pytest data/plugins/kira_plugin_hippocampus_memory/tests/ -v
```

8 项测试覆盖：路径管理、目录结构、TOML CRUD、SHA-256 去重、实体画像 + 昵称归档、recall 链路、衰减引擎、召回 query 信封清洗。

测试需用装了 `pytest` + `jieba` + `tomli_w` 的 Python 运行（KiraAI 运行用的 `venv` 不一定装了 `pytest`）。

## 致谢

核心算法全部移植自 KiraAI-lightning。适配层（`adapters/llm.py`, `adapters/migration.py`, `adapters/sender_cache.py`）是为了把 lightning 的 standalone 代码桥接到 KiraAI 的插件宿主而新写的。

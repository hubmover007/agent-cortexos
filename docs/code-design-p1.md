# Agent-CortexOS 记忆服务 代码设计 P1

> 对应方案：docs/memory-service-design-v2.md（v2.1 已冻结）
> 日期：2026-08-04 | 状态：实施中
> 本文档定义：包结构、数据模型、核心算法（伪代码）、技术选型、测试策略

---

## 1. 包结构

```
agent-cortexos/
├── pyproject.toml              # 包名 agent-cortexos，Python >=3.10
├── cortexos/
│   ├── __init__.py             # 对外入口：CortexOS 类（嵌入模式）+ serve()
│   ├── config.py               # 配置加载（yaml + env override）
│   ├── models.py               # 领域模型（Entry/Fact/Edge/Zone/Agent/AgentKey）
│   ├── storage/
│   │   ├── __init__.py         # StorageBackend ABC + get_backend() 工厂
│   │   ├── sqlite_backend.py   # LocalBackend：SQLite + FTS5 + 向量
│   │   ├── jsonl_backend.py    # JSONL 导出/备份/导入
│   │   └── s3_backend.py       # ObjectBackend：S3 兼容（P5 实施，先留接口）
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── llm_extractor.py    # LLM 提取（实体/事实/关系/时间线索）
│   │   └── heuristic.py        # 无 LLM 降级（关键词/正则）
│   ├── zones/
│   │   ├── router.py           # 三层路由（实体→语义→_inbox）
│   │   ├── engine.py           # 涌现/合并/归档/重力
│   │   └── cluster.py          # 语义聚类（连通聚类）
│   ├── recall/
│   │   ├── __init__.py
│   │   ├── hybrid.py           # 多通道候选 + RRF 融合
│   │   ├── scoring.py          # 多因子评分
│   │   └── graph.py            # 图遍历（BFS + 衰减）
│   ├── lifecycle/
│   │   ├── consolidate.py      # 整合引擎（四阶段 + 三门控）
│   │   └── resolve.py          # 冲突消解（时间窗口截断）
│   ├── auth/
│   │   ├── pairing.py          # 配对流程（pair_code → agent_key）
│   │   ├── keys.py             # key 管理（hash 存储/吊销/轮换）
│   │   └── permissions.py      # 权限判定（scope 主 + zone 覆盖）
│   ├── api/
│   │   ├── server.py           # FastAPI 应用
│   │   ├── deps.py             # 认证/scope 依赖注入
│   │   └── routes/             # memories/retrieve/zones/pair/keys/consolidate
│   ├── cli/
│   │   └── main.py             # cortexos 命令（serve/pair/store/recall/...）
│   └── embedding/
│       ├── __init__.py
│       ├── base.py             # Embedder ABC
│       └── openai_compat.py    # easyrouter / OpenAI 兼容 embedding
├── docs/
│   ├── memory-service-design-v2.md   # 方案（已冻结）
│   └── code-design-p1.md             # 本文档
└── tests/
    ├── unit/                   # 单测（路由/涌现/评分/权限/冲突）
    ├── integration/            # 集成（API + SQLite）
    └── fixtures/
```

---

## 2. 数据模型（SQLite Schema）

```sql
-- 记忆条目
CREATE TABLE entries (
    id          TEXT PRIMARY KEY,          -- ULID
    scope       TEXT NOT NULL,             -- 归属域 agent:xxx / team:xxx
    zone        TEXT NOT NULL DEFAULT '_inbox',
    layer       TEXT NOT NULL DEFAULT 'episode',  -- raw|fact|facet|episode|project
    content     TEXT NOT NULL,
    entities    TEXT NOT NULL DEFAULT '[]',  -- JSON list[str]
    metadata    TEXT NOT NULL DEFAULT '{}',  -- JSON dict
    created_at  REAL NOT NULL,               -- epoch seconds
    updated_at  REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',  -- active|superseded|archived
    valid_until REAL                          -- NULL = 永不过期
);
CREATE INDEX idx_entries_scope ON entries(scope, status);
CREATE INDEX idx_entries_zone  ON entries(zone, status);
CREATE INDEX idx_entries_created ON entries(created_at);

-- 结构化事实（三元组）
CREATE TABLE facts (
    id          TEXT PRIMARY KEY,
    entry_id    TEXT NOT NULL REFERENCES entries(id),
    scope       TEXT NOT NULL,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  REAL NOT NULL,
    valid_until REAL,
    status      TEXT NOT NULL DEFAULT 'active'   -- active|superseded|conflict
);
CREATE INDEX idx_facts_spo ON facts(subject, predicate, object);
CREATE INDEX idx_facts_scope ON facts(scope, status);

-- 关系边（图遍历）
CREATE TABLE edges (
    id          TEXT PRIMARY KEY,
    entry_id    TEXT NOT NULL REFERENCES entries(id),
    scope       TEXT NOT NULL,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    valid_until REAL,
    created_at  REAL NOT NULL
);
CREATE INDEX idx_edges_source ON edges(source);
CREATE INDEX idx_edges_target ON edges(target);
CREATE INDEX idx_edges_scope ON edges(scope);

-- Zone
CREATE TABLE zones (
    name         TEXT PRIMARY KEY,
    scope        TEXT NOT NULL,              -- zone 属于 scope
    description  TEXT DEFAULT '',
    entities     TEXT NOT NULL DEFAULT '[]', -- JSON list[str]
    keywords     TEXT NOT NULL DEFAULT '[]', -- JSON list[str]
    centroid     TEXT,                       -- JSON embedding（质心向量）
    gravity      REAL NOT NULL DEFAULT 1.0,
    entry_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'active',  -- active|dormant|archived
    pinned       INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    last_access  REAL NOT NULL
);
CREATE INDEX idx_zones_scope ON zones(scope, status);

-- 词法索引（FTS5）
CREATE VIRTUAL TABLE entries_fts USING fts5(
    content, entities, zone, scope, content='entries', content_rowid='rowid'
);
-- 注意：FTS5 需要 rowid，entries 表加 rowid 隐式列即可

-- Agent 与密钥
CREATE TABLE agents (
    agent_id    TEXT PRIMARY KEY,
    agent_name  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active'   -- active|revoked
);
CREATE TABLE agent_keys (
    key_id      TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL REFERENCES agents(agent_id),
    key_hash    TEXT NOT NULL,               -- sha256(secret)，不存明文
    scope_permissions TEXT NOT NULL,         -- JSON {"scope": "read|write|readwrite"}
    zone_overrides    TEXT NOT NULL DEFAULT '{}', -- JSON {"scope:zone": "read"}
    rate_limit  INTEGER NOT NULL DEFAULT 100,
    expires_at  REAL,
    created_at  REAL NOT NULL,
    last_used   REAL,
    status      TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX idx_keys_agent ON agent_keys(agent_id);

-- 配对码（短时效）
CREATE TABLE pair_codes (
    code        TEXT PRIMARY KEY,            -- 随机 8 字符
    agent_id    TEXT NOT NULL,
    agent_name  TEXT NOT NULL,
    expires_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'  -- pending|approved|expired
);

-- 审计日志
CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    agent_id    TEXT,
    key_id      TEXT,
    action      TEXT NOT NULL,               -- pair_request/pair_approve/store/retrieve/delete/revoke/...
    scope       TEXT,
    detail      TEXT DEFAULT ''
);
CREATE INDEX idx_audit_ts ON audit_log(ts);

-- 向量索引（内存 HNSW，<10万条量级）：
-- sqlite-vec 或自研近似；启动时从 entries 加载 embedding 列
-- 方案：entries 表增加 embedding TEXT（JSON float[]）列，内存构建索引
```

---

## 3. 核心算法设计（伪代码）

### 3.1 Zone 三层路由

```
route(entry, zones, embedder):
    # Layer 1: 实体精确匹配
    best_zone, best_overlap = None, 0
    for z in active_zones:
        overlap = |entry.entities ∩ z.entities|
        if overlap > best_overlap:
            best_zone, best_overlap = z, overlap
    if best_zone and best_overlap >= 1:
        return best_zone.name

    # Layer 2: 语义匹配（质心余弦）
    if entry.embedding:
        best_sim = 0
        for z in active_zones:
            if z.centroid is None: continue
            sim = cosine(entry.embedding, z.centroid)
            if sim > best_sim:
                best_sim, best_zone = sim, z
        if best_sim >= config.semantic_threshold:   # 默认 0.72
            return best_zone.name

    # Layer 3: 兜底
    return '_inbox'
```

### 3.2 Zone 涌现（连通聚类 + 自适应阈值）

```
emergence_scan(inbox_entries, zones, embedder, total_entries):
    # 质量门
    candidates = [e for e in inbox_entries
                  if len(e.content) >= min_content_len(20)
                  and e.embedding is not None]

    # 增量：先尝试归入现有 zone（相似度 > semantic_threshold）
    remaining = []
    for e in candidates:
        z = route_semantic_only(e, zones)
        if z: add_to_zone(z, e); continue
        remaining.append(e)

    # 剩余条目：连通聚类
    clusters = []
    for e in remaining:
        matched = None
        for c in clusters:
            if centroid_sim(c, e) >= cluster_similarity(0.75):
                matched = c; break
        if matched: matched.append(e)
        else: clusters.append([e])

    # 自适应阈值
    base = config.base_threshold          # 5
    scale = max(1.0, log2(total_entries / 100))
    threshold = base * scale

    # 涌现判定
    for c in clusters:
        if len(c) >= threshold:
            zone = create_zone_from_cluster(c)   # 名称=LLM/关键词，质心=均值
            move_entries(c, zone)
        # 不够阈值的留在 _inbox（下次扫描继续累积）

复杂度：O(n × m)，n=候选数，m=簇数（簇数通常很小）
```

### 3.3 多因子评分

```
score(entry, query_vec, query_entities, now):
    text_sim = rrf_fusion(entry)          # 语义+词法通道的 RRF 融合分（0~1）
    recency  = exp(-ln2 * age_days / half_life)      # half_life=7
    gravity  = zone_gravity(entry.zone)   # 缓存于 zones.gravity
    freq     = 1 - exp(-access_count / 50)
    scope_boost = 0.1 if entry.scope == query.scope else 0
    graph_path = graph_score(entry)       # 图遍历通道（见 3.4）
    return (0.35*text_sim + 0.25*recency + 0.15*gravity
            + 0.10*freq + 0.05*scope_boost + 0.10*graph_path)
```

### 3.4 图遍历（BFS ≤2 跳 + 衰减）

```
graph_search(query_entities, hop=2, decay=0.5):
    # 锚点 = 查询实体
    anchors = query_entities
    visited = set()
    scores = {}   # entry_id -> graph score

    def bfs(entity, depth):
        if depth > hop: return
        for edge in edges_from(entity):       # source=entity 或 target=entity
            other = edge.target if edge.source == entity else edge.source
            path_score = edge.weight * (0.5 ** (depth - 1))
            # 关系语义相似度（可选，用 edge.embedding vs query_vec）
            for entry in entries_linked_to(other):
                scores[entry.id] = max(scores.get(entry.id, 0), path_score)
            if depth < hop and other not in visited:
                visited.add(other)
                bfs(other, depth + 1)

    for a in anchors: bfs(a, 1)
    return normalize(scores)
```

### 3.5 冲突消解（时间窗口截断）

```
resolve_fact(new_fact, existing_facts):
    # 同 scope + 同 subject+predicate 的活跃事实
    for old in existing_facts:
        if old.object == new_fact.object:
            # 一致 → 合并：置信度取高，valid_until 取远
            old.confidence = max(old.confidence, new_fact.confidence)
            old.valid_until = max(old.valid_until, new_fact.valid_until)
        else:
            # 矛盾 → 时间窗口截断（Graphiti 模式）
            if new_fact.valid_from >= old.valid_from:
                old.valid_until = new_fact.valid_from   # 旧事实在新区间前失效
                old.status = 'superseded' if old.valid_until <= now else 'active'
                new_fact.status = 'active'
            else:
                # 无法判断新旧 → 都保留，标记 conflict
                new_fact.status = 'conflict'
    return new_fact
```

### 3.6 整合引擎（四阶段 + 三门控）

```
consolidate(scope):
    # 门控
    if not time_gate(24h): return skip
    if new_entries < content_gate(50) and not manual: return skip
    acquire_lock()                        # 锁门（进程内互斥）

    # Orient：读取 zone 现状
    zones = load_zones(scope)

    # Gather：扫描新增 raw 条目
    raws = scan_entries(scope, layer='raw', since=last_consolidate)

    # Consolidate：
    for group in cluster_similar(raws):   # 相似度 > 0.8 的 raw 分组
        if len(group) >= 3:
            episode = llm_summarize(group)      # LLM 总结（可配置关）
            store_episode(episode, source_ids=group.ids)
            mark_superseded(group.ids)          # raw 标记 superseded

    # 冲突处理（§3.5）
    resolve_all_facts(scope)

    # Prune：
    archive_dormant_zones(scope)          # dormant 30天/archive 90天
    trim_index(scope, budget=200/25KB)    # 索引预算修剪
    release_lock()
```

### 3.7 配对 + 权限判定

```
pair_request(agent_name):
    agent_id = 'ag_' + ulid()
    code = random_code(8)                 # 大写字母数字，排除易混字符
    store pair_code(code, agent_id, agent_name, expires=15min)
    return code

pair_approve(code, admin_scopes):        # CLI 管理员操作
    # 校验 code 有效 → 生成 agent_key
    secret = secrets.token_urlsafe(32)
    key_hash = sha256(secret)
    store key(agent_id, key_hash, scope_permissions=admin_scopes)
    return {key_id, secret}               # secret 仅此一次可见

check_permission(key, scope, zone, action):
    # ① zone 覆盖（单调递减：override 权限 ≤ scope 权限）
    ov = key.zone_overrides.get(f"{scope}:{zone}")
    if ov:
        base = key.scope_permissions.get(scope)
        return rank(ov) <= rank(base) and rank(ov) >= rank(action)
    # ② scope 权限
    perm = key.scope_permissions.get(scope)
    return perm and rank(perm) >= rank(action)

rank: read=1, write=2, readwrite=2        # write 与 readwrite 同级
单调递减校验：zone 覆盖的 rank 必须 ≤ scope 的 rank
```

---

## 4. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 服务框架 | FastAPI + uvicorn | 异步、自动 OpenAPI 文档 |
| 本地存储 | SQLite（WAL）+ FTS5 | 零依赖、单文件、够用 |
| 向量索引 | sqlite-vec（若兼容）否则内存 HNSW（numpy 自研近似） | 量级 <10 万条 |
| Embedding | OpenAI 兼容接口（easyrouter 统一 key） | 已有基础设施 |
| LLM 提取 | OpenAI 兼容 chat/completions（easyrouter） | 同上 |
| 图遍历 | 内存邻接表（启动加载 edges） | 量级小，BFS 快 |
| 对象存储 | boto3（S3 兼容：OSS/MinIO） | P5 实施 |
| ULID | python-ulid | 时间有序 ID |
| 配置 | PyYAML + 环境变量覆盖 | 全参数化 |

**Embedding 本地降级**：未配置 embedding API 时用 TF-IDF 向量（自研，scipy/numpy），保证零依赖可用。

---

## 5. 测试策略

| 层 | 内容 |
|----|------|
| 单元 | 路由（三层各分支）、涌现（阈值/质量门/聚类）、评分（各因子权重）、图遍历（衰减/跳数）、冲突消解（一致/矛盾/窗口截断）、权限（单调递减/越权拒绝）、配对（过期/重复使用） |
| 集成 | API 全流程：配对→写入→检索→zone 涌现→整合→权限拒绝 |
| 性能 | 写入 10k 条：路由 + 索引耗时；检索 p95 < 100ms |
| 边界 | 空内容、超长内容、无 LLM 降级、scope 不存在、key 吊销后访问 |

**关键测试用例（先行）**：
1. 路由：实体重叠最多者胜；语义阈值边界（0.719/0.720/0.721）
2. 涌现：49 条不涌现、50 条涌现（threshold=5×scale）
3. 冲突：新事实 valid_from 截断旧事实 valid_until
4. 权限：zone 覆盖 write 越权被拒；scope read 时 zone 覆盖 write 被拒（单调递减）
5. 配对：code 过期拒绝；code 二次兑换拒绝

---

## 6. 实施顺序

| 步 | 内容 | 验证 |
|----|------|------|
| 1 | pyproject + config + models + sqlite schema | 建表 + CRUD 单测 |
| 2 | embedding（openai_compat + tfidf 降级） | 向量相似度单测 |
| 3 | zones（router + engine + cluster） | 路由/涌现单测 |
| 4 | recall（hybrid + scoring + graph） | 检索单测 |
| 5 | extract（llm_extractor + heuristic） | 提取单测（mock LLM） |
| 6 | lifecycle（consolidate + resolve） | 冲突/整合单测 |
| 7 | auth（pairing + keys + permissions） | 配对/权限单测 |
| 8 | api（FastAPI 全套路由） | 集成测试 |
| 9 | cli（serve/pair/store/recall/zones） | 端到端冒烟 |
| 10 | sdk（CortexOS 类统一入口） | 嵌入模式冒烟 |

---

*P1 代码设计结束。按 §6 顺序实施，每步跑通测试再进下一步。*

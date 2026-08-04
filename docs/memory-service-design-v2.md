# Agent-CortexOS 通用记忆服务设计方案 v2.0

> 作者：张云飞 & 小白 | 日期：2026-08-04 | 状态：**草案，待确认**
> 定位：通用记忆服务（Memory-as-a-Service），**不绑定任何具体 Agent**，任何 Agent 均可接入读写统一记忆。
> 本文档为方案设计；**经确认后才进入代码设计**。

---

## 0. 版本变更

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-13 | CortexOS 初版（Zone 三层路由 / 多因子评分 / JSONL 存储） |
| v2.0 | 2026-08-04 | 重构为通用记忆服务：接入配对安全体系、存储可插拔（本地/对象存储）、Zone 涌现算法优化、融合业界最佳实践（Mem0/Graphiti/Letta） |
| v2.1 | 2026-08-04 | Scope/Zone 核心区分、Agent 维度、图遍历纳入、四源融合溯源明确（+M-FLOW / +OpenClaw 记忆系统）、待确认项给出建议值 |

---

## 1. 背景与目标

### 1.1 问题

v1.0 CortexOS 验证了核心概念（Zone 自动组织、多因子召回），但存在明显不足：

1. **耦合设计**：部分设计隐含特定场景（如运维类 Agent），不是通用记忆服务
2. **无安全体系**：任何调用者都能读写全部记忆，无法安全地开放给"任意 Agent"
3. **存储单一**：仅 JSONL 本地文件，无法满足云端部署、多副本、对象存储需求
4. **Zone 涌现算法粗糙**：仅 TF-IDF + 阈值，缺少语义聚类、时间衰减、冲突消解
5. **无冲突/过期处理**：记忆更新时旧事实如何处理？事实有"保质期"吗？

### 1.2 目标

1. **通用**：服务本身不感知任何具体 Agent 类型；提供标准接入方式（REST/SDK/MCP），任意 Agent 均可读写
2. **安全**：配对（Pairing）机制 + API Key + Scope 权限模型，多 Agent 隔离与受控共享
3. **存储可插拔**：本地部署（SQLite/JSONL）或云端对象存储（S3/OSS），二选一即可运行，接口统一
4. **记忆质量**：Zone 涌现算法优化、冲突消解、时间有效性（事实过期）、整合压缩
5. **检索**：混合检索（语义向量 + 词法 + 图/关系），多因子评分，时间衰减

### 1.3 设计溯源（四源融合）⭐

本方案不是从零设计，而是融合了四套成熟方案 + 业界最佳实践。每个关键设计的出处明确标注，避免"重新发明轮子"：

| 设计点 | 来源 | 融入位置 |
|--------|------|---------|
| Zone 三层路由（实体→语义→_inbox） | CortexOS v1.0 | §4.2 |
| 多因子评分（text_sim/recency/gravity/freq） | CortexOS v1.0 | §6.2 |
| Zone 生命周期（dormant/archive/merge） | CortexOS v1.0 | §5.3 |
| 四层抽象拓扑（FACTS→FACETS→EPISODES→PROJECTS） | Agent-Loop M-FLOW | entry.layer 字段（§2.1） |
| 图遍历 + 代价传播（锚点→子图→路径得分） | Agent-Loop M-FLOW | §6.1 图遍历通道 |
| 关系边数据模型（edges） | Agent-Loop M-FLOW | §2.1 Edge |
| 整合四阶段（Orient/Gather/Consolidate/Prune） | OpenClaw dream-consolidator | §9.2 |
| 三门控（时间≥24h/内容≥N/锁） | OpenClaw dream-consolidator | §9.1 |
| 矛盾消解规则（新者优先/结构化优先/信息量胜） | OpenClaw | §4.3 |
| 索引预算（≤200条/25KB） | OpenClaw MEMORY.md | §9.3 |
| 原始日志→结构化提炼（raw→episode） | OpenClaw（日志→topic 三层） | §9.2a |
| 时间有效性窗口（事实保质期） | Graphiti/Zep（业界） | §2.1 valid_until |
| 提取→冲突检测→图更新流水线 | Mem0（业界） | §4 |
| 分层记忆思想 | Letta（业界） | layer 层级 |
| 混合检索（语义+词法+实体+图） | 业界标准 | §6.1 |

> **融合原则**：CortexOS 提供"自动组织 + 多因子召回"骨架，M-FLOW 提供"层级抽象 + 图推理"深度，OpenClaw 提供"整合进化 + 预算控制"机制，业界方案补齐"时间有效性 + 冲突检测"缺口。四者各司其职，无重复设计。

### 1.4 非目标（v2.0）

- 不做 Agent 运行时（只做记忆）
- 不做分布式一致性（单实例部署，多实例靠对象存储+锁）
- 不做跨语言 SDK 全集（先 Python + 通用 REST；TS SDK 后续）

---

## 2. 核心概念

### 2.1 领域模型

```
MemoryEntry（记忆条目）
├── id: str                    # 唯一 ID（ULID，含时间序）
├── content: str               # 记忆内容（文本）
├── entities: list[str]        # 提取的实体（LLM 提取）
├── facts: list[Fact]          # 结构化事实（可选，LLM 提取）
├── scope: str                 # 归属域（agent/user/team 标识，通用）
├── zone: str                  # 所属 Zone（自动路由）
├── layer: str                 # 抽象层级：raw|fact|facet|episode|project
├── metadata: dict             # 自定义元数据（来源、类型、标签）
├── created_at / updated_at    # 时间戳
├── access_count               # 访问计数（评分用）
├── valid_until: datetime|null # 时间有效性窗口（Graphiti 借鉴）
└── status: active|superseded|archived   # 生命周期状态

Fact（结构化事实）
├── subject / predicate / object   # 三元组
├── confidence: float              # 置信度
├── valid_from / valid_until       # 有效性窗口（关键：事实会过期）
└── source_entry_id                # 来源条目

Edge（关系边，图检索用）
├── source: str                    # 源实体
├── target: str                    # 目标实体
├── relation: str                  # 关系描述（自然语言）
├── embedding: list[float]         # 关系语义向量
├── scope: str                     # 归属域
├── weight: float                  # 权重（随访问/时间调整）
└── created_at / valid_until       # 时间有效性

Zone（记忆域）
├── name: str
├── description: str
├── entities: list[str]            # 绑定实体（路由用）
├── scope_keywords: list[str]      # 范围关键词（路由用）
├── gravity: float                 # 重力（评分用，随时间衰减）
├── status: active|dormant|archived
└── created_at / last_access_at
```

### 2.2 Scope（归属域）与 Zone（记忆域）——核心概念区分 ⭐

**这是本方案最重要的设计决策，必须严格区分：**

| | **Scope（归属域）** | **Zone（记忆域）** |
|---|---|---|
| 回答的问题 | **谁的数据？**（WHO） | **数据是什么？**（WHAT） |
| 本质 | 所有权/主权边界 | 内容/语义组织 |
| 类比 | 用户账户（你的 vs 我的） | 文件夹分类（工作 vs 生活） |
| 稳定性 | **稳定**（权限锚点） | **动态**（自动涌现/合并/归档） |
| 跨域 | 跨 scope = 越权（需授权） | 跨 zone = 正常（同 scope 内） |
| 生命周期 | 人工管理（配对/吊销） | 自动管理（Zone 引擎） |

**铁律**：
1. 权限**只能锚定在 Scope** 上，绝不能锚定 Zone（Zone 会动态变化，权限会乱）
2. Zone 组织变化（涌现/合并/归档）**零权限副作用**
3. Scope 是字符串标识，服务不解析语义，零场景硬编码

### 2.3 Agent 维度（配对实体）

服务唯一感知的"人"是 Agent（配对实体）：

```
Agent
├── agent_id                     # 配对时注册的唯一 ID
├── own_scope: "agent:{id}"      # 自动拥有的专属 scope（不可剥夺）
├── grants: [                    # 管理员授予的其他 scope 权限
│     { scope, permission: read|write|readwrite }
│   ]
└── zone_overrides: [            # Zone 级收紧（可选增强）
      { scope, zone, permission: read }   # 只允许收紧，不允许越权
    ]
```

- 每个 Agent 配对即获得 `agent:{id}` 专属域，天然隔离
- 共享 = 管理员授予 grants（如 `team:infra` 只读）
- 多 Agent 协作：
  - 隔离：只持有 own_scope
  - 共享读：被授予 `team:*` 只读
  - 跨写：被授予目标 scope 的 write（需显式授权）

### 2.4 权限判定（严格顺序）

```
请求 (scope, zone, action)
→ ① 命中 zone_overrides？→ 用覆盖权限（最精确）
→ ② 命中 own_scope/grants？→ 用 scope 权限
→ ③ 都没有 → 拒绝（403）

额外约束：
- zone_overrides 只允许 read（收紧），不允许 write 越权
- 写操作默认要求 scope 级 write
- 删除/导出/归档为敏感操作，scope write + 可选二次确认
```

---

## 3. 总体架构

```
┌──────────────────────────────────────────────────────────┐
│                    任意 Agent（调用方）                     │
│  OpenClaw │ Agent-Loop │ Pi │ Claude Code │ 自定义 Agent  │
└──────────┬────────────┬──────┬────────────┬──────────────┘
           │            │      │            │
   ┌───────▼────┐ ┌─────▼────┐ │ ┌──────────▼─────────┐
   │ Python SDK │ │ REST API │ │ │ MCP Server（可选）   │
   └───────┬────┘ └─────┬────┘ │ └──────────┬─────────┘
           └────────────┴──────┴────────────┘
                        │
        ┌───────────────▼────────────────┐
        │        Memory Service          │
        │  (FastAPI 独立进程/容器)         │
        ├────────────────────────────────┤
        │  Auth: 配对 + API Key + Scope   │
        │  Write Pipeline: 提取→路由→存储 │
        │  Zone Engine: 涌现/合并/归档    │
        │  Recall Engine: 混合检索+评分   │
        │  Lifecycle: 整合/冲突/过期      │
        └───────────────┬────────────────┘
                        │
        ┌───────────────▼────────────────┐
        │        Storage Backend         │
        │  LocalBackend: SQLite+JSONL    │ ← 本地部署（默认）
        │  ObjectBackend: S3/OSS + 索引  │ ← 云端对象存储（可选）
        └────────────────────────────────┘
```

---

## 4. 写入流水线（Write Pipeline）

每次 `POST /v1/memories` 经过四阶段：

```
写入请求 → ① 提取(Extract) → ② 路由(Route) → ③ 冲突消解(Resolve) → ④ 存储(Store)
```

### 4.1 提取（Extract）

- 调用 LLM（**默认开启**，可配置关闭）提取：
  - 实体（entities）
  - 结构化事实三元组（facts，可配置开关）
  - **实体间关系（edges，图检索用）**：如 (服务器A) -[部署了]-> (服务B)
  - 时间线索（valid_until 推断，如"临时方案""下月失效"）
- 提取失败或未配置 LLM → 降级为纯文本存储（实体为空，靠词法检索）
- **设计原则**：LLM 默认开启保证记忆质量；可配置关闭实现零依赖自举

### 4.2 路由（Route）——Zone 分配

三层路由（继承 v1.0 并优化）：

```
Layer 1: 实体精确匹配    entry.entities ∩ zone.entities ≠ ∅ → 取重叠最多者
Layer 2: 语义匹配        内容 embedding 与 zone 中心向量相似度 > 阈值
Layer 3: 兜底            → _inbox（等待涌现）
```

- Zone 中心向量：该 Zone 内条目 embedding 的滚动质心（增量更新）
- 阈值参数化：`route.semantic_threshold`（默认 0.72，可配置）

### 4.3 冲突消解（Resolve）——Mem0 借鉴

写入前对同 scope 内相似事实做检测：

```
新事实 F_new vs 已有事实 F_old（同 subject+predicate）
├─ 一致 → 合并（更新 valid_until、增加 confidence）
├─ 矛盾 → 裁决：
│   ├─ F_new.valid_from ≥ F_old.valid_from → 新事实生效，
│   │    F_old.valid_until = F_new.valid_from（时间窗口截断，Graphiti 模式）
│   └─ 无法判断 → 两条保留，status 标记 conflict，待整合引擎处理
└─ 无冲突 → 直接写入
```

- 冲突检测可关闭（`resolve.enabled=false`，纯 append 模式）

### 4.4 存储（Store）

- 写入 Storage Backend + 更新索引（词法/向量/关系）
- 返回 `memory_id`（ULID）

---

## 5. Zone 涌现算法（v2.0 深度优化）

### 5.1 问题（v1.0 粗糙点）

1. 仅靠 TF-IDF 词重叠，语义相近但用词不同的条目聚不到一起
2. 阈值固定（5），不随数据规模自适应
3. 无时间维度：陈旧话题仍会涌现，噪音 Zone 累积
4. 无质量门：低质量条目（过短/无实体）也能触发涌现

### 5.2 v2.0 涌现算法

`_inbox` 条目积累后，周期性（可配置，默认每小时）触发：

```
Step 1: 候选过滤（质量门）
  过滤 content 长度 < min_len(20) 或 无实体且 embedding 为空的条目

Step 2: 聚类（两阶段）
  a) 增量聚类：新条目与现有 Zone 质心相似度 > threshold → 归入
  b) 剩余条目：按 embedding 相似度做连通聚类（相似度矩阵 > 0.75 连边）
     得到候选簇

Step 3: 涌现判定（自适应阈值）
  cluster_size ≥ emergence_threshold  → 创建新 Zone
  emergence_threshold 自适应：
    base = 5
    scale = max(1, log2(total_entries / 100))   # 数据越多阈值越高
    threshold = base * scale

Step 4: Zone 初始化
  名称/描述由 LLM 生成（可选）；无 LLM 时取簇中心条目关键词
  绑定实体 = 簇内高频实体（Top-5）
  质心向量 = 簇内条目 embedding 均值
  重力 = 簇内平均新鲜度（新簇重力高）

Step 5: 质量复核（可选，LLM）
  对低置信簇做一次 LLM 判定：是否值得建 Zone；不值得则留在 _inbox
```

### 5.3 Zone 生命周期（参数全部可配置）

```
active ── 30 天无访问（dormant_days）──→ dormant
dormant ── 90 天无访问（archive_days）──→ archived（不参与检索，可恢复）
active/dormant ── 与另一 Zone 质心相似度 > 0.7（merge_threshold）──→ 合并
合并规则：实体并集、关键词并集、重力加权平均、条目全部归入
```

### 5.4 Zone 重力（Gravity）

```
gravity = 新鲜度因子 × 活跃度因子 × 规模因子
新鲜度：exp(-λ × days_since_last_access)，λ 可配置（默认 0.02）
活跃度：1 - exp(-access_count / k)，k 可配置（默认 50）
规模：  1 - exp(-entry_count / m)，m 可配置（默认 100）
```

---

## 6. 检索与评分（Recall Engine）

### 6.1 混合检索（五通道并行）

| 通道 | 实现 | 适用 |
|------|------|------|
| 语义向量 | embedding 余弦相似度 | 语义近似 |
| 词法 | FTS5 / 倒排索引（BM25） | 精确关键词 |
| 实体 | 查询实体 → 关联条目 | 实体级精确 |
| 图遍历 | 锚点实体 → 一跳/二跳邻居 → 路径得分 | 关联推理（如"X 部署在 Y 上，Y 出过事故 Z"） |
| 时间 | valid_until 窗口过滤 | 时效性 |

图遍历算法（M-FLOW 风格）：
```
1. 查询提取实体 → 锚点
2. 锚点在 edges 图中扩散（BFS，hop ≤ 2）
3. 路径得分 = Σ(边权重 × 关系语义相似度) - 跳跃惩罚
4. 命中条目的图得分纳入多因子评分
```

### 6.2 多因子评分（v2.0 优化）

```
score = w1×text_sim + w2×recency + w3×gravity + w4×freq + w5×scope_boost + w6×graph_path

text_sim   语义+词法融合相似度（RRF 融合各通道候选）
recency    exp(-ln2 × age_days / half_life)，half_life 默认 7 天
gravity    Zone 重力（§5.4）
freq       访问频率归一化
scope_boost 当前 scope 精确命中加分（默认 0.1，可配置为 0 关闭）
graph_path 图遍历路径得分（默认 0.1）

权重默认：w1=0.35, w2=0.25, w3=0.15, w4=0.1, w5=0.05, w6=0.1（全部可配置）
```

### 6.3 时间有效性过滤

- `valid_until` 已过期的 Fact 默认不参与检索（可带 `include_expired=true` 强制）
- 检索结果附带 `valid_until`，供调用方判断

### 6.4 上下文组装（Context Assembly）

```
POST /v1/context { query, budget_tokens, scope, include_zones }
→ 检索 top-k → 按预算裁剪 → 组装为结构化上下文块：
  [{zone, layer, content, score, valid_until}]
→ 供 Agent 直接注入 system prompt / 上下文
```

---

## 7. 接入方式与配对安全体系 ⭐

### 7.1 设计目标

- 任意 Agent 可接入（不预设类型）
- 接入需要**显式配对**（防止任何人读写你的记忆）
- 配对后按 Scope 授权，最小权限原则

### 7.2 配对流程（Pairing）

类似设备配对（如蓝牙/智能家居）：

```
Agent 侧：                          Memory Service 侧：
  1. 请求配对（带 agent 名称/类型）  →  生成一次性配对码 pair_code（短时效，如 15 分钟）
  2. 人工确认：在服务管理端 approve  ←  管理员确认（CLI/Web/API）
  3. 凭 pair_code 换取凭证          →  颁发 agent_key + scope 权限
  4. 后续所有请求带 agent_key        →  校验通过，按 scope 权限执行
```

### 7.3 凭证模型

```
AgentKey
├── key_id: str                    # 公开标识
├── key_secret: str                # 密钥（仅颁发时可见一次）
├── scope_permissions: dict        # { scope: "read" | "write" | "readwrite" }
├── allowed_zones: list[str] | "*" # 可访问 Zone 白名单（可选）
├── rate_limit: int                # 限流（默认 100 req/min）
├── expires_at: datetime | null    # 过期时间（可轮换）
└── created_at / last_used_at
```

### 7.4 请求认证

```
Authorization: Bearer <key_secret>
X-Scope: agent:pi-coding        # 本次操作的目标 scope
```

- 每次请求校验：密钥有效 + scope 在权限内 + 限流
- 敏感操作（删除/归档/导出）要求 scope 权限为 write 且可选二次确认

### 7.5 多 Agent 协作模型（通用，不预设场景）

```
隔离：   agent A 只有 scope "agent:a" 权限 → 只读写自己的记忆
共享：   管理员给 agent A 配 scope "team:infra" read 权限 → A 可读团队知识
委托：   agent A 写入时指定 scope "agent:b"（需 write 权限）→ 跨 agent 写入
```

全部由 Scope 权限矩阵表达，**服务本身无任何场景硬编码**。

### 7.6 安全边界

- 密钥存储：服务端只存 hash（sha256），不存明文
- 传输：HTTPS（生产强制）
- 审计：所有写操作 + 配对操作记录审计日志
- 可撤销：管理员可随时吊销 agent_key

---

## 8. 存储设计（本地 / 对象存储二选一）

### 8.1 Storage Backend 抽象

```python
class StorageBackend(ABC):
    # 条目
    async def upsert_entry(self, entry) -> None
    async def get_entry(self, entry_id) -> Entry | None
    async def delete_entry(self, entry_id) -> None
    async def list_entries(self, *, scope=None, zone=None, limit, offset) -> list[Entry]
    # 检索
    async def search_lexical(self, query, *, scope, top_k) -> list[Scored]
    async def search_vector(self, embedding, *, scope, top_k) -> list[Scored]
    # Zone
    async def upsert_zone(self, zone) -> None
    async def list_zones(self, *, status=None) -> list[Zone]
    # 批量（整合用）
    async def scan_entries(self, *, since, scope=None) -> AsyncIterator[Entry]
    async def bulk_update(self, updates: list[tuple[id, dict]]) -> None
```

### 8.2 LocalBackend（本地部署，默认）

- **主存储**：SQLite（WAL 模式）
  - 表：entries / facts / zones / edges（关系）/ keys / audit_log
  - FTS5 虚拟表（词法索引）
  - 向量：本地近似（如 sqlite-vec 或内存 HNSW，量级 <10 万条足够）
- **导出/备份**：JSONL 按月分片（v1.0 保留，作为备份与迁移格式）
- 适用：单机、内网、个人使用

### 8.3 ObjectBackend（云端对象存储）

- **对象存储**：S3 / 阿里云 OSS / 兼容实现（MinIO）
  - 对象键：`memories/{scope}/{zone}/{yyyy}/{mm}/{entry_id}.json`
  - 追加写入：单对象小（<64KB 建议），或按批合并
  - 版本化：对象版本即历史（可回溯）
- **索引**：SQLite/Postgres（元数据 + FTS + 向量），对象存储只放原始数据
- **生命周期**：对象存储自带归档策略（过期/冷存储）对接 Zone 归档
- 适用：多实例、云端、需要备份容灾

### 8.4 选择建议

| 场景 | 选择 |
|------|------|
| 个人/单机/开发 | LocalBackend（SQLite） |
| 生产/多实例/团队 | ObjectBackend（OSS/S3 + 索引库） |
| 两者可随时切换 | 配置 `storage.backend=local|object`，数据可迁移工具 |

---

## 9. 记忆生命周期（整合引擎）

### 9.1 门控（可配置）

- 时间门：距上次整合 ≥ 24h
- 内容门：新增条目 ≥ 50 条 或 手动触发
- 锁门：单实例内互斥（避免并发整合）

### 9.2 整合流程（四阶段）

```
Orient    → 读取 Zone 现状、统计
Gather    → 扫描新增 raw 条目
Consolidate → 
   a) raw 相似条目 → 合并为 episode（LLM 总结，可选）
   b) 事实冲突 → 时间窗口截断/标注（§4.3）
   c) 冗余条目 → 压缩（保留信息量最大者）
Prune     → Zone 归档、索引预算修剪、软删除
```

### 9.3 索引预算（借鉴 OpenClaw）

- 每个 scope 的"索引视图"（供 Agent 快速浏览）限制：默认 200 条 / 25KB
- 超出自动修剪（保留高重力条目）

---

## 10. API 总览（v1）

```
# 配对与凭证
POST   /v1/pair/request          # 请求配对（返回 pair_code）
POST   /v1/pair/confirm           # 管理员确认配对
POST   /v1/pair/exchange          # 用 pair_code 换 agent_key
GET    /v1/keys                   # 列出我的 key（管理员）
DELETE /v1/keys/{key_id}          # 吊销 key

# 记忆读写
POST   /v1/memories               # 写入（自动提取/路由/消解）
GET    /v1/memories/{id}          # 读取
DELETE /v1/memories/{id}          # 删除（软删）
POST   /v1/retrieve               # 语义检索 {query, top_k, scope}
POST   /v1/search                 # 词法检索 {query, top_k, scope}
POST   /v1/context                # 上下文组装 {query, budget_tokens}
GET    /v1/scopes                 # 我有权限的 scope 列表

# Zone
GET    /v1/zones                  # 列表（按 scope/status 过滤）
POST   /v1/zones                  # 手动建 Zone（可选，一般自动涌现）
POST   /v1/zones/{name}/pin       # 固定 Zone（防归档）

# 生命周期
POST   /v1/consolidate            # 触发整合（异步，返回 task_id）
GET    /v1/tasks/{task_id}        # 查询异步任务
GET    /v1/stats                  # 统计（条目/Zone/存储量）

# 健康
GET    /healthz                   # 健康检查
```

---

## 11. SDK 与 MCP

### 11.1 Python SDK（v2.0 首发）

```python
from cortexos import CortexOS

cx = CortexOS(
    base_url="http://localhost:8200",
    agent_key="ak_xxx",            # 配对获得
    scope="agent:my-agent",        # 默认 scope
)

# 写入
mid = await cx.store("K8s Pod 重启：检查 OOMKilled，调整内存限制")

# 检索
results = await cx.recall("容器内存问题", top_k=5)

# 上下文组装
ctx = await cx.context("当前任务上下文", budget_tokens=4000)

# Zone 管理
zones = await cx.zones()
```

### 11.2 REST 示例（curl，任何语言可用）

```bash
# 配对（管理员确认后）
curl -X POST http://localhost:8200/v1/pair/request \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "my-agent"}'

# 写入
curl -X POST http://localhost:8200/v1/memories \
  -H "Authorization: Bearer ak_xxx" \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "scope": "agent:my-agent"}'

# 检索
curl -X POST http://localhost:8200/v1/retrieve \
  -H "Authorization: Bearer ak_xxx" \
  -d '{"query": "...", "top_k": 5, "scope": "agent:my-agent"}'
```

### 11.3 MCP Server（可选，v2.1）

- 暴露为 MCP 2026-07-28 无状态服务器
- 工具：`memory_store` / `memory_retrieve` / `memory_search` / `memory_forget` / `memory_context`
- 任何支持 MCP 的 Agent（Claude Code/Pi/OpenClaw）零代码接入
- 认证复用 agent_key（MCP 请求头带 Authorization）

---

## 12. 配置（全参数化，无硬编码）

```yaml
# config.yaml 示例（全部有默认值，可省略）
server:
  host: 0.0.0.0
  port: 8200
  tls: false

storage:
  backend: local            # local | object
  local:
    path: ./data/memory.db
  object:
    bucket: my-agent-memory
    endpoint: oss-cn-beijing.aliyuncs.com   # S3/OSS/MinIO 兼容
    region: cn-beijing
    index_db: ./data/index.db
  jsonl_export: ./data/export   # 备份导出目录

llm:                        # 提取用 LLM（可选，不配则降级）
  provider: easyrouter      # 任意 OpenAI 兼容
  base_url: ""
  api_key_env: LLM_API_KEY
  model: ""
  extract_entities: true
  extract_facts: false

zone:
  emergence:
    base_threshold: 5
    semantic_threshold: 0.72
    min_content_len: 20
    cluster_similarity: 0.75
  lifecycle:
    dormant_days: 30
    archive_days: 90
    merge_threshold: 0.7
  gravity:
    decay_lambda: 0.02
    activity_k: 50
    scale_m: 100

recall:
  weights: { text_sim: 0.4, recency: 0.3, gravity: 0.15, freq: 0.1, scope_boost: 0.05 }
  recency_half_life_days: 7.0
  rrf_k: 60

resolve:
  enabled: true
  conflict_window_days: 30

consolidate:
  time_gate_hours: 24
  content_gate_count: 50
```

---

## 13. 部署

### 13.1 本地部署

```bash
pip install agent-cortexos
cortexos serve --config config.yaml    # 启动服务
cortexos pair approve <pair_code>      # 管理端配对确认
```

### 13.2 Docker

```yaml
# docker-compose.yml
services:
  memory:
    image: cortexos:latest
    ports: ["8200:8200"]
    volumes: ["./data:/data"]
    environment:
      - STORAGE_BACKEND=local
```

### 13.3 云端

- 对象存储模式 + 单容器服务（或后续多实例）
- 反向代理 TLS + 限流

---

## 14. 路线图

| 阶段 | 内容 | 交付 |
|------|------|------|
| P0 | 方案确认（本文档） | 评审通过 |
| P1 | 领域模型 + Storage Backend（local） | 可存储/检索 |
| P2 | 配对安全体系 + REST API | 可安全接入 |
| P3 | Zone 引擎（涌现/生命周期/重力） | 自动组织 |
| P4 | 提取/冲突消解/整合引擎 | 记忆质量 |
| P5 | ObjectBackend + 迁移工具 | 云端部署 |
| P6 | Python SDK + 文档 + 示例 Agent | 开箱即用 |
| P7 | MCP Server（可选） | 生态接入 |

---

## 15. 待确认问题（请主人拍板）

**已确认（2026-08-04 主人）**：
- ✅ LLM 提取默认开启（可配置关闭）
- ✅ 对象存储选 S3 兼容协议（OSS/MinIO 通吃）
- ✅ 配对确认方式 CLI 先行
- ✅ 图遍历纳入 v2.0（非预留）
- ✅ 权限模型：Scope 主 + Zone 覆盖两层（见 §2.3/§2.4）

**剩余待确认（小白建议已给出，等主人确认）**：

| # | 问题 | 小白建议 | 理由 |
|---|------|---------|------|
| 1 | Zone 级权限允许 write？ | **允许，但单调递减约束**：zone 覆盖只能在 scope 权限内细化，不能越权（scope 只有 read → zone 最高 read） | 支持"只往某 zone 写"协作场景，同时杜绝越权路径 |
| 2 | 图遍历 hop 深度 | **默认 2 跳 + 每跳衰减 0.5**（得分 × 0.5^(hop-1)），hop 可配置 1-3 | 1 跳太浅，3 跳噪音大；2 跳是关联推理甜点位 |
| 3 | 关系提取默认开启？ | **默认开，分级 + 成本控制**：facts/edges 都开；只对高价值条目提取（长度≥50 或决策/事件类），短日志跳过 | 图遍历需要 edges 数据；选择性提取控制 LLM 成本 |
| 4 | 部署形态 | **独立容器为主 + 嵌入模式为辅**：同一代码库两种用法，接口一致 | 通用性要求 REST 服务；嵌入模式方便本地开发 |

---

*方案 v2.1 结束。待主人确认 4 项建议后冻结，进入代码设计（P1）。*

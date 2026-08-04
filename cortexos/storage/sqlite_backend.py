"""SQLite 存储后端 —— WAL 模式 + FTS5 + 完整 Schema。

技术选型理由：
  选择 sqlite3 + 线程池（asyncio.to_thread / run_in_executor），
  而不是 aiosqlite。理由：
  1. SQLite 本身是线程安全的（单写者+WAL），aiosqlite 只是简单封装
  2. 内置 sqlite3 无需额外 C 扩展，部署简单
  3. 线程池模式对写密集型操作（upsert_entry）与 aiosqlite 性能无差异
  4. Python 3.10+ 内置 asyncio.to_thread 足够用
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from cortexos.models import Edge, Entry, Fact, Zone


class SqliteBackend:
    """SQLite + WAL + FTS5 存储后端。"""

    def __init__(self, db_path: str = "./data/memory.db"):
        """初始化 SQLite 后端。

        Args:
            db_path: 数据库文件路径。自动创建父目录。
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── 生命周期 ──

    async def initialize(self) -> None:
        """初始化数据库：创建目录、打开连接、启用 WAL、建表。"""
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = await asyncio.to_thread(
            sqlite3.connect, self._db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

        # 启用 WAL 模式（并发读 + 单写者，比 DELETE 模式快 10x）
        await self._execute("PRAGMA journal_mode=WAL")
        await self._execute("PRAGMA synchronous=NORMAL")
        await self._execute("PRAGMA foreign_keys=ON")

        await self._create_schema()
        await self._migrate()

    async def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """异步执行 SQL。"""
        return await asyncio.to_thread(self._conn.execute, sql, params)

    async def _execute_many(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """异步批量执行 SQL。"""
        return await asyncio.to_thread(self._conn.executemany, sql, params_list)

    async def _fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """异步查询单行。"""
        cursor = await self._execute(sql, params)
        return await asyncio.to_thread(cursor.fetchone)

    async def _fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """异步查询多行。"""
        cursor = await self._execute(sql, params)
        return await asyncio.to_thread(cursor.fetchall)

    async def _commit(self) -> None:
        """异步提交事务。"""
        await asyncio.to_thread(self._conn.commit)

    # ── Schema ──

    async def _create_schema(self) -> None:
        """创建完整数据库 Schema（含索引）。"""
        await self._execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id          TEXT PRIMARY KEY,
                scope       TEXT NOT NULL,
                zone        TEXT NOT NULL DEFAULT '_inbox',
                layer       TEXT NOT NULL DEFAULT 'raw',
                content     TEXT NOT NULL,
                entities    TEXT NOT NULL DEFAULT '[]',
                embedding   TEXT,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'active',
                valid_until REAL
            )
        """)
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_scope ON entries(scope, status)"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_zone ON entries(zone, status)"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at)"
        )

        # FTS5 词法索引（独立表，手动同步）
        await self._execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                entry_id UNINDEXED, content, entities, zone, scope
            )
        """)

        # Facts 表
        await self._execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id          TEXT PRIMARY KEY,
                entry_id    TEXT NOT NULL,
                scope       TEXT NOT NULL,
                subject     TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                object      TEXT NOT NULL,
                confidence  REAL NOT NULL DEFAULT 1.0,
                valid_from  REAL NOT NULL,
                valid_until REAL,
                status      TEXT NOT NULL DEFAULT 'active'
            )
        """)
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_spo ON facts(subject, predicate, object)"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(scope, status)"
        )

        # Edges 表
        await self._execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id          TEXT PRIMARY KEY,
                entry_id    TEXT NOT NULL,
                scope       TEXT NOT NULL,
                source      TEXT NOT NULL,
                target      TEXT NOT NULL,
                relation    TEXT NOT NULL,
                weight      REAL NOT NULL DEFAULT 1.0,
                valid_until REAL,
                created_at  REAL NOT NULL,
                embedding   TEXT
            )
        """)
        await self._execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        await self._execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        await self._execute("CREATE INDEX IF NOT EXISTS idx_edges_scope ON edges(scope)")

        # Zones 表（复合主键 (scope, name)：zone 归属 scope，跨 scope 允许同名）
        await self._execute("""
            CREATE TABLE IF NOT EXISTS zones (
                name         TEXT NOT NULL,
                scope        TEXT NOT NULL,
                description  TEXT DEFAULT '',
                entities     TEXT NOT NULL DEFAULT '[]',
                keywords     TEXT NOT NULL DEFAULT '[]',
                centroid     TEXT,
                gravity      REAL NOT NULL DEFAULT 1.0,
                entry_count  INTEGER NOT NULL DEFAULT 0,
                status       TEXT NOT NULL DEFAULT 'active',
                pinned       INTEGER NOT NULL DEFAULT 0,
                created_at   REAL NOT NULL,
                last_access  REAL NOT NULL,
                PRIMARY KEY (scope, name)
            )
        """)
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_zones_scope ON zones(scope, status)"
        )

        # Agents 表
        await self._execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id    TEXT PRIMARY KEY,
                agent_name  TEXT NOT NULL,
                created_at  REAL NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active'
            )
        """)

        # Agent Keys 表
        await self._execute("""
            CREATE TABLE IF NOT EXISTS agent_keys (
                key_id      TEXT PRIMARY KEY,
                agent_id    TEXT NOT NULL,
                key_hash    TEXT NOT NULL,
                scope_permissions TEXT NOT NULL DEFAULT '{}',
                zone_overrides    TEXT NOT NULL DEFAULT '{}',
                rate_limit  INTEGER NOT NULL DEFAULT 100,
                expires_at  REAL,
                created_at  REAL NOT NULL,
                last_used   REAL,
                status      TEXT NOT NULL DEFAULT 'active'
            )
        """)
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_keys_agent ON agent_keys(agent_id)"
        )

        # Pair Codes 表（scope_permissions：管理员批准时授予的权限，exchange 时写入 key）
        await self._execute("""
            CREATE TABLE IF NOT EXISTS pair_codes (
                code        TEXT PRIMARY KEY,
                agent_id    TEXT NOT NULL,
                agent_name  TEXT NOT NULL,
                expires_at  REAL NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                scope_permissions TEXT NOT NULL DEFAULT '{}'
            )
        """)

        # 审计日志表
        await self._execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                agent_id    TEXT,
                key_id      TEXT,
                action      TEXT NOT NULL,
                scope       TEXT,
                detail      TEXT DEFAULT ''
            )
        """)
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)"
        )

        await self._commit()

    async def _migrate(self) -> None:
        """旧库结构迁移（幂等）。

        处理两种情况：
        1. pair_codes 缺 scope_permissions 列（v2.0 早期 schema）→ ALTER 补列
        2. zones 表是 name 单主键（v2.0 早期 schema）→ 重建为 (scope, name) 复合主键
        """
        # 1. pair_codes.scope_permissions
        cols = await self._fetchall("PRAGMA table_info(pair_codes)")
        col_names = [c["name"] for c in cols]
        if "scope_permissions" not in col_names:
            await self._execute(
                "ALTER TABLE pair_codes ADD COLUMN scope_permissions TEXT NOT NULL DEFAULT '{}'"
            )

        # 2. zones 复合主键检测：name 列 pk=1 且 scope 列 pk=0 → 旧结构
        zcols = await self._fetchall("PRAGMA table_info(zones)")
        name_pk = next((c for c in zcols if c["name"] == "name"), None)
        scope_pk = next((c for c in zcols if c["name"] == "scope"), None)
        is_old_pk = bool(name_pk and name_pk["pk"] == 1 and (scope_pk is None or scope_pk["pk"] == 0))
        if is_old_pk:
            # 重建：改名旧表 → 建新表 → 拷贝数据（同 scope+name 保留 last_access 最新）→ 删旧表
            await self._execute("ALTER TABLE zones RENAME TO zones_old")
            await self._execute("""
                CREATE TABLE zones (
                    name         TEXT NOT NULL,
                    scope        TEXT NOT NULL,
                    description  TEXT DEFAULT '',
                    entities     TEXT NOT NULL DEFAULT '[]',
                    keywords     TEXT NOT NULL DEFAULT '[]',
                    centroid     TEXT,
                    gravity      REAL NOT NULL DEFAULT 1.0,
                    entry_count  INTEGER NOT NULL DEFAULT 0,
                    status       TEXT NOT NULL DEFAULT 'active',
                    pinned       INTEGER NOT NULL DEFAULT 0,
                    created_at   REAL NOT NULL,
                    last_access  REAL NOT NULL,
                    PRIMARY KEY (scope, name)
                )
            """)
            await self._execute("""
                INSERT INTO zones (name, scope, description, entities, keywords,
                                   centroid, gravity, entry_count, status, pinned,
                                   created_at, last_access)
                SELECT z.name, z.scope, z.description, z.entities, z.keywords,
                       z.centroid, z.gravity, z.entry_count, z.status, z.pinned,
                       z.created_at, z.last_access
                FROM zones_old z
                JOIN (
                    SELECT scope, name, MAX(last_access) AS m
                    FROM zones_old GROUP BY scope, name
                ) k ON z.scope = k.scope AND z.name = k.name AND z.last_access = k.m
            """)
            await self._execute("DROP TABLE zones_old")

        await self._commit()

    # ── Entry CRUD ──

    async def upsert_entry(self, entry: Entry) -> None:
        """写入或更新条目（UPSERT）。"""
        data = entry.to_dict()
        await self._execute("""
            INSERT INTO entries (id, scope, zone, layer, content, entities, embedding,
                                 metadata, created_at, updated_at, access_count, status, valid_until)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                scope=excluded.scope, zone=excluded.zone, layer=excluded.layer,
                content=excluded.content, entities=excluded.entities,
                embedding=excluded.embedding, metadata=excluded.metadata,
                updated_at=excluded.updated_at, access_count=excluded.access_count,
                status=excluded.status, valid_until=excluded.valid_until
        """, (
            data["id"], data["scope"], data["zone"], data["layer"], data["content"],
            data["entities"], data["embedding"], data["metadata"],
            data["created_at"], data["updated_at"], data["access_count"],
            data["status"], data["valid_until"]
        ))
        # 同步 FTS5 索引：先删后插
        await self._execute(
            "DELETE FROM entries_fts WHERE entry_id = ?",
            (entry.id,)
        )
        await self._execute(
            "INSERT INTO entries_fts(entry_id, content, entities, zone, scope) VALUES(?, ?, ?, ?, ?)",
            (entry.id, entry.content, json.dumps(entry.entities, ensure_ascii=False), entry.zone, entry.scope)
        )
        await self._commit()

    async def get_entry(self, entry_id: str, active_only: bool = False) -> Optional[Entry]:
        """读取单条条目。

        Args:
            entry_id: 条目 ID。
            active_only: 如果 True，只返回 status='active' 的条目。
        """
        row = await self._fetchone("SELECT * FROM entries WHERE id = ?", (entry_id,))
        if row is None:
            return None
        entry = Entry.from_row(dict(row))
        if active_only and entry.status != "active":
            return None
        return entry

    async def delete_entry(self, entry_id: str) -> bool:
        """软删除条目（status → archived），返回是否删除成功。"""
        entry = await self.get_entry(entry_id)
        if not entry:
            return False
        now = time.time()
        await self._execute(
            "UPDATE entries SET status='archived', updated_at=? WHERE id=?",
            (now, entry_id)
        )
        await self._commit()
        return True

    async def list_entries(
        self,
        *,
        scope: Optional[str] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entry]:
        """列出条目。"""
        conditions = []
        params: List[Any] = []
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if zone:
            conditions.append("zone = ?")
            params.append(zone)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if layer:
            conditions.append("layer = ?")
            params.append(layer)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM entries{where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = await self._fetchall(sql, tuple(params))
        return [Entry.from_row(dict(r)) for r in rows]

    async def count_entries(
        self,
        *,
        scope: Optional[str] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> int:
        """统计条目数。"""
        conditions = []
        params: List[Any] = []
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if zone:
            conditions.append("zone = ?")
            params.append(zone)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if layer:
            conditions.append("layer = ?")
            params.append(layer)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        row = await self._fetchone(f"SELECT COUNT(*) as cnt FROM entries{where}", tuple(params))
        return row["cnt"] if row else 0

    # ── 检索 ──

    async def search_lexical(
        self, query: str, *, scope: Optional[str] = None, top_k: int = 20
    ) -> List[tuple[Entry, float]]:
        """FTS5 词法检索 —— 返回 (Entry, bm25_score)。"""
        # FTS5 搜索，带 scope 过滤
        where_clause = ""
        params_scope: List[Any] = []
        if scope:
            where_clause = " AND e.scope = ?"
            params_scope.append(scope)

        # 使用 FTS5 rank（bm25 近似），只返回 active 条目，e.scope 明确限定
        # 注意：MATCH 必须引用 FTS 表真名（别名对 MATCH 无效）
        sql = f"""
            SELECT e.*, f.rank AS rank
            FROM entries_fts f
            JOIN entries e ON e.id = f.entry_id
            WHERE entries_fts MATCH ? AND e.status = 'active'{where_clause}
            ORDER BY rank
            LIMIT ?
        """
        params = [query] + params_scope + [top_k]
        try:
            rows = await self._fetchall(sql, tuple(params))
        except Exception:  # noqa: BLE001 - FTS5 语法错误（特殊字符）降级 LIKE
            # FTS5 对特殊字符（" * - AND 等）会抛 malformed MATCH expression，
            # 降级为子串 LIKE 搜索保证查询不 500
            like = f"%{query}%"
            sql2 = f"""
                SELECT e.* FROM entries e
                WHERE e.content LIKE ? AND e.status = 'active'{where_clause}
                ORDER BY e.created_at DESC LIMIT ?
            """
            rows = await self._fetchall(sql2, tuple([like] + params_scope + [top_k]))
        results: List[tuple[Entry, float]] = []
        for row in rows:
            d = dict(row)
            rank = d.pop("rank", 0.0)
            # 归一化 bm25 分数为 0~1（近似）
            score = 1.0 / (1.0 + abs(rank)) if rank else 0.5
            results.append((Entry.from_row(d), score))
        return results

    async def search_all_embeddings(
        self, *, scope: Optional[str] = None
    ) -> List[tuple[str, Optional[List[float]]]]:
        """批量获取所有条目的 embedding（用于内存向量索引构建）。"""
        if scope:
            rows = await self._fetchall(
                "SELECT id, embedding FROM entries WHERE embedding IS NOT NULL AND scope = ? AND status = 'active'",
                (scope,)
            )
        else:
            rows = await self._fetchall(
                "SELECT id, embedding FROM entries WHERE embedding IS NOT NULL AND status = 'active'"
            )
        results: List[tuple[str, Optional[List[float]]]] = []
        for row in rows:
            emb = row["embedding"]
            if isinstance(emb, str) and emb:
                emb = json.loads(emb)
            results.append((row["id"], emb))
        return results

    # ── Facts ──

    async def upsert_fact(self, fact: Fact) -> None:
        """写入/更新事实。"""
        data = fact.to_dict()
        await self._execute("""
            INSERT INTO facts (id, entry_id, scope, subject, predicate, object,
                               confidence, valid_from, valid_until, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                confidence=excluded.confidence, valid_until=excluded.valid_until,
                status=excluded.status
        """, (
            data["id"], data["entry_id"], data["scope"],
            data["subject"], data["predicate"], data["object"],
            data["confidence"], data["valid_from"], data["valid_until"], data["status"]
        ))
        await self._commit()

    async def find_facts(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        scope: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Fact]:
        """查询事实。"""
        conditions = ["1=1"]
        params: List[Any] = []
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = " AND ".join(conditions)
        rows = await self._fetchall(f"SELECT * FROM facts WHERE {where}", tuple(params))
        return [Fact.from_row(dict(r)) for r in rows]

    # ── Edges ──

    async def upsert_edge(self, edge: Edge) -> None:
        """写入/更新关系边。"""
        data = edge.to_dict()
        await self._execute("""
            INSERT INTO edges (id, entry_id, scope, source, target, relation,
                               weight, valid_until, created_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                weight=excluded.weight, valid_until=excluded.valid_until
        """, (
            data["id"], data["entry_id"], data["scope"],
            data["source"], data["target"], data["relation"],
            data["weight"], data["valid_until"], data["created_at"], data["embedding"]
        ))
        await self._commit()

    async def find_edges(
        self,
        *,
        source: Optional[str] = None,
        target: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> List[Edge]:
        """查询关系边。"""
        conditions = ["1=1"]
        params: List[Any] = []
        if source:
            conditions.append("source = ?")
            params.append(source)
        if target:
            conditions.append("target = ?")
            params.append(target)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        where = " AND ".join(conditions)
        rows = await self._fetchall(f"SELECT * FROM edges WHERE {where}", tuple(params))
        return [Edge.from_row(dict(r)) for r in rows]

    async def list_edges_from(self, entity: str) -> List[Edge]:
        """查询从指定实体出发的边。"""
        rows = await self._fetchall(
            "SELECT * FROM edges WHERE source = ?", (entity,)
        )
        return [Edge.from_row(dict(r)) for r in rows]

    async def list_edges_to(self, entity: str) -> List[Edge]:
        """查询指向指定实体的边。"""
        rows = await self._fetchall(
            "SELECT * FROM edges WHERE target = ?", (entity,)
        )
        return [Edge.from_row(dict(r)) for r in rows]

    # ── Zones ──

    async def upsert_zone(self, zone: Zone) -> None:
        """写入/更新 Zone（复合主键 scope+name）。"""
        data = zone.to_dict()
        await self._execute("""
            INSERT INTO zones (name, scope, description, entities, keywords,
                               centroid, gravity, entry_count, status, pinned,
                               created_at, last_access)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, name) DO UPDATE SET
                description=excluded.description,
                entities=excluded.entities, keywords=excluded.keywords,
                centroid=excluded.centroid, gravity=excluded.gravity,
                entry_count=excluded.entry_count, status=excluded.status,
                pinned=excluded.pinned, last_access=excluded.last_access
        """, (
            data["name"], data["scope"], data["description"],
            data["entities"], data["keywords"],
            data["centroid"], data["gravity"], data["entry_count"],
            data["status"], data["pinned"],
            data["created_at"], data["last_access"]
        ))
        await self._commit()

    async def get_zone(self, name: str, scope: Optional[str] = None) -> Optional[Zone]:
        """读取单个 Zone。

        Args:
            name: Zone 名称。
            scope: 归属 scope（复合主键；None 时按 name 模糊取一条，兼容旧调用）。
        """
        if scope is not None:
            row = await self._fetchone(
                "SELECT * FROM zones WHERE name = ? AND scope = ?", (name, scope)
            )
        else:
            row = await self._fetchone(
                "SELECT * FROM zones WHERE name = ? LIMIT 1", (name,)
            )
        if row is None:
            return None
        return Zone.from_row(dict(row))

    async def list_zones(
        self,
        *,
        scope: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Zone]:
        """列出 Zone。"""
        conditions = []
        params: List[Any] = []
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = await self._fetchall(
            f"SELECT * FROM zones{where} ORDER BY created_at DESC", tuple(params)
        )
        return [Zone.from_row(dict(r)) for r in rows]

    # ── 批量操作 ──

    async def scan_entries(
        self,
        *,
        since: Optional[float] = None,
        scope: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> AsyncIterator[Entry]:
        """流式扫描条目（整合用）。

        注意：SQLite 不支持异步流，这里用分页模拟，
        每批 100 条，避免内存爆炸。
        """
        conditions = ["status = 'active'"]
        params: List[Any] = []
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if layer:
            conditions.append("layer = ?")
            params.append(layer)
        where = " AND ".join(conditions)
        offset = 0
        while True:
            rows = await self._fetchall(
                f"SELECT * FROM entries WHERE {where} ORDER BY created_at LIMIT 100 OFFSET ?",
                tuple(params + [offset])
            )
            if not rows:
                break
            for row in rows:
                yield Entry.from_row(dict(row))
            offset += 100

    async def scan_zone(
        self,
        *,
        scope: str = None,
        zone: str = None,
        limit: int = 50,
    ) -> List[Entry]:
        """扫描 zone 内条目。

        Args:
            scope: scope。
            zone: zone 名称（None 表示所有 zone）。
            limit: 返回条数上限。

        Returns:
            条目列表。
        """
        conditions = ["status = 'active'"]
        params: List[Any] = []
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if zone:
            conditions.append("zone = ?")
            params.append(zone)
        where = " AND ".join(conditions)
        rows = await self._fetchall(
            f"SELECT * FROM entries WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        )
        return [Entry.from_row(dict(r)) for r in rows]

    async def bulk_update_status(self, updates: List[tuple[str, str]]) -> None:
        """批量更新条目状态 [(entry_id, new_status), ...]"""
        now = time.time()
        await self._execute_many(
            "UPDATE entries SET status=?, updated_at=? WHERE id=?",
            [(status, now, eid) for eid, status in updates]
        )
        await self._commit()

    # ── Agent / 密钥 / 配对 ──

    async def upsert_agent(self, agent: Dict[str, Any]) -> None:
        """写入 Agent。"""
        await self._execute("""
            INSERT INTO agents (agent_id, agent_name, created_at, status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                agent_name=excluded.agent_name, status=excluded.status
        """, (agent["agent_id"], agent["agent_name"], agent.get("created_at", time.time()),
              agent.get("status", "active")))
        await self._commit()

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """读取 Agent。"""
        row = await self._fetchone("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        return dict(row) if row else None

    async def upsert_agent_key(self, key_data: Dict[str, Any]) -> None:
        """写入密钥。

        兼容两种输入：scope_permissions/zone_overrides 为 dict 或 JSON 字符串
        （从 DB 读回再写回时是字符串，避免双重编码）。
        """
        sp = key_data.get("scope_permissions", {})
        if isinstance(sp, str):
            sp = json.loads(sp)
        zo = key_data.get("zone_overrides", {})
        if isinstance(zo, str):
            zo = json.loads(zo)
        await self._execute("""
            INSERT INTO agent_keys (key_id, agent_id, key_hash, scope_permissions,
                                    zone_overrides, rate_limit, expires_at, created_at,
                                    last_used, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key_id) DO UPDATE SET
                status=excluded.status, last_used=excluded.last_used
        """, (
            key_data["key_id"], key_data["agent_id"], key_data["key_hash"],
            json.dumps(sp, ensure_ascii=False),
            json.dumps(zo, ensure_ascii=False),
            key_data.get("rate_limit", 100), key_data.get("expires_at"),
            key_data.get("created_at", time.time()), key_data.get("last_used"),
            key_data.get("status", "active"),
        ))
        await self._commit()

    async def get_agent_key_by_id(self, key_id: str) -> Optional[Dict[str, Any]]:
        """按 key_id 查询密钥。"""
        row = await self._fetchone(
            "SELECT * FROM agent_keys WHERE key_id = ?", (key_id,)
        )
        return dict(row) if row else None

    async def get_agent_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """按 key_hash 查询密钥。"""
        row = await self._fetchone(
            "SELECT * FROM agent_keys WHERE key_hash = ?", (key_hash,)
        )
        return dict(row) if row else None

    async def list_agent_keys(self, agent_id: str) -> List[Dict[str, Any]]:
        """列出 Agent 的所有密钥。"""
        rows = await self._fetchall(
            "SELECT * FROM agent_keys WHERE agent_id = ?", (agent_id,)
        )
        return [dict(r) for r in rows]

    async def upsert_pair_code(self, code_data: Dict[str, Any]) -> None:
        """写入配对码（含批准时授予的 scope 权限）。"""
        sp = code_data.get("scope_permissions", {})
        if isinstance(sp, str):
            sp = json.loads(sp)
        await self._execute("""
            INSERT INTO pair_codes (code, agent_id, agent_name, expires_at, status, scope_permissions)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                status=excluded.status, expires_at=excluded.expires_at,
                scope_permissions=excluded.scope_permissions
        """, (
            code_data["code"], code_data["agent_id"], code_data["agent_name"],
            code_data["expires_at"], code_data.get("status", "pending"),
            json.dumps(sp, ensure_ascii=False),
        ))
        await self._commit()

    async def get_pair_code(self, code: str) -> Optional[Dict[str, Any]]:
        """读取配对码。"""
        row = await self._fetchone(
            "SELECT * FROM pair_codes WHERE code = ?", (code,)
        )
        return dict(row) if row else None

    async def write_audit_log(self, log_data: Dict[str, Any]) -> None:
        """写入审计日志。"""
        await self._execute("""
            INSERT INTO audit_log (ts, agent_id, key_id, action, scope, detail)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            log_data.get("ts", time.time()),
            log_data.get("agent_id"),
            log_data.get("key_id"),
            log_data["action"],
            log_data.get("scope"),
            log_data.get("detail", ""),
        ))
        await self._commit()

    # ── 统计 ──

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        total = await self._fetchone("SELECT COUNT(*) as cnt FROM entries")
        active = await self._fetchone(
            "SELECT COUNT(*) as cnt FROM entries WHERE status='active'"
        )
        zones = await self._fetchone("SELECT COUNT(*) as cnt FROM zones")
        facts = await self._fetchone("SELECT COUNT(*) as cnt FROM facts")
        edges = await self._fetchone("SELECT COUNT(*) as cnt FROM edges")
        try:
            db_size = os.path.getsize(self._db_path)
        except OSError:
            db_size = 0
        return {
            "total_entries": total["cnt"] if total else 0,
            "active_entries": active["cnt"] if active else 0,
            "total_zones": zones["cnt"] if zones else 0,
            "total_facts": facts["cnt"] if facts else 0,
            "total_edges": edges["cnt"] if edges else 0,
            "db_size_bytes": db_size,
        }

    # ── Scopes ──

    async def list_scopes(self) -> List[str]:
        """列出所有出现过数据的 scope（entries + zones 并集）。"""
        rows = await self._fetchall(
            "SELECT DISTINCT scope FROM entries UNION SELECT DISTINCT scope FROM zones"
        )
        return [r["scope"] for r in rows]

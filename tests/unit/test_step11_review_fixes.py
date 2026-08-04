"""Step 11 回归测试 —— 代码审核发现的问题修复验证。

覆盖：
1. 配对流程 scope_permissions 丢失（关键 bug）
2. 密钥 scope_permissions 双重编码（authenticate 更新 last_used 时）
3. zones 跨 scope 同名冲突（复合主键）
4. zones 旧 schema 迁移
5. get_stats 字段统一 + db_size_bytes
6. list_scopes
7. search_lexical 排除 archived
8. store 写入管线（embedding 生成 + 语义路由 + 质心更新）
"""

import json
import os
import tempfile
import time

import pytest

from cortexos.config import Config
from cortexos.models import Entry, Zone


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


@pytest.mark.asyncio
class TestPairingPermissions:
    """配对流程权限回归。"""

    async def _backend(self):
        from cortexos.storage.sqlite_backend import SqliteBackend
        path = _temp_db()
        db = SqliteBackend(path)
        await db.initialize()
        return db, path

    async def test_exchanged_key_carries_scopes(self):
        """关键回归：approve 授予的 scope 权限必须到达 exchange 颁发的 key。"""
        from cortexos.auth.pairing import pair_request, pair_approve, pair_exchange
        db, path = await self._backend()
        try:
            cfg = Config()
            req = await pair_request("my-agent", db, cfg)
            approved = await pair_approve(
                req["code"], {"agent:my-agent": "readwrite"}, db, cfg,
            )
            assert approved is not None
            exchanged = await pair_exchange(req["code"], db, cfg)
            assert exchanged is not None

            keys = await db.list_agent_keys(exchanged["agent_id"])
            assert len(keys) == 1
            sp = json.loads(keys[0]["scope_permissions"])
            assert sp == {"agent:my-agent": "readwrite"}
        finally:
            await db.close()
            os.unlink(path)

    async def test_authenticate_no_double_encode(self):
        """回归：authenticate 更新 last_used 后 scope_permissions 不被双重编码。"""
        from cortexos.auth.pairing import pair_request, pair_approve, pair_exchange
        from cortexos.auth.keys import authenticate
        db, path = await self._backend()
        try:
            cfg = Config()
            req = await pair_request("my-agent", db, cfg)
            await pair_approve(req["code"], {"agent:my-agent": "read"}, db, cfg)
            exchanged = await pair_exchange(req["code"], db, cfg)

            # 认证一次（触发 last_used 更新回写）
            key_data = await authenticate(exchanged["secret"], db)
            assert key_data is not None
            assert key_data["scope_permissions"] == {"agent:my-agent": "read"}

            # 再认证一次，确认没有累积损坏
            key_data2 = await authenticate(exchanged["secret"], db)
            assert key_data2["scope_permissions"] == {"agent:my-agent": "read"}
        finally:
            await db.close()
            os.unlink(path)


@pytest.mark.asyncio
class TestZoneScopeIsolation:
    """zones 复合主键回归。"""

    async def test_same_zone_name_different_scopes(self):
        """回归：两个 scope 可以有同名 zone 互不覆盖。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()

            z1 = Zone(name="k8s", scope="agent:a", description="A 的 k8s")
            z2 = Zone(name="k8s", scope="agent:b", description="B 的 k8s")
            await db.upsert_zone(z1)
            await db.upsert_zone(z2)

            got1 = await db.get_zone("k8s", scope="agent:a")
            got2 = await db.get_zone("k8s", scope="agent:b")
            assert got1 is not None and got1.description == "A 的 k8s"
            assert got2 is not None and got2.description == "B 的 k8s"
            assert got1.id != got2.id

            # 更新其中一个不影响另一个
            z1.description = "A 更新"
            await db.upsert_zone(z1)
            got2 = await db.get_zone("k8s", scope="agent:b")
            assert got2.description == "B 的 k8s"

            await db.close()
        finally:
            os.unlink(path)

    async def test_migration_from_old_schema(self):
        """回归：旧 schema（name 单主键）初始化时自动迁移为复合主键。"""
        import sqlite3
        path = _temp_db()
        try:
            # 手工创建旧 schema zones 表（name 单主键）+ 数据
            conn = sqlite3.connect(path)
            conn.execute("""
                CREATE TABLE zones (
                    name TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    entities TEXT NOT NULL DEFAULT '[]',
                    keywords TEXT NOT NULL DEFAULT '[]',
                    centroid TEXT,
                    gravity REAL NOT NULL DEFAULT 1.0,
                    entry_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL
                )
            """)
            now = time.time()
            conn.execute(
                "INSERT INTO zones VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("k8s", "agent:a", "old", "[]", "[]", None, 1.0, 3, "active", 0, now, now),
            )
            conn.execute(
                "INSERT INTO zones VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("mysql", "agent:b", "old-b", "[]", "[]", None, 1.0, 5, "active", 0, now, now),
            )
            conn.commit()
            conn.close()

            from cortexos.storage.sqlite_backend import SqliteBackend
            db = SqliteBackend(path)
            await db.initialize()

            # 迁移后两个 scope 的 zone 都保留
            zones = await db.list_zones()
            assert len(zones) == 2
            got = await db.get_zone("mysql", scope="agent:b")
            assert got is not None and got.entry_count == 5

            # 新表可写复合主键
            z = Zone(name="mysql", scope="agent:c")
            await db.upsert_zone(z)
            assert await db.get_zone("mysql", scope="agent:c") is not None

            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestStatsAndScopes:
    """stats / scopes 端点回归。"""

    async def test_get_stats_unified_fields(self):
        """回归：get_stats 返回统一字段 + db_size_bytes。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            await db.upsert_entry(Entry(content="hello", scope="agent:x"))

            stats = await db.get_stats()
            assert stats["total_entries"] == 1
            assert stats["active_entries"] == 1
            assert "total_zones" in stats
            assert "total_facts" in stats
            assert "total_edges" in stats
            assert stats["db_size_bytes"] > 0
            await db.close()
        finally:
            os.unlink(path)

    async def test_list_scopes(self):
        """回归：list_scopes 返回有数据的 scope。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            await db.upsert_entry(Entry(content="a", scope="agent:x"))
            await db.upsert_entry(Entry(content="b", scope="agent:y"))
            scopes = await db.list_scopes()
            assert set(scopes) == {"agent:x", "agent:y"}
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestLexicalActiveOnly:
    """词法检索只返回 active 条目。"""

    async def test_search_excludes_archived(self):
        """回归：软删除（archived）条目不出现在词法检索结果。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()

            e = Entry(content="MySQL 主从延迟告警处理", scope="agent:test")
            await db.upsert_entry(e)
            await db.delete_entry(e.id)  # → archived

            results = await db.search_lexical("MySQL", scope="agent:test")
            assert all(r[0].id != e.id for r in results)
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestStorePipeline:
    """store 写入管线回归。"""

    async def _setup(self):
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.embedding.tfidf import TfidfEmbedder
        path = _temp_db()
        db = SqliteBackend(path)
        await db.initialize()
        embedder = TfidfEmbedder()
        return db, embedder, path

    async def test_pipeline_generates_embedding_and_entities(self):
        """回归：管线生成 embedding 并提取实体。"""
        from cortexos.store import store_entry
        db, embedder, path = await self._setup()
        try:
            entry = Entry(content="Kubernetes pod 重启排查", scope="agent:p")
            await store_entry(entry, db, embedder, Config())

            assert entry.embedding is not None
            assert len(entry.embedding) > 0
            assert "Kubernetes" in entry.entities or "kubernetes" in entry.entities
            assert entry.zone == "_inbox"  # 无匹配 zone → 兜底
            await db.close()
        finally:
            os.unlink(path)

    async def test_pipeline_routes_to_existing_zone(self):
        """回归：有匹配 zone 时 Layer 1 实体路由命中，entry_count 增长。"""
        from cortexos.store import store_entry
        db, embedder, path = await self._setup()
        try:
            # 先手动创建带实体的 Zone（zone 涌现是 consolidate 阶段的事）
            zone = Zone(
                name="k8s-ops", scope="agent:p",
                entities=["Kubernetes", "pod"],
            )
            await db.upsert_zone(zone)

            # 写入含实体的条目 → 应命中 Layer 1 实体路由
            entry = Entry(content="Kubernetes pod 重启排查流程", scope="agent:p")
            await store_entry(entry, db, embedder, Config())

            assert entry.zone == "k8s-ops"
            got = await db.get_zone("k8s-ops", scope="agent:p")
            assert got is not None
            assert got.entry_count >= 1
            await db.close()
        finally:
            os.unlink(path)

    async def test_pipeline_explicit_zone_respected(self):
        """回归：显式指定 zone 时不重新路由。"""
        from cortexos.store import store_entry
        db, embedder, path = await self._setup()
        try:
            entry = Entry(content="随便一条", scope="agent:p", zone="custom-zone")
            await store_entry(entry, db, embedder, Config())
            assert entry.zone == "custom-zone"
            got = await db.get_entry(entry.id)
            assert got.zone == "custom-zone"
            await db.close()
        finally:
            os.unlink(path)

    async def test_pipeline_no_llm_warning_with_tfidf(self, caplog):
        """回归：TF-IDF embedder 下不产生 LLM 提取失败警告。"""
        from cortexos.store import store_entry
        import logging
        db, embedder, path = await self._setup()
        try:
            with caplog.at_level(logging.WARNING, logger="cortexos.store"):
                entry = Entry(content="Kubernetes pod 重启排查", scope="agent:p")
                await store_entry(entry, db, embedder, Config())
            # 不应有 LLM 提取失败的警告（TF-IDF 直接跳过，不报 warning）
            assert "LLM 提取失败" not in caplog.text
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestKeysJsonSerialization:
    """密钥 JSON 字段序列化回归。"""

    async def test_list_keys_parses_json_fields(self):
        """回归：list_keys 返回已解析的 scope_permissions/zone_overrides。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.auth.keys import list_keys as do_list
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            # 写入密钥（scope_permissions 为 dict）
            await db.upsert_agent_key({
                "key_id": "ak_test",
                "agent_id": "ag_test",
                "key_hash": "abc123",
                "scope_permissions": {"agent:x": "readwrite"},
                "zone_overrides": {"agent:x:secret": "read"},
                "rate_limit": 100,
                "created_at": 1.0,
                "status": "active",
            })
            keys = await do_list("ag_test", db)
            assert len(keys) == 1
            assert keys[0]["scope_permissions"] == {"agent:x": "readwrite"}
            assert isinstance(keys[0]["scope_permissions"], dict)
            assert keys[0]["zone_overrides"] == {"agent:x:secret": "read"}
            assert isinstance(keys[0]["zone_overrides"], dict)
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestStoreFactResolution:
    """store 管线事实冲突消解回归（C1）。"""

    async def _setup(self):
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.embedding.tfidf import TfidfEmbedder
        path = _temp_db()
        db = SqliteBackend(path)
        await db.initialize()
        return db, TfidfEmbedder(), path

    async def test_store_resolves_fact_conflict(self, monkeypatch):
        """回归：store 落 facts 前做冲突消解，旧矛盾 fact 被时间窗口截断。"""
        from cortexos.store import store_entry, extract_entities
        from cortexos.models import Entry, Fact
        db, embedder, path = await self._setup()
        try:
            # 第一条：事实 (server-A, version, 1.0)
            await db.upsert_fact(Fact(
                subject="server-A", predicate="version", object="1.0",
                scope="agent:p", confidence=0.9,
                valid_from=time.time() - 100, valid_until=None,
            ))

            # monkeypatch：第二条提取出矛盾事实 (server-A, version, 2.0)
            async def fake_extract(content, embedder, config):
                return {
                    "entities": ["server-A"],
                    "facts": [{
                        "subject": "server-A", "predicate": "version",
                        "object": "2.0", "confidence": 0.95,
                    }],
                    "edges": [], "valid_until": None,
                }
            monkeypatch.setattr("cortexos.store.extract_entities", fake_extract)

            entry = Entry(content="server-A 升级到 2.0", scope="agent:p")
            await store_entry(entry, db, embedder, Config())

            # 旧 fact 应被截断为 superseded，新 fact active
            facts = await db.find_facts(scope="agent:p", status="active")
            assert len(facts) == 1
            assert facts[0].object == "2.0"
            olds = await db.find_facts(scope="agent:p", subject="server-A", predicate="version")
            superseded = [f for f in olds if f.status == "superseded"]
            assert len(superseded) == 1
            assert superseded[0].valid_until is not None
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestConsolidateWritesBackOldFacts:
    """consolidate 事实消解写回回归（C2）。"""

    async def test_resolve_scope_facts_writes_back_olds(self):
        """回归：_resolve_scope_facts 中被截断的旧 fact 必须写回 DB。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.lifecycle.consolidate import ConsolidateEngine
        from cortexos.models import Fact
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            now = time.time()
            # 两条矛盾事实（同 subject+predicate）
            await db.upsert_fact(Fact(
                id="f_old", subject="svc", predicate="status", object="down",
                scope="agent:c", confidence=0.8,
                valid_from=now - 100, valid_until=None, status="active",
            ))
            await db.upsert_fact(Fact(
                id="f_new", subject="svc", predicate="status", object="up",
                scope="agent:c", confidence=0.9,
                valid_from=now, valid_until=None, status="active",
            ))

            engine = ConsolidateEngine(db, Config())
            await engine._resolve_scope_facts("agent:c")

            # 旧 fact 必须已在 DB 中变为 superseded（valid_until 截断落库）
            olds = await db.find_facts(scope="agent:c", subject="svc", predicate="status")
            old = next((f for f in olds if f.id == "f_old"), None)
            assert old is not None
            assert old.status == "superseded"
            assert old.valid_until is not None and old.valid_until <= now + 1
            await db.close()
        finally:
            os.unlink(path)

    async def test_content_gate_blocks_episode(self):
        """回归：raw 条目不足 content_gate_count 时不创建 episode（内容门）。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.lifecycle.consolidate import ConsolidateEngine
        from cortexos.models import Entry
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            # 只写 3 条 raw（< content_gate_count=50）
            for i in range(3):
                await db.upsert_entry(Entry(
                    content=f"Kubernetes 问题 {i}", scope="agent:g",
                    zone="_inbox", layer="raw",
                ))
            cfg = Config()
            engine = ConsolidateEngine(db, cfg)
            # 直接绕过时间门：手动设 last_consolidate 为很久以前
            engine._last_consolidate["agent:g"] = time.time() - 999999
            stats = await engine.consolidate("agent:g")
            assert stats["episodes_created"] == 0
            # 3 条 raw 都还在（未被 supersede）
            entries = [e async for e in db.scan_entries(scope="agent:g")]
            assert len(entries) == 3
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestPairingApproveRobustness:
    """pair_approve 健壮性回归（C4/C5）。"""

    async def test_approve_does_not_corrupt_approved_code(self):
        """回归：已 approved 的码再次 approve 不改变状态。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.auth.pairing import pair_request, pair_approve
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            cfg = Config()
            req = await pair_request("a1", db, cfg)
            await pair_approve(req["code"], {"agent:x": "read"}, db, cfg)

            # 第二次 approve（码已 approved）→ 返回 None 且状态不变
            again = await pair_approve(req["code"], {"agent:x": "readwrite"}, db, cfg)
            assert again is None
            pair = await db.get_pair_code(req["code"])
            assert pair["status"] == "approved"
            sp = pair["scope_permissions"]
            if isinstance(sp, str):
                sp = json.loads(sp)
            assert sp == {"agent:x": "read"}
            await db.close()
        finally:
            os.unlink(path)

    async def test_approve_rejects_invalid_permission_value(self):
        """回归：非法权限值（非 read/write/readwrite）被拒绝。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.auth.pairing import pair_request, pair_approve
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            cfg = Config()
            req = await pair_request("a2", db, cfg)
            result = await pair_approve(req["code"], {"agent:x": "admin"}, db, cfg)
            assert result is None
            pair = await db.get_pair_code(req["code"])
            assert pair["status"] == "pending"  # 状态未被破坏
            await db.close()
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestLexicalFallback:
    """FTS 特殊字符降级回归。"""

    async def test_search_lexical_special_chars_fallback(self):
        """回归：FTS5 特殊字符查询降级 LIKE，不抛异常。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.models import Entry
        path = _temp_db()
        try:
            db = SqliteBackend(path)
            await db.initialize()
            await db.upsert_entry(Entry(content="MySQL 主从延迟", scope="agent:s"))

            # 各种 FTS5 会报错的特殊字符
            for q in ['"', '*', 'AND OR NOT', 'a"b', '-', '(', ')']:
                results = await db.search_lexical(q, scope="agent:s")
                assert isinstance(results, list)
            await db.close()
        finally:
            os.unlink(path)

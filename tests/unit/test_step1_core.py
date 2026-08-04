"""Step 1 单元测试：config + models + sqlite_backend。"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from cortexos.config import (
    Config,
    load_config,
)
from cortexos.models import AgentKey, Edge, Entry, Fact, Zone


# ────────────────────── Config 测试 ──────────────────────


class TestConfig:
    """配置模块测试。"""

    def test_default_config(self):
        """默认配置应与设计文档一致。"""
        cfg = Config()
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.port == 8200
        assert cfg.storage.backend == "local"
        assert cfg.storage.local.path == "./data/memory.db"
        assert cfg.zone.emergence.semantic_threshold == 0.72
        assert cfg.zone.emergence.base_threshold == 5
        assert cfg.zone.emergence.cluster_similarity == 0.75
        assert cfg.zone.emergence.min_content_len == 20
        assert cfg.zone.lifecycle.dormant_days == 30
        assert cfg.zone.lifecycle.archive_days == 90
        assert cfg.zone.lifecycle.merge_threshold == 0.7
        assert cfg.zone.gravity.decay_lambda == 0.02
        assert cfg.zone.gravity.activity_k == 50.0
        assert cfg.zone.gravity.scale_m == 100.0
        assert cfg.recall.weights.text_sim == 0.35
        assert cfg.recall.weights.recency == 0.25
        assert cfg.recall.weights.gravity == 0.15
        assert cfg.recall.weights.freq == 0.10
        assert cfg.recall.weights.scope_boost == 0.05
        assert cfg.recall.weights.graph_path == 0.10
        assert cfg.recall.rrf_k == 60
        assert cfg.recall.recency_half_life_days == 7.0
        assert cfg.recall.graph_hop == 2
        assert cfg.recall.graph_decay == 0.5
        assert cfg.consolidate.time_gate_hours == 24
        assert cfg.consolidate.content_gate_count == 50
        assert cfg.consolidate.index_budget_count == 200
        assert cfg.consolidate.index_budget_bytes == 25600
        assert cfg.pair.code_length == 8
        assert cfg.pair.code_expire_minutes == 15

    def test_from_yaml(self, tmp_path):
        """从 YAML 加载配置。"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("""
server:
  host: 127.0.0.1
  port: 9999
zone:
  emergence:
    semantic_threshold: 0.80
recall:
  weights:
    text_sim: 0.50
""")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.port == 9999
        assert cfg.zone.emergence.semantic_threshold == 0.80
        assert cfg.recall.weights.text_sim == 0.50
        # 未覆盖的保持默认
        assert cfg.recall.weights.recency == 0.25
        assert cfg.zone.emergence.base_threshold == 5

    def test_env_override(self, monkeypatch):
        """环境变量覆盖配置。"""
        monkeypatch.setenv("CORTEXOS_SERVER_HOST", "10.0.0.1")
        monkeypatch.setenv("CORTEXOS_SERVER_PORT", "8888")
        monkeypatch.setenv("CORTEXOS_ZONE_EMERGENCE_SEMANTIC_THRESHOLD", "0.85")
        cfg = Config.from_env()
        assert cfg.server.host == "10.0.0.1"
        assert cfg.server.port == 8888
        assert cfg.zone.emergence.semantic_threshold == 0.85


# ────────────────────── Models 测试 ──────────────────────


class TestEntry:
    """Entry 模型测试。"""

    def test_create_entry(self):
        """创建 Entry 应有默认值并正确序列化。"""
        entry = Entry(content="测试记忆内容", scope="agent:test")
        assert entry.id
        assert entry.zone == "_inbox"
        assert entry.layer == "raw"
        assert entry.entities == []
        assert entry.status == "active"
        assert entry.valid_until is None
        assert entry.access_count == 0

    def test_entry_to_dict_and_from_row(self):
        """Entry 序列化/反序列化往返一致。"""
        entry = Entry(
            content="K8s Pod 重启",
            scope="agent:ops",
            zone="k8s_troubleshooting",
            entities=["k8s", "pod"],
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "alert"},
        )
        d = entry.to_dict()
        restored = Entry.from_row(d)
        assert restored.id == entry.id
        assert restored.content == entry.content
        assert restored.zone == entry.zone
        assert restored.entities == entry.entities
        assert restored.embedding == entry.embedding
        assert restored.metadata == entry.metadata

    def test_entry_touch(self):
        """touch 应增加访问计数。"""
        entry = Entry(content="test")
        assert entry.access_count == 0
        entry.touch()
        assert entry.access_count == 1
        entry.touch()
        assert entry.access_count == 2


class TestFact:
    """Fact 模型测试。"""

    def test_fact_serialization(self):
        """Fact 序列化/反序列化往返一致。"""
        fact = Fact(
            subject="nginx",
            predicate="deployed_on",
            object="server-01",
            scope="agent:ops",
            confidence=0.95,
            valid_from=time.time(),
            status="active",
        )
        d = fact.to_dict()
        restored = Fact.from_row(d)
        assert restored.subject == "nginx"
        assert restored.predicate == "deployed_on"
        assert restored.object == "server-01"
        assert restored.confidence == 0.95


class TestEdge:
    """Edge 模型测试。"""

    def test_edge_serialization(self):
        """Edge 序列化往返一致。"""
        edge = Edge(
            source="nginx",
            target="server-01",
            relation="运行在",
            scope="agent:ops",
            weight=0.8,
        )
        d = edge.to_dict()
        restored = Edge.from_row(d)
        assert restored.source == "nginx"
        assert restored.target == "server-01"
        assert restored.relation == "运行在"
        assert restored.weight == 0.8


class TestZone:
    """Zone 模型测试。"""

    def test_zone_serialization(self):
        """Zone 序列化往返一致。"""
        zone = Zone(
            name="k8s_incidents",
            scope="agent:ops",
            description="K8s 事故记录",
            entities=["k8s", "pod", "node"],
            keywords=["OOMKilled", "CrashLoopBackOff"],
            centroid=[0.1, 0.2, 0.3],
            gravity=1.5,
            entry_count=25,
        )
        d = zone.to_dict()
        restored = Zone.from_row(d)
        assert restored.name == "k8s_incidents"
        assert restored.entities == ["k8s", "pod", "node"]
        assert restored.centroid == [0.1, 0.2, 0.3]


class TestAgentKey:
    """AgentKey 权限判定测试。"""

    def test_has_scope_permission_read(self):
        """scope 级权限检查。"""
        key = AgentKey(
            key_id="k1", agent_id="ag1", key_hash="hash1",
            scope_permissions={"agent:test": "read"}
        )
        assert key.has_scope_permission("agent:test", "read")
        assert not key.has_scope_permission("agent:test", "write")
        assert not key.has_scope_permission("agent:other", "read")

    def test_has_scope_permission_readwrite(self):
        """readwrite 应包含 read 和 write。"""
        key = AgentKey(
            key_id="k2", agent_id="ag1", key_hash="hash2",
            scope_permissions={"agent:test": "readwrite"}
        )
        assert key.has_scope_permission("agent:test", "read")
        assert key.has_scope_permission("agent:test", "write")

    def test_zone_override(self):
        """Zone 覆盖权限读取。"""
        key = AgentKey(
            key_id="k3", agent_id="ag1", key_hash="hash3",
            scope_permissions={"agent:test": "readwrite"},
            zone_overrides={"agent:test:k8s_zone": "read"}
        )
        assert key.get_zone_override("agent:test", "k8s_zone") == "read"
        assert key.get_zone_override("agent:test", "other_zone") is None

    def test_key_expired(self):
        """过期密钥判定。"""
        expired_key = AgentKey(
            key_id="k4", agent_id="ag1", key_hash="hash4",
            expires_at=time.time() - 100,
        )
        assert expired_key.is_expired

        valid_key = AgentKey(
            key_id="k5", agent_id="ag1", key_hash="hash5",
            expires_at=time.time() + 3600,
        )
        assert not valid_key.is_expired

        no_expiry = AgentKey(
            key_id="k6", agent_id="ag1", key_hash="hash6",
            expires_at=None,
        )
        assert not no_expiry.is_expired


# ────────────────────── SQLite Backend 测试 ──────────────────────

# 使用临时文件数据库而非 :memory: 以避免 FTS5 问题
def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


@pytest.mark.asyncio
class TestSqliteBackend:
    """SQLite 存储后端测试。"""

    async def test_initialize_and_close(self):
        """初始化应建表成功，关闭后连接释放。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()
            await db.close()
        finally:
            os.unlink(db_path)

    async def test_upsert_and_get_entry(self):
        """写入和读取条目。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            entry = Entry(
                content="测试记忆",
                scope="agent:test",
                zone="_inbox",
                entities=["test", "memory"],
                embedding=[0.1, 0.2, 0.3],
                metadata={"lang": "zh"},
            )
            await db.upsert_entry(entry)

            restored = await db.get_entry(entry.id)
            assert restored is not None
            assert restored.content == "测试记忆"
            assert restored.scope == "agent:test"
            assert restored.entities == ["test", "memory"]
            assert restored.embedding == [0.1, 0.2, 0.3]
            assert restored.metadata == {"lang": "zh"}
            assert restored.status == "active"

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_list_entries(self):
        """列出条目（分页过滤）。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            for i in range(5):
                entry = Entry(content=f"entry {i}", scope="agent:test",
                            zone=f"zone_{i % 2}")
                await db.upsert_entry(entry)

            all_entries = await db.list_entries(scope="agent:test", limit=10)
            assert len(all_entries) == 5

            zone0 = await db.list_entries(scope="agent:test", zone="zone_0")
            assert len(zone0) >= 2

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_count_entries(self):
        """统计条目数。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            for i in range(3):
                entry = Entry(content=f"entry {i}", scope="agent:test")
                await db.upsert_entry(entry)

            cnt = await db.count_entries(scope="agent:test")
            assert cnt == 3

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_delete_entry(self):
        """软删除条目。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            entry = Entry(content="可删", scope="agent:test")
            await db.upsert_entry(entry)
            await db.delete_entry(entry.id)

            restored = await db.get_entry(entry.id)
            assert restored.status == "archived"

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_search_lexical(self):
        """FTS5 词法检索。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            e1 = Entry(content="Kubernetes Pod restart troubleshooting", scope="agent:test")
            e2 = Entry(content="MySQL database connection pool config", scope="agent:test")
            e3 = Entry(content="Redis cache expiration strategy", scope="agent:test")
            await db.upsert_entry(e1)
            await db.upsert_entry(e2)
            await db.upsert_entry(e3)

            results = await db.search_lexical("Pod restart", scope="agent:test", top_k=5)
            assert len(results) > 0
            assert any("Pod" in r[0].content for r in results)

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_facts_crud(self):
        """Fact 写入和查询。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            fact = Fact(
                subject="nginx",
                predicate="deployed_on",
                object="server-01",
                scope="agent:test",
            )
            await db.upsert_fact(fact)

            facts = await db.find_facts(subject="nginx", scope="agent:test")
            assert len(facts) == 1
            assert facts[0].predicate == "deployed_on"
            assert facts[0].object == "server-01"

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_edges_crud(self):
        """Edge 写入和查询。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            edge = Edge(
                source="nginx",
                target="server-01",
                relation="运行在",
                scope="agent:test",
            )
            await db.upsert_edge(edge)

            from_edges = await db.list_edges_from("nginx")
            assert len(from_edges) == 1
            assert from_edges[0].target == "server-01"

            to_edges = await db.list_edges_to("server-01")
            assert len(to_edges) == 1

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_zones_crud(self):
        """Zone 写入和查询。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            zone = Zone(
                name="k8s",
                scope="agent:test",
                description="K8s 相关",
                entities=["k8s", "pod"],
                centroid=[0.1, 0.2],
            )
            await db.upsert_zone(zone)

            restored = await db.get_zone("k8s")
            assert restored is not None
            assert restored.description == "K8s 相关"
            assert restored.entities == ["k8s", "pod"]

            all_zones = await db.list_zones(scope="agent:test")
            assert len(all_zones) >= 1

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_agent_and_key_management(self):
        """Agent 和密钥管理。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            await db.upsert_agent({
                "agent_id": "ag_test1",
                "agent_name": "Test Agent",
                "created_at": time.time(),
            })
            agent = await db.get_agent("ag_test1")
            assert agent is not None
            assert agent["agent_name"] == "Test Agent"

            await db.upsert_agent_key({
                "key_id": "key_test1",
                "agent_id": "ag_test1",
                "key_hash": "sha256hash123",
                "scope_permissions": {"agent:ag_test1": "readwrite"},
                "created_at": time.time(),
            })
            key = await db.get_agent_key_by_id("key_test1")
            assert key is not None
            assert key["key_hash"] == "sha256hash123"

            keys = await db.list_agent_keys("ag_test1")
            assert len(keys) == 1

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_pair_code(self):
        """配对码创建和查询。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            await db.upsert_pair_code({
                "code": "ABC12345",
                "agent_id": "ag_test2",
                "agent_name": "Pair Test",
                "expires_at": time.time() + 900,
                "status": "pending",
            })

            code = await db.get_pair_code("ABC12345")
            assert code is not None
            assert code["agent_id"] == "ag_test2"
            assert code["status"] == "pending"

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_audit_log(self):
        """审计日志写入。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            await db.write_audit_log({
                "action": "store",
                "scope": "agent:test",
                "detail": "wrote entry",
            })

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_get_stats(self):
        """统计信息。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            for i in range(10):
                e = Entry(content=f"entry{i}", scope="agent:test")
                await db.upsert_entry(e)

            stats = await db.get_stats()
            assert stats["total_entries"] == 10
            assert stats["active_entries"] == 10

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_bulk_update_status(self):
        """批量更新状态。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            entries = []
            for i in range(3):
                e = Entry(content=f"entry{i}", scope="agent:test")
                await db.upsert_entry(e)
                entries.append(e)

            await db.bulk_update_status([(e.id, "superseded") for e in entries])

            for e in entries:
                restored = await db.get_entry(e.id)
                assert restored.status == "superseded"

            await db.close()
        finally:
            os.unlink(db_path)

    async def test_scan_entries(self):
        """流式扫描条目。"""
        from cortexos.storage.sqlite_backend import SqliteBackend
        db_path = _temp_db()
        try:
            db = SqliteBackend(db_path)
            await db.initialize()

            for i in range(5):
                e = Entry(content=f"entry{i}", scope="agent:test", layer="raw")
                await db.upsert_entry(e)

            scanned = []
            async for e in db.scan_entries(scope="agent:test", layer="raw"):
                scanned.append(e)

            assert len(scanned) == 5

            await db.close()
        finally:
            os.unlink(db_path)

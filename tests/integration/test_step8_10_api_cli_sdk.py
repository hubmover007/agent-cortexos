"""Steps 8-10 集成测试：FastAPI API + CLI + SDK。

测试：
- REST API 端到端冒烟（配对→写入→检索→zones→权限拒绝）
- 认证/权限/限流（401/403/429）
- CLI 冒烟（serve + store + recall）
- SDK 冒烟（CortexOS 本地模式 store + recall + search）
"""

import time
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from cortexos.config import Config
from cortexos.models import Entry
from cortexos.storage.sqlite_backend import SqliteBackend
from cortexos.api.routes import create_app_v1


# ────────────────────── 测试夹具 ──────────────────────


@pytest.fixture
def app_client():
    """创建带测试 DB 的 FastAPI TestClient。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    import asyncio
    async def init():
        db = SqliteBackend(path)
        await db.initialize()
        return db

    db = asyncio.run(init())

    config = Config()
    app = create_app_v1(db, config)
    client = TestClient(app)

    yield client, db, path

    # 清理
    client.close()
    try:
        os.unlink(path)
    except OSError:
        pass


def _pair(client, agent_name: str, scopes: dict) -> dict:
    """完整配对流程：request → confirm → exchange。

    Returns:
        {"headers": {...}, "secret": str, "agent_id": str, "key_id": str}
    """
    r1 = client.post("/v1/pair/request", json={"agent_name": agent_name})
    assert r1.status_code == 200
    code = r1.json()["code"]

    r2 = client.post("/v1/pair/confirm", json={
        "code": code,
        "scope_permissions": scopes,
    })
    assert r2.status_code == 200

    r3 = client.post("/v1/pair/exchange", json={"code": code})
    assert r3.status_code == 200
    data = r3.json()

    return {
        "headers": {"Authorization": f"Bearer {data['secret']}"},
        "secret": data["secret"],
        "agent_id": data["agent_id"],
        "key_id": data["key_id"],
    }


# ────────────────────── REST API 端到端测试 ──────────────────────


class TestHealthz:
    """健康检查测试。"""

    def test_healthz_ok(self, app_client):
        client, db, path = app_client
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"


class TestPairingAPI:
    """配对 API 端到端测试。"""

    def test_pair_request(self, app_client):
        client, db, path = app_client
        resp = client.post("/v1/pair/request", json={"agent_name": "test-bot"})
        assert resp.status_code == 200
        data = resp.json()
        assert "code" in data
        assert data["agent_id"].startswith("ag_")

    def test_pair_confirm(self, app_client):
        client, db, path = app_client
        r1 = client.post("/v1/pair/request", json={"agent_name": "bot1"})
        code = r1.json()["code"]

        r2 = client.post("/v1/pair/confirm", json={
            "code": code,
            "scope_permissions": {"agent:bot1": "readwrite"},
        })
        assert r2.status_code == 200
        assert r2.json()["code"] == code

    def test_pair_exchange(self, app_client):
        client, db, path = app_client
        r1 = client.post("/v1/pair/request", json={"agent_name": "bot2"})
        code = r1.json()["code"]

        r2 = client.post("/v1/pair/confirm", json={
            "code": code,
            "scope_permissions": {"agent:bot2": "readwrite"},
        })
        assert r2.status_code == 200

        r3 = client.post("/v1/pair/exchange", json={"code": code})
        assert r3.status_code == 200
        data = r3.json()
        assert data["key_id"].startswith("ak_")
        assert data["secret"]

    def test_pair_exchange_key_has_scopes(self, app_client):
        """回归：exchange 颁发的 key 必须携带 approve 时授予的 scope 权限。"""
        client, db, path = app_client
        paired = _pair(client, "bot-scope", {"agent:scope-a": "readwrite"})

        # 用新 key 查 keys 列表（脱敏）验证权限
        resp = client.get("/v1/keys", headers=paired["headers"])
        assert resp.status_code == 200
        keys = resp.json()["keys"]
        assert len(keys) >= 1
        assert keys[0]["scope_permissions"] == {"agent:scope-a": "readwrite"}

    def test_pair_invalid_code(self, app_client):
        client, db, path = app_client
        resp = client.post("/v1/pair/confirm", json={
            "code": "INVALID",
            "scope_permissions": {},
        })
        assert resp.status_code == 400

    def test_pair_double_exchange_rejected(self, app_client):
        """二次兑换被拒绝。"""
        client, db, path = app_client
        r1 = client.post("/v1/pair/request", json={"agent_name": "bot3"})
        code = r1.json()["code"]

        client.post("/v1/pair/confirm", json={
            "code": code,
            "scope_permissions": {"agent:bot3": "readwrite"},
        })
        client.post("/v1/pair/exchange", json={"code": code})

        # 第二次
        r4 = client.post("/v1/pair/exchange", json={"code": code})
        assert r4.status_code == 400


class TestAuthAPI:
    """认证 / 权限 / 限流测试。"""

    def test_no_token_401(self, app_client):
        client, db, path = app_client
        resp = client.post("/v1/memories", json={
            "content": "无 token 写入", "scope": "agent:auth",
        })
        assert resp.status_code == 401

    def test_invalid_token_401(self, app_client):
        client, db, path = app_client
        resp = client.get("/v1/stats", headers={"Authorization": "Bearer bad-token"})
        assert resp.status_code == 401

    def test_wrong_scope_403(self, app_client):
        """key 只有 scope A 权限，写 scope B → 403。"""
        client, db, path = app_client
        paired = _pair(client, "bot-a", {"agent:a": "readwrite"})

        resp = client.post("/v1/memories", headers=paired["headers"], json={
            "content": "越权写入", "scope": "agent:b",
        })
        assert resp.status_code == 403

        # 读自己 scope → 200
        resp2 = client.post("/v1/retrieve", headers=paired["headers"], json={
            "query": "x", "scope": "agent:a",
        })
        assert resp2.status_code == 200

    def test_readonly_key_cannot_write(self, app_client):
        """read 权限的 key 不能写，能读。"""
        client, db, path = app_client
        paired = _pair(client, "bot-ro", {"agent:ro": "read"})

        resp = client.post("/v1/memories", headers=paired["headers"], json={
            "content": "只读 key 尝试写入", "scope": "agent:ro",
        })
        assert resp.status_code == 403

        resp2 = client.post("/v1/retrieve", headers=paired["headers"], json={
            "query": "anything", "scope": "agent:ro",
        })
        assert resp2.status_code == 200

    def test_rate_limit_429(self, app_client):
        """超限流 → 429。"""
        client, db, path = app_client
        # 直接用后端创建限流=2 的 key（配对流程固定用默认限流）
        import asyncio
        from cortexos.auth.keys import create_key

        async def _mk():
            agent = await db.upsert_agent({
                "agent_id": "ag_rate", "agent_name": "rate-bot",
                "created_at": time.time(), "status": "active",
            })
            return await create_key(
                "ag_rate", {"agent:rate": "read"}, db,
                rate_limit=2,
            )

        key = asyncio.run(_mk())
        headers = {"Authorization": f"Bearer {key['secret']}"}

        # 前 2 次 OK
        for _ in range(2):
            r = client.get("/v1/stats", headers=headers)
            assert r.status_code == 200
        # 第 3 次 429
        r = client.get("/v1/stats", headers=headers)
        assert r.status_code == 429


class TestMemoryAPI:
    """记忆 CRUD API 测试（需认证）。"""

    def _setup(self, client, scope="agent:recall"):
        return _pair(client, "mem-bot", {scope: "readwrite"})

    def test_store_memory(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        resp = client.post("/v1/memories", headers=paired["headers"], json={
            "content": "nginx 在 server-01 上运行",
            "scope": "agent:recall",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"]
        assert data["zone"] != ""

    def test_store_memory_generates_embedding(self, app_client):
        """回归：写入管线必须生成 embedding（语义层可用）。"""
        client, db, path = app_client
        paired = self._setup(client)
        resp = client.post("/v1/memories", headers=paired["headers"], json={
            "content": "Kubernetes pod 重启排查流程",
            "scope": "agent:recall",
        })
        assert resp.status_code == 200
        entry_id = resp.json()["id"]

        import asyncio
        entry = asyncio.run(db.get_entry(entry_id))
        assert entry is not None
        assert entry.embedding is not None
        assert len(entry.embedding) > 0

    def test_get_memory(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        r1 = client.post("/v1/memories", headers=paired["headers"], json={
            "content": "测试记忆", "scope": "agent:recall",
        })
        entry_id = r1.json()["id"]

        r2 = client.get(f"/v1/memories/{entry_id}", headers=paired["headers"])
        assert r2.status_code == 200
        assert r2.json()["content"] == "测试记忆"

    def test_get_memory_404(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        resp = client.get("/v1/memories/nonexistent", headers=paired["headers"])
        assert resp.status_code == 404

    def test_get_memory_cross_scope_403(self, app_client):
        """回归：无权限 scope 的条目读取 → 403。"""
        client, db, path = app_client
        paired = self._setup(client, scope="agent:recall")

        # 用另一个 key 写入不同 scope
        other = _pair(client, "other-bot", {"agent:other": "readwrite"})
        r = client.post("/v1/memories", headers=other["headers"], json={
            "content": "别人的记忆", "scope": "agent:other",
        })
        other_id = r.json()["id"]

        # 第一个 key 读取 → 403
        resp = client.get(f"/v1/memories/{other_id}", headers=paired["headers"])
        assert resp.status_code == 403

    def test_delete_memory(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        r1 = client.post("/v1/memories", headers=paired["headers"], json={
            "content": "删除测试", "scope": "agent:recall",
        })
        entry_id = r1.json()["id"]

        r2 = client.delete(f"/v1/memories/{entry_id}", headers=paired["headers"])
        assert r2.status_code == 200

        r3 = client.get(f"/v1/memories/{entry_id}", headers=paired["headers"])
        assert r3.status_code == 404

    def test_retrieve_memories(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        for i in range(3):
            client.post("/v1/memories", headers=paired["headers"], json={
                "content": f"nginx 部署在 k8s 集群 node-{i}",
                "scope": "agent:recall",
            })

        resp = client.post("/v1/retrieve", headers=paired["headers"], json={
            "query": "nginx k8s", "scope": "agent:recall",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert len(data["items"]) > 0

    def test_search(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "MySQL 数据库主从同步延迟告警", "scope": "agent:recall",
        })

        resp = client.post("/v1/search", headers=paired["headers"], json={
            "query": "MySQL 延迟", "scope": "agent:recall",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_context(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "上下文测试内容", "scope": "agent:recall",
        })

        resp = client.get("/v1/context?scope=agent:recall", headers=paired["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert "entries" in data


class TestZonesAPI:
    """Zones API 测试。"""

    def _setup(self, client, scope="agent:test"):
        return _pair(client, "zone-bot", {scope: "readwrite"})

    def test_list_zones(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "zone test", "scope": "agent:test",
        })

        resp = client.get("/v1/zones?scope=agent:test", headers=paired["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert "zones" in data

    def test_pin_zone(self, app_client):
        client, db, path = app_client
        paired = self._setup(client)
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "zone pin test", "scope": "agent:test",
        })

        resp = client.post(
            "/v1/zones/agent:test/_inbox/pin", headers=paired["headers"], json={},
        )
        # 可能 200 或 404（zone 不存在）
        assert resp.status_code in (200, 404)

    def test_scopes_filtered_by_permission(self, app_client):
        """回归：/v1/scopes 只返回 key 有权限的 scope。"""
        client, db, path = app_client
        paired = _pair(client, "scope-bot", {"agent:mine": "readwrite"})

        # 往 agent:mine 写一条数据（scope 列表只显示有数据的 scope）
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "my data", "scope": "agent:mine",
        })

        # 另一个 key 写入别的 scope
        other = _pair(client, "other-scope", {"agent:theirs": "readwrite"})
        client.post("/v1/memories", headers=other["headers"], json={
            "content": "their data", "scope": "agent:theirs",
        })

        resp = client.get("/v1/scopes", headers=paired["headers"])
        assert resp.status_code == 200
        assert "agent:mine" in resp.json()["scopes"]
        assert "agent:theirs" not in resp.json()["scopes"]


class TestStatsAPI:
    """统计 API 测试。"""

    def test_stats(self, app_client):
        client, db, path = app_client
        paired = _pair(client, "stats-bot", {"agent:test": "readwrite"})
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "stats test", "scope": "agent:test",
        })

        resp = client.get("/v1/stats", headers=paired["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entries"] >= 1
        assert data["total_zones"] >= 0
        assert data["db_size_bytes"] > 0


# ────────────────────── SDK 集成测试 ──────────────────────


@pytest.mark.asyncio
class TestSDK:
    """CortexOS SDK 集成测试。"""

    async def _make_cortex(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        from cortexos import CortexOS
        cortex = CortexOS(path)
        await cortex._ensure_init()
        return cortex, path

    async def test_sdk_store_and_recall(self):
        """SDK store + recall。"""
        cortex, db_path = await self._make_cortex()
        try:
            entry_id = await cortex.store(
                "Kubernetes 集群升级到 v1.30",
                scope="agent:sdk-test",
            )
            assert entry_id

            results = await cortex.recall("Kubernetes", scope="agent:sdk-test")
            assert len(results) > 0
        finally:
            await cortex.close()
            os.unlink(db_path)

    async def test_sdk_get(self):
        """SDK get。"""
        cortex, db_path = await self._make_cortex()
        try:
            entry_id = await cortex.store("读取测试", scope="agent:sdk-test")
            entry = await cortex.get(entry_id)
            assert entry is not None
            assert entry.content == "读取测试"
        finally:
            await cortex.close()
            os.unlink(db_path)

    async def test_sdk_delete(self):
        """SDK delete。"""
        cortex, db_path = await self._make_cortex()
        try:
            entry_id = await cortex.store("删除测试", scope="agent:sdk-test")
            ok = await cortex.delete(entry_id)
            assert ok
            gone = await cortex.get(entry_id)
            assert gone is None
        finally:
            await cortex.close()
            os.unlink(db_path)

    async def test_sdk_search(self):
        """回归：SDK 词法搜索（曾因参数名 limit/top_k 不匹配崩溃）。"""
        cortex, db_path = await self._make_cortex()
        try:
            await cortex.store("Redis 缓存雪崩排查", scope="agent:sdk-test")
            results = await cortex.search("Redis", scope="agent:sdk-test", limit=5)
            assert len(results) > 0
            assert "Redis" in results[0]["content"]
        finally:
            await cortex.close()
            os.unlink(db_path)

    async def test_sdk_store_embeds(self):
        """回归：SDK store 生成 embedding。"""
        cortex, db_path = await self._make_cortex()
        try:
            entry_id = await cortex.store("nginx 部署", scope="agent:sdk-test")
            entry = await cortex.get(entry_id)
            assert entry.embedding is not None
        finally:
            await cortex.close()
            os.unlink(db_path)

    async def test_sdk_zones(self):
        """SDK zones。"""
        cortex, db_path = await self._make_cortex()
        try:
            await cortex.store("zone test", scope="agent:sdk-test")
            zones = await cortex.list_zones("agent:sdk-test")
            assert len(zones) >= 0
        finally:
            await cortex.close()
            os.unlink(db_path)

    async def test_sdk_stats(self):
        """SDK stats。"""
        cortex, db_path = await self._make_cortex()
        try:
            await cortex.store("stats test", scope="agent:sdk-test")
            stats = await cortex.stats()
            assert stats["total_entries"] >= 1
            assert "db_size_bytes" in stats
        finally:
            await cortex.close()
            os.unlink(db_path)


# ────────────────────── CLI 冒烟测试 ──────────────────────


class TestCLI:
    """CLI 冒烟测试。"""

    def test_cli_store_recall(self):
        """CLI store + recall。"""
        import subprocess
        import sys

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            # store
            r1 = subprocess.run(
                [sys.executable, "-m", "cortexos", "store",
                 "nginx 部署到 k8s 集群", "--scope", "agent:cli-test",
                 "--db", path],
                capture_output=True, text=True,
            )
            assert r1.returncode == 0, r1.stderr

            # recall
            r2 = subprocess.run(
                [sys.executable, "-m", "cortexos", "recall",
                 "nginx k8s", "--scope", "agent:cli-test",
                 "--db", path],
                capture_output=True, text=True,
            )
            assert r2.returncode == 0, r2.stderr
            items = json.loads(r2.stdout)
            assert len(items) > 0

            # stats
            r3 = subprocess.run(
                [sys.executable, "-m", "cortexos", "stats",
                 "--db", path],
                capture_output=True, text=True,
            )
            assert r3.returncode == 0, r3.stderr
            stats = json.loads(r3.stdout)
            assert stats["total_entries"] >= 1
            assert "db_size_bytes" in stats

            # zones
            r4 = subprocess.run(
                [sys.executable, "-m", "cortexos", "zones",
                 "--scope", "agent:cli-test", "--db", path],
                capture_output=True, text=True,
            )
            assert r4.returncode == 0, r4.stderr
        finally:
            os.unlink(path)

    def test_cli_pair_flow(self):
        """CLI pair request + pair approve + pair exchange。"""
        import subprocess
        import sys

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            # pair request
            r1 = subprocess.run(
                [sys.executable, "-m", "cortexos", "pair-request",
                 "cli-bot", "--db", path],
                capture_output=True, text=True,
            )
            assert r1.returncode == 0, r1.stderr
            data = json.loads(r1.stdout)
            code = data["code"]

            # pair approve
            r2 = subprocess.run(
                [sys.executable, "-m", "cortexos", "pair-approve",
                 code, "--scopes", '{"agent:cli-bot":"readwrite"}',
                 "--db", path],
                capture_output=True, text=True,
            )
            assert r2.returncode == 0, r2.stderr

            # pair exchange
            r3 = subprocess.run(
                [sys.executable, "-m", "cortexos", "pair-exchange",
                 code, "--db", path],
                capture_output=True, text=True,
            )
            assert r3.returncode == 0, r3.stderr
            exchanged = json.loads(r3.stdout)
            assert exchanged["key_id"].startswith("ak_")

            # 回归：兑换的 key 必须带 approve 时授予的 scope 权限
            import asyncio
            from cortexos.storage.sqlite_backend import SqliteBackend
            async def _check():
                db = SqliteBackend(path)
                await db.initialize()
                keys = await db.list_agent_keys(exchanged["agent_id"])
                await db.close()
                return keys
            keys = asyncio.run(_check())
            assert keys[0]["scope_permissions"] == '{"agent:cli-bot": "readwrite"}'
        finally:
            os.unlink(path)

    def test_cli_consolidate(self):
        """CLI consolidate。"""
        import subprocess
        import sys

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            # store something first
            subprocess.run(
                [sys.executable, "-m", "cortexos", "store",
                 "consolidate test1", "--scope", "agent:cli-consolidate",
                 "--db", path],
                capture_output=True, text=True,
            )

            r = subprocess.run(
                [sys.executable, "-m", "cortexos", "consolidate",
                 "--scope", "agent:cli-consolidate", "--db", path],
                capture_output=True, text=True,
            )
            # consolidate 可能因为时间门控返回 0
            assert r.returncode == 0, r.stderr
        finally:
            os.unlink(path)

    def test_cli_help(self):
        """CLI help。"""
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-m", "cortexos", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "serve" in r.stdout.lower() or "pair" in r.stdout.lower()


class TestAuthzHardening:
    """越权面加固回归：delete_key 跨 agent / consolidate 权限。"""

    def test_delete_key_cross_agent_forbidden(self, app_client):
        """回归：A 不能吊销 B 的 key（403）。"""
        client, db, path = app_client
        a = _pair(client, "agent-a", {"agent:sa": "readwrite"})
        b = _pair(client, "agent-b", {"agent:sb": "readwrite"})

        # A 尝试吊销 B 的 key
        resp = client.delete(f"/v1/keys/{b['key_id']}", headers=a["headers"])
        assert resp.status_code == 403

        # B 的 key 仍然有效
        resp = client.get("/v1/keys", headers=b["headers"])
        assert resp.status_code == 200
        assert any(k["key_id"] == b["key_id"] for k in resp.json()["keys"])

    def test_delete_own_key_ok(self, app_client):
        """回归：吊销自己的 key 成功。"""
        client, db, path = app_client
        a = _pair(client, "agent-c", {"agent:sc": "readwrite"})
        resp = client.delete(f"/v1/keys/{a['key_id']}", headers=a["headers"])
        assert resp.status_code == 200
        # 吊销后 key 失效
        resp = client.get("/v1/keys", headers=a["headers"])
        assert resp.status_code == 401

    def test_consolidate_requires_scope_write(self, app_client):
        """回归：无 scope 写权限时触发 consolidate → 403。"""
        client, db, path = app_client
        paired = _pair(client, "consolidate-bot", {"agent:readonly": "read"})

        resp = client.post(
            "/v1/consolidate?scope=agent:readonly",
            headers=paired["headers"],
        )
        assert resp.status_code == 403

    def test_consolidate_with_write_ok(self, app_client):
        """回归：有写权限时 consolidate 可执行（不 403）。"""
        client, db, path = app_client
        paired = _pair(client, "consolidate-bot2", {"agent:rw": "readwrite"})
        resp = client.post(
            "/v1/consolidate?scope=agent:rw",
            headers=paired["headers"],
        )
        assert resp.status_code == 200

    def test_search_special_chars_no_500(self, app_client):
        """回归：搜索含 FTS5 特殊字符不 500。"""
        client, db, path = app_client
        paired = _pair(client, "search-bot", {"agent:sx": "readwrite"})
        client.post("/v1/memories", headers=paired["headers"], json={
            "content": "MySQL 主从延迟处理", "scope": "agent:sx",
        })

        for q in ['"', '*', 'AND OR NOT', 'a"b']:
            resp = client.post("/v1/search", headers=paired["headers"], json={
                "query": q, "scope": "agent:sx", "top_k": 5,
            })
            assert resp.status_code == 200

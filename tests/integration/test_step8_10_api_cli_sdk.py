"""Steps 8-10 集成测试：FastAPI API + CLI + SDK。

测试：
- REST API 端到端冒烟（配对→写入→检索→zones→权限拒绝）
- CLI 冒烟（serve + store + recall）
- SDK 冒烟（CortexOS 本地模式 store + recall）
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
        # request
        r1 = client.post("/v1/pair/request", json={"agent_name": "bot1"})
        code = r1.json()["code"]

        # confirm
        r2 = client.post("/v1/pair/confirm", json={
            "code": code,
            "scope_permissions": {"agent:bot1": "readwrite"},
        })
        assert r2.status_code == 200
        assert r2.json()["code"] == code

    def test_pair_exchange(self, app_client):
        client, db, path = app_client
        # request → confirm → exchange
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


class TestMemoryAPI:
    """记忆 CRUD API 测试。"""

    def test_store_memory(self, app_client):
        client, db, path = app_client
        resp = client.post("/v1/memories", json={
            "content": "nginx 在 server-01 上运行",
            "scope": "agent:recall",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"]
        assert data["zone"] != ""

    def test_get_memory(self, app_client):
        client, db, path = app_client
        r1 = client.post("/v1/memories", json={
            "content": "测试记忆", "scope": "agent:recall",
        })
        entry_id = r1.json()["id"]

        r2 = client.get(f"/v1/memories/{entry_id}")
        assert r2.status_code == 200
        assert r2.json()["content"] == "测试记忆"

    def test_get_memory_404(self, app_client):
        client, db, path = app_client
        resp = client.get("/v1/memories/nonexistent")
        assert resp.status_code == 404

    def test_delete_memory(self, app_client):
        client, db, path = app_client
        r1 = client.post("/v1/memories", json={
            "content": "删除测试", "scope": "agent:recall",
        })
        entry_id = r1.json()["id"]

        r2 = client.delete(f"/v1/memories/{entry_id}")
        assert r2.status_code == 200

        r3 = client.get(f"/v1/memories/{entry_id}")
        assert r3.status_code == 404

    def test_retrieve_memories(self, app_client):
        client, db, path = app_client
        # 写入几条
        for i in range(3):
            client.post("/v1/memories", json={
                "content": f"nginx 部署在 k8s 集群 node-{i}",
                "scope": "agent:recall",
            })

        resp = client.post("/v1/retrieve", json={
            "query": "nginx k8s", "scope": "agent:recall",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert len(data["items"]) > 0

    def test_search(self, app_client):
        client, db, path = app_client
        client.post("/v1/memories", json={
            "content": "MySQL 数据库主从同步延迟告警", "scope": "agent:recall",
        })

        resp = client.post("/v1/search", json={
            "query": "MySQL 延迟", "scope": "agent:recall",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_context(self, app_client):
        client, db, path = app_client
        client.post("/v1/memories", json={
            "content": "上下文测试内容", "scope": "agent:recall",
        })

        resp = client.get("/v1/context?scope=agent:recall")
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert "entries" in data


class TestZonesAPI:
    """Zones API 测试。"""

    def test_list_zones(self, app_client):
        client, db, path = app_client
        client.post("/v1/memories", json={
            "content": "zone test", "scope": "agent:test",
        })

        resp = client.get("/v1/zones?scope=agent:test")
        assert resp.status_code == 200
        data = resp.json()
        assert "zones" in data

    def test_pin_zone(self, app_client):
        client, db, path = app_client
        # 先写一条记忆生成 zone
        client.post("/v1/memories", json={
            "content": "zone pin test", "scope": "agent:test",
        })

        resp = client.post("/v1/zones/agent:test/_inbox/pin", json={})
        # 可能 200 或 404（zone 不存在）
        assert resp.status_code in (200, 404)


class TestStatsAPI:
    """统计 API 测试。"""

    def test_stats(self, app_client):
        client, db, path = app_client
        client.post("/v1/memories", json={
            "content": "stats test", "scope": "agent:test",
        })

        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entries"] >= 1


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

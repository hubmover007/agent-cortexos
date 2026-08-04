"""Steps 6+7 单元测试：lifecycle(resolve+consolidate) + auth(pairing+keys+permissions)。

关键测试用例（code-design-p1.md §5）：
- 冲突时间窗口截断
- 权限单调递减越权拒绝
- 配对过期/二次兑换拒绝
"""

import time

import pytest

from cortexos.config import Config
from cortexos.models import AgentKey, Fact


# ────────────────────── 冲突消解测试 ──────────────────────


class TestResolve:
    """冲突消解测试。"""

    @pytest.mark.asyncio
    async def test_same_object_merge(self):
        """一致事实合并：置信度取高，valid_until 取远。"""
        from cortexos.lifecycle.resolve import resolve_fact
        cfg = Config()

        old = Fact(
            id="f1", subject="nginx", predicate="deployed_on",
            object="server-01", scope="agent:test",
            confidence=0.8, valid_from=100, valid_until=200,
            status="active",
        )
        new = Fact(
            id="f2", subject="nginx", predicate="deployed_on",
            object="server-01", scope="agent:test",
            confidence=0.9, valid_from=150, valid_until=250,
        )
        resolved = await resolve_fact(new, [old], cfg, now=300)
        # old 的置信度和 valid_until 应被更新
        assert old.confidence == pytest.approx(0.9)
        assert old.valid_until == pytest.approx(250)
        # new 标记为 superseded
        assert resolved.status == "superseded"

    @pytest.mark.asyncio
    async def test_conflict_time_window_truncation(self):
        """矛盾事实时间窗口截断：new.valid_from >= old.valid_from → 截断旧事实。"""
        from cortexos.lifecycle.resolve import resolve_fact
        cfg = Config()

        old = Fact(
            id="f1", subject="nginx", predicate="deployed_on",
            object="server-01", scope="agent:test",
            confidence=0.8, valid_from=100, valid_until=500,
            status="active",
        )
        new = Fact(
            id="f2", subject="nginx", predicate="deployed_on",
            object="server-02", scope="agent:test",  # 不同 object
            confidence=0.9, valid_from=300, valid_until=600,
        )
        resolved = await resolve_fact(new, [old], cfg, now=400)
        # 旧事实的 valid_until 被截断为新事实的 valid_from
        assert old.valid_until == pytest.approx(300)
        # old.valid_until(300) < now(400) → 旧事实在当前时刻已失效 → superseded
        assert old.status == "superseded"
        # 新事实生效
        assert resolved.status == "active"

    @pytest.mark.asyncio
    async def test_unresolvable_conflict(self):
        """无法判断新旧 → new 标记 conflict。"""
        from cortexos.lifecycle.resolve import resolve_fact
        cfg = Config()

        old = Fact(
            id="f1", subject="nginx", predicate="deployed_on",
            object="server-01", scope="agent:test",
            confidence=0.8, valid_from=500, valid_until=800,  # 未来
            status="active",
        )
        new = Fact(
            id="f2", subject="nginx", predicate="deployed_on",
            object="server-02", scope="agent:test",
            confidence=0.9, valid_from=300, valid_until=600,  # new.valid_from < old.valid_from
        )
        resolved = await resolve_fact(new, [old], cfg, now=400)
        assert resolved.status == "conflict"

    @pytest.mark.asyncio
    async def test_resolve_disabled(self):
        """冲突消解关闭 → 不处理。"""
        from cortexos.lifecycle.resolve import resolve_fact
        cfg = Config()
        cfg.resolve.enabled = False

        old = Fact(
            id="f1", subject="nginx", predicate="deployed_on",
            object="server-01", scope="agent:test",
            confidence=0.8, valid_from=100, status="active",
        )
        new = Fact(
            id="f2", subject="nginx", predicate="deployed_on",
            object="server-02", scope="agent:test",
            confidence=0.9, valid_from=300,
        )
        resolved = await resolve_fact(new, [old], cfg)
        assert resolved.status == "active"  # 不修改


# ────────────────────── 权限判定测试 ──────────────────────


class TestPermissions:
    """权限判定测试。"""

    def test_scope_read_allowed(self):
        """scope read 允许读操作。"""
        from cortexos.auth.permissions import check_permission

        key = AgentKey(
            key_id="k1", agent_id="ag1", key_hash="h1",
            scope_permissions={"agent:test": "read"},
        )
        assert check_permission(key, "agent:test", "zone_a", "read")
        assert not check_permission(key, "agent:test", "zone_a", "write")

    def test_scope_readwrite_allows_read_and_write(self):
        """scope readwrite 允许读写。"""
        from cortexos.auth.permissions import check_permission

        key = AgentKey(
            key_id="k2", agent_id="ag1", key_hash="h2",
            scope_permissions={"agent:test": "readwrite"},
        )
        assert check_permission(key, "agent:test", "zone_a", "read")
        assert check_permission(key, "agent:test", "zone_a", "write")

    def test_no_scope_permission_denied(self):
        """无 scope → 拒绝。"""
        from cortexos.auth.permissions import check_permission

        key = AgentKey(
            key_id="k3", agent_id="ag1", key_hash="h3",
            scope_permissions={},
        )
        assert not check_permission(key, "agent:test", "zone_a", "read")
        assert not check_permission(key, "agent:test", "zone_a", "write")

    def test_zone_override_read_only_violation(self):
        """单调递减约束：zone 覆盖 write 越权被拒。"""
        from cortexos.auth.permissions import check_permission

        key = AgentKey(
            key_id="k4", agent_id="ag1", key_hash="h4",
            scope_permissions={"agent:test": "read"},  # scope 只读
            zone_overrides={"agent:test:k8s_zone": "readwrite"},  # zone 覆盖写
        )
        # zone 覆盖 rank(2) > scope rank(1) → 违反单调递减约束 → 拒绝
        assert not check_permission(key, "agent:test", "k8s_zone", "write")
        # 但 read 操作应该拒绝（因为 override_rank > scope_rank）
        # 等等，代码中如果 override_rank > scope_rank，整个返回 False
        assert not check_permission(key, "agent:test", "k8s_zone", "read")

    def test_zone_override_read_valid(self):
        """单调递减：zone 覆盖 read 在 scope readwrite 下有效。"""
        from cortexos.auth.permissions import check_permission

        key = AgentKey(
            key_id="k5", agent_id="ag1", key_hash="h5",
            scope_permissions={"agent:test": "readwrite"},
            zone_overrides={"agent:test:k8s_zone": "read"},
        )
        assert check_permission(key, "agent:test", "k8s_zone", "read")
        # zone 覆盖设为 read → write 被拒
        assert not check_permission(key, "agent:test", "k8s_zone", "write")

    def test_zone_override_no_effect_when_no_match(self):
        """Zone 覆盖不匹配时走 scope 权限。"""
        from cortexos.auth.permissions import check_permission

        key = AgentKey(
            key_id="k6", agent_id="ag1", key_hash="h6",
            scope_permissions={"agent:test": "readwrite"},
            zone_overrides={"agent:test:k8s_zone": "read"},
        )
        # other_zone 无覆盖 → scope 权限生效
        assert check_permission(key, "agent:test", "other_zone", "write")
        assert check_permission(key, "agent:test", "other_zone", "read")

    def test_validate_zone_override_invalid(self):
        """验证 zone 覆盖越权判定。"""
        from cortexos.auth.permissions import validate_zone_override

        # readwrite 在 read 上无效
        assert not validate_zone_override("readwrite", "read")
        # read 在 readwrite 上有效
        assert validate_zone_override("read", "readwrite")
        # 同级有效
        assert validate_zone_override("read", "read")
        assert validate_zone_override("readwrite", "readwrite")


# ────────────────────── 配对测试 ──────────────────────


@pytest.mark.asyncio
class TestPairing:
    """配对流程测试。"""

    async def _make_backend(self):
        """创建测试用 SQLite 后端。"""
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        from cortexos.storage.sqlite_backend import SqliteBackend
        db = SqliteBackend(path)
        await db.initialize()
        return db, path

    async def test_pair_request_creates_code(self):
        """配对请求创建配对码。"""
        from cortexos.auth.pairing import pair_request
        db, path = await self._make_backend()
        try:
            cfg = Config()
            result = await pair_request("test-agent", db, cfg)
            assert "code" in result
            assert len(result["code"]) == cfg.pair.code_length
            assert "agent_id" in result

            # 验证 code 存入了 DB
            pair = await db.get_pair_code(result["code"])
            assert pair is not None
            assert pair["agent_name"] == "test-agent"
            assert pair["status"] == "pending"
        finally:
            import os; os.unlink(path)

    async def test_pair_code_expired(self):
        """配对码过期后拒绝。"""
        from cortexos.auth.pairing import pair_request, pair_exchange
        db, path = await self._make_backend()
        try:
            cfg = Config()
            cfg.pair.code_expire_minutes = -1  # 立即过期
            result = await pair_request("test-agent", db, cfg)
            code = result["code"]

            # 尝试兑换 → 过期应返回 None
            exchanged = await pair_exchange(code, db, cfg)
            assert exchanged is None
        finally:
            import os; os.unlink(path)

    async def test_pair_code_double_exchange_rejected(self):
        """配对码二次兑换拒绝。"""
        from cortexos.auth.pairing import (
            pair_request, pair_approve, pair_exchange,
        )
        db, path = await self._make_backend()
        try:
            cfg = Config()
            result = await pair_request("test-agent", db, cfg)
            code = result["code"]

            # 批准
            approved = await pair_approve(code, {"agent:test-agent": "readwrite"}, db, cfg)
            assert approved is not None

            # 第一次兑换
            key1 = await pair_exchange(code, db, cfg)
            assert key1 is not None

            # 第二次兑换 → 应拒绝（状态已变）
            key2 = await pair_exchange(code, db, cfg)
            assert key2 is None
        finally:
            import os; os.unlink(path)

    async def test_pair_full_flow(self):
        """完整配对流程：request → approve → exchange。"""
        from cortexos.auth.pairing import (
            pair_request, pair_approve, pair_exchange,
        )
        db, path = await self._make_backend()
        try:
            cfg = Config()
            # Request
            req = await pair_request("my-agent", db, cfg)
            assert req["code"]

            # Approve（只批准，不颁发密钥）
            approved = await pair_approve(
                req["code"],
                {"agent:my-agent": "readwrite"},
                db, cfg,
            )
            assert approved is not None
            assert approved["code"] == req["code"]
            assert approved["agent_id"]
            assert approved["scopes"] == ["agent:my-agent"]

            # Exchange（颁发密钥，secret 仅此一次可见）
            exchanged = await pair_exchange(req["code"], db, cfg)
            assert exchanged is not None
            assert exchanged["key_id"]
            assert len(exchanged["secret"]) > 20  # secrets.token_urlsafe(32)

            # 二次兑换被拒（配对码已 used）
            again = await pair_exchange(req["code"], db, cfg)
            assert again is None
        finally:
            import os; os.unlink(path)

    async def test_approve_invalid_code(self):
        """批准无效 code → None。"""
        from cortexos.auth.pairing import pair_approve
        db, path = await self._make_backend()
        try:
            result = await pair_approve("INVALID", {}, db, Config())
            assert result is None
        finally:
            import os; os.unlink(path)


# ────────────────────── 密钥管理测试 ──────────────────────


@pytest.mark.asyncio
class TestKeys:
    """密钥管理测试。"""

    async def _make_setup(self):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        from cortexos.storage.sqlite_backend import SqliteBackend
        db = SqliteBackend(path)
        await db.initialize()
        # 注册 Agent
        await db.upsert_agent({
            "agent_id": "ag_test", "agent_name": "Test",
            "created_at": time.time(),
        })
        return db, path

    async def test_create_key(self):
        """创建密钥。"""
        from cortexos.auth.keys import create_key
        db, path = await self._make_setup()
        try:
            result = await create_key(
                "ag_test",
                {"agent:ag_test": "readwrite"},
                db,
            )
            assert result["key_id"].startswith("ak_")
            assert len(result["secret"]) > 20

            # 验证 hash 存储
            from cortexos.auth.pairing import _hash_secret
            stored = await db.get_agent_key_by_hash(_hash_secret(result["secret"]))
            assert stored is not None
        finally:
            import os; os.unlink(path)

    async def test_revoke_key(self):
        """吊销密钥。"""
        from cortexos.auth.keys import create_key, revoke_key
        db, path = await self._make_setup()
        try:
            result = await create_key("ag_test", {}, db)
            ok = await revoke_key(result["key_id"], db)
            assert ok

            key = await db.get_agent_key_by_id(result["key_id"])
            assert key["status"] == "revoked"
        finally:
            import os; os.unlink(path)

    async def test_list_keys(self):
        """列出密钥。"""
        from cortexos.auth.keys import create_key, list_keys
        db, path = await self._make_setup()
        try:
            await create_key("ag_test", {"a": "read"}, db)
            await create_key("ag_test", {"b": "write"}, db)
            keys = await list_keys("ag_test", db)
            assert len(keys) == 2
            # 不包含 hash
            assert "key_hash" not in keys[0]
        finally:
            import os; os.unlink(path)

    async def test_authenticate_valid_key(self):
        """有效密钥认证。"""
        from cortexos.auth.keys import create_key, authenticate
        db, path = await self._make_setup()
        try:
            result = await create_key("ag_test", {"a": "read"}, db)
            key_data = await authenticate(result["secret"], db)
            assert key_data is not None
            assert key_data["key_id"] == result["key_id"]
        finally:
            import os; os.unlink(path)

    async def test_authenticate_invalid_key(self):
        """无效密钥认证。"""
        from cortexos.auth.keys import authenticate
        db, path = await self._make_setup()
        try:
            key_data = await authenticate("invalid_secret_xxx", db)
            assert key_data is None
        finally:
            import os; os.unlink(path)

    async def test_authenticate_revoked_key(self):
        """吊销密钥认证失败。"""
        from cortexos.auth.keys import create_key, revoke_key, authenticate
        db, path = await self._make_setup()
        try:
            result = await create_key("ag_test", {}, db)
            await revoke_key(result["key_id"], db)
            key_data = await authenticate(result["secret"], db)
            assert key_data is None
        finally:
            import os; os.unlink(path)

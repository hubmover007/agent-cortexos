"""API 模块 —— 依赖项（Bearer 认证 + scope 权限校验 + 限流）。

通过 FastAPI Depends 自动注入 backend 和 config。

认证模型：
- pair/request、pair/confirm、pair/exchange 为引导流程，公开访问（无 token）
- 其余所有数据端点必须携带 Bearer token（AgentKey）
- verify_bearer：token 有效性 + 限流（429）
- require_scope / assert_scope：scope 级权限判定（403）
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

# 简单内存限流：key_id → [timestamps]
_rate_buckets: Dict[str, list] = {}
_rate_lock = asyncio.Lock()


def get_backend(request: Request):
    """从 app.state 注入后端。"""
    backend = getattr(request.app.state, "backend", None)
    if not backend:
        raise HTTPException(500, "未注入 backend")
    return backend


def get_config(request: Request):
    """从 app.state 注入配置。"""
    config = getattr(request.app.state, "config", None)
    if not config:
        raise HTTPException(500, "未注入 config")
    return config


def get_embedder(request: Request):
    """从 app.state 注入 embedder。"""
    embedder = getattr(request.app.state, "embedder", None)
    if not embedder:
        raise HTTPException(500, "未注入 embedder")
    return embedder


async def _check_rate_limit(key_id: str, rate_limit: int) -> bool:
    """检查限流（每分钟允许请求数）。"""
    now = time.time()
    async with _rate_lock:
        bucket = _rate_buckets.setdefault(key_id, [])
        # 只保留最近 60 秒的时间戳，同时防止 bucket 无限增长
        bucket[:] = [t for t in bucket if now - t < 60]
        if len(bucket) >= rate_limit:
            return False
        bucket.append(now)
        return True


def _parse_key(key_data: Dict):
    """将 DB 行字典解析为 AgentKey 模型（JSON 字段反序列化）。"""
    from cortexos.models import AgentKey
    return AgentKey.from_row(key_data)


async def verify_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None,
) -> Dict:
    """验证 Bearer token，返回解析后的 agent_key 信息。

    Raises:
        HTTPException(401): token 缺失或无效。
        HTTPException(429): 超过限流。
    """
    if credentials is None:
        raise HTTPException(401, "Missing bearer token")
    if request is None:
        raise HTTPException(401, "No request context")

    backend = get_backend(request)
    from cortexos.auth.keys import authenticate as auth_key
    key_data = await auth_key(credentials.credentials, backend)
    if not key_data:
        raise HTTPException(401, "Invalid token")

    # 限流（按 key 的 rate_limit）
    rate_limit = int(key_data.get("rate_limit", 100))
    if not await _check_rate_limit(key_data["key_id"], rate_limit):
        raise HTTPException(429, "Rate limit exceeded")

    key = _parse_key(key_data)
    return {
        "key_id": key.key_id,
        "agent_id": key.agent_id,
        "key_hash": key.key_hash,
        "scope_permissions": key.scope_permissions,
        "zone_overrides": key.zone_overrides,
        "rate_limit": key.rate_limit,
        "expires_at": key.expires_at,
        "created_at": key.created_at,
        "last_used": key.last_used,
        "status": key.status,
    }


def assert_scope(key_data: Dict, scope: str, action: str) -> None:
    """校验 key 对 scope 是否有 action 权限，无则抛 403。

    Args:
        key_data: verify_bearer 返回的 key 信息。
        scope: 目标 scope。
        action: "read" | "write"。
    """
    from cortexos.auth.permissions import check_permission
    key = _parse_key(key_data)
    if not check_permission(key, scope, "*", action):
        raise HTTPException(
            403, f"No '{action}' permission on scope '{scope}'"
        )


def require_scope(scope: str, action: str):
    """依赖工厂：要求 key 对固定 scope 有指定权限。

    Args:
        scope: 目标 scope（静态值）。
        action: "read" | "write"。
    """
    async def _dep(key_data: Dict = Depends(verify_bearer)) -> Dict:
        assert_scope(key_data, scope, action)
        return key_data
    return _dep

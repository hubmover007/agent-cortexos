"""API 模块 —— 依赖项（Bearer 认证 + scope 校验 + 限流）。
通过 FastAPI Depends 自动注入 backend 和 config。
"""

from __future__ import annotations

import time
from typing import Dict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import asyncio
import hashlib
import json

security = HTTPBearer()

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
        bucket = [t for t in bucket if now - t < 60]
        if len(bucket) >= rate_limit:
            return False
        bucket.append(now)
        _rate_buckets[key_id] = bucket
        return True


async def verify_bearer(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> Dict:
    """验证 Bearer token，返回 agent_key 信息。

    Raises:
        HTTPException(401): token 无效。
    """
    if request is None:
        raise HTTPException(401, "No request context")

    backend = get_backend(request)
    from cortexos.auth.keys import authenticate as auth_key
    key_data = await auth_key(credentials.credentials, backend)
    if not key_data:
        raise HTTPException(401, "Invalid token")
    return key_data


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()

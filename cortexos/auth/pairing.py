"""配对流程 —— pair_code → agent_key。

严格按照 code-design-p1.md §3.7 伪代码实现。

配对流程：
1. pair_request：Agent 发起配对 → 生成 8 字符 code（15 分钟过期）
2. pair_approve：管理员确认 → 生成 agent_key（仅此一次可见）
3. pair_exchange：Agent 用 code 换取 agent_key
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import string
import time
from typing import Dict, Optional, Tuple

from cortexos.config import Config
from cortexos.storage import StorageBackend

# 配对码字符集（排除易混字符 0/O/I/l/1）
_PAIR_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def _generate_pair_code(length: int = 8) -> str:
    """生成随机配对码。

    Args:
        length: 码长度（默认 8）。

    Returns:
        配对码字符串。
    """
    return "".join(secrets.choice(_PAIR_ALPHABET) for _ in range(length))


def _generate_agent_id() -> str:
    """生成 Agent ID（ag_ + 12 随机字符）。"""
    return "ag_" + "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(12))


def _generate_key_id() -> str:
    """生成 Key ID（ak_ + 16 随机字符）。"""
    return "ak_" + "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(16))


def _hash_secret(secret: str) -> str:
    """sha256 哈希密钥。

    Args:
        secret: 原始密钥。

    Returns:
        hex 摘要字符串。
    """
    return hashlib.sha256(secret.encode()).hexdigest()


async def pair_request(
    agent_name: str,
    backend: StorageBackend,
    config: Config,
) -> Dict[str, str]:
    """发起配对请求 —— 生成一次性配对码。

    Args:
        agent_name: Agent 名称。
        backend: 存储后端。
        config: 配置。

    Returns:
        {"code": "...", "agent_id": "...", "expires_in": seconds}
    """
    agent_id = _generate_agent_id()
    code = _generate_pair_code(config.pair.code_length)
    expires_at = time.time() + config.pair.code_expire_minutes * 60

    await backend.upsert_agent({
        "agent_id": agent_id,
        "agent_name": agent_name,
        "created_at": time.time(),
        "status": "active",
    })

    await backend.upsert_pair_code({
        "code": code,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "expires_at": expires_at,
        "status": "pending",
    })

    await backend.write_audit_log({
        "ts": time.time(),
        "agent_id": agent_id,
        "action": "pair_request",
        "detail": f"Agent {agent_name} 请求配对",
    })

    return {
        "code": code,
        "agent_id": agent_id,
        "expires_in": config.pair.code_expire_minutes * 60,
    }


async def pair_approve(
    code: str,
    scope_permissions: Dict[str, str],
    backend: StorageBackend,
    config: Config,
) -> Optional[Dict[str, str]]:
    """管理员确认配对 —— 将配对码标记为 approved（不直接颁发密钥）。

    Args:
        code: 配对码。
        scope_permissions: 授予的 scope 权限 {scope: "read"|"write"|"readwrite"}。
        backend: 存储后端。
        config: 配置。

    Returns:
        {"code": "...", "agent_id": "...", "scopes": [...]} 或 None（code 无效/过期）。
    """
    pair = await backend.get_pair_code(code)
    if pair is None:
        return None

    now = time.time()
    if now > pair["expires_at"] or pair["status"] != "pending":
        # 标记过期
        await backend.upsert_pair_code({
            **pair,
            "status": "expired",
        })
        await backend.write_audit_log({
            "ts": now,
            "agent_id": pair["agent_id"],
            "action": "pair_approve_failed",
            "detail": f"配对码 {code} 已过期或已使用",
        })
        return None

    # 批准：记录管理员授予的 scope 权限，等待 exchange 时颁发密钥
    await backend.upsert_pair_code({
        **pair,
        "status": "approved",
        "scope_permissions": scope_permissions,
    })

    await backend.write_audit_log({
        "ts": now,
        "agent_id": pair["agent_id"],
        "action": "pair_approve",
        "detail": f"Agent {pair['agent_name']} 配对已批准",
    })

    return {
        "code": code,
        "agent_id": pair["agent_id"],
        "scopes": list(scope_permissions.keys()),
    }


async def pair_exchange(
    code: str,
    backend: StorageBackend,
    config: Config,
) -> Optional[Dict[str, str]]:
    """Agent 凭配对码换取 agent_key（secret 仅此一次可见）。

    Args:
        code: 配对码。
        backend: 存储后端。
        config: 配置。

    Returns:
        {"key_id": "...", "secret": "...", "agent_id": "..."} 或 None。
    """
    pair = await backend.get_pair_code(code)
    if pair is None:
        return None

    now = time.time()
    if now > pair["expires_at"]:
        return None

    if pair["status"] != "approved":
        return None

    # 颁发 agent_key（secret 仅此一次可见）
    key_id = _generate_key_id()
    secret = secrets.token_urlsafe(32)
    key_hash = _hash_secret(secret)
    scope_permissions = pair.get("scope_permissions") or {}

    await backend.upsert_agent_key({
        "key_id": key_id,
        "agent_id": pair["agent_id"],
        "key_hash": key_hash,
        "scope_permissions": scope_permissions,
        "zone_overrides": {},
        "rate_limit": config.rate_limit.default,
        "expires_at": None,
        "created_at": now,
        "last_used": None,
        "status": "active",
    })

    # 标记配对码已使用（防二次兑换）
    await backend.upsert_pair_code({
        **pair,
        "status": "used",
    })

    await backend.write_audit_log({
        "ts": now,
        "agent_id": pair["agent_id"],
        "key_id": key_id,
        "action": "pair_exchange",
        "detail": f"Agent {pair['agent_name']} 已兑换密钥",
    })

    return {
        "key_id": key_id,
        "secret": secret,
        "agent_id": pair["agent_id"],
    }

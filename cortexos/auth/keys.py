"""密钥管理 —— hash 存储 / 吊销 / 轮换 / 限流。

key 只存 sha256 hash，不存明文。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from cortexos.auth.pairing import _generate_key_id, _generate_agent_id, _hash_secret
from cortexos.storage import StorageBackend

import secrets


async def create_key(
    agent_id: str,
    scope_permissions: Dict[str, str],
    backend: StorageBackend,
    rate_limit: int = 100,
    expires_at: Optional[float] = None,
) -> Dict[str, str]:
    """创建新密钥。

    Args:
        agent_id: Agent ID。
        scope_permissions: 权限 {scope: "read"|"write"|"readwrite"}。
        backend: 存储后端。
        rate_limit: 限流（默认 100）。
        expires_at: 过期时间戳。

    Returns:
        {"key_id": "...", "secret": "..."}（secret 仅此一次可见）。
    """
    key_id = _generate_key_id()
    secret = secrets.token_urlsafe(32)
    key_hash = _hash_secret(secret)

    await backend.upsert_agent_key({
        "key_id": key_id,
        "agent_id": agent_id,
        "key_hash": key_hash,
        "scope_permissions": scope_permissions,
        "zone_overrides": {},
        "rate_limit": rate_limit,
        "expires_at": expires_at,
        "created_at": time.time(),
        "last_used": None,
        "status": "active",
    })

    await backend.write_audit_log({
        "ts": time.time(),
        "agent_id": agent_id,
        "key_id": key_id,
        "action": "key_create",
        "detail": f"为 Agent {agent_id} 创建新密钥",
    })

    return {"key_id": key_id, "secret": secret}


async def revoke_key(key_id: str, backend: StorageBackend) -> bool:
    """吊销密钥。

    Args:
        key_id: 密钥 ID。
        backend: 存储后端。

    Returns:
        是否成功。
    """
    key = await backend.get_agent_key_by_id(key_id)
    if not key:
        return False

    await backend.upsert_agent_key({
        **key,
        "status": "revoked",
    })

    await backend.write_audit_log({
        "ts": time.time(),
        "agent_id": key["agent_id"],
        "key_id": key_id,
        "action": "key_revoke",
        "detail": f"吊销密钥 {key_id}",
    })

    return True


async def rotate_key(
    key_id: str,
    backend: StorageBackend,
) -> Optional[Dict[str, str]]:
    """轮换密钥（吊销旧的 + 生成新的）。

    Args:
        key_id: 要轮换的密钥 ID。
        backend: 存储后端。

    Returns:
        新密钥信息 {"key_id": "...", "secret": "..."} 或 None。
    """
    key = await backend.get_agent_key_by_id(key_id)
    if not key:
        return None

    # 吊销旧 key
    await revoke_key(key_id, backend)

    # 创建新 key（相同权限）
    return await create_key(
        agent_id=key["agent_id"],
        scope_permissions=key.get("scope_permissions", {}),
        backend=backend,
    )


async def list_keys(agent_id: str, backend: StorageBackend) -> List[Dict]:
    """列出 Agent 的所有密钥。

    Args:
        agent_id: Agent ID。
        backend: 存储后端。

    Returns:
        密钥列表（不含 hash）。
    """
    keys = await backend.list_agent_keys(agent_id)
    # 脱敏：不返回 key_hash
    result = []
    for k in keys:
        result.append({
            "key_id": k["key_id"],
            "agent_id": k["agent_id"],
            "scope_permissions": k["scope_permissions"],
            "rate_limit": k["rate_limit"],
            "expires_at": k["expires_at"],
            "created_at": k["created_at"],
            "last_used": k["last_used"],
            "status": k["status"],
        })
    return result


async def authenticate(
    secret: str,
    backend: StorageBackend,
) -> Optional[Dict]:
    """通过密钥认证。

    Args:
        secret: 密钥明文。
        backend: 存储后端。

    Returns:
        密钥信息字典或 None。
    """
    key_hash = _hash_secret(secret)
    key_data = await backend.get_agent_key_by_hash(key_hash)
    if not key_data:
        return None

    if key_data["status"] != "active":
        return None

    if key_data.get("expires_at") and time.time() > key_data["expires_at"]:
        return None

    # 更新 last_used
    await backend.upsert_agent_key({
        **key_data,
        "last_used": time.time(),
    })

    return key_data

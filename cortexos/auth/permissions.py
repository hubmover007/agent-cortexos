"""权限判定 —— scope 主 + zone 覆盖两层模型。

严格按照 code-design-p1.md §3.7 伪代码实现。

判定流程：
1. Zone 覆盖命中 → 用覆盖权限（单调递减约束）
2. Scope 权限命中 → 用 scope 权限
3. 都不命中 → 拒绝（403）

单调递减约束：zone 覆盖 rank ≤ scope rank，zone 覆盖只允许 read。
"""

from __future__ import annotations

from typing import Dict, Optional

from cortexos.models import AgentKey


def _rank(permission: str) -> int:
    """权限等级：read=1, write=2, readwrite=2。"""
    if permission in ("write", "readwrite"):
        return 2
    if permission == "read":
        return 1
    return 0


def check_permission(
    key: AgentKey,
    scope: str,
    zone: str,
    action: str,
) -> bool:
    """检查 AgentKey 是否有权限执行操作。

    判定顺序（code-design-p1.md §3.7）：
    1. Zone 覆盖（最精确，单调递减约束）
    2. Scope 权限
    3. 拒绝

    Args:
        key: Agent 密钥。
        scope: 目标 scope。
        zone: 目标 zone。
        action: 操作（"read"|"write"）。

    Returns:
        是否允许。
    """
    # ① Zone 覆盖
    zone_override = key.get_zone_override(scope, zone)
    if zone_override is not None:
        # 单调递减约束：zone 覆盖 rank ≤ scope rank（不能越权）
        scope_perm = key.scope_permissions.get(scope)
        scope_rank = _rank(scope_perm) if scope_perm else 0
        override_rank = _rank(zone_override)
        # zone 覆盖只允许 ≤ scope rank
        if override_rank > scope_rank:
            return False
        # zone 覆盖默认收紧为只读；readwrite 覆盖（且 scope 允许）才放行写
        if action == "write":
            return zone_override == "readwrite"
        return _rank(zone_override) >= _rank(action)

    # ② Scope 权限
    perm = key.scope_permissions.get(scope)
    if perm:
        return _rank(perm) >= _rank(action)

    # ③ 拒绝
    return False


def validate_zone_override(
    override_permission: str,
    scope_permission: str,
) -> bool:
    """验证 Zone 覆盖是否满足单调递减约束。

    约束：override rank ≤ scope rank。
    即 scope 只读时，zone 覆盖不能写。

    Args:
        override_permission: Zone 覆盖权限。
        scope_permission: Scope 权限。

    Returns:
        是否有效。
    """
    override_rank = _rank(override_permission)
    scope_rank = _rank(scope_permission)
    return override_rank <= scope_rank

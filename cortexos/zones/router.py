"""Zone 三层路由 —— 实体精确匹配 → 语义质心匹配 → _inbox 兜底。

严格按照 code-design-p1.md §3.1 伪代码实现。
"""

from __future__ import annotations

import math
from typing import List, Optional

from cortexos.config import Config
from cortexos.embedding.base import Embedder
from cortexos.models import Entry, Zone


async def route_entry(
    entry: Entry,
    zones: List[Zone],
    embedder: "Embedder",
    config: Config,
) -> str:
    """为条目执行三层路由，返回 Zone 名称。

    Args:
        entry: 待路由条目。
        zones: 当前 active zones 列表。
        embedder: Embedder 实例。
        config: 配置（含 semantic_threshold）。

    Returns:
        Zone 名称。
    """
    threshold = config.zone.emergence.semantic_threshold

    # Layer 1: 实体精确匹配
    zone_name = _route_by_entity(entry, zones)
    if zone_name:
        return zone_name

    # Layer 2: 语义匹配（质心余弦相似度）
    if entry.embedding:
        zone_name = _route_by_semantic(entry, zones, embedder, threshold)
        if zone_name:
            return zone_name

    # Layer 3: 兜底
    return "_inbox"


def _route_by_entity(entry: Entry, zones: List[Zone]) -> Optional[str]:
    """Layer 1: 实体精确匹配 —— 取重叠实体最多的 zone。

    Args:
        entry: 条目。
        zones: active zones。

    Returns:
        Zone 名称或 None。
    """
    if not entry.entities:
        return None

    best_zone: Optional[str] = None
    best_overlap = 0

    for z in zones:
        if z.status != "active":
            continue
        overlap = len(set(entry.entities) & set(z.entities))
        if overlap > best_overlap:
            best_zone = z.name
            best_overlap = overlap

    if best_zone and best_overlap >= 1:
        return best_zone
    return None


def _route_by_semantic(
    entry: Entry,
    zones: List[Zone],
    embedder: Embedder,
    threshold: float,
) -> Optional[str]:
    """Layer 2: 语义匹配 —— embedding 与 zone 质心余弦相似度。

    Args:
        entry: 条目（需有 embedding）。
        zones: active zones。
        embedder: Embedder。
        threshold: 语义匹配阈值（默认 0.72）。

    Returns:
        Zone 名称或 None。
    """
    if not entry.embedding:
        return None

    best_zone: Optional[str] = None
    best_sim = 0.0

    for z in zones:
        if z.status != "active" or z.centroid is None:
            continue
        sim = embedder.cosine_similarity(entry.embedding, z.centroid)
        if sim > best_sim:
            best_sim = sim
            best_zone = z.name

    if best_zone and best_sim >= threshold:
        return best_zone
    return None


def route_semantic_only(
    entry: Entry,
    zones: List[Zone],
    embedder: Embedder,
    threshold: float = 0.72,
) -> Optional[str]:
    """仅语义路由（用于涌现阶段增量归并）。

    Args:
        entry: 条目。
        zones: active zones。
        embedder: Embedder。
        threshold: 阈值。

    Returns:
        Zone 名称或 None。
    """
    return _route_by_semantic(entry, zones, embedder, threshold)

"""Zone 引擎 —— 涌现、合并、归档、重力计算。

严格按照 code-design-p1.md §3.2 伪代码实现。
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

from cortexos.config import Config
from cortexos.embedding.base import Embedder
from cortexos.models import Entry, Zone
from cortexos.zones.cluster import (
    centroid_similarity,
    cluster_entries,
    compute_centroid,
    find_best_cluster,
)
from cortexos.zones.router import route_semantic_only


# ── 涌现 ──


async def emergence_scan(
    inbox_entries: List[Entry],
    zones: List[Zone],
    embedder: "Embedder",
    config: Config,
    total_entries: int,
) -> List[Zone]:
    """扫描 _inbox，执行涌现流程。

    算法流程（code-design-p1.md §3.2）：
    1. 质量门过滤
    2. 增量归入现有 zone
    3. 剩余条目连通聚类
    4. 自适应阈值涌现判定
    5. 创建新 zone

    Args:
        inbox_entries: _inbox 中的条目列表。
        zones: 当前 active zones。
        embedder: Embedder。
        config: 配置。
        total_entries: 系统总条目数。

    Returns:
        新创建的 Zone 列表。
    """
    new_zones: List[Zone] = []

    # Step 1: 质量门
    candidates = [
        e for e in inbox_entries
        if len(e.content) >= config.zone.emergence.min_content_len
        and e.embedding is not None
    ]
    if not candidates:
        return new_zones

    # Step 2: 增量归并（先尝试归入现有 zone）
    remaining: List[Entry] = []
    for e in candidates:
        matched = route_semantic_only(
            e, zones, embedder, config.zone.emergence.semantic_threshold
        )
        if matched:
            # 更新 zone 质心（增量滚动更新）
            for z in zones:
                if z.name == matched:
                    _update_zone_centroid_incremental(z, e)
                    z.entry_count += 1
                    break
        else:
            remaining.append(e)

    # Step 3: 剩余条目连通聚类
    clusters = cluster_entries(
        remaining,
        similarity_threshold=config.zone.emergence.cluster_similarity,
        embedder=embedder,
    )

    # Step 4-5: 自适应阈值 + 涌现判定
    base = config.zone.emergence.base_threshold
    scale = max(1.0, math.log2(max(1, total_entries / 100)))
    threshold = base * scale

    for cluster in clusters:
        if len(cluster) >= threshold:
            zone = _create_zone_from_cluster(cluster, config)
            new_zones.append(zone)

    return new_zones


# ── 合并 ──


async def merge_zones(
    zones: List[Zone],
    embedder: "Embedder",
    config: Config,
) -> List[tuple[str, str]]:
    """检测并合并相似 Zone。

    返回 [(source_zone_name, target_zone_name)] 的合并列表。

    Args:
        zones: 所有 zones。
        embedder: Embedder。
        config: 配置。

    Returns:
        合并对列表。
    """
    threshold = config.zone.lifecycle.merge_threshold
    active_zones = [z for z in zones if z.status == "active"]
    merges: List[tuple[str, str]] = []

    for i, z1 in enumerate(active_zones):
        if z1.centroid is None:
            continue
        for z2 in active_zones[i + 1:]:
            if z2.centroid is None:
                continue
            sim = embedder.cosine_similarity(z1.centroid, z2.centroid)
            if sim >= threshold:
                # 小 zone 合并到大 zone
                if z1.entry_count >= z2.entry_count:
                    merges.append((z2.name, z1.name))
                else:
                    merges.append((z1.name, z2.name))

    return merges


# ── 生命周期 ──


async def lifecycle_check(
    zones: List[Zone],
    config: Config,
    now: Optional[float] = None,
) -> List[tuple[str, str]]:
    """检查并更新 Zone 生命周期状态。

    规则：
    - active → dormant：last_access 超过 dormant_days 天
    - dormant → archived：last_access 超过 archive_days 天
    - pinned zones 不归档

    Args:
        zones: 所有 zones。
        config: 配置。
        now: 当前时间戳（默认 time.time()）。

    Returns:
        [(zone_name, new_status)] 的状态变更列表。
    """
    if now is None:
        now = time.time()

    dormant_sec = config.zone.lifecycle.dormant_days * 86400
    archive_sec = config.zone.lifecycle.archive_days * 86400
    changes: List[tuple[str, str]] = []

    for z in zones:
        if z.pinned:
            continue
        idle = now - z.last_access
        if z.status == "active" and idle >= dormant_sec:
            changes.append((z.name, "dormant"))
        elif z.status == "dormant" and idle >= archive_sec:
            changes.append((z.name, "archived"))

    return changes


# ── 重力 ──


def compute_gravity(
    entry_count: int,
    access_count: int,
    last_access: float,
    config: Config,
    now: Optional[float] = None,
) -> float:
    """计算 Zone 重力值。

    重力公式（方案文档 §5.4）：
    gravity = 新鲜度因子 × 活跃度因子 × 规模因子

    新鲜度：exp(-λ × days_since_last_access)
    活跃度：1 - exp(-access_count / k)
    规模：1 - exp(-entry_count / m)

    Args:
        entry_count: 条目数。
        access_count: 访问次数。
        last_access: 最后访问时间戳。
        config: 配置。
        now: 当前时间戳。

    Returns:
        重力值。
    """
    if now is None:
        now = time.time()

    lam = config.zone.gravity.decay_lambda
    k = config.zone.gravity.activity_k
    m = config.zone.gravity.scale_m

    days_since = (now - last_access) / 86400.0
    freshness = math.exp(-lam * days_since)
    activity = 1.0 - math.exp(-access_count / k)
    scale = 1.0 - math.exp(-entry_count / m)

    return freshness * activity * scale


def update_zone_gravity(zone: Zone, config: Config) -> None:
    """更新 Zone 的重力值。

    Args:
        zone: Zone 对象（原地修改）。
        config: 配置。
    """
    zone.gravity = compute_gravity(
        zone.entry_count, zone.entry_count, zone.last_access, config
    )


# ── 辅助 ──


async def _update_zone_centroid_incremental(zone: Zone, entry: Entry) -> None:
    """增量更新 Zone 质心（滚动均值）。

    公式：new_centroid = (n × old_centroid + new_vec) / (n + 1)

    Args:
        zone: Zone（原地修改）。
        entry: 新增条目。
    """
    if entry.embedding is None:
        return
    if zone.centroid is None:
        zone.centroid = entry.embedding[:]
    else:
        n = zone.entry_count
        dim = len(zone.centroid)
        for i in range(dim):
            zone.centroid[i] = (n * zone.centroid[i] + entry.embedding[i]) / (n + 1)


def _create_zone_from_cluster(
    cluster: List[Entry],
    config: Config,
) -> Zone:
    """从簇创建新 Zone。

    名称 = 簇中第一条目内容摘要（无 LLM 时用前 30 字符）
    质心 = 簇内 embedding 均值
    重力 = 平均新鲜度

    Args:
        cluster: 条目簇。
        config: 配置。

    Returns:
        新 Zone 对象。
    """
    # 名称：第一条目内容前 30 字符
    name_base = cluster[0].content[:30].replace("\n", " ").strip()
    scope = cluster[0].scope

    # 确保名称唯一
    name = name_base
    now = time.time()

    # 实体：簇内 Top-5 高频实体
    from collections import Counter
    entity_counter = Counter()
    for e in cluster:
        for ent in e.entities:
            entity_counter[ent] += 1
    top_entities = [e for e, _ in entity_counter.most_common(5)]

    # 关键词：同实体
    keywords = top_entities[:]

    # 质心：embedding 均值
    centroid = compute_centroid([e.embedding for e in cluster if e.embedding])

    # 重力
    gravity = compute_gravity(
        entry_count=len(cluster),
        access_count=len(cluster),
        last_access=now,
        config=config,
    )

    zone = Zone(
        name=name,
        scope=scope,
        description=f"自动涌现 Zone（{len(cluster)} 条）",
        entities=top_entities,
        keywords=keywords,
        centroid=centroid,
        gravity=gravity,
        entry_count=len(cluster),
        status="active",
        pinned=0,
        created_at=now,
        last_access=now,
    )
    return zone

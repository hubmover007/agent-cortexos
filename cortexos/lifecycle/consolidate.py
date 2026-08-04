"""整合引擎 —— 四阶段 Orient/Gather/Consolidate/Prune + 三门控。

严格按照 code-design-p1.md §3.6 伪代码实现。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from cortexos.config import Config
from cortexos.models import Entry, Zone
from cortexos.storage import StorageBackend


class ConsolidateEngine:
    """记忆整合引擎。

    四阶段：
    1. Orient：读取 Zone 现状
    2. Gather：扫描新增 raw 条目
    3. Consolidate：相似 raw 分组 → episode 总结（可选 LLM）
    4. Prune：Zone 归档 + 索引预算修剪

    三门控：
    - 时间门：距上次整合 ≥ time_gate_hours 小时
    - 内容门：新增条目 ≥ content_gate_count 条
    - 锁门：进程内互斥
    """

    def __init__(self, backend: StorageBackend, config: Config):
        """初始化整合引擎。

        Args:
            backend: 存储后端。
            config: 配置。
        """
        self._backend = backend
        self._config = config
        self._last_consolidate: Dict[str, float] = {}  # scope → last_ts
        self._lock: Dict[str, bool] = {}

    async def consolidate(self, scope: str) -> Dict[str, int]:
        """对指定 scope 执行整合。

        Args:
            scope: 目标 scope。

        Returns:
            {"raw_scanned": int, "episodes_created": int, "archived_zones": int}
        """
        cfg = self._config.consolidate
        now = time.time()

        # 门控
        if not self._check_gates(scope, now):
            return {"raw_scanned": 0, "episodes_created": 0, "archived_zones": 0}

        # 获取锁
        if not self._acquire_lock(scope):
            return {"raw_scanned": 0, "episodes_created": 0, "archived_zones": 0}

        try:
            stats = {"raw_scanned": 0, "episodes_created": 0, "archived_zones": 0}

            # ── Orient ──
            zones = await self._backend.list_zones(scope=scope, status="active")

            # ── Gather ──
            since = self._last_consolidate.get(scope, 0)
            raw_entries: List[Entry] = []
            async for e in self._backend.scan_entries(
                scope=scope, layer="raw", since=since,
            ):
                raw_entries.append(e)
                stats["raw_scanned"] += 1

            # ── Consolidate（内容门：新增 raw 达到阈值才生成 episode）──
            if len(raw_entries) >= cfg.content_gate_count:
                from cortexos.zones.cluster import cluster_entries
                clusters = cluster_entries(
                    raw_entries, similarity_threshold=cfg.similarity_threshold,
                )
                superseded_ids: List[tuple[str, str]] = []
                for cluster in clusters:
                    if len(cluster) >= cfg.raw_summary_threshold:
                        # 创建 episode 条目（简化：取第一条内容作为总结）
                        episode_content = _make_episode_from_cluster(cluster)
                        episode = Entry(
                            content=episode_content,
                            scope=scope,
                            zone=cluster[0].zone,
                            layer="episode",
                            entities=list(set(
                                e2 for e in cluster for e2 in e.entities
                            )),
                            metadata={"source_count": len(cluster)},
                        )
                        await self._backend.upsert_entry(episode)
                        stats["episodes_created"] += 1
                        for e in cluster:
                            superseded_ids.append((e.id, "superseded"))

                if superseded_ids:
                    await self._backend.bulk_update_status(superseded_ids)

            # 事实冲突处理
            await self._resolve_scope_facts(scope)

            # ── Prune ──
            archived = await self._prune_zones(scope)
            stats["archived_zones"] = archived

            self._last_consolidate[scope] = now
            return stats

        finally:
            self._release_lock(scope)

    def _check_gates(self, scope: str, now: float) -> bool:
        """检查门控条件。

        Args:
            scope: scope。
            now: 当前时间。

        Returns:
            是否允许整合。
        """
        cfg = self._config.consolidate
        last = self._last_consolidate.get(scope, 0)
        hours_since = (now - last) / 3600.0
        return hours_since >= cfg.time_gate_hours

    def _acquire_lock(self, scope: str) -> bool:
        """尝试获取作用域锁。

        Args:
            scope: scope。

        Returns:
            是否获取成功。
        """
        if self._lock.get(scope, False):
            return False
        self._lock[scope] = True
        return True

    def _release_lock(self, scope: str) -> None:
        """释放作用域锁。"""
        self._lock[scope] = False

    async def _resolve_scope_facts(self, scope: str) -> None:
        """处理 scope 内的所有事实冲突。

        每组 subject+predicate 只消解一次（最新一条作为新事实），
        被截断/合并修改的旧事实必须写回，否则时间窗口截断不落库。
        """
        from cortexos.lifecycle.resolve import resolve_fact
        facts = await self._backend.find_facts(scope=scope, status="active")
        # 按 subject+predicate 分组
        groups: Dict[tuple, List] = {}
        for f in facts:
            key = (f.subject, f.predicate)
            groups.setdefault(key, []).append(f)

        for group in groups.values():
            if len(group) < 2:
                continue
            # 按生效时间升序：最新一条作为新事实，其余作为旧事实
            group.sort(key=lambda f: f.valid_from or 0)
            new_f = group[-1]
            others = group[:-1]
            before = {old.id: old.valid_until for old in others}
            resolved = await resolve_fact(new_f, others, self._config)
            await self._backend.upsert_fact(resolved)
            # 写回被截断/合并修改的旧事实
            for old in others:
                if old.status != "active" or old.valid_until != before.get(old.id):
                    await self._backend.upsert_fact(old)

    async def _prune_zones(self, scope: str) -> int:
        """归档过期 Zone。"""
        from cortexos.zones.engine import lifecycle_check
        zones = await self._backend.list_zones(scope=scope)
        changes = await lifecycle_check(zones, self._config)
        for zone_name, new_status in changes:
            zone = await self._backend.get_zone(zone_name, scope=scope)
            if zone:
                zone.status = new_status
                await self._backend.upsert_zone(zone)
        return len([c for c in changes if c[1] == "archived"])


def _make_episode_from_cluster(cluster: List[Entry]) -> str:
    """从 raw 条目簇生成 episode 内容（简化：拼接摘要）。

    Args:
        cluster: raw 条目簇。

    Returns:
        总结文本。
    """
    lines = [f"## 整合 episode（{len(cluster)} 条原始记录）\n"]
    for e in cluster[:5]:  # 最多取前 5 条
        lines.append(f"- {e.content[:100]}")
    return "\n".join(lines)

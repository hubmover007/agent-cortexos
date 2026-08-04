"""图遍历 —— BFS ≤2 跳 + 衰减。

严格按照 code-design-p1.md §3.4 伪代码实现。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set

from cortexos.models import Edge


class GraphIndex:
    """内存邻接表图索引。

    启动时从 edges 表加载，用于 BFS 图遍历检索。
    """

    def __init__(self):
        """初始化空图索引。"""
        # 邻接表：entity → [(target_entity, edge_weight), ...]
        self._out_edges: Dict[str, List[tuple[str, float]]] = defaultdict(list)
        self._in_edges: Dict[str, List[tuple[str, float]]] = defaultdict(list)
        # entry_id → entities（实体属于哪些条目）
        self._entry_entities: Dict[str, List[str]] = defaultdict(list)
        # entity → entry_ids（实体出现在哪些条目）
        self._entity_entries: Dict[str, List[str]] = defaultdict(list)

    def add_edge(self, edge: Edge) -> None:
        """添加关系边到图索引。

        Args:
            edge: 关系边。
        """
        self._out_edges[edge.source].append((edge.target, edge.weight))
        self._in_edges[edge.target].append((edge.source, edge.weight))
        if edge.entry_id:
            self._entry_entities[edge.entry_id].append(edge.source)
            self._entry_entities[edge.entry_id].append(edge.target)
            self._entity_entries[edge.source].append(edge.entry_id)
            self._entity_entries[edge.target].append(edge.entry_id)

    def load_edges(self, edges: List[Edge]) -> None:
        """批量加载边。

        Args:
            edges: 边列表。
        """
        for edge in edges:
            self.add_edge(edge)

    def graph_search(
        self,
        query_entities: List[str],
        hop: int = 2,
        decay: float = 0.5,
    ) -> Dict[str, float]:
        """BFS 图遍历检索。

        算法（code-design-p1.md §3.4）：
        1. 锚点 = 查询实体
        2. BFS 扩散，每跳衰减 0.5
        3. 路径得分 = Σ(边权重 × 0.5^(depth-1))

        Args:
            query_entities: 查询实体列表（锚点）。
            hop: 最大跳数（默认 2）。
            decay: 每跳衰减系数（默认 0.5）。

        Returns:
            {entry_id: graph_score} 映射。
        """
        scores: Dict[str, float] = {}
        visited: Set[str] = set()

        def bfs(entity: str, depth: int, cum_score: float) -> None:
            if depth > hop:
                return
            # 当前实体对应的条目
            for entry_id in self._entity_entries.get(entity, []):
                scores[entry_id] = max(scores.get(entry_id, 0.0), cum_score)

            if depth < hop:
                for target, weight in self._out_edges.get(entity, []):
                    if target not in visited:
                        visited.add(target)
                        bfs(target, depth + 1, cum_score + weight * (decay ** depth))
                for source, weight in self._in_edges.get(entity, []):
                    if source not in visited:
                        visited.add(source)
                        bfs(source, depth + 1, cum_score + weight * (decay ** depth))

        for anchor in query_entities:
            visited.add(anchor)
            bfs(anchor, 1, 1.0)  # depth=1 起始（锚点本身权重 1.0）

        return scores

    def entity_count(self) -> int:
        """实体数量。"""
        return len(self._out_edges)

    def edge_count(self) -> int:
        """边数量。"""
        return sum(len(v) for v in self._out_edges.values())

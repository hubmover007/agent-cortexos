"""混合检索 —— 四通道候选 + RRF 融合。

通道：向量余弦 / 词法 FTS5 / 实体精确 / 图遍历
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from cortexos.config import Config
from cortexos.embedding.base import Embedder
from cortexos.models import Entry, Zone
from cortexos.recall.graph import GraphIndex
from cortexos.recall.scoring import rank_entries, rrf_fusion
from cortexos.storage import StorageBackend


async def hybrid_retrieve(
    query: str,
    embedder: "Embedder",
    backend: StorageBackend,
    graph_index: "GraphIndex",
    config: Config,
    *,
    scope: Optional[str] = None,
    top_k: Optional[int] = None,
    query_entities: Optional[List[str]] = None,
    zone_gravities: Optional[Dict[str, float]] = None,
) -> List[Tuple[Entry, float]]:
    """混合检索主入口。

    四通道候选：
    1. 语义向量（embedding 余弦相似度）
    2. 词法 FTS5
    3. 实体精确匹配
    4. 图遍历

    各通道候选通过 RRF 融合，最后多因子评分排序。

    Args:
        query: 查询文本。
        embedder: Embedder 实例。
        backend: 存储后端。
        graph_index: 图索引。
        config: 配置。
        scope: 限定 scope。
        top_k: 返回数量（默认 20）。
        query_entities: 查询实体列表（用于实体通道和图通道）。
        zone_gravities: {zone_name: gravity} 映射。

    Returns:
        排序后的 [(entry, score)] 列表。
    """
    if top_k is None:
        top_k = config.recall.top_k

    k = config.recall.rrf_k
    half_life = config.recall.recency_half_life_days
    hop = config.recall.graph_hop
    decay = config.recall.graph_decay

    # ── 通道 1: 语义向量 ──
    ranked_semantic: List[str] = []
    try:
        query_vec = await embedder.embed_query(query)
        all_embs = await backend.search_all_embeddings(scope=scope)
        # 内存余弦相似度排序
        scored: List[Tuple[str, float]] = []
        for eid, emb in all_embs:
            if emb:
                sim = embedder.cosine_similarity(query_vec, emb)
                scored.append((eid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_semantic = [eid for eid, _ in scored[:top_k * 3]]
    except Exception:
        ranked_semantic = []

    # ── 通道 2: 词法 FTS5 ──
    lexical_results = await backend.search_lexical(query, scope=scope, top_k=top_k * 3)
    ranked_lexical = [r[0].id for r in lexical_results]

    # ── 通道 3: 实体精确 ──
    ranked_entity: List[str] = []
    if query_entities:
        # 简陋实现：扫描（实际应维护倒排索引）
        seen = set()
        entries = await backend.list_entries(scope=scope, limit=top_k * 10)
        for e in entries:
            if e.entities and any(e2 in query_entities for e2 in e.entities):
                if e.id not in seen:
                    seen.add(e.id)
                    ranked_entity.append(e.id)
        ranked_entity = ranked_entity[:top_k * 3]

    # ── 通道 4: 图遍历 ──
    graph_scores: Dict[str, float] = {}
    if query_entities:
        graph_scores = graph_index.graph_search(query_entities, hop=hop, decay=decay)
        ranked_graph = sorted(graph_scores, key=graph_scores.get, reverse=True)[:top_k * 3]  # type: ignore[arg-type]
    else:
        ranked_graph = []

    # ── RRF 融合 ──
    rrf_scores = rrf_fusion(
        [ranked_semantic, ranked_lexical, ranked_entity, ranked_graph],
        k=k,
    )

    # ── 收集所有候选条目 ──
    candidate_ids = set()
    for lid in [ranked_semantic, ranked_lexical, ranked_entity, ranked_graph]:
        for eid in lid:
            candidate_ids.add(eid)

    candidates: List[Entry] = []
    for eid in list(candidate_ids)[:top_k * 5]:
        entry = await backend.get_entry(eid)
        if entry and entry.status == "active":
            candidates.append(entry)

    # ── 加载 Zone 重力 ──
    if zone_gravities is None:
        zone_gravities = {}
        zones = await backend.list_zones(scope=scope, status="active")
        for z in zones:
            zone_gravities[z.name] = z.gravity

    # ── 多因子评分排序 ──
    results = rank_entries(
        candidates,
        rrf_scores,
        graph_scores,
        query_scope=scope or "",
        zone_gravities=zone_gravities,
        weights=config.recall.weights,
        half_life_days=half_life,
        top_k=top_k,
    )

    return results

"""多因子评分 —— RRF 融合 + 6 因子加权。

严格按照 code-design-p1.md §3.3 伪代码实现。
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from cortexos.config import Config, RecallWeights
from cortexos.models import Entry


def rrf_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> Dict[str, float]:
    """Reciprocal Rank Fusion（RRF）融合多通道候选。

    公式：RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        ranked_lists: 各通道排序的 entry_id 列表。
        k: 平滑参数（默认 60）。

    Returns:
        {entry_id: rrf_score} 映射。
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, eid in enumerate(ranked):
            score = 1.0 / (k + rank + 1)
            scores[eid] = scores.get(eid, 0.0) + score
    # 归一化到 [0, 1]
    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {eid: s / max_score for eid, s in scores.items()}
    return scores


def compute_text_sim(
    entry_id: str,
    semantic_scores: Dict[str, float],
    lexical_scores: Dict[str, float],
) -> float:
    """计算文本相似度：取语义余弦与词法 bm25 归一化分数的较大值。

    不再使用 RRF 排名分（排名只反映相对顺序，丢失相似度强度，
    会导致不相关条目也拿到接近满分）。

    Args:
        entry_id: 条目 ID。
        semantic_scores: 语义通道余弦相似度映射。
        lexical_scores: 词法通道归一化 bm25 分数映射。

    Returns:
        text_sim (0~1)。
    """
    return max(
        semantic_scores.get(entry_id, 0.0),
        lexical_scores.get(entry_id, 0.0),
    )


def compute_recency(created_at: float, half_life_days: float = 7.0) -> float:
    """计算时间新鲜度。

    公式：exp(-ln2 × age_days / half_life_days)

    Args:
        created_at: 创建时间戳（epoch 秒）。
        half_life_days: 半衰期天数。

    Returns:
        recency (0~1)。
    """
    now = time.time()
    age_days = (now - created_at) / 86400.0
    return math.exp(-math.log(2) * age_days / half_life_days)


def compute_freq(access_count: int) -> float:
    """计算访问频率归一化。

    公式：1 - exp(-access_count / 50)

    Args:
        access_count: 访问次数。

    Returns:
        freq (0~1)。
    """
    return 1.0 - math.exp(-access_count / 50.0)


def score_entry(
    entry: Entry,
    rrf_scores: Dict[str, float],
    graph_scores: Dict[str, float],
    query_scope: str,
    zone_gravity: float = 1.0,
    weights: Optional[RecallWeights] = None,
    half_life_days: float = 7.0,
    semantic_scores: Optional[Dict[str, float]] = None,
    lexical_scores: Optional[Dict[str, float]] = None,
) -> float:
    """多因子评分。

    公式（code-design-p1.md §3.3）：
    score = w1×text_sim + w2×recency + w3×gravity + w4×freq
           + w5×scope_boost + w6×graph_path

    Args:
        entry: 条目。
        rrf_scores: RRF 融合分（保留用于兼容/调试）。
        graph_scores: 图遍历得分。
        query_scope: 查询 scope。
        zone_gravity: Zone 重力值。
        weights: 权重配置。
        half_life_days: 半衰期天数。
        semantic_scores: 语义通道真实余弦相似度。
        lexical_scores: 词法通道归一化分数。

    Returns:
        综合得分。
    """
    if weights is None:
        weights = RecallWeights()

    text_sim = compute_text_sim(
        entry.id,
        semantic_scores or {},
        lexical_scores or {},
    )
    recency = compute_recency(entry.created_at, half_life_days)
    gravity = zone_gravity
    freq = compute_freq(entry.access_count)
    scope_boost = 0.1 if entry.scope == query_scope else 0.0
    graph_path = graph_scores.get(entry.id, 0.0)

    total = (
        weights.text_sim * text_sim
        + weights.recency * recency
        + weights.gravity * gravity
        + weights.freq * freq
        + weights.scope_boost * scope_boost
        + weights.graph_path * graph_path
    )
    return total


def rank_entries(
    entries: List[Entry],
    rrf_scores: Dict[str, float],
    graph_scores: Dict[str, float],
    query_scope: str,
    zone_gravities: Dict[str, float],
    weights: Optional[RecallWeights] = None,
    half_life_days: float = 7.0,
    top_k: int = 20,
    semantic_scores: Optional[Dict[str, float]] = None,
    lexical_scores: Optional[Dict[str, float]] = None,
) -> List[Tuple[Entry, float]]:
    """多因子排序：对条目列表评分并排序。

    Args:
        entries: 候选条目列表。
        rrf_scores: RRF 融合文本分。
        graph_scores: 图遍历分。
        query_scope: 查询 scope。
        zone_gravities: {zone_name: gravity} 映射。
        weights: 权重。
        half_life_days: 半衰期。
        top_k: 返回前 k 条。
        semantic_scores: 语义通道真实余弦相似度。
        lexical_scores: 词法通道归一化分数。

    Returns:
        排序后的 [(entry, score)] 列表。
    """
    scored: List[Tuple[Entry, float]] = []
    for entry in entries:
        zone_gravity = zone_gravities.get(entry.zone, 1.0)
        s = score_entry(
            entry, rrf_scores, graph_scores, query_scope,
            zone_gravity, weights, half_life_days,
            semantic_scores, lexical_scores,
        )
        scored.append((entry, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

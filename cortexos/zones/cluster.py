"""语义聚类 —— 连通聚类算法。

基于 embedding 相似度矩阵做连通聚类，
相似度 > threshold 的两个条目连边，形成连通分量即簇。
"""

from __future__ import annotations

from typing import List, Tuple

from cortexos.models import Entry, Zone


def cluster_entries(
    entries: List[Entry],
    similarity_threshold: float = 0.75,
    embedder: "Embedder" = None,
) -> List[List[Entry]]:
    """对条目做连通聚类。

    算法：
    1. 计算条目间 embedding 相似度矩阵
    2. 相似度 > threshold → 连边
    3. 找出连通分量 → 每个分量一个簇

    Args:
        entries: 待聚类条目列表（均需有 embedding）。
        similarity_threshold: 相似度阈值（默认 0.75）。
        embedder: Embedder 实例（用于 cosine_similarity）。

    Returns:
        簇列表，每个簇是条目列表。
    """
    if not entries:
        return []

    n = len(entries)
    if n == 1:
        return [entries]

    # 构建邻接表（无向图）
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if entries[i].embedding and entries[j].embedding:
                if embedder:
                    sim = embedder.cosine_similarity(
                        entries[i].embedding, entries[j].embedding
                    )
                else:
                    from cortexos.embedding.base import Embedder
                    sim = Embedder.cosine(entries[i].embedding, entries[j].embedding)
                if sim >= similarity_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

    # 连通分量（DFS）
    visited = set()
    clusters: List[List[Entry]] = []
    for i in range(n):
        if i not in visited:
            comp: List[int] = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    stack.extend(adjacency[node])
            if comp:
                clusters.append([entries[idx] for idx in comp])

    return clusters


def centroid_similarity(
    cluster: List[Entry],
    entry: Entry,
    embedder: "Embedder" = None,
) -> float:
    """计算条目与簇质心的相似度。

    Args:
        cluster: 已有条目簇。
        entry: 待评估条目。
        embedder: Embedder 实例。

    Returns:
        余弦相似度。
    """
    if not entry.embedding:
        return 0.0

    # 计算簇质心
    centroid = compute_centroid([e.embedding for e in cluster if e.embedding])
    if centroid is None:
        return 0.0

    if embedder:
        return embedder.cosine_similarity(entry.embedding, centroid)
    from cortexos.embedding.base import Embedder
    return Embedder.cosine(entry.embedding, centroid)


def compute_centroid(embeddings: List[List[float]]) -> List[float]:
    """计算向量列表的质心（均值向量）。

    Args:
        embeddings: 向量列表。

    Returns:
        质心向量。
    """
    if not embeddings:
        return []
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for emb in embeddings:
        for i, val in enumerate(emb):
            centroid[i] += val
    n = len(embeddings)
    return [v / n for v in centroid]


def find_best_cluster(
    entry: Entry,
    clusters: List[List[Entry]],
    similarity_threshold: float = 0.75,
    embedder: "Embedder" = None,
) -> int:
    """为条目找到最佳匹配簇（返回索引，-1 表示未匹配）。

    Args:
        entry: 待匹配条目。
        clusters: 已有簇列表。
        similarity_threshold: 匹配阈值。
        embedder: Embedder。

    Returns:
        簇索引（-1 表示未匹配）。
    """
    if not entry.embedding or not clusters:
        return -1

    best_idx = -1
    best_sim = 0.0
    for idx, cluster in enumerate(clusters):
        sim = centroid_similarity(cluster, entry, embedder)
        if sim >= similarity_threshold and sim > best_sim:
            best_sim = sim
            best_idx = idx
    return best_idx

"""Embedder 抽象基类 —— 定义向量化接口。"""

from __future__ import annotations

import abc
from typing import List


class Embedder(abc.ABC):
    """语义向量化抽象基类。

    所有 embedding 实现（OpenAI 兼容 / TF-IDF 降级）必须实现此接口。
    """

    # 相似度尺度提示：不同向量化方式的余弦相似度分布不同。
    # 语义 embedding（如 OpenAI text-embedding）相似度普遍偏高，
    # 稀疏 TF-IDF 向量相似度普遍偏低。聚类/路由阈值应据此适配。
    similarity_scale: float = 0.75  # 默认按语义 embedding 尺度

    @abc.abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """对文本列表生成 embedding 向量。

        Args:
            texts: 输入文本列表。

        Returns:
            等长向量列表，每个向量维度一致。
        """
        ...

    @abc.abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """对单条查询文本生成 embedding 向量。

        Args:
            text: 查询文本（单条）。

        Returns:
            向量。
        """
        ...

    @abc.abstractmethod
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度。

        Args:
            a: 向量 A。
            b: 向量 B。

        Returns:
            余弦相似度 (0~1)。
        """
        ...

    @abc.abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""
        ...

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        """静态余弦相似度计算（方便各实现复用）。

        Args:
            a: 向量 A。
            b: 向量 B。

        Returns:
            余弦相似度 (0~1)。
        """
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

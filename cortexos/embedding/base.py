"""Embedder 抽象基类 —— 定义向量化接口。"""

from __future__ import annotations

import abc
from typing import List


class Embedder(abc.ABC):
    """语义向量化抽象基类。

    所有 embedding 实现（OpenAI 兼容 / TF-IDF 降级）必须实现此接口。
    """

    @property
    def is_available(self) -> bool:
        """Embedder 是否可用（提供语义向量能力）。

        对仅支持本地计算的实现（如 TF-IDF），返回 True 表示
        向量化本身可用但无 LLM 能力；对 OpenAI，返回 True
        仅当配置了 API key。

        LLM 提取需额外检查（见 extract_entities 的 guard）。
        """
        return True

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

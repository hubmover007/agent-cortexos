"""TF-IDF 降级 Embedder —— 无需 API key，零依赖可用。

当未配置 embedding API key 时自动降级使用。
基于 numpy 实现，提供固定维度的语义近似向量。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from cortexos.embedding.base import Embedder


class TfidfEmbedder(Embedder):
    """TF-IDF 降级向量化器。

    策略：
    1. 对中文使用 2-gram 字符切分，英文使用空格分词
    2. 基于已见语料构建词汇表并基于词汇表累计 IDF
    3. 向量维度固定为 max_features（默认 512）
    4. 通过特征哈希（feature hashing）映射到固定维度

    复杂度：O(n × max_features)，适合 <10 万条目场景。

    注意：稀疏 TF-IDF 向量的余弦相似度分布与语义 embedding 不同
    （普遍偏低），因此 similarity_scale 设为 0.3，供聚类/路由阈值适配。
    """

    similarity_scale: float = 0.3

    def __init__(self, max_features: int = 512):
        """初始化 TF-IDF Embedder。

        Args:
            max_features: 向量维度（固定）。
        """
        self._max_features = max_features
        self._doc_count = 0
        self._df: Dict[int, int] = {}  # feature hash → document frequency

    # ── 分词 ──

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """混合分词：中文用 2-gram 字符级，英文用空格+标点分词。

        Args:
            text: 输入文本。

        Returns:
            token 列表。
        """
        if not text:
            return []
        # 判断文本类型
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        if has_chinese:
            # 中英文混合：先提取英文词，再对中文做 2-gram
            tokens: List[str] = []
            # 提取英文/数字词
            en_words = re.findall(r"[a-zA-Z0-9]+", text)
            tokens.extend(w.lower() for w in en_words)
            # 对中文部分做 2-gram
            chinese_chars = re.sub(r"[a-zA-Z0-9\s]", "", text)
            for i in range(len(chinese_chars) - 1):
                tokens.append(chinese_chars[i:i + 2])
            if chinese_chars:
                tokens.append(chinese_chars[-1:])  # 最后一个单字
            return tokens
        else:
            # 纯英文：按空格和标点分词
            return re.findall(r"[a-zA-Z0-9]+", text.lower())

    # ── 特征哈希 ──

    def _hash_feature(self, token: str) -> int:
        """将 token 哈希到 [0, max_features) 范围。

        使用 hashlib md5（确定性）而非 Python 内置 hash()，
        因为内置 hash() 受 PYTHONHASHSEED 随机化影响，
        会导致同一文本在不同进程产生不同向量（不可接受）。
        """
        import hashlib
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._max_features

    # ── 向量化 ──

    def _text_to_vec(self, tf: Dict[int, float], idf: Dict[int, float]) -> List[float]:
        """将 TF + IDF 字典转为稠密向量。

        Args:
            tf:特征哈希 → TF 值。
            idf:特征哈希 → IDF 值。

        Returns:
            稠密向量（max_features 维）。
        """
        vec = np.zeros(self._max_features, dtype=np.float32)
        if not tf:
            return vec.tolist()
        for feat, tf_val in tf.items():
            idf_val = idf.get(feat, 1.0)
            vec[feat] = tf_val * idf_val
        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。

        注意：IDF 在 embed 时更新（基于当前这批文本的 DF）。

        Args:
            texts: 输入文本列表。

        Returns:
            等长向量列表。
        """
        if not texts:
            return []

        # 计算这批文本的 DF
        batch_df: Dict[int, int] = {}
        docs_tf: List[Dict[int, float]] = []

        for text in texts:
            tokens = self._tokenize(text)
            tf = self._compute_tf(tokens)
            docs_tf.append(tf)
            for feat in tf:
                batch_df[feat] = batch_df.get(feat, 0) + 1

        # 更新全局 DF
        for feat, count in batch_df.items():
            self._df[feat] = self._df.get(feat, 0) + count
        self._doc_count += len(texts)

        # 计算 IDF
        N = max(1, self._doc_count)
        idf: Dict[int, float] = {}
        for feat in batch_df:
            idf[feat] = math.log((N + 1) / (self._df.get(feat, 1) + 1)) + 1.0

        # 生成向量
        return [self._text_to_vec(tf, idf) for tf in docs_tf]

    def _compute_tf(self, tokens: List[str]) -> Dict[int, float]:
        """计算 TF（词频归一化）。

        Args:
            tokens: 分词结果。

        Returns:
            特征哈希 → TF 值。
        """
        if not tokens:
            return {}
        counter = Counter(tokens)
        max_freq = max(counter.values())
        tf: Dict[int, float] = {}
        for token, count in counter.items():
            feat = self._hash_feature(token)
            # 子线性 TF 缩放: 1 + log(tf)（平滑版）
            tf[feat] = tf.get(feat, 0.0) + (1.0 + math.log(count)) / max_freq
        return tf

    async def embed_query(self, text: str) -> List[float]:
        """单条查询向量化。

        查询时仍使用训练阶段累计的 IDF。

        Args:
            text: 查询文本。

        Returns:
            向量。
        """
        results = await self.embed([text])
        return results[0] if results else []

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度。"""
        return Embedder.cosine(a, b)

    def dimension(self) -> int:
        """向量维度。"""
        return self._max_features

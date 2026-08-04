"""Embedding 模块。"""

from __future__ import annotations

from typing import Any

from cortexos.embedding.base import Embedder

__all__ = ["Embedder", "build_embedder"]


def build_embedder(config: Any) -> Embedder:
    """根据配置构建 Embedder（工厂，复用 config 中的唯一工厂）。

    - 配置了 LLM base_url 且环境变量中有 API key → OpenAICompatEmbedder
    - 否则 → TfidfEmbedder（零依赖降级）

    Args:
        config: Config 对象（含 llm 配置段）。

    Returns:
        Embedder 实例。
    """
    from cortexos.config import _make_embedder
    return _make_embedder(config)

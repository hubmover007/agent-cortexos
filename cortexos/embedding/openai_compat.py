"""OpenAI 兼容 Embedding + Chat 客户端。

通过 base_url 可配置，支持 OpenAI / easyrouter / 任意兼容服务。
当未配置 API key 时不可用，需降级到 TF-IDF。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from cortexos.embedding.base import Embedder


class OpenAICompatEmbedder(Embedder):
    """OpenAI 兼容 Embedding 实现。

    支持：
    - /v1/embeddings（向量化）
    - /v1/chat/completions（LLM 提取用，可选）
    - base_url 可配置任意兼容服务
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "LLM_API_KEY",
        model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ):
        """初始化 OpenAI 兼容 Embedder。

        Args:
            base_url: API 端点（含 /v1）。
            api_key: API key（不传则从环境变量读取）。
            api_key_env: 环境变量名（api_key 未传时从此读取）。
            model: embedding 模型名。
            chat_model: chat 模型名（LLM 提取用）。
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get(api_key_env, "")
        self._model = model
        self._chat_model = chat_model
        self._dim: Optional[int] = None

    @property
    def is_available(self) -> bool:
        """是否有 API key（即是否可用）。"""
        return bool(self._api_key)

    # ── Embedder 接口 ──

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。"""
        if not texts:
            return []
        if not self.is_available:
            raise RuntimeError("OpenAI 兼容 Embedder 不可用：缺少 API key")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            if self._dim is None and embeddings:
                self._dim = len(embeddings[0])
            return embeddings

    async def embed_query(self, text: str) -> List[float]:
        """单条查询向量化。"""
        results = await self.embed([text])
        return results[0] if results else []

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度。"""
        return Embedder.cosine(a, b)

    def dimension(self) -> int:
        """向量维度。"""
        return self._dim or 1536

    # ── Chat（LLM 提取用） ──

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """调用 Chat API（用于 LLM 提取）。

        Args:
            messages: 消息列表 [{"role": "system", "content": "..."}, ...]。
            model: 模型名（不传则用默认 chat_model）。
            temperature: 温度。
            max_tokens: 最大输出 token。

        Returns:
            LLM 返回的文本内容。
        """
        if not self.is_available:
            raise RuntimeError("OpenAI 兼容 Embedder 不可用：缺少 API key")

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model or self._chat_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """调用 Chat API 并要求 JSON 输出（用于结构化提取）。

        Args:
            messages: 消息列表。
            model: 模型名。

        Returns:
            解析后的 JSON 字典。
        """
        raw = await self.chat(messages, model=model, temperature=0.0)
        # 提取 JSON 块（可能包裹在 ```json ... ``` 或纯文本中）
        raw_stripped = raw.strip()
        if raw_stripped.startswith("```"):
            # 去除 ```json ... ``` 包裹
            lines = raw_stripped.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_stripped = "\n".join(lines).strip()
        try:
            return json.loads(raw_stripped)
        except json.JSONDecodeError:
            # 尝试提取第一个 {...} 块
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"error": "JSON parse failed", "raw": raw}

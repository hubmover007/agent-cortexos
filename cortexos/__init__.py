"""Agent-CortexOS v2.0 —— 通用记忆服务（Memory-as-a-Service）。

任意 Agent 可接入的读写统一记忆服务。

用法（本地嵌入模式）：
    from cortexos import CortexOS

    cortex = CortexOS("mydb.db")
    await cortex.store("今天部署了 nginx", scope="agent:demo")
    results = await cortex.recall("nginx")

用法（REST 模式）：
    from cortexos import CortexOS

    cortex = CortexOS("mydb.db", base_url="http://localhost:8000/v1")
    # 自动走 REST API
"""

from __future__ import annotations

__version__ = "2.0.0"

import json
from typing import Dict, List, Optional

from cortexos.config import Config, _make_embedder
from cortexos.models import Entry


class CortexOS:
    """CortexOS 客户端 —— 嵌入模式（本地后端）或 REST 模式。

    Attributes:
        db_path: SQLite 数据库路径。
        base_url: REST API 基础 URL（可选，不传则本地模式）。
        config: 配置对象。
    """

    def __init__(
        self,
        db_path: str = "cortexos.db",
        base_url: Optional[str] = None,
        config: Optional[Config] = None,
        api_key: Optional[str] = None,
    ):
        """初始化 CortexOS 客户端。

        Args:
            db_path: SQLite 数据库路径（本地模式）。
            base_url: REST API 基础 URL（可选，传入则走 REST 模式）。
            config: 配置（可选，默认 Config()）。
            api_key: AgentKey 明文（REST 模式认证，Bearer 头）。
        """
        self.db_path = db_path
        self.base_url = base_url
        self.config = config or Config()
        self.api_key = api_key
        self._backend = None
        self._initialized = False

    async def _ensure_init(self):
        """延迟初始化后端。"""
        if self._initialized:
            return
        from cortexos.storage.sqlite_backend import SqliteBackend
        from cortexos.config import _make_embedder
        self._backend = SqliteBackend(self.db_path)
        await self._backend.initialize()
        self._embedder = _make_embedder(self.config)
        self._initialized = True

    @property
    def backend(self):
        """获取后端实例。"""
        if not self._backend:
            raise RuntimeError("未初始化，请先 await cortex._ensure_init()")
        return self._backend

    # ── 记忆 CRUD ──

    async def store(
        self,
        content: str,
        scope: str = "default",
        layer: str = "raw",
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        zone: Optional[str] = None,
    ) -> str:
        """写入记忆。

        Args:
            content: 内容。
            scope: 作用域。
            layer: 层级（raw/episode/summary）。
            entities: 实体列表。
            metadata: 元数据。
            zone: 目标 zone（可选，自动路由）。

        Returns:
            条目 ID。
        """
        if self.base_url:
            return await self._rest_store(
                content, scope, layer, entities, metadata, zone,
            )

        await self._ensure_init()

        # 完整写入管线：实体提取 → embedding → 路由 → 质心更新 → facts/edges
        from cortexos.store import store_entry
        entry = Entry(
            content=content,
            scope=scope,
            layer=layer,
            entities=entities or [],
            metadata=metadata or {},
            zone=zone or "_inbox",
        )
        await store_entry(entry, self.backend, self._embedder, self.config)

        return entry.id

    async def get(self, entry_id: str) -> Optional[Entry]:
        """获取单条记忆（仅 active，软删除后返回 None）。"""
        if self.base_url:
            return await self._rest_get(entry_id)
        await self._ensure_init()
        entry = await self.backend.get_entry(entry_id)
        if entry and entry.status != "active":
            return None
        return entry

    async def delete(self, entry_id: str) -> bool:
        """删除记忆。"""
        if self.base_url:
            return await self._rest_delete(entry_id)
        await self._ensure_init()
        return await self.backend.delete_entry(entry_id)

    # ── 检索 ──

    async def recall(
        self,
        query: str,
        scope: str = "default",
        top_k: int = 10,
    ) -> List[Dict]:
        """检索记忆（四通道混合）。

        Args:
            query: 查询文本。
            scope: 作用域。
            top_k: 返回条数。

        Returns:
            [{"id": ..., "content": ..., "score": ..., "zone": ...}, ...]
        """
        if self.base_url:
            return await self._rest_recall(query, scope, top_k)

        await self._ensure_init()

        from cortexos.recall.hybrid import hybrid_retrieve
        from cortexos.recall.graph import GraphIndex
        from cortexos.extract.heuristic import heuristic_extract

        edges = await self.backend.find_edges(scope=scope)
        gi = GraphIndex()
        gi.load_edges(edges)

        query_entities = heuristic_extract(query).get("entities") or None

        results = await hybrid_retrieve(
            query=query,
            embedder=self._embedder,
            backend=self.backend,
            graph_index=gi,
            config=self.config,
            scope=scope,
            top_k=top_k,
            query_entities=query_entities,
        )

        return [
            {"id": e.id, "content": e.content, "score": round(s, 4), "zone": e.zone}
            for e, s in results
        ]

    async def search(self, query: str, scope: str = "default", limit: int = 10) -> List[Dict]:
        """词法搜索。"""
        if self.base_url:
            return await self._rest_search(query, scope, limit)
        await self._ensure_init()
        rows = await self.backend.search_lexical(query, scope=scope, top_k=limit)
        return [
            {"id": e.id, "content": e.content, "score": round(s, 4), "zone": e.zone}
            for e, s in rows
        ]

    # ── Zones ──

    async def list_zones(self, scope: str = "default") -> List[Dict]:
        """列出 zone。"""
        if self.base_url:
            return await self._rest_zones(scope)
        await self._ensure_init()
        zones = await self.backend.list_zones(scope=scope)
        return [
            {"name": z.name, "status": z.status, "entries": z.entry_count,
             "gravity": round(z.gravity, 2)}
            for z in zones
        ]

    # ── Admin ──

    async def stats(self) -> Dict:
        """获取统计。"""
        if self.base_url:
            return await self._rest_stats()
        await self._ensure_init()
        return await self.backend.get_stats()

    async def consolidate(self, scope: str = "default") -> Dict:
        """触发整合。"""
        if self.base_url:
            return await self._rest_consolidate(scope)
        await self._ensure_init()
        from cortexos.lifecycle.consolidate import ConsolidateEngine
        engine = ConsolidateEngine(self.backend, self.config)
        return await engine.consolidate(scope)

    # ── REST 模式 ──

    def _headers(self) -> Dict:
        """REST 请求头（带 Bearer 认证）。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _rest_store(self, content, scope, layer, entities, metadata, zone):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.base_url}/memories",
                json={"content": content, "scope": scope, "layer": layer,
                      "entities": entities or [], "metadata": metadata or {},
                      "zone": zone},
                headers=self._headers(),
            ) as resp:
                data = await resp.json()
                return data["id"]

    async def _rest_get(self, entry_id):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                f"{self.base_url}/memories/{entry_id}", headers=self._headers(),
            ) as resp:
                if resp.status == 404:
                    return None
                return await resp.json()

    async def _rest_delete(self, entry_id):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.delete(
                f"{self.base_url}/memories/{entry_id}", headers=self._headers(),
            ) as resp:
                return resp.status == 200

    async def _rest_recall(self, query, scope, top_k):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.base_url}/retrieve",
                json={"query": query, "scope": scope, "top_k": top_k},
                headers=self._headers(),
            ) as resp:
                data = await resp.json()
                return data["items"]

    async def _rest_search(self, query, scope, limit):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.base_url}/search",
                json={"query": query, "scope": scope, "top_k": limit},
                headers=self._headers(),
            ) as resp:
                data = await resp.json()
                return data["items"]

    async def _rest_zones(self, scope):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                f"{self.base_url}/zones?scope={scope}", headers=self._headers(),
            ) as resp:
                data = await resp.json()
                return data["zones"]

    async def _rest_stats(self):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                f"{self.base_url}/stats", headers=self._headers(),
            ) as resp:
                return await resp.json()

    async def _rest_consolidate(self, scope):
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.base_url}/consolidate?scope={scope}",
                headers=self._headers(),
            ) as resp:
                return await resp.json()

    async def close(self):
        """关闭连接。"""
        if self._backend:
            await self._backend.close()
            self._initialized = False

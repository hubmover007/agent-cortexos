"""存储后端抽象 + 工厂函数。"""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, List, Optional

from cortexos.models import Edge, Entry, Fact, Zone


class StorageBackend(abc.ABC):
    """存储后端抽象基类。

    所有存储实现（SQLite / JSONL / S3）必须实现此接口。
    """

    # ── 条目 CRUD ──

    @abc.abstractmethod
    async def upsert_entry(self, entry: Entry) -> None:
        """写入/更新条目。"""
        ...

    @abc.abstractmethod
    async def get_entry(self, entry_id: str) -> Optional[Entry]:
        """读取单条条目。"""
        ...

    @abc.abstractmethod
    async def delete_entry(self, entry_id: str) -> None:
        """软删除条目。"""
        ...

    @abc.abstractmethod
    async def list_entries(
        self,
        *,
        scope: Optional[str] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entry]:
        """列出条目。"""
        ...

    @abc.abstractmethod
    async def count_entries(
        self,
        *,
        scope: Optional[str] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> int:
        """统计条目数。"""
        ...

    # ── 检索 ──

    @abc.abstractmethod
    async def search_lexical(
        self, query: str, *, scope: Optional[str] = None, top_k: int = 20
    ) -> List[tuple[Entry, float]]:
        """FTS5 词法检索。"""
        ...

    @abc.abstractmethod
    async def search_all_embeddings(
        self, *, scope: Optional[str] = None
    ) -> List[tuple[str, Optional[List[float]]]]:
        """批量获取 embedding（内存向量索引构建用）。"""
        ...

    # ── Facts ──

    @abc.abstractmethod
    async def upsert_fact(self, fact: Fact) -> None:
        """写入/更新事实。"""
        ...

    @abc.abstractmethod
    async def find_facts(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        scope: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Fact]:
        """查询事实。"""
        ...

    # ── Edges ──

    @abc.abstractmethod
    async def upsert_edge(self, edge: Edge) -> None:
        """写入/更新关系边。"""
        ...

    @abc.abstractmethod
    async def find_edges(
        self,
        *,
        source: Optional[str] = None,
        target: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> List[Edge]:
        """查询关系边。"""
        ...

    @abc.abstractmethod
    async def list_edges_from(self, entity: str) -> List[Edge]:
        """查询从指定实体出发的边。"""
        ...

    @abc.abstractmethod
    async def list_edges_to(self, entity: str) -> List[Edge]:
        """查询指向指定实体的边。"""
        ...

    # ── Zones ──

    @abc.abstractmethod
    async def upsert_zone(self, zone: Zone) -> None:
        """写入/更新 Zone。"""
        ...

    @abc.abstractmethod
    async def get_zone(self, name: str, scope: Optional[str] = None) -> Optional[Zone]:
        """读取单个 Zone（复合主键 scope+name）。"""
        ...

    @abc.abstractmethod
    async def list_zones(
        self,
        *,
        scope: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Zone]:
        """列出 Zone。"""
        ...

    # ── 批量 / 扫描 ──

    @abc.abstractmethod
    async def scan_entries(
        self,
        *,
        since: Optional[float] = None,
        scope: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> AsyncIterator[Entry]:
        """流式扫描条目（整合用）。"""
        ...

    @abc.abstractmethod
    async def bulk_update_status(
        self, updates: List[tuple[str, str]]
    ) -> None:
        """批量更新条目状态 [(entry_id, new_status), ...]"""
        ...

    # ── Agent / 密钥 / 配对 ──

    @abc.abstractmethod
    async def upsert_agent(self, agent: Dict[str, Any]) -> None:
        """写入 Agent。"""
        ...

    @abc.abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """读取 Agent。"""
        ...

    @abc.abstractmethod
    async def upsert_agent_key(self, key_data: Dict[str, Any]) -> None:
        """写入密钥。"""
        ...

    @abc.abstractmethod
    async def get_agent_key_by_id(self, key_id: str) -> Optional[Dict[str, Any]]:
        """按 key_id 查询密钥。"""
        ...

    @abc.abstractmethod
    async def get_agent_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """按 key_hash 查询密钥。"""
        ...

    @abc.abstractmethod
    async def list_agent_keys(self, agent_id: str) -> List[Dict[str, Any]]:
        """列出 Agent 的所有密钥。"""
        ...

    @abc.abstractmethod
    async def upsert_pair_code(self, code_data: Dict[str, Any]) -> None:
        """写入配对码。"""
        ...

    @abc.abstractmethod
    async def get_pair_code(self, code: str) -> Optional[Dict[str, Any]]:
        """读取配对码。"""
        ...

    @abc.abstractmethod
    async def write_audit_log(self, log_data: Dict[str, Any]) -> None:
        """写入审计日志。"""
        ...

    # ── Scopes / 统计 ──

    @abc.abstractmethod
    async def list_scopes(self) -> List[str]:
        """列出所有出现过数据的 scope。"""
        ...

    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        ...

    # ── 启动/关闭 ──

    @abc.abstractmethod
    async def initialize(self) -> None:
        """初始化：建表/创建索引。"""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """关闭连接。"""
        ...


def get_backend(config: Any) -> StorageBackend:
    """工厂函数：根据配置返回存储后端。

    Args:
        config: Config 对象。

    Returns:
        StorageBackend 实例。
    """
    from cortexos.config import Config
    cfg: Config = config

    if cfg.storage.backend == "local":
        from cortexos.storage.sqlite_backend import SqliteBackend
        return SqliteBackend(db_path=cfg.storage.local.path)
    else:
        raise ValueError(f"不支持的存储后端: {cfg.storage.backend}")

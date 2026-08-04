"""领域模型 —— Entry / Fact / Edge / Zone / Agent / AgentKey。

所有模型均为 dataclass，可序列化/反序列化，
通过 SQLite backend 持久化。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ulid


def _now() -> float:
    """当前 epoch 秒（float）。"""
    return time.time()


def _new_id() -> str:
    """生成 ULID（时间有序唯一 ID）。"""
    return str(ulid.new())


# ── Entry ──


@dataclass
class Entry:
    """记忆条目 —— 核心存储单元。

    Attributes:
        id: ULID 唯一标识。
        scope: 归属域（如 agent:xxx、team:yyy）。
        zone: Zone 名称（自动路由或 _inbox）。
        layer: 抽象层级 raw|fact|facet|episode|project。
        content: 记忆内容文本。
        entities: 提取的实体列表。
        embedding: 语义向量（JSON float[]，可选）。
        metadata: 自定义元数据字典。
        created_at / updated_at: 时间戳（epoch 秒）。
        access_count: 访问次数。
        status: active|superseded|archived。
        valid_until: 过期时间戳（None=永不过期）。
    """
    content: str
    scope: str = "default"
    id: str = field(default_factory=_new_id)
    zone: str = "_inbox"
    layer: str = "raw"
    entities: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    access_count: int = 0
    status: str = "active"
    valid_until: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "id": self.id,
            "scope": self.scope,
            "zone": self.zone,
            "layer": self.layer,
            "content": self.content,
            "entities": json.dumps(self.entities, ensure_ascii=False),
            "embedding": json.dumps(self.embedding) if self.embedding else None,
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "status": self.status,
            "valid_until": self.valid_until,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Entry":
        """从 SQLite 查询行构建 Entry。"""
        entities = row.get("entities", "[]")
        if isinstance(entities, str):
            entities = json.loads(entities)
        meta = row.get("metadata", "{}")
        if isinstance(meta, str):
            meta = json.loads(meta)
        emb = row.get("embedding")
        if isinstance(emb, str) and emb:
            emb = json.loads(emb)
        return cls(
            id=row["id"],
            scope=row["scope"],
            zone=row["zone"],
            layer=row["layer"],
            content=row["content"],
            entities=entities,
            embedding=emb,
            metadata=meta,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row.get("access_count", 0),
            status=row.get("status", "active"),
            valid_until=row.get("valid_until"),
        )

    def touch(self) -> None:
        """更新访问统计。"""
        self.access_count += 1
        self.updated_at = _now()


# ── Fact ──


@dataclass
class Fact:
    """结构化事实（三元组）。

    Attributes:
        subject / predicate / object: SPO 三元组。
        confidence: 置信度 (0~1)。
        valid_from / valid_until: 有效性时间窗口。
        scope: 归属域。
        entry_id: 来源条目 ID。
        status: active|superseded|conflict。
    """
    subject: str
    predicate: str
    object: str
    scope: str = "default"
    id: str = field(default_factory=_new_id)
    entry_id: str = ""
    confidence: float = 1.0
    valid_from: float = field(default_factory=_now)
    valid_until: Optional[float] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entry_id": self.entry_id,
            "scope": self.scope,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "status": self.status,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Fact":
        return cls(
            id=row["id"],
            entry_id=row.get("entry_id", ""),
            scope=row["scope"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row.get("confidence", 1.0),
            valid_from=row["valid_from"],
            valid_until=row.get("valid_until"),
            status=row.get("status", "active"),
        )


# ── Edge ──


@dataclass
class Edge:
    """关系边（图遍历用）。

    Attributes:
        source / target: 源/目标实体。
        relation: 关系描述。
        weight: 边权重。
        scope: 归属域。
        entry_id: 来源条目 ID。
        valid_until: 过期时间戳。
    """
    source: str
    target: str
    relation: str
    scope: str = "default"
    id: str = field(default_factory=_new_id)
    entry_id: str = ""
    weight: float = 1.0
    valid_until: Optional[float] = None
    created_at: float = field(default_factory=_now)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entry_id": self.entry_id,
            "scope": self.scope,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "valid_until": self.valid_until,
            "created_at": self.created_at,
            "embedding": json.dumps(self.embedding) if self.embedding else None,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Edge":
        emb = row.get("embedding")
        if isinstance(emb, str) and emb:
            emb = json.loads(emb)
        return cls(
            id=row["id"],
            entry_id=row.get("entry_id", ""),
            scope=row["scope"],
            source=row["source"],
            target=row["target"],
            relation=row["relation"],
            weight=row.get("weight", 1.0),
            valid_until=row.get("valid_until"),
            created_at=row.get("created_at", _now()),
            embedding=emb,
        )


# ── Zone ──


@dataclass
class Zone:
    """记忆域（自动组织单元）。

    Attributes:
        name: Zone 名称（唯一）。
        scope: 归属域。
        description: 描述。
        entities: 绑定实体列表。
        keywords: 范围关键词列表。
        centroid: 质心向量（JSON float[]）。
        gravity: 重力值。
        entry_count: 条目数。
        status: active|dormant|archived。
        pinned: 是否固定（防归档）。
        created_at / last_access: 时间戳。
    """
    name: str
    scope: str = "default"
    id: str = field(default_factory=_new_id)
    description: str = ""
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    gravity: float = 1.0
    entry_count: int = 0
    status: str = "active"
    pinned: int = 0
    created_at: float = field(default_factory=_now)
    last_access: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "scope": self.scope,
            "description": self.description,
            "entities": json.dumps(self.entities, ensure_ascii=False),
            "keywords": json.dumps(self.keywords, ensure_ascii=False),
            "centroid": json.dumps(self.centroid) if self.centroid else None,
            "gravity": self.gravity,
            "entry_count": self.entry_count,
            "status": self.status,
            "pinned": self.pinned,
            "created_at": self.created_at,
            "last_access": self.last_access,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Zone":
        entities = row.get("entities", "[]")
        if isinstance(entities, str):
            entities = json.loads(entities)
        keywords = row.get("keywords", "[]")
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        centroid = row.get("centroid")
        if isinstance(centroid, str) and centroid:
            centroid = json.loads(centroid)
        return cls(
            id=row.get("id", _new_id()),
            name=row["name"],
            scope=row["scope"],
            description=row.get("description", ""),
            entities=entities,
            keywords=keywords,
            centroid=centroid,
            gravity=row.get("gravity", 1.0),
            entry_count=row.get("entry_count", 0),
            status=row.get("status", "active"),
            pinned=row.get("pinned", 0),
            created_at=row.get("created_at", _now()),
            last_access=row.get("last_access", _now()),
        )


# ── Agent ──


@dataclass
class Agent:
    """配对实体 —— 接入记忆服务的身份标识。

    Attributes:
        agent_id: 唯一标识（ag_ 前缀）。
        agent_name: 名称。
        created_at: 注册时间。
        status: active|revoked。
    """
    agent_id: str
    agent_name: str
    created_at: float = field(default_factory=_now)
    status: str = "active"

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Agent":
        return cls(
            agent_id=row["agent_id"],
            agent_name=row["agent_name"],
            created_at=row["created_at"],
            status=row.get("status", "active"),
        )


# ── AgentKey ──


@dataclass
class AgentKey:
    """Agent 密钥 —— 认证凭证。

    Attributes:
        key_id: 公开标识。
        agent_id: 所属 Agent。
        key_hash: sha256(secret)，不存明文。
        scope_permissions: {scope: "read"|"write"|"readwrite"}。
        zone_overrides: {scope:zone: "read"} Zone 级权限覆盖。
        rate_limit: 限流（请求/分钟）。
        expires_at / created_at / last_used: 时间戳。
        status: active|revoked。
    """
    key_id: str
    agent_id: str
    key_hash: str
    scope_permissions: Dict[str, str] = field(default_factory=dict)
    zone_overrides: Dict[str, str] = field(default_factory=dict)
    rate_limit: int = 100
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=_now)
    last_used: Optional[float] = None
    status: str = "active"

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "AgentKey":
        sp = row.get("scope_permissions", "{}")
        if isinstance(sp, str):
            sp = json.loads(sp)
        zo = row.get("zone_overrides", "{}")
        if isinstance(zo, str):
            zo = json.loads(zo)
        return cls(
            key_id=row["key_id"],
            agent_id=row["agent_id"],
            key_hash=row["key_hash"],
            scope_permissions=sp,
            zone_overrides=zo,
            rate_limit=row.get("rate_limit", 100),
            expires_at=row.get("expires_at"),
            created_at=row["created_at"],
            last_used=row.get("last_used"),
            status=row.get("status", "active"),
        )

    def has_scope_permission(self, scope: str, required: str) -> bool:
        """检查 scope 级权限。"""
        perm = self.scope_permissions.get(scope)
        if not perm:
            return False
        return _rank(perm) >= _rank(required)

    def get_zone_override(self, scope: str, zone: str) -> Optional[str]:
        """获取 Zone 级覆盖权限。"""
        key = f"{scope}:{zone}"
        return self.zone_overrides.get(key)

    @property
    def is_expired(self) -> bool:
        """检查密钥是否过期。"""
        if self.expires_at is None:
            return False
        return _now() > self.expires_at

    @property
    def is_revoked(self) -> bool:
        """检查密钥是否已吊销。"""
        return self.status == "revoked"


# ── 权限 rank ──


def _rank(permission: str) -> int:
    """权限等级：read=1, write=2, readwrite=2。"""
    if permission == "readwrite":
        return 2
    if permission == "write":
        return 2
    if permission == "read":
        return 1
    return 0

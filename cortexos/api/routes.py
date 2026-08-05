"""API 模块 —— FastAPI 路由（17 业务端点 + healthz）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from pydantic import BaseModel

from cortexos.api.deps import (
    get_backend, get_config, get_embedder, verify_bearer, assert_scope,
)
from cortexos.config import Config
from cortexos.models import Entry
from cortexos.storage import StorageBackend


# ── Request/Response Models ──

class PairRequestReq(BaseModel):
    agent_name: str

class PairRequestResp(BaseModel):
    code: str
    agent_id: str
    expires_in: int

class PairConfirmReq(BaseModel):
    code: str
    scope_permissions: Dict[str, str]

class PairConfirmResp(BaseModel):
    code: str
    agent_id: str
    scopes: List[str]

class PairExchangeReq(BaseModel):
    code: str

class PairExchangeResp(BaseModel):
    key_id: str
    secret: str
    agent_id: str

class StoreMemoryReq(BaseModel):
    content: str
    scope: str = "default"
    layer: str = "raw"
    entities: List[str] = []
    metadata: Dict = {}
    zone: Optional[str] = None

class MemoryResp(BaseModel):
    id: str
    content: str
    scope: str
    zone: str
    layer: str
    entities: List[str]
    created_at: float

class RetrieveReq(BaseModel):
    query: str
    scope: str = "default"
    top_k: int = 10
    use_semantic: bool = True
    use_lexical: bool = True
    use_graph: bool = True

class RetrieveResp(BaseModel):
    items: List[Dict]
    total: int

class SearchReq(BaseModel):
    query: str
    scope: str = "default"
    top_k: int = 10

class ZoneInfo(BaseModel):
    name: str
    scope: str
    status: str
    entry_count: int
    gravity: float

class StatsResp(BaseModel):
    total_entries: int
    total_facts: int
    total_zones: int
    total_edges: int
    db_size_bytes: int

class HealthResp(BaseModel):
    status: str
    version: str
    uptime: float

class KeyListResp(BaseModel):
    keys: List[Dict]

class ContextReq(BaseModel):
    scope: str = "default"
    zone: Optional[str] = None
    max_tokens: int = 4000

class ContextResp(BaseModel):
    context: str
    entries: List[Dict]

class PinZoneReq(BaseModel):
    gravity: Optional[float] = None


# ── Helper ──

def _has_any_scope_perm(key_data: Dict, scope: str) -> bool:
    """key 是否对 scope 有任意权限（含 zone 覆盖）。"""
    from cortexos.auth.permissions import _rank
    perm = key_data.get("scope_permissions", {}).get(scope)
    if perm and _rank(perm) > 0:
        return True
    for key, ov in (key_data.get("zone_overrides", {}) or {}).items():
        if key.startswith(f"{scope}:") and _rank(ov) > 0:
            return True
    return False


# ── Standalone routes ──

async def healthz(request: Request):
    """健康检查探测端点。"""
    uptime = getattr(request.app.state, "start_time", 0)
    return HealthResp(
        status="ok",
        version="2.0.0",
        uptime=time.time() - uptime if uptime else 0,
    )


async def pair_request(
    req: PairRequestReq,
    backend=Depends(get_backend),
    config=Depends(get_config),
):
    """发起配对请求 —— 生成一次性配对码。"""
    from cortexos.auth.pairing import pair_request as do_request
    result = await do_request(req.agent_name, backend, config)
    return PairRequestResp(**result)


async def pair_confirm(
    req: PairConfirmReq,
    request: Request,
    backend=Depends(get_backend),
    config=Depends(get_config),
):
    """管理员确认配对（需 X-Admin-Token 管理令牌）。

    - 服务端未配置 admin_token 时，API 确认一律拒绝（403），
      只能通过服务端本地 CLI `cortexos pair-approve` 确认。
    - 远程场景必须配置 CORTEXOS_SERVER_ADMIN_TOKEN 并携带该令牌。
    """
    admin_token = (config.server.admin_token or "").strip()
    if not admin_token:
        raise HTTPException(
            403, "API pairing confirm disabled: 请配置 CORTEXOS_SERVER_ADMIN_TOKEN "
                 "或使用服务端本地 CLI cortexos pair-approve"
        )
    provided = request.headers.get("X-Admin-Token", "")
    if provided != admin_token:
        raise HTTPException(403, "Invalid admin token")

    from cortexos.auth.pairing import pair_approve
    result = await pair_approve(
        req.code, req.scope_permissions,
        backend, config,
    )
    if not result:
        raise HTTPException(400, "Invalid or expired pairing code")
    return PairConfirmResp(**result)


async def pair_exchange(
    req: PairExchangeReq,
    backend=Depends(get_backend),
    config=Depends(get_config),
):
    """Agent 用配对码换取 agent_key。"""
    from cortexos.auth.pairing import pair_exchange as do_exchange
    result = await do_exchange(req.code, backend, config)
    if not result:
        raise HTTPException(400, "Invalid, expired, or unapproved pairing code")
    return PairExchangeResp(**result)


# ── Key management ──

async def list_keys(
    key_data: Dict = Depends(verify_bearer),
    backend=Depends(get_backend),
):
    """列出 Agent 的所有密钥（脱敏）。"""
    from cortexos.auth.keys import list_keys as do_list
    keys = await do_list(key_data["agent_id"], backend)
    return KeyListResp(keys=keys)


async def delete_key(
    key_id: str,
    key_data: Dict = Depends(verify_bearer),
    backend=Depends(get_backend),
):
    """吊销密钥（只能吊销自己的 key，防跨 agent 越权）。"""
    from cortexos.auth.keys import revoke_key
    key = await backend.get_agent_key_by_id(key_id)
    if not key:
        raise HTTPException(404, "Key not found")
    if key["agent_id"] != key_data["agent_id"]:
        raise HTTPException(403, "Cannot revoke another agent's key")
    ok = await revoke_key(key_id, backend)
    if not ok:
        raise HTTPException(404, "Key not found")
    return {"status": "revoked"}


# ── Memory CRUD ──

async def store_memory(
    req: StoreMemoryReq,
    backend=Depends(get_backend),
    config=Depends(get_config),
    embedder=Depends(get_embedder),
    key_data: Dict = Depends(verify_bearer),
):
    """写入记忆（需要 scope 写权限）。

    完整写入管线：实体提取 → embedding → 三层 zone 路由 → 质心更新 → facts/edges。
    """
    from cortexos.store import store_entry
    assert_scope(key_data, req.scope, "write")

    entry = Entry(
        content=req.content,
        scope=req.scope,
        layer=req.layer,
        entities=req.entities,
        metadata=req.metadata,
        zone=req.zone or "_inbox",
    )

    await store_entry(entry, backend, embedder, config)

    return MemoryResp(
        id=entry.id, content=entry.content,
        scope=entry.scope, zone=entry.zone or "_inbox",
        layer=entry.layer, entities=entry.entities,
        created_at=entry.created_at,
    )


async def get_memory(
    memory_id: str,
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """读取单条记忆（仅 active；需要该条目 scope 的读权限）。"""
    entry = await backend.get_entry(memory_id)
    if not entry or entry.status != "active":
        raise HTTPException(404, "Memory not found")
    assert_scope(key_data, entry.scope, "read")
    return MemoryResp(
        id=entry.id, content=entry.content,
        scope=entry.scope, zone=entry.zone or "_inbox",
        layer=entry.layer, entities=entry.entities,
        created_at=entry.created_at,
    )


async def delete_memory(
    memory_id: str,
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """删除记忆（需要该条目 scope 的写权限）。"""
    entry = await backend.get_entry(memory_id)
    if not entry or entry.status != "active":
        raise HTTPException(404, "Memory not found")
    assert_scope(key_data, entry.scope, "write")
    ok = await backend.delete_entry(memory_id)
    if not ok:
        raise HTTPException(404, "Memory not found")
    return {"status": "deleted"}


async def retrieve_memories(
    req: RetrieveReq,
    backend=Depends(get_backend),
    config=Depends(get_config),
    embedder=Depends(get_embedder),
    key_data: Dict = Depends(verify_bearer),
):
    """四通道混合检索（需要 scope 读权限）。"""
    assert_scope(key_data, req.scope, "read")
    from cortexos.recall.hybrid import hybrid_retrieve
    from cortexos.recall.graph import GraphIndex

    # 构建图索引（始终构建，空图时图通道自然返回空）
    gi = GraphIndex()
    edges = await backend.find_edges(scope=req.scope)
    gi.load_edges(edges)

    # 查询实体（用于实体通道 + 图通道）
    query_entities = None
    if req.query:
        try:
            from cortexos.extract.heuristic import heuristic_extract
            query_entities = heuristic_extract(req.query).get("entities") or None
        except Exception:
            query_entities = None

    results = await hybrid_retrieve(
        query=req.query,
        embedder=embedder,
        backend=backend,
        graph_index=gi,
        config=config,
        scope=req.scope,
        top_k=req.top_k,
        query_entities=query_entities,
    )

    items = [
        {"id": e.id, "content": e.content, "zone": e.zone,
         "score": s, "layer": e.layer}
        for e, s in results
    ]
    return RetrieveResp(items=items, total=len(items))


async def search_memories(
    req: SearchReq,
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """搜索记忆（词法 FTS5，需要 scope 读权限）。"""
    assert_scope(key_data, req.scope, "read")
    results = await backend.search_lexical(req.query, scope=req.scope, top_k=req.top_k)
    items = [
        {"id": e.id, "content": e.content,
         "zone": e.zone or "_inbox", "score": round(float(s), 4),
         "layer": e.layer}
        for e, s in results
    ]
    return RetrieveResp(items=items, total=len(items))


async def get_context(
    scope: str = "default",
    zone: Optional[str] = None,
    max_tokens: int = 4000,
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """获取上下文（需要 scope 读权限）。"""
    assert_scope(key_data, scope, "read")
    entries = await backend.list_entries(scope=scope, zone=zone, limit=50)
    items = [
        {"id": e.id, "content": e.content, "zone": e.zone or "_inbox",
         "layer": e.layer, "created_at": e.created_at}
        for e in entries
    ]
    context = "\n\n".join(e.content for e in entries)
    if len(context) > max_tokens * 4:
        context = context[:max_tokens * 4]
    return ContextResp(context=context, entries=items)


async def get_zone_context(
    scope: str,
    zone: str,
    limit: int = 20,
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """获取 zone 上下文（需要 scope 读权限）。"""
    assert_scope(key_data, scope, "read")
    entries = await backend.list_entries(scope=scope, zone=zone, limit=limit)
    items = [
        {"id": e.id, "content": e.content, "layer": e.layer}
        for e in entries
    ]
    context = "\n\n".join(e.content for e in entries)
    return {"context": context, "entries": items}


async def list_scopes(
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """列出当前 key 有权限的活跃 scope。"""
    scopes = await backend.list_scopes()
    allowed = [s for s in scopes if _has_any_scope_perm(key_data, s)]
    return {"scopes": allowed}


async def list_zones(
    scope: str = "default",
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """列出 scope 下所有 zone（需要 scope 读权限）。"""
    assert_scope(key_data, scope, "read")
    zones = await backend.list_zones(scope=scope)
    return {
        "zones": [
            ZoneInfo(
                name=z.name, scope=z.scope, status=z.status,
                entry_count=z.entry_count, gravity=z.gravity,
            )
            for z in zones
        ]
    }


async def pin_zone(
    scope: str,
    name: str,
    req: PinZoneReq,
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """固定 zone（需要 scope 写权限）。"""
    assert_scope(key_data, scope, "write")
    zone = await backend.get_zone(name, scope=scope)
    if not zone:
        raise HTTPException(404, "Zone not found")
    if req.gravity is not None:
        zone.gravity = req.gravity
    else:
        zone.gravity = zone.gravity + 1.0 if zone.gravity else 1.0
    await backend.upsert_zone(zone)
    return {"zone": name, "gravity": zone.gravity}


async def trigger_consolidate(
    scope: str = "default",
    backend=Depends(get_backend),
    config=Depends(get_config),
    key_data: Dict = Depends(verify_bearer),
):
    """触发整合（需要 scope 写权限，防止越权影响他人数据）。"""
    assert_scope(key_data, scope, "write")
    from cortexos.lifecycle.consolidate import ConsolidateEngine
    engine = ConsolidateEngine(backend, config)
    stats = await engine.consolidate(scope)
    return stats


async def get_stats(
    backend=Depends(get_backend),
    key_data: Dict = Depends(verify_bearer),
):
    """获取统计信息（需有效密钥）。"""
    stats = await backend.get_stats()
    return StatsResp(
        total_entries=stats.get("total_entries", 0),
        total_facts=stats.get("total_facts", 0),
        total_zones=stats.get("total_zones", 0),
        total_edges=stats.get("total_edges", 0),
        db_size_bytes=stats.get("db_size_bytes", 0),
    )


# ── App factory ──

def create_app_v1(backend: StorageBackend, config: Config, embedder=None) -> FastAPI:
    """创建 FastAPI v1 路由应用。

    Args:
        backend: 存储后端。
        config: 配置。
        embedder: Embedder 实例（不传则按配置自动构建）。
    """
    app = FastAPI(title="CortexOS v2.0", version="2.0.0")

    if embedder is None:
        from cortexos.embedding import build_embedder
        embedder = build_embedder(config)

    app.state.backend = backend
    app.state.config = config
    app.state.embedder = embedder
    app.state.start_time = time.time()

    # Healthz
    app.get("/healthz", response_model=HealthResp)(healthz)

    # Pairing
    app.post("/v1/pair/request", response_model=PairRequestResp)(pair_request)
    app.post("/v1/pair/confirm", response_model=PairConfirmResp)(pair_confirm)
    app.post("/v1/pair/exchange", response_model=PairExchangeResp)(pair_exchange)

    # Keys
    app.get("/v1/keys", response_model=KeyListResp)(list_keys)
    app.delete("/v1/keys/{key_id}")(delete_key)

    # Memories
    app.post("/v1/memories", response_model=MemoryResp)(store_memory)
    app.get("/v1/memories/{memory_id}", response_model=MemoryResp)(get_memory)
    app.delete("/v1/memories/{memory_id}")(delete_memory)

    # Retrieval
    app.post("/v1/retrieve", response_model=RetrieveResp)(retrieve_memories)
    app.post("/v1/search", response_model=RetrieveResp)(search_memories)
    app.get("/v1/context")(get_context)
    app.get("/v1/zones/{scope}/{zone}/context")(get_zone_context)

    # Scopes & Zones
    app.get("/v1/scopes")(list_scopes)
    app.get("/v1/zones")(list_zones)
    app.post("/v1/zones/{scope}/{name}/pin")(pin_zone)

    # Admin
    app.post("/v1/consolidate")(trigger_consolidate)
    app.get("/v1/stats", response_model=StatsResp)(get_stats)

    return app

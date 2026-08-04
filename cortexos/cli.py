"""CortexOS CLI —— cortexos serve / pair / store / recall / zones / stats / consolidate。

使用 argparse 实现，兼容 Python 3.10+。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

from cortexos.config import _get_or_create_embedder


def _get_backend_and_config(path: Optional[str] = None):
    """初始化后端和配置。"""
    from cortexos.config import Config
    from cortexos.storage.sqlite_backend import SqliteBackend

    db_path = path or "cortexos.db"
    config = Config()
    backend = SqliteBackend(db_path)
    return backend, config


async def cmd_serve(args):
    """启动 API 服务。"""
    import uvicorn
    from cortexos.api import create_app_v1
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()

    app = create_app_v1(backend, config)
    config_uv = uvicorn.Config(
        app, host=args.host, port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config_uv)
    await server.serve()


async def cmd_pair_request(args):
    """发起配对请求。"""
    from cortexos.auth.pairing import pair_request
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()
    result = await pair_request(args.agent_name, backend, config)
    print(json.dumps(result, indent=2))


async def cmd_pair_approve(args):
    """批准配对。"""
    from cortexos.auth.pairing import pair_approve
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()
    result = await pair_approve(args.code, json.loads(args.scopes), backend, config)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("ERROR: Invalid or expired code", file=sys.stderr)
        sys.exit(1)


async def cmd_pair_exchange(args):
    """兑换密钥。"""
    from cortexos.auth.pairing import pair_exchange
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()
    result = await pair_exchange(args.code, backend, config)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("ERROR: Invalid or unapproved code", file=sys.stderr)
        sys.exit(1)


async def cmd_store(args):
    """存储记忆。"""
    from cortexos.models import Entry
    from cortexos.store import store_entry
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()

    entry = Entry(
        content=args.text,
        scope=args.scope,
        layer="raw",
        entities=json.loads(args.entities) if args.entities else [],
        zone=args.zone if args.zone else "_inbox",
    )
    # 完整写入管线：实体提取 → embedding → 路由 → 质心更新
    await store_entry(entry, backend, _get_or_create_embedder(config), config)

    print(json.dumps({"id": entry.id, "zone": entry.zone or "_inbox"}, indent=2))


async def cmd_recall(args):
    """检索记忆。"""
    from cortexos.recall.hybrid import hybrid_retrieve
    from cortexos.recall.graph import GraphIndex
    from cortexos.zones.engine import compute_gravity
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()

    from cortexos.embedding import build_embedder
    from cortexos.extract.heuristic import heuristic_extract
    embedder = build_embedder(config)

    edges = await backend.find_edges(scope=args.scope)
    gi = GraphIndex()
    gi.load_edges(edges)

    query_entities = heuristic_extract(args.query).get("entities") or None

    results = await hybrid_retrieve(
        query=args.query,
        embedder=embedder,
        backend=backend,
        graph_index=gi,
        config=config,
        scope=args.scope,
        top_k=args.top_k,
        query_entities=query_entities,
    )

    items = [
        {"id": e.id, "content": e.content, "score": round(s, 4), "zone": e.zone}
        for e, s in results
    ]
    print(json.dumps(items, indent=2, ensure_ascii=False))


async def cmd_zones(args):
    """查看 zone 列表。"""
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()
    zones = await backend.list_zones(scope=args.scope)
    items = [
        {"name": z.name, "status": z.status, "entries": z.entry_count, "gravity": round(z.gravity, 2)}
        for z in zones
    ]
    print(json.dumps(items, indent=2))


async def cmd_stats(args):
    """查看统计。"""
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()
    stats = await backend.get_stats()
    print(json.dumps(stats, indent=2))


async def cmd_consolidate(args):
    """触发整合。"""
    from cortexos.lifecycle.consolidate import ConsolidateEngine
    backend, config = _get_backend_and_config(args.db)
    await backend.initialize()
    engine = ConsolidateEngine(backend, config)
    stats = await engine.consolidate(args.scope)
    print(json.dumps(stats, indent=2))


def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        prog="cortexos",
        description="CortexOS v2.0 通用记忆服务",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # serve
    p = sub.add_parser("serve", help="启动 API 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default="cortexos.db")

    # pair request
    p = sub.add_parser("pair-request", help="发起配对请求")
    p.add_argument("agent_name")
    p.add_argument("--db", default="cortexos.db")

    # pair approve
    p = sub.add_parser("pair-approve", help="批准配对")
    p.add_argument("code")
    p.add_argument("--scopes", default='{}')
    p.add_argument("--db", default="cortexos.db")

    # pair exchange
    p = sub.add_parser("pair-exchange", help="兑换密钥")
    p.add_argument("code")
    p.add_argument("--db", default="cortexos.db")

    # store
    p = sub.add_parser("store", help="存储记忆")
    p.add_argument("text")
    p.add_argument("--scope", default="default")
    p.add_argument("--zone", default=None)
    p.add_argument("--entities", default=None)
    p.add_argument("--db", default="cortexos.db")

    # recall
    p = sub.add_parser("recall", help="检索记忆")
    p.add_argument("query")
    p.add_argument("--scope", default="default")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--db", default="cortexos.db")

    # zones
    p = sub.add_parser("zones", help="查看 zone 列表")
    p.add_argument("--scope", default="default")
    p.add_argument("--db", default="cortexos.db")

    # stats
    p = sub.add_parser("stats", help="查看统计")
    p.add_argument("--db", default="cortexos.db")

    # consolidate
    p = sub.add_parser("consolidate", help="触发整合")
    p.add_argument("--scope", default="default")
    p.add_argument("--db", default="cortexos.db")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "serve": cmd_serve,
        "pair-request": cmd_pair_request,
        "pair-approve": cmd_pair_approve,
        "pair-exchange": cmd_pair_exchange,
        "store": cmd_store,
        "recall": cmd_recall,
        "zones": cmd_zones,
        "stats": cmd_stats,
        "consolidate": cmd_consolidate,
    }

    handler = handlers.get(args.command)
    if handler:
        asyncio.run(handler(args))
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

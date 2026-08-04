"""记忆写入管线 —— entities 提取 + embedding + zone 路由 + 质心更新 + facts/edges 落库。

所有写入入口（API / SDK / CLI）统一走此管线，保证：
1. 实体提取（LLM 可用且开启时优先，失败降级启发式）
2. 语义向量生成（失败时降级为无 embedding，路由退回实体/_inbox）
3. 三层 zone 路由（实体 → 语义质心 → _inbox）
4. Zone 质心/条目数增量更新
5. LLM 提取的 facts / edges 落库（配置开启时）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from cortexos.config import Config
from cortexos.embedding.base import Embedder
from cortexos.extract.heuristic import heuristic_extract
from cortexos.models import Edge, Entry, Fact
from cortexos.storage import StorageBackend

logger = logging.getLogger(__name__)


async def extract_entities(
    content: str,
    embedder: Embedder,
    config: Config,
) -> Dict[str, Any]:
    """提取结构化信息（实体/事实/边/有效期）。

    LLM 配置开启且可用时走 LLM 提取，任何异常降级为启发式。
    """
    try:
        from cortexos.extract.llm_extractor import LLMExtractor
        from cortexos.embedding.openai_compat import OpenAICompatEmbedder
        # 仅 OpenAI 兼容 embedder 支持 LLM 提取；TF-IDF 无此能力
        if not isinstance(embedder, OpenAICompatEmbedder) or not embedder.is_available:
            return heuristic_extract(content)
        extractor = LLMExtractor(embedder, config.llm)
        result = await extractor.extract(content)
        if result.get("entities"):
            return result
    except Exception as exc:  # noqa: BLE001 - 提取失败必须降级
        logger.warning("LLM 提取失败，降级启发式: %s", exc)
    return heuristic_extract(content)


async def store_entry(
    entry: Entry,
    backend: StorageBackend,
    embedder: Embedder,
    config: Config,
) -> Entry:
    """执行完整写入管线（原地修改 entry 并落库）。

    Args:
        entry: 待写入条目（content/scope/layer 必填，其余可选）。
        backend: 存储后端。
        embedder: Embedder 实例。
        config: 配置。

    Returns:
        写入后的 entry（zone/embedding/entities 已填充）。
    """
    # 1. 实体提取（调用方未显式指定时）
    extracted: Dict[str, Any] = {}
    if not entry.entities:
        extracted = await extract_entities(entry.content, embedder, config)
        entry.entities = extracted.get("entities", []) or []

    # 2. 语义向量
    if entry.embedding is None:
        try:
            vecs = await embedder.embed([entry.content])
            if vecs and vecs[0]:
                entry.embedding = vecs[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding 生成失败（跳过语义层）: %s", exc)
            entry.embedding = None

    # 3. 时间有效性（LLM 提取的时间线索）
    if entry.valid_until is None and extracted.get("valid_until"):
        entry.valid_until = extracted["valid_until"]

    # 4. 三层 zone 路由（未显式指定 zone 时）
    if not entry.zone or entry.zone == "_inbox":
        from cortexos.zones.router import route_entry
        zones = await backend.list_zones(scope=entry.scope, status="active")
        entry.zone = await route_entry(entry, zones, embedder, config) or "_inbox"

    # 5. Zone 质心增量更新 + 条目数 + 最近访问
    if entry.zone and entry.embedding:
        zone = await backend.get_zone(entry.zone, scope=entry.scope)
        if zone is not None:
            from cortexos.zones.engine import _update_zone_centroid_incremental
            await _update_zone_centroid_incremental(zone, entry)
            zone.entry_count += 1
            zone.last_access = time.time()
            await backend.upsert_zone(zone)

    # 6. 条目落库
    await backend.upsert_entry(entry)

    # 7. facts / edges 落库（LLM 提取且配置开启时）
    now = time.time()
    for f in extracted.get("facts", []) or []:
        try:
            fact = Fact(
                subject=str(f.get("subject", "")),
                predicate=str(f.get("predicate", "")),
                object=str(f.get("object", "")),
                scope=entry.scope,
                entry_id=entry.id,
                confidence=float(f.get("confidence", 1.0)),
                valid_from=now,
                valid_until=entry.valid_until,
            )
            # 冲突消解：同 scope+subject+predicate 的活跃事实先做时间窗口截断/合并
            if config.resolve.enabled:
                from cortexos.lifecycle.resolve import resolve_fact
                existing = await backend.find_facts(
                    scope=entry.scope,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    status="active",
                )
                before = {old.id: old.valid_until for old in existing}
                fact = await resolve_fact(fact, existing, config, now=now)
                # 写回被截断/合并修改的旧事实
                for old in existing:
                    if old.status != "active" or old.valid_until != before.get(old.id):
                        await backend.upsert_fact(old)
            await backend.upsert_fact(fact)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact 落库失败: %s", exc)
    for e in extracted.get("edges", []) or []:
        try:
            await backend.upsert_edge(Edge(
                source=str(e.get("source", "")),
                target=str(e.get("target", "")),
                relation=str(e.get("relation", "")),
                scope=entry.scope,
                entry_id=entry.id,
                weight=float(e.get("weight", 1.0)),
                valid_until=entry.valid_until,
                created_at=now,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge 落库失败: %s", exc)

    return entry

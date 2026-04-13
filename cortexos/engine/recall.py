"""Recall engine — the retrieval system combining TF-IDF, recency, gravity, and frequency."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..config import Config
from ..models.entry import Entry
from ..models.zone import Zone
from .index import MemoryIndex


class RecallEngine:
    """Multi-factor recall engine.

    Scoring formula:
        score = w1*text_sim + w2*recency + w3*zone_gravity + w4*access_freq

    Default weights: text_sim=0.4, recency=0.3, zone_gravity=0.2, access_freq=0.1
    """

    def __init__(self, index: MemoryIndex, zones: Dict[str, Zone], config: Config):
        self.index = index
        self.zones = zones
        self.config = config

    def recall(
        self,
        query: str,
        budget: int = 10,
        zones: Optional[List[str]] = None,
        mem_types: Optional[List[str]] = None,
    ) -> List[Entry]:
        """Retrieve the most relevant entries for a query.

        Args:
            query: The search query.
            budget: Maximum number of entries to return.
            zones: Optional zone name filter.
            mem_types: Optional memory type filter.

        Returns:
            List of Entry objects, sorted by relevance, up to budget.
        """
        # Step 1: Get TF-IDF candidates
        tfidf_results = self.index.tfidf_search(query, zones=zones, mem_types=mem_types)

        # Step 2: Also get entity-match candidates
        from .router import ZoneRouter
        entities = ZoneRouter.extract_entities(None, query)
        entity_ids = self.index.entity_lookup(entities) if entities else set()

        # Merge candidate sets
        all_candidates: Dict[str, float] = {}
        for eid, score in tfidf_results:
            all_candidates[eid] = score

        for eid in entity_ids:
            if eid not in all_candidates:
                all_candidates[eid] = 0.3  # Entity match gets a base score

        if not all_candidates:
            return []

        # Step 3: Multi-factor scoring
        w = self.config.recall_weights
        now = datetime.now(timezone.utc)
        scored: List[tuple] = []

        # Find max access count for normalization
        max_access = max(
            (self.index.entries[eid].access_count for eid in all_candidates if eid in self.index.entries),
            default=1,
        ) or 1

        # Find max gravity for normalization
        max_gravity = max(
            (z.gravity for z in self.zones.values()),
            default=1.0,
        ) or 1.0

        for eid, text_score in all_candidates.items():
            entry = self.index.entries.get(eid)
            if entry is None:
                continue

            # Apply zone/type filters for entity-matched entries
            if zones and entry.zone not in zones and entry.zone != "_inbox":
                continue
            if mem_types and entry.mem_type not in mem_types:
                continue

            # Recency score (exponential decay)
            try:
                entry_dt = datetime.fromisoformat(entry.ts)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                age_days = (now - entry_dt).total_seconds() / 86400
            except (ValueError, TypeError):
                age_days = 30.0

            half_life = self.config.recency_half_life_days
            recency_score = math.exp(-0.693 * age_days / half_life)

            # Zone gravity score
            zone = self.zones.get(entry.zone)
            gravity_score = (zone.gravity / max_gravity) if zone else 0.0

            # Access frequency score
            freq_score = entry.access_count / max_access

            # Combined score
            final_score = (
                w.text_similarity * text_score
                + w.recency * recency_score
                + w.zone_gravity * gravity_score
                + w.access_frequency * freq_score
            )

            scored.append((entry, final_score))

        # Sort and truncate
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [entry for entry, _ in scored[:budget]]

        # Touch recalled entries (update access stats)
        for entry in results:
            entry.touch()

        return results

    def session_context(self, budget: int = 500) -> str:
        """Generate session startup context string.

        Includes: active zone summaries, pending tasks, recent entries.

        Args:
            budget: Approximate token budget for the context string.

        Returns:
            Formatted context string.
        """
        parts = []

        # Active zones (top 5 by gravity)
        active_zones = sorted(
            [z for z in self.zones.values() if z.status.value == "active"],
            key=lambda z: z.gravity,
            reverse=True,
        )[:5]

        if active_zones:
            parts.append("## Active Knowledge Zones")
            for z in active_zones:
                parts.append(f"- **{z.name}** (gravity: {z.gravity:.1f}): {z.scope}")

        # Recent entries (top 5 by time)
        all_entries = sorted(
            self.index.entries.values(),
            key=lambda e: e.ts,
            reverse=True,
        )[:5]

        if all_entries:
            parts.append("\n## Recent Memories")
            for e in all_entries:
                content_preview = e.content[:100] + "..." if len(e.content) > 100 else e.content
                parts.append(f"- [{e.zone}] {content_preview}")

        result = "\n".join(parts)

        # Rough token budget enforcement (1 token ≈ 4 chars)
        max_chars = budget * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...(truncated)"

        return result

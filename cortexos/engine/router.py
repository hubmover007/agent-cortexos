"""Three-layer Zone router: entity lookup → TF-IDF scope match → _inbox fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..models.entry import Entry
from ..models.zone import Zone
from .index import MemoryIndex, tokenize


class ZoneRouter:
    """Routes incoming entries to the best-matching Zone.

    Three-layer routing strategy:
    1. Entity lookup: if entry entities match a zone's known entities, route there.
    2. TF-IDF scope match: compare entry content with zone scope keywords.
    3. Fallback: route to _inbox.
    """

    def __init__(self, zones: Dict[str, Zone], index: MemoryIndex):
        self.zones = zones
        self.index = index

    def route(self, entry: Entry) -> str:
        """Determine the best zone for an entry.

        Args:
            entry: The entry to route.

        Returns:
            Zone name (string). Falls back to '_inbox'.
        """
        # Layer 1: Entity reverse lookup
        zone_name = self._entity_match(entry)
        if zone_name:
            return zone_name

        # Layer 2: TF-IDF scope matching
        zone_name = self._scope_match(entry)
        if zone_name:
            return zone_name

        # Layer 3: Fallback to _inbox
        return "_inbox"

    def _entity_match(self, entry: Entry) -> Optional[str]:
        """Check if any of the entry's entities are bound to a zone."""
        if not entry.entities:
            return None

        best_zone = None
        best_overlap = 0

        for zone in self.zones.values():
            if zone.status.value != "active":
                continue
            overlap = len(
                set(e.lower() for e in entry.entities)
                & set(e.lower() for e in zone.entities)
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_zone = zone.name

        return best_zone if best_overlap > 0 else None

    def _scope_match(self, entry: Entry, threshold: float = 0.15) -> Optional[str]:
        """Match entry content against zone scope keywords using TF-IDF similarity."""
        entry_tokens = set(tokenize(entry.content))
        if not entry_tokens:
            return None

        best_zone = None
        best_score = threshold  # minimum threshold

        for zone in self.zones.values():
            if zone.status.value != "active":
                continue
            if not zone.keywords:
                continue

            zone_tokens = set(t.lower() for t in zone.keywords)
            if not zone_tokens:
                continue

            # Jaccard-like similarity weighted by gravity
            intersection = entry_tokens & zone_tokens
            union = entry_tokens | zone_tokens

            if union:
                similarity = len(intersection) / len(union)
                # Weight by zone gravity (normalized)
                weighted_score = similarity * min(zone.gravity, 5.0) / 5.0

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_zone = zone.name

        return best_zone

    def extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction heuristic.

        Extracts capitalized words and common patterns.
        In production, plug in NER model here.
        """
        import re
        entities = []

        # Capitalized words (likely proper nouns) — skip sentence starters
        words = text.split()
        for i, word in enumerate(words):
            clean = re.sub(r"[^\w]", "", word)
            if clean and clean[0].isupper() and len(clean) > 2 and i > 0:
                entities.append(clean)

        # Technical patterns: URLs, IPs, paths
        urls = re.findall(r"https?://\S+", text)
        entities.extend(urls)

        ips = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)
        entities.extend(ips)

        return list(set(entities))

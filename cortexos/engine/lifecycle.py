"""Lifecycle management — consolidate (warm path) and garden (cold path)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from ..config import Config
from ..models.entry import Entry
from ..models.zone import Zone, ZoneStatus
from .index import MemoryIndex, tokenize
from .storage import JSONLBackend


class LifecycleManager:
    """Manages the warm-path and cold-path lifecycle operations.

    - consolidate(): warm path — zone discovery, dedup, scope updates
    - garden(): cold path — gravity decay, dormant/archive transitions
    """

    def __init__(
        self,
        storage: JSONLBackend,
        index: MemoryIndex,
        zones: Dict[str, Zone],
        config: Config,
    ):
        self.storage = storage
        self.index = index
        self.zones = zones
        self.config = config

    def consolidate(self) -> Dict[str, int]:
        """Warm-path operations: run between sessions.

        Returns:
            Summary dict with counts of operations performed.
        """
        stats = {"zones_discovered": 0, "scopes_updated": 0, "duplicates_found": 0}

        # 1. Discover new zones from _inbox
        discovered = self._discover_zones()
        stats["zones_discovered"] = len(discovered)

        # 2. Update zone scope keywords
        stats["scopes_updated"] = self._update_scopes()

        return stats

    def garden(self) -> Dict[str, int]:
        """Cold-path operations: run periodically (cron or manual).

        Returns:
            Summary dict with counts of operations performed.
        """
        stats = {"gravity_decayed": 0, "zones_dormant": 0, "zones_archived": 0}
        now = datetime.now(timezone.utc)
        dormant_threshold = timedelta(days=self.config.zone_config.dormant_days)
        archive_threshold = timedelta(days=self.config.zone_config.archive_days)

        for zone in list(self.zones.values()):
            if zone.name == "_inbox":
                continue

            # Check last access
            if zone.last_access:
                try:
                    last = datetime.fromisoformat(zone.last_access)
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    age = now - last
                except (ValueError, TypeError):
                    age = dormant_threshold + timedelta(days=1)
            else:
                age = dormant_threshold + timedelta(days=1)

            if zone.status == ZoneStatus.ACTIVE:
                # Gravity decay for inactive zones
                if age > dormant_threshold:
                    zone.decay_gravity(self.config.zone_config.gravity_decay)
                    stats["gravity_decayed"] += 1

                    # Transition to dormant
                    if zone.gravity < 0.1:
                        zone.status = ZoneStatus.DORMANT
                        stats["zones_dormant"] += 1

            elif zone.status == ZoneStatus.DORMANT:
                if age > archive_threshold:
                    zone.status = ZoneStatus.ARCHIVED
                    stats["zones_archived"] += 1

        return stats

    def _discover_zones(self) -> List[Zone]:
        """Detect patterns in _inbox and create new zones."""
        inbox_ids = self.index.zone_entries("_inbox")
        if len(inbox_ids) < self.config.zone_config.emergence_threshold:
            return []

        # Group inbox entries by shared entities
        entity_groups: Dict[str, List[str]] = defaultdict(list)
        for eid in inbox_ids:
            entry = self.index.entries.get(eid)
            if entry:
                for entity in entry.entities:
                    entity_groups[entity.lower()].append(eid)

        discovered = []
        threshold = self.config.zone_config.emergence_threshold

        for entity, entry_ids in entity_groups.items():
            if len(entry_ids) < threshold:
                continue
            if entity in self.zones:
                continue

            # Create new zone from entity cluster
            entries = [self.index.entries[eid] for eid in entry_ids if eid in self.index.entries]
            all_tokens = []
            for e in entries:
                all_tokens.extend(tokenize(e.content))

            # Top keywords
            from collections import Counter
            token_counts = Counter(all_tokens)
            keywords = [t for t, _ in token_counts.most_common(10)]

            zone = Zone(
                name=entity,
                scope=f"Zone auto-discovered from entity '{entity}'",
                keywords=keywords,
                entities=[entity],
                entry_count=len(entry_ids),
            )
            self.zones[zone.name] = zone

            # Reroute entries to new zone
            for eid in entry_ids:
                entry = self.index.entries.get(eid)
                if entry and entry.zone == "_inbox":
                    # Update zone assignment
                    self.index.zone_index["_inbox"].discard(eid)
                    entry.zone = zone.name
                    self.index.zone_index[zone.name].add(eid)

            discovered.append(zone)

        return discovered

    def _update_scopes(self) -> int:
        """Update zone scope keywords based on current entries."""
        updated = 0
        for zone in self.zones.values():
            if zone.status != ZoneStatus.ACTIVE:
                continue

            entry_ids = self.index.zone_entries(zone.name)
            if not entry_ids:
                continue

            all_tokens = []
            all_entities = set()
            for eid in entry_ids:
                entry = self.index.entries.get(eid)
                if entry:
                    all_tokens.extend(tokenize(entry.content))
                    all_entities.update(e.lower() for e in entry.entities)

            if all_tokens:
                from collections import Counter
                token_counts = Counter(all_tokens)
                new_keywords = [t for t, _ in token_counts.most_common(10)]

                if set(new_keywords) != set(zone.keywords):
                    zone.keywords = new_keywords
                    zone.entities = list(all_entities)
                    zone.entry_count = len(entry_ids)
                    updated += 1

        return updated

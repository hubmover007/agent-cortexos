"""Zone CRUD + discovery + evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import Config
from ..models.zone import Zone, ZoneStatus
from ..engine.index import MemoryIndex


class ZoneManager:
    """Manages Zone CRUD operations, discovery, and evolution.

    Zones metadata is persisted in zones.yaml.
    """

    def __init__(self, zones: Dict[str, Zone], index: MemoryIndex, config: Config):
        self._zones = zones
        self._index = index
        self._config = config

    def list(self, include_dormant: bool = True) -> List[Zone]:
        """List all zones.

        Args:
            include_dormant: If False, exclude dormant and archived zones.
        """
        result = list(self._zones.values())
        if not include_dormant:
            result = [z for z in result if z.status == ZoneStatus.ACTIVE]
        return sorted(result, key=lambda z: z.gravity, reverse=True)

    def create(self, name: str, scope: str = "", **kwargs) -> Zone:
        """Manually create a zone.

        Args:
            name: Unique zone name.
            scope: Description of what this zone covers.

        Returns:
            The newly created Zone.

        Raises:
            ValueError: If a zone with this name already exists.
        """
        if name in self._zones:
            raise ValueError(f"Zone '{name}' already exists")

        zone = Zone(name=name, scope=scope, **kwargs)
        self._zones[name] = zone
        return zone

    def get(self, name: str) -> Optional[Zone]:
        """Get a zone by name."""
        return self._zones.get(name)

    def stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific zone.

        Returns:
            Dict with entry_count, gravity, last_access, status.
        """
        zone = self._zones.get(name)
        if zone is None:
            raise ValueError(f"Zone '{name}' not found")

        entry_ids = self._index.zone_entries(name)
        return {
            "name": zone.name,
            "entry_count": len(entry_ids),
            "gravity": zone.gravity,
            "last_access": zone.last_access,
            "status": zone.status.value,
            "keywords": zone.keywords[:5],
        }

    def discover(self) -> List[Zone]:
        """Trigger zone emergence detection from _inbox.

        Delegates to LifecycleManager._discover_zones().
        Returns list of newly discovered zones.
        """
        from ..engine.lifecycle import LifecycleManager
        from ..engine.storage import JSONLBackend

        # Create a temporary lifecycle manager for discovery
        storage = JSONLBackend(self._config.memory_dir)
        lm = LifecycleManager(storage, self._index, self._zones, self._config)
        return lm._discover_zones()

    def save(self, path: Optional[Path] = None) -> None:
        """Persist zones metadata to YAML file."""
        path = path or self._config.zones_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: zone.to_dict() for name, zone in self._zones.items()}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def load(path: Path) -> Dict[str, Zone]:
        """Load zones metadata from YAML file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {name: Zone.from_dict(d) for name, d in data.items()}

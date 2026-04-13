"""Zone data model — the knowledge domain container."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ZoneStatus(str, enum.Enum):
    """Zone lifecycle status."""
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"


@dataclass
class Zone:
    """A knowledge domain zone in CortexOS.

    Zones are the gravitational fields of memory. Entries are routed
    to zones based on semantic similarity, entity overlap, and scope matching.

    Attributes:
        name: Unique zone identifier.
        scope: Description of what this zone covers.
        status: Current lifecycle status.
        gravity: Attraction strength (higher = more entries get routed here).
        keywords: TF-IDF keywords representing this zone's scope.
        entities: Known entities belonging to this zone.
        entry_count: Number of entries in this zone.
        created_at: When this zone was created/emerged.
        last_access: Last time an entry was routed or recalled from this zone.
        meta: Arbitrary metadata.
    """
    name: str
    scope: str = ""
    status: ZoneStatus = ZoneStatus.ACTIVE
    gravity: float = 1.0
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    entry_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_access: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "scope": self.scope,
            "status": self.status.value,
            "gravity": self.gravity,
            "keywords": self.keywords,
            "entities": self.entities,
            "entry_count": self.entry_count,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        """Deserialize from dictionary."""
        status = data.get("status", "active")
        if isinstance(status, str):
            status = ZoneStatus(status)
        return cls(
            name=data["name"],
            scope=data.get("scope", ""),
            status=status,
            gravity=data.get("gravity", 1.0),
            keywords=data.get("keywords", []),
            entities=data.get("entities", []),
            entry_count=data.get("entry_count", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_access=data.get("last_access"),
            meta=data.get("meta", {}),
        )

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = datetime.now(timezone.utc).isoformat()

    def boost_gravity(self, amount: float = 0.1) -> None:
        """Increase gravity when a new entry is routed here."""
        self.gravity += amount
        self.touch()

    def decay_gravity(self, factor: float = 0.95) -> None:
        """Apply gravity decay during garden operations."""
        self.gravity *= factor

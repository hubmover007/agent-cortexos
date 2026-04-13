"""Entry data model — the fundamental unit of memory in CortexOS."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Entry:
    """A single memory entry in CortexOS.

    Attributes:
        id: Unique identifier (UUID).
        content: The actual memory content text.
        zone: Zone this entry belongs to.
        mem_type: Type of memory (experience, decision, fact, note, etc.).
        entities: Extracted entities from content.
        ts: Timestamp of creation.
        agent_id: The agent that created this entry.
        access_count: Number of times this entry was recalled.
        last_accessed: Last time this entry was recalled.
        meta: Arbitrary metadata.
    """
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    zone: str = "_inbox"
    mem_type: str = "note"
    entities: List[str] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_id: str = "default"
    access_count: int = 0
    last_accessed: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "content": self.content,
            "zone": self.zone,
            "type": self.mem_type,
            "entities": self.entities,
            "ts": self.ts,
            "agent_id": self.agent_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entry":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data["content"],
            zone=data.get("zone", "_inbox"),
            mem_type=data.get("type", "note"),
            entities=data.get("entities", []),
            ts=data.get("ts", datetime.now(timezone.utc).isoformat()),
            agent_id=data.get("agent_id", "default"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            meta=data.get("meta", {}),
        )

    def touch(self) -> None:
        """Update access stats when this entry is recalled."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()

"""Task and FollowUp data models."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TaskStatus(str, enum.Enum):
    """Task lifecycle status."""
    TODO = "todo"
    DOING = "doing"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass
class FollowUp:
    """A scheduled follow-up action for a task.

    Attributes:
        id: Unique identifier.
        action: Description of what to do.
        due: When this follow-up should be triggered (ISO 8601).
        done: Whether this follow-up has been completed.
    """
    action: str
    due: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "action": self.action, "due": self.due, "done": self.done}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FollowUp":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            action=data["action"],
            due=data.get("due"),
            done=data.get("done", False),
        )


@dataclass
class Task:
    """A task tracked by CortexOS.

    Attributes:
        id: Unique identifier (UUID).
        summary: Brief description of the task.
        status: Current status.
        priority: Priority level (1=highest, 5=lowest).
        due: Due date (ISO 8601).
        zone: Related zone.
        follow_ups: Scheduled follow-up actions.
        created_at: Creation timestamp.
        completed_at: Completion timestamp.
        agent_id: The agent that owns this task.
        meta: Arbitrary metadata.
    """
    summary: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.TODO
    priority: int = 3
    due: Optional[str] = None
    zone: Optional[str] = None
    follow_ups: List[FollowUp] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    agent_id: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "status": self.status.value,
            "priority": self.priority,
            "due": self.due,
            "zone": self.zone,
            "follow_ups": [f.to_dict() for f in self.follow_ups],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "agent_id": self.agent_id,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        status = data.get("status", "todo")
        if isinstance(status, str):
            status = TaskStatus(status)
        follow_ups = [FollowUp.from_dict(f) for f in data.get("follow_ups", [])]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            summary=data["summary"],
            status=status,
            priority=data.get("priority", 3),
            due=data.get("due"),
            zone=data.get("zone"),
            follow_ups=follow_ups,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=data.get("completed_at"),
            agent_id=data.get("agent_id", "default"),
            meta=data.get("meta", {}),
        )

"""Task CRUD + follow-up scheduling."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import Config
from ..models.task import Task, TaskStatus, FollowUp


class TaskManager:
    """Manages tasks and follow-up actions.

    Tasks are persisted in tasks.yaml.
    """

    def __init__(self, tasks: Dict[str, Task], config: Config):
        self._tasks = tasks
        self._config = config

    def create(
        self,
        summary: str,
        priority: int = 3,
        due: Optional[str] = None,
        zone: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> Task:
        """Create a new task.

        Args:
            summary: Brief description of the task.
            priority: 1 (highest) to 5 (lowest).
            due: Due date in ISO 8601 format.
            zone: Related zone name.
            agent_id: Owning agent ID.

        Returns:
            The newly created Task.
        """
        task = Task(
            summary=summary,
            priority=priority,
            due=due,
            zone=zone,
            agent_id=agent_id or self._config.agent_id,
            **kwargs,
        )
        self._tasks[task.id] = task
        return task

    def list(
        self,
        status: Optional[TaskStatus] = None,
        zone: Optional[str] = None,
    ) -> List[Task]:
        """List tasks with optional filters.

        Args:
            status: Filter by status. None = all statuses.
            zone: Filter by zone. None = all zones.

        Returns:
            Sorted list of tasks (by priority, then due date).
        """
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if zone is not None:
            tasks = [t for t in tasks if t.zone == zone]

        return sorted(tasks, key=lambda t: (t.priority, t.due or "9999"))

    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> Task:
        """Update task fields.

        Args:
            task_id: The task ID.
            **kwargs: Fields to update (summary, priority, due, status, zone).

        Returns:
            The updated Task.

        Raises:
            ValueError: If task not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found")

        for key, value in kwargs.items():
            if key == "status" and isinstance(value, str):
                value = TaskStatus(value)
            if hasattr(task, key):
                setattr(task, key, value)

        return task

    def complete(self, task_id: str) -> Task:
        """Mark a task as completed.

        Args:
            task_id: The task ID.

        Returns:
            The completed Task.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found")

        task.status = TaskStatus.DONE
        task.completed_at = datetime.now(timezone.utc).isoformat()
        return task

    def add_follow_up(
        self,
        task_id: str,
        action: str,
        due: Optional[str] = None,
    ) -> FollowUp:
        """Add a follow-up action to a task.

        Args:
            task_id: The parent task ID.
            action: Description of the follow-up action.
            due: When this follow-up should be triggered.

        Returns:
            The created FollowUp.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found")

        follow_up = FollowUp(action=action, due=due)
        task.follow_ups.append(follow_up)
        return follow_up

    def pending_follow_ups(self) -> List[dict]:
        """Get all pending follow-ups that are due.

        Returns:
            List of dicts with task_id, task_summary, and follow_up info.
        """
        now = datetime.now(timezone.utc).isoformat()
        results = []

        for task in self._tasks.values():
            if task.status == TaskStatus.DONE:
                continue
            for fu in task.follow_ups:
                if fu.done:
                    continue
                if fu.due and fu.due <= now:
                    results.append({
                        "task_id": task.id,
                        "task_summary": task.summary,
                        "follow_up_id": fu.id,
                        "action": fu.action,
                        "due": fu.due,
                    })

        return results

    def save(self, path: Optional[Path] = None) -> None:
        """Persist tasks to YAML file."""
        path = path or self._config.tasks_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {tid: task.to_dict() for tid, task in self._tasks.items()}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def load(path: Path) -> Dict[str, Task]:
        """Load tasks from YAML file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {tid: Task.from_dict(d) for tid, d in data.items()}

"""CortexOS — the main facade class providing unified API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config
from .models.entry import Entry
from .models.zone import Zone
from .engine.storage import JSONLBackend
from .engine.index import MemoryIndex
from .engine.router import ZoneRouter
from .engine.recall import RecallEngine
from .engine.lifecycle import LifecycleManager
from .zones.manager import ZoneManager
from .tasks.manager import TaskManager


class CortexOS:
    """The Cognitive Operating System for AI Agents.

    Provides a unified API for:
    - store/recall: memory management
    - zones: knowledge domain management
    - tasks: task and follow-up management
    - lifecycle: consolidation and gardening
    - session_context: startup context injection

    Usage:
        import cortexos
        cx = cortexos.init()
        cx.store("learned something important")
        results = cx.recall("what did I learn?")
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Any = None,
    ):
        # Load configuration
        self._config = Config.load(workspace=workspace, agent_id=agent_id, config=config)

        # Initialize storage backend
        self._storage = JSONLBackend(self._config.memory_dir)

        # Load persisted zones and tasks
        zone_data = ZoneManager.load(self._config.zones_file)
        task_data = TaskManager.load(self._config.tasks_file)

        # Build in-memory index
        self._index = MemoryIndex()
        entries = self._storage.load_all()
        self._index.build(entries)

        # Initialize zone router
        self._zones_data = zone_data
        self._router = ZoneRouter(self._zones_data, self._index)

        # Initialize recall engine
        self._recall_engine = RecallEngine(self._index, self._zones_data, self._config)

        # Initialize managers
        self.zones = ZoneManager(self._zones_data, self._index, self._config)
        self.tasks = TaskManager(task_data, self._config)
        self.lifecycle = LifecycleManager(
            self._storage, self._index, self._zones_data, self._config
        )

    def store(
        self,
        content: str,
        mem_type: str = "note",
        entities: Optional[List[str]] = None,
        zone: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Entry:
        """Store a new memory entry.

        The entry is automatically routed to the best matching zone
        unless a zone is explicitly specified.

        Args:
            content: The memory content text.
            mem_type: Type of memory (note, experience, decision, fact).
            entities: Extracted entities. If None, auto-extracted.
            zone: Explicit zone assignment. If None, auto-routed.
            meta: Arbitrary metadata dict.

        Returns:
            The stored Entry with assigned zone and ID.
        """
        # Auto-extract entities if not provided
        if entities is None:
            entities = self._router.extract_entities(content)

        entry = Entry(
            content=content,
            mem_type=mem_type,
            entities=entities,
            agent_id=self._config.agent_id,
            meta=meta or {},
        )

        # Route to zone (auto or explicit)
        if zone:
            entry.zone = zone
        else:
            entry.zone = self._router.route(entry)

        # Persist to JSONL
        self._storage.append(entry)

        # Update in-memory index
        self._index.add(entry)

        # Boost zone gravity if routed to an existing zone
        if entry.zone in self._zones_data and entry.zone != "_inbox":
            self._zones_data[entry.zone].boost_gravity()
            self._zones_data[entry.zone].entry_count += 1

        return entry

    def recall(
        self,
        query: str,
        budget: int = 10,
        zones: Optional[List[str]] = None,
        mem_types: Optional[List[str]] = None,
    ) -> List[Entry]:
        """Retrieve relevant memories for a query.

        Uses multi-factor scoring: TF-IDF similarity, recency,
        zone gravity, and access frequency.

        Args:
            query: Search query text.
            budget: Maximum number of entries to return.
            zones: Optional zone filter.
            mem_types: Optional memory type filter.

        Returns:
            List of Entry objects, sorted by relevance.
        """
        return self._recall_engine.recall(
            query=query, budget=budget, zones=zones, mem_types=mem_types
        )

    def session_context(self, budget: int = 500) -> str:
        """Generate context string for session startup injection.

        Args:
            budget: Approximate token budget.

        Returns:
            Formatted context string for system prompt injection.
        """
        return self._recall_engine.session_context(budget=budget)

    def stats(self) -> Dict[str, Any]:
        """Get overall system statistics.

        Returns:
            Dict with total_entries, zone_count, task_count, etc.
        """
        active_zones = [z for z in self._zones_data.values() if z.status.value == "active"]
        pending_tasks = [t for t in self.tasks._tasks.values() if t.status.value in ("todo", "doing")]

        return {
            "total_entries": self._index.total_entries,
            "zone_count": len(self._zones_data),
            "active_zones": len(active_zones),
            "task_count": len(self.tasks._tasks),
            "pending_tasks": len(pending_tasks),
            "workspace": str(self._config.workspace),
            "agent_id": self._config.agent_id,
        }

    def save(self) -> None:
        """Persist all metadata (zones, tasks) to disk.

        Entry data is already persisted on store().
        Call this to save zone and task metadata changes.
        """
        self.zones.save()
        self.tasks.save()

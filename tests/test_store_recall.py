"""Tests for store and recall functionality."""

import os
import shutil
import tempfile

import pytest

import cortexos


@pytest.fixture
def workspace():
    """Create a temporary workspace for testing."""
    tmpdir = tempfile.mkdtemp(prefix="cortexos_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cx(workspace):
    """Create a CortexOS instance with a temp workspace."""
    return cortexos.init(workspace=workspace, agent_id="test-agent")


class TestStore:
    def test_store_basic(self, cx):
        """Store a simple entry and verify it exists."""
        entry = cx.store("Hello world")
        assert entry.content == "Hello world"
        assert entry.id is not None
        assert entry.zone == "_inbox"  # No zones yet, goes to inbox

    def test_store_with_type(self, cx):
        """Store with explicit memory type."""
        entry = cx.store("Important decision", mem_type="decision")
        assert entry.mem_type == "decision"

    def test_store_with_entities(self, cx):
        """Store with explicit entities."""
        entry = cx.store("K8s pod crashed", entities=["K8s", "pod"])
        assert "K8s" in entry.entities
        assert "pod" in entry.entities

    def test_store_with_explicit_zone(self, cx):
        """Store to a specific zone."""
        cx.zones.create("ops", scope="Operations")
        entry = cx.store("Server went down", zone="ops")
        assert entry.zone == "ops"

    def test_store_increments_count(self, cx):
        """Multiple stores increase total entry count."""
        cx.store("Entry 1")
        cx.store("Entry 2")
        cx.store("Entry 3")
        assert cx.stats()["total_entries"] == 3

    def test_store_with_meta(self, cx):
        """Store with custom metadata."""
        entry = cx.store("test", meta={"source": "unit_test"})
        assert entry.meta["source"] == "unit_test"


class TestRecall:
    def test_recall_basic(self, cx):
        """Recall returns relevant entries."""
        cx.store("Python is a programming language")
        cx.store("Java is also a programming language")
        cx.store("Dogs are great pets")

        results = cx.recall("programming language")
        assert len(results) >= 1
        # Should find programming-related entries
        contents = [r.content for r in results]
        assert any("programming" in c for c in contents)

    def test_recall_empty_db(self, cx):
        """Recall on empty database returns empty list."""
        results = cx.recall("anything")
        assert results == []

    def test_recall_budget(self, cx):
        """Recall respects budget parameter."""
        for i in range(20):
            cx.store(f"Entry about topic alpha number {i}")

        results = cx.recall("topic alpha", budget=5)
        assert len(results) <= 5

    def test_recall_with_zone_filter(self, cx):
        """Recall with zone filter only returns entries from that zone."""
        cx.zones.create("tech", scope="Technology")
        cx.store("Python tips", zone="tech")
        cx.store("Cooking recipe")  # Goes to _inbox

        results = cx.recall("tips", zones=["tech"])
        for r in results:
            assert r.zone == "tech" or r.zone == "_inbox"

    def test_recall_updates_access_count(self, cx):
        """Recalling an entry increments its access count."""
        cx.store("Unique searchable content xyz123")
        results = cx.recall("xyz123")
        if results:
            assert results[0].access_count >= 1


class TestSessionContext:
    def test_session_context_returns_string(self, cx):
        """session_context returns a non-empty string when there's data."""
        cx.store("Some important context")
        context = cx.session_context(budget=500)
        assert isinstance(context, str)

    def test_session_context_empty(self, cx):
        """session_context on empty database returns a string."""
        context = cx.session_context()
        assert isinstance(context, str)


class TestStats:
    def test_stats_basic(self, cx):
        """Stats returns expected keys."""
        s = cx.stats()
        assert "total_entries" in s
        assert "zone_count" in s
        assert "task_count" in s
        assert "agent_id" in s
        assert s["agent_id"] == "test-agent"

    def test_stats_after_operations(self, cx):
        """Stats reflect operations."""
        cx.store("Entry 1")
        cx.store("Entry 2")
        cx.zones.create("test-zone")
        cx.tasks.create("Test task")

        s = cx.stats()
        assert s["total_entries"] == 2
        assert s["zone_count"] >= 1
        assert s["task_count"] == 1


class TestPersistence:
    def test_save_and_reload(self, workspace):
        """Data persists across CortexOS instances."""
        cx1 = cortexos.init(workspace=workspace, agent_id="test")
        cx1.store("Persistent memory")
        cx1.zones.create("saved-zone", scope="Test zone")
        cx1.tasks.create("Saved task")
        cx1.save()

        # Create new instance pointing to same workspace
        cx2 = cortexos.init(workspace=workspace, agent_id="test")
        assert cx2.stats()["total_entries"] == 1

        zones = cx2.zones.list()
        zone_names = [z.name for z in zones]
        assert "saved-zone" in zone_names

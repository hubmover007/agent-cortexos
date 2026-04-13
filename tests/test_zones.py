"""Tests for Zone lifecycle and management."""

import shutil
import tempfile

import pytest

import cortexos
from cortexos.models.zone import ZoneStatus


@pytest.fixture
def workspace():
    tmpdir = tempfile.mkdtemp(prefix="cortexos_test_zones_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cx(workspace):
    return cortexos.init(workspace=workspace)


class TestZoneCRUD:
    def test_create_zone(self, cx):
        zone = cx.zones.create("devops", scope="DevOps practices and tools")
        assert zone.name == "devops"
        assert zone.scope == "DevOps practices and tools"
        assert zone.status == ZoneStatus.ACTIVE
        assert zone.gravity == 1.0

    def test_create_duplicate_raises(self, cx):
        cx.zones.create("test")
        with pytest.raises(ValueError, match="already exists"):
            cx.zones.create("test")

    def test_list_zones(self, cx):
        cx.zones.create("a", scope="Zone A")
        cx.zones.create("b", scope="Zone B")
        zones = cx.zones.list()
        names = [z.name for z in zones]
        assert "a" in names
        assert "b" in names

    def test_list_excludes_dormant(self, cx):
        z = cx.zones.create("dormant-zone")
        z.status = ZoneStatus.DORMANT
        
        active = cx.zones.list(include_dormant=False)
        names = [z.name for z in active]
        assert "dormant-zone" not in names

    def test_zone_stats(self, cx):
        cx.zones.create("stats-zone", scope="For stats testing")
        cx.store("Entry in stats zone", zone="stats-zone")
        
        stats = cx.zones.stats("stats-zone")
        assert stats["name"] == "stats-zone"
        assert stats["entry_count"] >= 1
        assert stats["gravity"] > 1.0  # Boosted by store

    def test_zone_stats_not_found(self, cx):
        with pytest.raises(ValueError, match="not found"):
            cx.zones.stats("nonexistent")


class TestZoneRouting:
    def test_route_to_inbox_by_default(self, cx):
        entry = cx.store("Random content with no zone match")
        assert entry.zone == "_inbox"

    def test_route_by_entity(self, cx):
        zone = cx.zones.create("k8s", scope="Kubernetes operations")
        zone.entities = ["kubernetes", "k8s"]
        
        entry = cx.store("K8s pod is restarting", entities=["k8s"])
        assert entry.zone == "k8s"

    def test_explicit_zone_overrides_routing(self, cx):
        cx.zones.create("override", scope="Test")
        entry = cx.store("Some content", zone="override")
        assert entry.zone == "override"


class TestZoneGravity:
    def test_gravity_boost_on_store(self, cx):
        zone = cx.zones.create("boost-test")
        initial_gravity = zone.gravity

        cx.store("Content for boost test", zone="boost-test")
        assert zone.gravity > initial_gravity

    def test_gravity_decay(self, cx):
        zone = cx.zones.create("decay-test")
        zone.gravity = 5.0
        zone.decay_gravity(0.9)
        assert zone.gravity == pytest.approx(4.5, rel=1e-2)


class TestZoneDiscovery:
    def test_discover_zones(self, cx):
        """When inbox has enough entries with shared entity, discover new zone."""
        # Store several entries with the same entity
        for i in range(6):
            cx.store(f"Docker container issue #{i}", entities=["docker"])
        
        discovered = cx.zones.discover()
        # Should discover a 'docker' zone
        zone_names = [z.name for z in discovered]
        assert "docker" in zone_names

    def test_discover_below_threshold(self, cx):
        """Below threshold, no zones are discovered."""
        cx.store("Single docker entry", entities=["docker"])
        discovered = cx.zones.discover()
        assert len(discovered) == 0

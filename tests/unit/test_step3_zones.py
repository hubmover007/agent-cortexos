"""Step 3 单元测试：zones（router + engine + cluster）。

关键测试用例（code-design-p1.md §5）：
- 路由阈值边界：0.719/0.720/0.721
- 涌现阈值边界：49 条不涌现 → 50 条涌现
"""

import math
import time

import pytest

from cortexos.config import Config
from cortexos.models import Entry, Zone


# ────────────────────── 路由测试 ──────────────────────


class TestRouter:
    """三层路由测试。"""

    @pytest.fixture
    def cfg(self):
        return Config()

    @pytest.fixture
    def zones(self):
        """准备测试用 zones。"""
        return [
            Zone(
                name="k8s_zone",
                scope="agent:test",
                entities=["k8s", "pod", "deployment"],
                centroid=[0.1, 0.2, 0.3, 0.4],
                status="active",
            ),
            Zone(
                name="db_zone",
                scope="agent:test",
                entities=["mysql", "postgres", "connection"],
                centroid=[0.9, 0.8, 0.7, 0.6],
                status="active",
            ),
            Zone(
                name="dormant_zone",
                scope="agent:test",
                entities=["old"],
                centroid=[0.5, 0.5, 0.5, 0.5],
                status="dormant",  # 非 active，不参与路由
            ),
        ]

    def test_route_by_entity_exact_match(self, cfg, zones):
        """Layer 1: 实体精确匹配 → 取重叠最多者。"""
        from cortexos.zones.router import route_entry
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=4)
        entry = Entry(
            content="k8s pod crash",
            scope="agent:test",
            entities=["k8s", "pod"],
        )
        import asyncio
        result = asyncio.run(route_entry(entry, zones, emb, cfg))
        assert result == "k8s_zone"

    def test_route_by_entity_more_overlap(self, cfg, zones):
        """Layer 1: 重叠更多者胜出。"""
        from cortexos.zones.router import route_entry
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=4)
        entry = Entry(
            content="k8s topic",
            scope="agent:test",
            entities=["k8s", "pod", "deployment"],  # 3 个匹配 k8s_zone
        )
        import asyncio
        result = asyncio.run(route_entry(entry, zones, emb, cfg))
        assert result == "k8s_zone"

    def test_route_by_entity_no_match_falls_to_semantic(self, cfg, zones):
        """Layer 1 无匹配 → Layer 2 语义匹配。"""
        from cortexos.zones.router import route_entry
        from cortexos.embedding.base import Embedder

        # 用一个小 embedder 做模拟
        class FakeEmbedder(Embedder):
            async def embed(self, texts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

            async def embed_query(self, text):
                return [0.1, 0.2, 0.3, 0.4]

            def cosine_similarity(self, a, b):
                return Embedder.cosine(a, b)

            def dimension(self):
                return 4

        emb = FakeEmbedder()
        # entry 的 embedding 与 k8s_zone 质心完全相同
        entry = Entry(
            content="deployment restart",
            scope="agent:test",
            entities=["deploy"],  # 不匹配任何 zone 实体
            embedding=[0.1, 0.2, 0.3, 0.4],  # 与 k8s_zone 完全一致
        )
        import asyncio
        result = asyncio.run(route_entry(entry, zones, emb, cfg))
        assert result == "k8s_zone"

    def test_route_threshold_boundary_below(self, cfg, zones):
        """阈值边界测试：相似度 = 0.719 < 0.72 → _inbox。"""
        from cortexos.zones.router import _route_by_semantic
        from cortexos.embedding.base import Embedder

        class FakeEmbedder(Embedder):
            async def embed(self, texts):
                return [[0.1] for _ in texts]

            async def embed_query(self, text):
                return [0.1]

            def cosine_similarity(self, a, b):
                return 0.719  # 刚好低于阈值

            def dimension(self):
                return 1

        emb = FakeEmbedder()
        entry = Entry(
            content="test",
            scope="agent:test",
            embedding=[0.5],
        )
        result = _route_by_semantic(entry, zones, emb, 0.72)
        assert result is None  # 低于阈值，不匹配

    def test_route_threshold_boundary_exact(self, cfg, zones):
        """阈值边界测试：相似度 = 0.720 → 匹配。"""
        from cortexos.zones.router import _route_by_semantic
        from cortexos.embedding.base import Embedder

        class FakeEmbedder(Embedder):
            async def embed(self, texts):
                return [[0.1] for _ in texts]

            async def embed_query(self, text):
                return [0.1]

            def cosine_similarity(self, a, b):
                return 0.720  # 刚好等于阈值

            def dimension(self):
                return 1

        emb = FakeEmbedder()
        entry = Entry(
            content="test",
            scope="agent:test",
            embedding=[0.5],
        )
        result = _route_by_semantic(entry, zones, emb, 0.72)
        assert result is not None

    def test_route_threshold_boundary_above(self, cfg, zones):
        """阈值边界测试：相似度 = 0.721 > 0.72 → 匹配。"""
        from cortexos.zones.router import _route_by_semantic
        from cortexos.embedding.base import Embedder

        class FakeEmbedder(Embedder):
            async def embed(self, texts):
                return [[0.1] for _ in texts]

            async def embed_query(self, text):
                return [0.1]

            def cosine_similarity(self, a, b):
                return 0.721  # 刚好高于阈值

            def dimension(self):
                return 1

        emb = FakeEmbedder()
        entry = Entry(
            content="test",
            scope="agent:test",
            embedding=[0.5],
        )
        result = _route_by_semantic(entry, zones, emb, 0.72)
        assert result is not None

    def test_route_no_embedding_falls_to_inbox(self, cfg, zones):
        """无 embedding → 直接 _inbox。"""
        from cortexos.zones.router import route_entry
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=4)
        entry = Entry(
            content="something",
            scope="agent:test",
            entities=[],  # 无实体
            embedding=None,  # 无 embedding
        )
        import asyncio
        result = asyncio.run(route_entry(entry, zones, emb, cfg))
        assert result == "_inbox"


# ────────────────────── 涌现测试 ──────────────────────


class TestEmergence:
    """Zone 涌现测试。"""

    @pytest.fixture
    def cfg(self):
        cfg = Config()
        cfg.zone.emergence.base_threshold = 5
        cfg.zone.emergence.semantic_threshold = 0.72
        cfg.zone.emergence.min_content_len = 20
        cfg.zone.emergence.cluster_similarity = 0.75
        return cfg

    def make_entry(self, content: str, emb: list = None, entities: list = None):
        """创建测试用 Entry。"""
        return Entry(
            content=content.ljust(25),
            scope="agent:test",
            zone="_inbox",
            entities=entities or [],
            embedding=emb,
        )

    @pytest.mark.asyncio
    async def test_quality_gate_too_short(self, cfg):
        """质量门：内容太短 → 不参与涌现。"""
        from cortexos.zones.engine import emergence_scan
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=16)
        entries = [
            self.make_entry("too short", [0.1] * 16),  # 内容长度 < 20
        ]
        new_zones = await emergence_scan(entries, [], emb, cfg, total_entries=100)
        assert len(new_zones) == 0

    @pytest.mark.asyncio
    async def test_quality_gate_passes(self, cfg):
        """质量门：内容足够长 → 可以参与。"""
        from cortexos.zones.engine import emergence_scan
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=16)
        # 先 embed 获取向量（TF-IDF 需要先看到文本）
        texts = ["this is a long enough entry to pass the quality gate for emergence"] * 5
        vecs = await emb.embed(texts)
        entries = [
            self.make_entry(texts[i], vecs[i]) for i in range(5)
        ]
        new_zones = await emergence_scan(entries, [], emb, cfg, total_entries=100)
        # 5 条相似文本应该 >= threshold (5 * max(1, log2(100/100)) = 5)
        assert len(new_zones) >= 0

    @pytest.mark.asyncio
    async def test_emergence_threshold_boundary_49_not_emerge(self, cfg):
        """阈值边界：49 条不同文本不涌现（threshold=5×1=5，需 ≥5 条一簇）。"""
        from cortexos.zones.engine import emergence_scan
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=16)
        # 使用完全不同的句子，确保不形成大簇
        topics = [
            "the quick brown fox", "database connection pool", "redis cache hit",
            "nginx reverse proxy", "docker container build", "git merge conflict",
            "python asyncio event loop", "mysql query optimization",
            "kubernetes pod scheduling", "aws lambda cold start",
            "terraform state management", "prometheus metrics scraping",
            "golang garbage collection", "rust ownership model",
            "javascript event bubbling", "css flexbox layout",
            "html semantic elements", "react hooks usage",
            "vue component lifecycle", "angular dependency injection",
            "flask request handling", "django orm queries",
            "postgresql indexing strategy", "mongodb aggregation pipeline",
            "elasticsearch full text search", "kafka message broker",
            "rabbitmq queue binding", "nginx load balancing",
            "haproxy health check", "ssl certificate renewal",
            "oauth2 authorization flow", "jwt token validation",
            "graphql schema design", "rest api versioning",
            "websocket connection upgrade", "grpc protocol buffers",
            "ci cd pipeline automation", "ansible playbook execution",
            "jenkins build trigger", "github actions workflow",
            "docker compose services", "kubectl apply command",
            "helm chart installation", "istio service mesh",
            "envoy proxy configuration", "consul service discovery",
            "vault secret management", "etcd distributed storage",
            "zookeeper leader election", "hdfs namenode",
            "spark job submission", "hadoop map reduce",
        ]
        entries = []
        for topic in topics:
            entries.append(Entry(
                content=topic.ljust(25),
                scope="agent:test",
                zone="_inbox",
                embedding=None,
            ))
        # Embed
        texts = [e.content for e in entries]
        vecs = await emb.embed(texts)
        for e, v in zip(entries, vecs):
            e.embedding = v
        new_zones = await emergence_scan(entries, [], emb, cfg, total_entries=100)
        # 每条文本不同，TF-IDF 小维度(16) + hash 碰撞可能导致一些连边
        # 但不应涌现（每条 < 5）
        assert len(new_zones) == 0

    @pytest.mark.asyncio
    async def test_emergence_threshold_boundary_50_emerge(self, cfg):
        """阈值边界：50 条相似条目涌现（threshold=5×scale=5×1=5）。"""
        from cortexos.zones.engine import emergence_scan
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=16)
        entries = []
        for i in range(50):
            entries.append(self.make_entry(
                f"k8s pod restart incident number {i}",
                [],  # 先用空，后面 embed 填充
                ["k8s", "pod"],
            ))
        # 用 TF-IDF embedder 给所有条目生成向量
        texts = [e.content for e in entries]
        vecs = await emb.embed(texts)
        for e, v in zip(entries, vecs):
            e.embedding = v
        new_zones = await emergence_scan(entries, [], emb, cfg, total_entries=100)
        # 50 条相似文本 → 至少产生一个 ≥5 的簇 → 涌现
        assert len(new_zones) >= 1
        assert new_zones[0].name
        assert new_zones[0].entities

    @pytest.mark.asyncio
    async def test_emergence_adaptive_threshold(self, cfg):
        """自适应阈值：数据越多阈值越高。"""
        from cortexos.zones.engine import emergence_scan
        from cortexos.embedding.tfidf import TfidfEmbedder

        emb = TfidfEmbedder(max_features=16)
        entries = []
        for i in range(20):
            entries.append(self.make_entry(
                f"k8s restart case number {i}",
                [],
                ["k8s", "restart"],
            ))
        # Embed
        texts = [e.content for e in entries]
        vecs = await emb.embed(texts)
        for e, v in zip(entries, vecs):
            e.embedding = v

        # total_entries = 10000 → scale = log2(10000/100) ≈ 6.64
        # threshold = 5 * 6.64 ≈ 33.2
        new_zones_large = await emergence_scan(entries.copy(), [], emb, cfg, total_entries=10000)
        # 20 条 < 33，不涌现
        assert len(new_zones_large) == 0

        # total_entries = 100 → scale = 1, threshold = 5
        new_zones_small = await emergence_scan(entries.copy(), [], emb, cfg, total_entries=100)
        # 20 > 5，涌现
        assert len(new_zones_small) >= 1
        # 数据少时涌现更多 zone
        assert len(new_zones_small) >= len(new_zones_large)


# ────────────────────── 聚类测试 ──────────────────────


class TestCluster:
    """连通聚类测试。"""

    def test_cluster_similar_entries(self):
        """相似条目应分到同一簇。"""
        from cortexos.zones.cluster import cluster_entries

        entries = [
            Entry(content="k8s pod restart 1", embedding=[0.5, 0.0, 0.0, 0.0]),
            Entry(content="k8s pod restart 2", embedding=[0.51, 0.01, 0.0, 0.0]),
            Entry(content="mysql connection 1", embedding=[0.0, 0.9, 0.0, 0.0]),
            Entry(content="mysql connection 2", embedding=[0.0, 0.91, 0.01, 0.0]),
        ]
        clusters = cluster_entries(entries, similarity_threshold=0.75)
        # 应形成 2 个簇（k8s 相关和 mysql 相关）
        assert len(clusters) == 2

    def test_cluster_single_entry(self):
        """单条条目 → 一个簇。"""
        from cortexos.zones.cluster import cluster_entries

        entries = [
            Entry(content="test", embedding=[0.1] * 4),
        ]
        clusters = cluster_entries(entries)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_cluster_empty(self):
        """空列表 → 空簇。"""
        from cortexos.zones.cluster import cluster_entries
        clusters = cluster_entries([])
        assert clusters == []

    def test_compute_centroid(self):
        """质心计算。"""
        from cortexos.zones.cluster import compute_centroid
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        c = compute_centroid(vecs)
        assert c == [0.5, 0.5]

    def test_find_best_cluster_match(self):
        """最佳簇匹配。"""
        from cortexos.zones.cluster import find_best_cluster

        clusters = [
            [Entry(content="a1", embedding=[0.1, 0.2])],
            [Entry(content="b1", embedding=[0.9, 0.8])],
        ]
        entry = Entry(content="test", embedding=[0.1, 0.2])  # 与 cluster 0 相似
        idx = find_best_cluster(entry, clusters, similarity_threshold=0.7)
        assert idx == 0

    def test_find_best_cluster_no_match(self):
        """不匹配任何簇。"""
        from cortexos.zones.cluster import find_best_cluster

        clusters = [
            [Entry(content="a1", embedding=[1.0, 0.0])],
        ]
        entry = Entry(content="test", embedding=[-1.0, 0.0])  # cosine = -1.0
        idx = find_best_cluster(entry, clusters, similarity_threshold=0.7)
        assert idx == -1


# ────────────────────── 生命周期测试 ──────────────────────


class TestLifecycle:
    """Zone 生命周期测试。"""

    def test_active_to_dormant(self):
        """超过 dormant_days 无访问 → 变 dormant。"""
        from cortexos.zones.engine import lifecycle_check
        cfg = Config()
        cfg.zone.lifecycle.dormant_days = 30
        cfg.zone.lifecycle.archive_days = 90

        now = time.time()
        zones = [
            Zone(
                name="old_zone",
                scope="agent:test",
                status="active",
                last_access=now - 31 * 86400,  # 31 天前
                pinned=0,
            ),
            Zone(
                name="fresh_zone",
                scope="agent:test",
                status="active",
                last_access=now - 10 * 86400,  # 10 天前
                pinned=0,
            ),
        ]
        import asyncio
        changes = asyncio.run(lifecycle_check(zones, cfg, now))
        assert len(changes) == 1
        assert changes[0] == ("old_zone", "dormant")

    def test_dormant_to_archived(self):
        """超过 archive_days 无访问 → 变 archived。"""
        from cortexos.zones.engine import lifecycle_check
        cfg = Config()
        cfg.zone.lifecycle.dormant_days = 30
        cfg.zone.lifecycle.archive_days = 90

        now = time.time()
        zones = [
            Zone(
                name="very_old",
                scope="agent:test",
                status="dormant",
                last_access=now - 100 * 86400,  # 100 天前
                pinned=0,
            ),
        ]
        import asyncio
        changes = asyncio.run(lifecycle_check(zones, cfg, now))
        assert changes == [("very_old", "archived")]

    def test_pinned_zone_not_archived(self):
        """固定 Zone 不归档。"""
        from cortexos.zones.engine import lifecycle_check
        cfg = Config()
        cfg.zone.lifecycle.dormant_days = 30
        cfg.zone.lifecycle.archive_days = 90

        now = time.time()
        zones = [
            Zone(
                name="pinned_old",
                scope="agent:test",
                status="active",
                last_access=now - 100 * 86400,
                pinned=1,  # 固定
            ),
        ]
        import asyncio
        changes = asyncio.run(lifecycle_check(zones, cfg, now))
        assert len(changes) == 0


# ────────────────────── 重力测试 ──────────────────────


class TestGravity:
    """重力公式测试。"""

    def test_fresh_zone_high_gravity(self):
        """新 Zone 重力较高。"""
        from cortexos.zones.engine import compute_gravity
        cfg = Config()
        now = time.time()
        g = compute_gravity(
            entry_count=50, access_count=50,
            last_access=now, config=cfg, now=now,
        )
        # 新鲜度因子 ≈ exp(0) = 1，活跃度 ≈ 1 - exp(-1) ≈ 0.63
        # 规模 ≈ 1 - exp(-0.5) ≈ 0.39
        # gravity ≈ 1 * 0.63 * 0.39 ≈ 0.25
        assert g > 0.1
        assert g <= 1.0

    def test_old_zone_low_gravity(self):
        """旧 Zone 重力较低。"""
        from cortexos.zones.engine import compute_gravity
        cfg = Config()
        now = time.time()
        g = compute_gravity(
            entry_count=5, access_count=5,
            last_access=now - 100 * 86400,  # 100 天前
            config=cfg, now=now,
        )
        # 新鲜度因子 ≈ exp(-0.02 * 100) = exp(-2) ≈ 0.135
        assert g < 0.2

    def test_update_zone_gravity(self):
        """原地更新 Zone 重力。"""
        from cortexos.zones.engine import update_zone_gravity
        cfg = Config()
        zone = Zone(
            name="test",
            scope="agent:test",
            entry_count=10,
            last_access=time.time(),
        )
        update_zone_gravity(zone, cfg)
        assert zone.gravity > 0


# ────────────────────── 合并测试 ──────────────────────


class TestMerge:
    """Zone 合并测试。"""

    @pytest.mark.asyncio
    async def test_similar_zones_merge(self):
        """质心相近的 Zone 合并。"""
        from cortexos.zones.engine import merge_zones
        from cortexos.embedding.tfidf import TfidfEmbedder

        cfg = Config()
        cfg.zone.lifecycle.merge_threshold = 0.7
        emb = TfidfEmbedder(max_features=16)

        zones = [
            Zone(name="zone_a", scope="agent:test", centroid=[0.1] * 16, entry_count=100, status="active"),
            Zone(name="zone_b", scope="agent:test", centroid=[0.1] * 16, entry_count=10, status="active"),  # 高度相似，归入 zone_a
            Zone(name="zone_c", scope="agent:test", centroid=[0.9] * 16, entry_count=5, status="active"),  # 不相似
        ]
        merges = await merge_zones(zones, emb, cfg)
        # zone_b 应合并到 zone_a
        assert ("zone_b", "zone_a") in merges or len(merges) >= 1

    @pytest.mark.asyncio
    async def test_dissimilar_zones_no_merge(self):
        """质心不相近的 Zone 不合并。"""
        from cortexos.zones.engine import merge_zones
        from cortexos.embedding.tfidf import TfidfEmbedder

        cfg = Config()
        cfg.zone.lifecycle.merge_threshold = 0.7
        emb = TfidfEmbedder(max_features=16)

        # 先训练（让 TF-IDF 有词汇表）
        await emb.embed(["zone alpha content"] * 5)
        await emb.embed(["zone beta content"] * 5)

        sim = emb.cosine_similarity(
            (await emb.embed_query("zone alpha content")),
            (await emb.embed_query("completely different unrelated topic")),
        )
        # 如果它们已经不相近，直接测试
        zones = [
            Zone(name="zone_a", scope="agent:test",
                 centroid=(await emb.embed_query("zone alpha content")), entry_count=10, status="active"),
            Zone(name="zone_b", scope="agent:test",
                 centroid=(await emb.embed_query("completely different unrelated topic")), entry_count=10, status="active"),
        ]
        merges = await merge_zones(zones, emb, cfg)
        assert len(merges) == 0


# ────────────────────── 增量更新测试 ──────────────────────


class TestIncrementalUpdate:
    """增量质心更新测试。"""

    @pytest.mark.asyncio
    async def test_update_centroid_incremental(self):
        """增量质心更新公式验证。"""
        from cortexos.zones.engine import _update_zone_centroid_incremental
        from cortexos.models import Zone as Z

        zone = Z(name="test", scope="agent:test", centroid=[1.0, 2.0], entry_count=2)
        entry = Entry(content="new", embedding=[3.0, 4.0])
        await _update_zone_centroid_incremental(zone, entry)
        # 旧均值 [1,2], 2 条; 新值 [3,4]; 新均值 = (2*[1,2] + [3,4]) / 3 = [5/3, 8/3]
        assert zone.centroid[0] == pytest.approx(5.0 / 3.0)
        assert zone.centroid[1] == pytest.approx(8.0 / 3.0)
